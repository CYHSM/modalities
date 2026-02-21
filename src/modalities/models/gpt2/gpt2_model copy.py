import logging
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Optional, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator, validator

from modalities.config.lookup_enum import LookupEnum
from modalities.config.utils import convert_base_model_config_to_dict
from modalities.models.components.layer_norms import (
    LayerNormConfig,
    PytorchRMSLayerNormConfig,
    RMSLayerNorm,
    RMSLayerNormConfig,
)
from modalities.models.model import ActivationType, NNModel, SwiGLU

from modalities.util import parse_enum_by_name


try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# =============================================================================
# Configs
# =============================================================================

def default_heterogeneous_experts() -> list["ExpertDefinition"]:
    return (
        [ExpertDefinition(ffn_hidden=1024, max_loops=1)] * 8 +
        [ExpertDefinition(ffn_hidden=512, max_loops=2)] * 4 +
        [ExpertDefinition(ffn_hidden=128, max_loops=8)] * 2
    )


# def default_heterogeneous_experts() -> list["ExpertDefinition"]:
#     return [
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=4096, max_loops=1),
#         ExpertDefinition(ffn_hidden=2048, max_loops=2),
#         ExpertDefinition(ffn_hidden=2048, max_loops=2),
#         ExpertDefinition(ffn_hidden=1024, max_loops=4),
#         ExpertDefinition(ffn_hidden=1024, max_loops=4),
#         ExpertDefinition(ffn_hidden=512, max_loops=8),
#         ExpertDefinition(ffn_hidden=512, max_loops=8),
#     ]

class ExpertDefinition(BaseModel):
    """Defines a single expert: just an MLP width + loop count."""
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    max_loops: Annotated[int, Field(strict=True, ge=1)] = 1

class AdaptiveComputationConfig(BaseModel):
    """Configuration for the shared-expert Mixture-of-Experts system."""
    experts: list[ExpertDefinition] = Field(default_factory=default_heterogeneous_experts)
    top_k: Annotated[int, Field(strict=True, ge=1)] = 1
    load_balance_weight: Annotated[float, Field(ge=0.0)] = 0.01
    ponder_penalty_weight: float = 0.01
    scheduler_type: str = "constant"

    @model_validator(mode="after")
    def check_top_k(self) -> "AdaptiveComputationConfig":
        if self.top_k > len(self.experts):
            raise ValueError(f"top_k ({self.top_k}) > num experts ({len(self.experts)})")
        return self

class LayerNorms(LookupEnum):
    rms_norm = RMSLayerNorm
    layer_norm = nn.LayerNorm
    pytorch_rms_norm = nn.RMSNorm


class LayerNormWrapperConfig(BaseModel):
    norm_type: LayerNorms
    config: PytorchRMSLayerNormConfig | RMSLayerNormConfig | LayerNormConfig


class PositionTypes(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    NOPE = "NOPE"


# =============================================================================
# QKV Transforms (unchanged)
# =============================================================================


class QueryKeyValueTransform(nn.Module):
    @abstractmethod
    def forward(self, q, k, v):
        raise NotImplementedError


class IdentityTransform(QueryKeyValueTransform):
    def forward(self, q, k, v):
        return q, k, v


class RotaryTransform(QueryKeyValueTransform):
    def __init__(self, n_embd: int, n_head: int, seq_length_dim: int = -2, base_freq: int = 10000):
        super().__init__()
        self.dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        self.base_freq = base_freq
        self.reset_parameters()

    def reset_parameters(self):
        device = self.inv_freq.device if hasattr(self, "inv_freq") else None
        inv_freq = 1.0 / (
            self.base_freq ** (torch.arange(0, self.dim_model, 2, device=device).float() / self.dim_model)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update_cos_sin_tables(self, x):
        seq_len = x.shape[self.seq_length_dim]
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_length_dim], device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)
        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[self.seq_length_dim], :]
        sin = sin[:, :, : x.shape[self.seq_length_dim], :]
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, q, k, v):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        q = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)
        return q, k, v


class QueryKeyValueTransformType(Enum):
    IdentityTransform = IdentityTransform
    RotaryTransform = RotaryTransform


class AttentionImplementation(str, Enum):
    MANUAL = "manual"
    PYTORCH_FLASH = "pytorch_flash"
    DAO_FLASH = "dao_flash"


class AttentionConfig(BaseModel):
    class QueryKeyValueTransformConfig(BaseModel):
        class IdentityTransformConfig(BaseModel):
            pass

        class RotaryTransformConfig(BaseModel):
            n_embd: Annotated[int, Field(strict=True, ge=0)]
            n_head: Annotated[int, Field(strict=True, ge=0)]
            seq_length_dim: Annotated[int, Field(strict=True)]
            base_freq: Annotated[int, Field(strict=True, ge=10000)]

        @validator("type_hint", pre=True, always=True)
        def parse_sharding_strategy_by_name(cls, name):
            return parse_enum_by_name(name=name, enum_type=QueryKeyValueTransformType)

        type_hint: QueryKeyValueTransformType
        config: RotaryTransformConfig | IdentityTransformConfig

    qkv_transforms: list[QueryKeyValueTransformConfig]
    qk_norm_config: Optional[LayerNormWrapperConfig] = None


class GPT2LLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    use_meta_device: Optional[bool] = False
    poe_type: PositionTypes
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[int, Field(strict=True, ge=1)]
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool
    attention_config: AttentionConfig
    attention_implementation: AttentionImplementation
    activation_type: ActivationType
    attention_norm_config: LayerNormWrapperConfig
    ffn_norm_config: LayerNormWrapperConfig
    lm_head_norm_config: LayerNormWrapperConfig
    use_weight_tying: bool
    seed: Optional[int] = None
    enforce_swiglu_hidden_dim_multiple_of: int = 256
    adaptive_config: Optional[AdaptiveComputationConfig] = None

    @model_validator(mode="after")
    def check_divisibility(self) -> "GPT2LLMConfig":
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError("n_head_q must be divisible by n_head_kv")
        return self

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd],
            ["ffn_hidden", "vocab_size", "n_embd"],
        ):
            if param % 128 != 0:
                raise ValueError(f"{param_name} with value {param} should be divisible by 128.")
        return self


# =============================================================================
# Attention (unchanged)
# =============================================================================


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head_q, n_head_kv, n_embd, attention_config, attention_impl, bias, dropout):
        super().__init__()
        assert n_embd % n_head_q == 0
        assert n_head_q % n_head_kv == 0

        self.n_rep = n_head_q // n_head_kv
        self.attention_impl = attention_impl

        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_attn = nn.Linear(n_embd, n_embd // self.n_rep, bias=bias)
        self.v_attn = nn.Linear(n_embd, n_embd // self.n_rep, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv
        self.n_embd = n_embd
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(self.dropout)

        self.qkv_transforms = nn.ModuleList(
            transform_config.type_hint.value(
                **convert_base_model_config_to_dict(transform_config.config)
            )
            for transform_config in attention_config.qkv_transforms
        )

        if attention_config.qk_norm_config is not None:
            self.q_norm = attention_config.qk_norm_config.norm_type.value(**dict(attention_config.qk_norm_config.config))
            self.k_norm = attention_config.qk_norm_config.norm_type.value(**dict(attention_config.qk_norm_config.config))
        else:
            self.q_norm = None
            self.k_norm = None

    def projection(self, x):
        return self.q_attn(x), self.k_attn(x), self.v_attn(x)

    @staticmethod
    def execute_qkv_transforms(q, k, v, qkv_transforms, n_head_q):
        B, T, D = q.size()
        n_head_dim = D // n_head_q
        q = q.view(B, T, n_head_q, n_head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, -1, n_head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, -1, n_head_dim).transpose(1, 2).contiguous()
        for transform in qkv_transforms:
            q, k, v = transform(q, k, v)
        return q, k, v

    @staticmethod
    def _repeat_kv(x, n_rep):
        B, nh_kv, T, hs = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    @classmethod
    def repeat_kv_heads(cls, q, k, v):
        n_head_q = q.shape[1]
        n_head_kv = k.shape[1]
        if n_head_q != n_head_kv:
            n_rep = n_head_q // n_head_kv
            k = cls._repeat_kv(k, n_rep)
            v = cls._repeat_kv(v, n_rep)
        return k, v

    @classmethod
    def execute_attention(cls, q, k, v, dropout, attention_impl):
        if attention_impl == AttentionImplementation.MANUAL:
            k, v = cls.repeat_kv_heads(q, k, v)
            y = manual_scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None, dropout_p=dropout, is_causal=True)
            y = y.transpose(1, 2).contiguous()
        elif attention_impl == AttentionImplementation.PYTORCH_FLASH:
            k, v = cls.repeat_kv_heads(q, k, v)
            y = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None, dropout_p=dropout, is_causal=True)
            y = y.transpose(1, 2).contiguous()
        elif attention_impl == AttentionImplementation.DAO_FLASH:
            if flash_attn_func is None:
                raise NotImplementedError("Dao Flash Attention is not installed.")
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            y = flash_attn_func(q, k, v, dropout_p=dropout, causal=True, softmax_scale=None, window_size=(-1, -1))
        else:
            raise NotImplementedError(f"Attention implementation {attention_impl} not supported")
        return y

    def forward(self, x):
        B, T, _ = x.size()
        q, k, v = self.projection(x)
        q, k, v = CausalSelfAttention.execute_qkv_transforms(q, k, v, self.qkv_transforms, self.n_head_q)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        y = CausalSelfAttention.execute_attention(q, k, v, self.dropout, self.attention_impl)
        y = y.reshape(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


# =============================================================================
# MLP Building Blocks (unchanged, reused by both GPT2Block and Expert)
# =============================================================================


class TransformerMLP(nn.Module):
    def __init__(self, n_embd, ffn_hidden, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ffn_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# =============================================================================
# GPT2Block — used only in the non-MoE fallback path
# =============================================================================


class GPT2Block(nn.Module):
    def __init__(
        self, n_embd, bias, n_head_q, n_head_kv, activation_type, attention_impl,
        attention_config, dropout, ffn_hidden, attention_norm, ffn_norm,
        enforce_swiglu_hidden_dim_multiple_of,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q, n_head_kv=n_head_kv, n_embd=n_embd,
            attention_config=attention_config, attention_impl=attention_impl,
            bias=bias, dropout=dropout,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(
                n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
        else:
            raise NotImplementedError("unimplemented activation")

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# =============================================================================
# Shared Expert MoE Components
# =============================================================================


class Expert(nn.Module):
    """A single expert: an MLP applied `max_loops` times with residual connections.

    Same architecture regardless of role — capacity experts use a wide FFN with
    max_loops=1, compute experts use a narrow FFN with max_loops>1.
    The expert returns the *delta* (accumulated MLP contributions), not h + input.
    """

    def __init__(
        self,
        n_embd: int,
        ffn_hidden: int,
        max_loops: int,
        bias: bool,
        dropout: float,
        activation_type: ActivationType,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
    ):
        super().__init__()
        self.max_loops = max_loops
        self.expert_norm = nn.RMSNorm(n_embd)

        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(
                n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
        else:
            raise NotImplementedError(f"Unsupported activation: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP `max_loops` times with residual, return delta only.

        Args:
            x: input tensor, shape (..., D). Works with any leading dims
               since MLP is token-independent.

        Returns:
            delta: the accumulated change, shape (..., D).
                   Caller adds this to the residual stream.
        """
        h = x
        for _ in range(self.max_loops):
            h = h + self.mlp(self.expert_norm(h))
        return h - x


class SharedExpertPool(nn.Module):
    """Pool of experts shared across all layers. Instantiated once, referenced by all."""

    def __init__(
        self,
        expert_configs: list[ExpertDefinition],
        n_embd: int,
        bias: bool,
        dropout: float,
        activation_type: ActivationType,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(
                n_embd=n_embd,
                ffn_hidden=cfg.ffn_hidden,
                max_loops=cfg.max_loops,
                bias=bias,
                dropout=dropout,
                activation_type=activation_type,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
            for cfg in expert_configs
        ])

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.experts[expert_idx](x)


class ExpertRouter(nn.Module):
    """Per-layer lightweight router. Maps each token to expert logits."""

    def __init__(self, n_embd: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — normalized hidden states.
        Returns:
            logits: (B, T, num_experts) — raw routing scores.
        """
        return self.gate(x)


def moe_dispatch(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    expert_pool: SharedExpertPool,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Route tokens to experts via top-k gating.

    Args:
        x:             (B, T, D) — hidden states (unnormed; experts handle their own norms).
        router_logits: (B, T, E) — raw logits from the per-layer router.
        expert_pool:   the shared expert pool.
        top_k:         number of experts each token is sent to.

    Returns:
        output:   (B, T, D) — weighted sum of expert deltas.
        aux_loss: scalar — load-balancing loss.
        metrics:  dict with routing diagnostics.
    """
    B, T, D = x.shape
    E = expert_pool.num_experts

    # Flatten to (N, D) and (N, E) for dispatch
    x_flat = x.reshape(B * T, D)
    logits_flat = router_logits.reshape(B * T, E)

    # Gating
    gates = F.softmax(logits_flat, dim=-1)                       # (N, E)
    top_values, top_indices = gates.topk(top_k, dim=-1)          # (N, k), (N, k)

    # Renormalize selected gate values so they sum to 1 per token
    if top_k > 1:
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)

    # Dispatch: gather tokens per expert, run expert, scatter back
    output_flat = torch.zeros_like(x_flat)

    for k_idx in range(top_k):
        indices_k = top_indices[:, k_idx]                        # (N,)
        weights_k = top_values[:, k_idx]                         # (N,)

        for e_idx in range(E):
            mask = indices_k == e_idx                             # (N,)
            if not mask.any():
                continue
            expert_input = x_flat[mask]                           # (n_tokens, D)
            expert_delta = expert_pool(e_idx, expert_input)       # (n_tokens, D)
            output_flat[mask] += weights_k[mask].unsqueeze(-1) * expert_delta

    output = output_flat.reshape(B, T, D)

    # ---- Load-balancing auxiliary loss (Switch Transformer style) ----
    # f_e = fraction of tokens routed to expert e (across top-k selections)
    # P_e = mean gate probability for expert e
    # L_balance = E * sum_e(f_e * P_e)
    #
    # This encourages uniform utilization without fighting the desired
    # capacity/compute specialization too aggressively.
    with torch.no_grad():
        # Count how often each expert appears in any top-k slot
        expert_counts = torch.zeros(E, device=x.device)
        for k_idx in range(top_k):
            for e_idx in range(E):
                expert_counts[e_idx] += (top_indices[:, k_idx] == e_idx).float().sum()
        f = expert_counts / (B * T * top_k)  # fraction of total assignments

    P = gates.mean(dim=0)  # (E,) — mean probability per expert (has grad)
    aux_loss = E * (f * P).sum()

    # ---- Metrics ----
    with torch.no_grad():
        # Per-expert token fractions for logging
        expert_load = expert_counts / (B * T * top_k)
        # Router entropy (higher = more uniform routing)
        router_entropy = -(gates * (gates + 1e-9).log()).sum(dim=-1).mean()

    metrics = {
        "expert_load": expert_load,           # (E,)
        "router_entropy": router_entropy,     # scalar
        "aux_loss": aux_loss.detach(),        # scalar
    }

    return output, aux_loss, metrics


# =============================================================================
# MoE Transformer Layer
# =============================================================================


class MoETransformerLayer(nn.Module):
    """Transformer layer: per-layer attention + routing to shared experts.

    Per-layer parameters:  attention weights, attention_norm, ffn_norm, router.
    Shared parameters:     expert pool (passed during forward, not owned).

    The expert pool is NOT stored as a submodule here to avoid duplicate
    registration in the state dict — it's owned once by GPT2LLM.
    """

    def __init__(
        self,
        n_embd: int,
        bias: bool,
        n_head_q: int,
        n_head_kv: int,
        attention_impl: AttentionImplementation,
        attention_config: AttentionConfig,
        dropout: float,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.top_k = top_k

        self.attn = CausalSelfAttention(
            n_head_q=n_head_q, n_head_kv=n_head_kv, n_embd=n_embd,
            attention_config=attention_config, attention_impl=attention_impl,
            bias=bias, dropout=dropout,
        )

        # Lightweight per-layer router
        self.router = ExpertRouter(n_embd, num_experts)

    def forward(
        self, x: torch.Tensor, expert_pool: SharedExpertPool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, T, D)
            expert_pool: shared expert pool (owned by GPT2LLM, passed here).
        Returns:
            x: (B, T, D) — updated hidden states.
            metrics: routing diagnostics for this layer.
        """
        # 1) Self-attention (per-layer params)
        x = x + self.attn(self.attention_norm(x))

        # 2) Route to shared experts
        x_normed = self.ffn_norm(x)
        router_logits = self.router(x_normed)
        expert_delta, aux_loss, metrics = moe_dispatch(
            x, router_logits, expert_pool, self.top_k,
        )
        x = x + expert_delta

        metrics["aux_loss_raw"] = aux_loss
        return x, metrics


# =============================================================================
# GPT2LLM — Main Model
# =============================================================================


class GPT2LLM(NNModel):
    def __init__(
        self, sample_key, prediction_key, poe_type, sequence_length, vocab_size,
        n_layer, n_head_q, n_head_kv, n_embd, ffn_hidden, dropout, bias,
        activation_type, attention_implementation, attention_config,
        attention_norm_config, ffn_norm_config, lm_head_norm_config,
        use_weight_tying, seed=None, enforce_swiglu_hidden_dim_multiple_of=256,
        adaptive_config=None,
    ):
        weight_decay_groups = {
            "linear": [
                ".attn", ".mlp", ".lm_head.weight",
                ".router.gate.weight",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm", ".expert_norm",
            ],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)

        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.poe_type = poe_type
        self.use_moe = adaptive_config is not None
        self.adaptive_config = adaptive_config
        print(self.adaptive_config)

        assert vocab_size is not None
        assert sequence_length is not None

        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(sequence_length, n_embd)
        elif poe_type is PositionTypes.NOPE:
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            c.type_hint.value for c in attention_config.qkv_transforms
        ]:
            raise ValueError('Use "RotaryTransform" together with "NOPE".')

        # ---- Build layers ----

        def make_norm(norm_config):
            return norm_config.norm_type.value(**dict(norm_config.config))

        if self.use_moe:
            # Shared expert pool — instantiated once
            self.expert_pool = SharedExpertPool(
                expert_configs=adaptive_config.experts,
                n_embd=n_embd,
                bias=bias,
                dropout=dropout,
                activation_type=activation_type,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )

            layers = {}
            for layer_idx in range(n_layer):
                layers[str(layer_idx)] = MoETransformerLayer(
                    n_embd=n_embd, bias=bias,
                    n_head_q=n_head_q, n_head_kv=n_head_kv,
                    attention_impl=attention_implementation,
                    attention_config=attention_config,
                    dropout=dropout,
                    attention_norm=make_norm(attention_norm_config),
                    ffn_norm=make_norm(ffn_norm_config),
                    num_experts=self.expert_pool.num_experts,
                    top_k=adaptive_config.top_k,
                )
        else:
            self.expert_pool = None
            layers = {}
            for layer_idx in range(n_layer):
                layers[str(layer_idx)] = GPT2Block(
                    n_embd=n_embd, bias=bias, n_head_q=n_head_q, n_head_kv=n_head_kv,
                    activation_type=activation_type, attention_impl=attention_implementation,
                    attention_config=attention_config, dropout=dropout, ffn_hidden=ffn_hidden,
                    attention_norm=make_norm(attention_norm_config),
                    ffn_norm=make_norm(ffn_norm_config),
                    enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
                )

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=wpe,
            drop=nn.Dropout(dropout),
            h=nn.ModuleDict(layers),
            lm_head_norm=make_norm(lm_head_norm_config),
            lm_head=nn.Linear(n_embd, vocab_size, bias=False),
        ))

        if use_weight_tying:
            self.transformer.wte.weight = self.transformer.lm_head.weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def forward(self, inputs):
        if isinstance(inputs, dict):
            result = self.forward_impl(inputs[self.sample_key])
            return {self.prediction_key: result} if isinstance(result, dict) else {self.prediction_key: result}
        return self.forward_impl(inputs)

    def forward_impl(self, inputs: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        device = inputs.device
        seq_len = inputs.size(1)
        assert seq_len <= self.sequence_length

        h = self.transformer.wte(inputs)

        if self.poe_type is PositionTypes.ABSOLUTE and hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            h = h + self.transformer.wpe(pos)

        h = self.transformer.drop(h)

        # ---- Layer loop ----
        all_layer_metrics: list[dict[str, torch.Tensor]] = []
        total_aux_loss = torch.tensor(0.0, device=device, dtype=h.dtype)

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer = self.transformer.h[layer_key]

            if self.use_moe:
                h, layer_metrics = layer(h, self.expert_pool)
                total_aux_loss = total_aux_loss + layer_metrics["aux_loss_raw"]
                all_layer_metrics.append(layer_metrics)
            else:
                h = layer(h)

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        if not self.use_moe:
            return logits

        # ---- Aggregate MoE metrics ----
        avg_aux_loss = total_aux_loss / self.n_layer
        weighted_aux_loss = avg_aux_loss * self.adaptive_config.load_balance_weight

        metrics_bag = {
            "scalars": {
                "aux_loss": avg_aux_loss.detach(),
                "router_entropy": torch.stack(
                    [m["router_entropy"] for m in all_layer_metrics]
                ).mean(),
            },
            "per_layer_vectors": {
                "expert_load": torch.stack(
                    [m["expert_load"] for m in all_layer_metrics]
                ),  # (L, E)
            },
        }

        return {
            "logits": logits,
            "ponder_loss": weighted_aux_loss,
            "metrics": metrics_bag,
        }


# =============================================================================
# Manual attention fallback (unchanged)
# =============================================================================


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value