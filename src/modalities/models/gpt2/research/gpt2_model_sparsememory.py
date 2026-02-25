import logging
import math
from abc import abstractmethod
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


# ==============================================================================
# CONFIG
# ==============================================================================


class AdaptiveComputationConfig(BaseModel):
    enable_adaptive: bool = False
    max_loops: int = 10
    halt_threshold: float = 0.99
    ponder_penalty_weight: float = 0.01
    # --- CHANGED: simplified memory config ---
    use_memory_bank: bool = True
    num_slots: int = 16384       # total knowledge slots
    top_k: int = 32              # how many to retrieve per token
    scheduler_type: str = "constant"
    frozen_gate: Optional[float] = None
    uses_new_names: Optional[bool] = True


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


# ==============================================================================
# ATTENTION COMPONENTS (unchanged)
# ==============================================================================


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


# ==============================================================================
# ATTENTION (unchanged)
# ==============================================================================


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
        self.resid_dropout = nn.Dropout(dropout)
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
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None, dropout_p=dropout, is_causal=True)
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


# ==============================================================================
# MLP (unchanged)
# ==============================================================================


class TransformerMLP(nn.Module):
    def __init__(self, n_embd, ffn_hidden, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ffn_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ==============================================================================
# TRANSFORMER BLOCK (unchanged)
# ==============================================================================


class GPT2Block(nn.Module):
    def __init__(
        self, n_embd, bias, n_head_q, n_head_kv, activation_type,
        attention_impl, attention_config, dropout, ffn_hidden,
        attention_norm, ffn_norm, enforce_swiglu_hidden_dim_multiple_of,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self._check_ffn_hidden_dim(n_embd, ffn_hidden)
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

    def _check_ffn_hidden_dim(self, n_embd, ffn_hidden):
        expected = 4 * n_embd
        if ffn_hidden != expected:
            logger.warning(f"Expected ffn_hidden={expected}, got n_embd={n_embd}, ffn_hidden={ffn_hidden}.")

    def forward(self, x, scale=1.0):
        x = x + scale * self.attn(self.attention_norm(x))
        x = x + scale * self.mlp(self.ffn_norm(x))
        return x


# ==============================================================================
# --- NEW: SPARSE MEMORY BANK ---
# ==============================================================================


class SparseMemoryBank(nn.Module):
    """
    Learnable key-value store with top-k sparse retrieval.
    FSDP-compatible: all parameter init happens in reset_parameters().

    Components:
      - keys/values: the learnable "knowledge" (nn.Parameter)
      - q_norm: normalizes the query for stable dot-product scores
      - k_norm: normalizes keys for stable dot-product scores
      - gate_proj: scalar sigmoid gate so the model can learn to ignore memory
    """

    def __init__(self, num_slots: int, n_embd: int, top_k: int = 32):
        super().__init__()
        self.num_slots = num_slots
        self.n_embd = n_embd
        self.top_k = top_k
        self.scale = n_embd ** -0.5

        # Knowledge storage
        self.keys = nn.Parameter(torch.empty(num_slots, n_embd))
        self.values = nn.Parameter(torch.empty(num_slots, n_embd))

        # Norms for stable similarity scoring
        self.q_norm = nn.LayerNorm(n_embd)
        self.k_norm = nn.LayerNorm(n_embd)

        # Scalar gate: input-dependent, per-token, starts near-closed
        self.gate_proj = nn.Linear(n_embd, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Small normal init keeps keys distinct, avoids saturation
        nn.init.normal_(self.keys, mean=0.0, std=0.02)
        nn.init.normal_(self.values, mean=0.0, std=0.02)
        # Gate starts near-closed so memory doesn't disrupt early training
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] hidden states (used as both query and gate input)
        Returns:
            out: [B, T, D] gated memory contribution (add this to residual)
            gate_avg: scalar, mean gate value for logging
        """
        B, T, D = x.shape

        # 1. Normalize query and keys for stable scoring
        q = self.q_norm(x)                              # [B, T, D]
        k = self.k_norm(self.keys)                      # [N, D]

        # 2. Score query against all keys
        scores = torch.einsum("btd,nd->btn", q, k) * self.scale  # [B, T, N]

        # 3. Top-K selection (the sparse part)
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)    # [B, T, K]

        # 4. Softmax over only the top-K
        weights = F.softmax(top_scores, dim=-1)                   # [B, T, K]

        # 5. Gather corresponding values and weighted-sum
        top_vals = self.values[top_idx]                           # [B, T, K, D]
        retrieved = (weights.unsqueeze(-1) * top_vals).sum(dim=2) # [B, T, D]

        # 6. Gate: let the model learn how much memory to inject
        gate = torch.sigmoid(self.gate_proj(x))                  # [B, T, 1]
        out = gate * retrieved                                    # [B, T, D]

        return out, gate.mean()


# ==============================================================================
# ADAPTIVE COMPUTATION COMPONENTS
# ==============================================================================


class AdaptiveRouter(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.net = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, x: torch.Tensor, step_normalized: float) -> torch.Tensor:
        B, T, _ = x.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=x.device, dtype=x.dtype)
        return torch.sigmoid(self.net(torch.cat([x, step_feat], dim=-1))).squeeze(-1)


# ==============================================================================
# --- CHANGED: SIMPLIFIED ADAPTIVE RECURSIVE BLOCK ---
# ==============================================================================


class AdaptiveRecursiveBlock(nn.Module):
    def __init__(
        self,
        block: GPT2Block,
        adaptive_config: AdaptiveComputationConfig,
        n_embd: int,
        layer_idx: int,
        memory_bank: Optional[SparseMemoryBank],
    ):
        super().__init__()
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.layer_idx = layer_idx
        self.memory_bank = memory_bank

        self.halt_router = AdaptiveRouter(n_embd)
        self.loop_scales = nn.Parameter(torch.full((self.max_loops,), -7.0))

        # --- CHANGED: single norm before memory query, no separate local/global ---
        if self.memory_bank is not None:
            self.mem_norm = nn.LayerNorm(n_embd)
        else:
            self.mem_norm = None

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        device = x.device

        h = x
        accumulated_output = torch.zeros_like(h)
        prob_remain = torch.ones(B, T, device=device, dtype=x.dtype)
        expected_steps = torch.zeros(B, T, device=device, dtype=x.dtype)
        total_cos_sim = torch.zeros(B, T, device=device, dtype=x.dtype)

        halt_probs_list = []
        gate_avgs = []
        denom = max(1, self.max_loops - 1)

        for step in range(self.max_loops):
            # --- CHANGED: single clean memory injection ---
            if self.memory_bank is not None:
                mem_query = self.mem_norm(h)
                mem_out, gate_avg = self.memory_bank(mem_query)
                h_enriched = h + mem_out
                gate_avgs.append(gate_avg)
            else:
                h_enriched = h

            # Transformer block
            scale = F.softplus(self.loop_scales[step])
            h_prev = h
            h_new = self.block(h_enriched, scale=scale)

            # Halting
            cos_sim = F.cosine_similarity(h_new, h_prev, dim=-1, eps=1e-8)
            halt_prob = self.halt_router(h_new, step / denom)
            halt_probs_list.append(halt_prob.detach().mean())

            if step == self.max_loops - 1:
                p_stop = prob_remain
                prob_remain = torch.zeros_like(prob_remain)
            else:
                p_stop = prob_remain * halt_prob
                prob_remain = prob_remain * (1.0 - halt_prob)

            accumulated_output = accumulated_output + h_new * p_stop.unsqueeze(-1)
            expected_steps = expected_steps + p_stop * (step + 1)
            total_cos_sim = total_cos_sim + cos_sim * p_stop
            h = h_new

            if not self.training and prob_remain.max() < 0.01:
                break

        if not self.training and prob_remain.sum() > 0:
            accumulated_output = accumulated_output + h * prob_remain.unsqueeze(-1)

        halt_probs_stacked = self._pad_to_max(halt_probs_list, device)
        avg_gate = torch.stack(gate_avgs).mean() if gate_avgs else torch.tensor(0.0, device=device)

        return (
            accumulated_output,
            expected_steps,
            total_cos_sim,
            self.loop_scales.detach(),
            halt_probs_stacked,
            avg_gate,
        )

    def _pad_to_max(self, lst, device):
        if not lst:
            return torch.zeros(self.max_loops, device=device)
        stacked = torch.stack(lst)
        if stacked.size(0) < self.max_loops:
            pad = torch.zeros(self.max_loops - stacked.size(0), device=device, dtype=stacked.dtype)
            stacked = torch.cat([stacked, pad])
        return stacked


# ==============================================================================
# --- CHANGED: MAIN MODEL ---
# ==============================================================================


class GPT2LLM(NNModel):
    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        sequence_length: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        attention_implementation: AttentionImplementation,
        attention_config: AttentionConfig,
        attention_norm_config: LayerNormWrapperConfig,
        ffn_norm_config: LayerNormWrapperConfig,
        lm_head_norm_config: LayerNormWrapperConfig,
        use_weight_tying: bool,
        seed: Optional[int] = None,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
        adaptive_config: Optional[AdaptiveComputationConfig] = None,
    ):
        weight_decay_groups = {
            "linear": [
                ".attn",
                ".mlp",
                ".lm_head.weight",
                ".halt_router.net.weight",
                ".gate_proj.weight",       # memory gate projection (linear)
            ],
            "embedding": [
                ".wte",
                ".wpe",
            ],
            "layernorm": [
                ".attention_norm",
                ".ffn_norm",
                ".lm_head_norm",
                ".mem_norm",               # pre-memory-query norm
                "memory_bank.q_norm",   # was ".q_norm"
                "memory_bank.k_norm",   # was ".k_norm"
                ".loop_scales",
                ".halt_router.net.bias",
                ".gate_proj.bias",
                ".keys",
                ".values",
                ".bias",
            ],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)

        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.poe_type = poe_type

        assert vocab_size is not None
        assert sequence_length is not None

        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=sequence_length, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention_config.qkv_transforms
        ]:
            raise ValueError('Use "RotaryTransform" together with "NOPE".')

        self.use_adaptive = adaptive_config is not None and adaptive_config.enable_adaptive
        self.adaptive_config = adaptive_config

        def create_block():
            return GPT2Block(
                n_embd=n_embd, bias=bias, n_head_q=n_head_q, n_head_kv=n_head_kv,
                activation_type=activation_type, attention_impl=attention_implementation,
                attention_config=attention_config, dropout=dropout, ffn_hidden=ffn_hidden,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )

        # --- CHANGED: single shared SparseMemoryBank ---
        if self.use_adaptive and adaptive_config.use_memory_bank:
            self.memory_bank = SparseMemoryBank(
                num_slots=adaptive_config.num_slots,
                n_embd=n_embd,
                top_k=adaptive_config.top_k,
            )
        else:
            self.memory_bank = None

        # Build layers
        layers = {}
        for layer_idx in range(n_layer):
            block = create_block()
            if self.use_adaptive:
                layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                    block=block,
                    adaptive_config=adaptive_config,
                    n_embd=n_embd,
                    layer_idx=layer_idx,
                    memory_bank=self.memory_bank,  # shared across all layers
                )
            else:
                layers[str(layer_idx)] = block

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=wpe,
                drop=nn.Dropout(dropout),
                h=nn.ModuleDict(layers),
                lm_head_norm=lm_head_norm_config.norm_type.value(**dict(lm_head_norm_config.config)),
                lm_head=nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False),
            )
        )
        if use_weight_tying:
            self.transformer.wte.weight = self.transformer.lm_head.weight

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...
    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def forward(self, inputs):
        if isinstance(inputs, dict):
            result = self.forward_impl(inputs[self.sample_key])
            if isinstance(result, dict):
                return {self.prediction_key: result}
            else:
                return {self.prediction_key: result}
        else:
            return self.forward_impl(inputs)

    def forward_impl(self, inputs: torch.Tensor):
        device = inputs.device
        seq_len = inputs.size(1)
        assert seq_len <= self.sequence_length

        h = self.transformer.wte(inputs)

        if self.poe_type is PositionTypes.ABSOLUTE and hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            h = h + self.transformer.wpe(pos)

        h = self.transformer.drop(h) if hasattr(self.transformer, "drop") else h

        # Tracking
        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)
        num_adaptive_layers = 0
        per_layer_ponder_costs = []
        per_layer_cos_sims = []
        per_layer_loop_scales = []
        per_layer_halt_probs = []
        per_layer_gate_avgs = []

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer_module = self.transformer.h[layer_key]
            layer_idx = int(layer_key)

            if self.use_adaptive:
                # --- CHANGED: 6 return values instead of 7 ---
                h, cost, cos_sim, scales, halt_probs, gate_avg = layer_module(h)

                cost_mean = cost.mean()
                sim_mean = cos_sim.mean()
                total_ponder_cost = total_ponder_cost + cost_mean
                per_layer_ponder_costs.append(cost_mean)
                per_layer_cos_sims.append(sim_mean)
                per_layer_loop_scales.append(scales)
                per_layer_halt_probs.append(halt_probs)
                per_layer_gate_avgs.append(gate_avg)
                num_adaptive_layers += 1
            else:
                lns_scale = 1.0 / (layer_idx + 1)
                h = layer_module(h, scale=lns_scale)

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        if self.use_adaptive:
            avg_ponder = total_ponder_cost / max(num_adaptive_layers, 1)
            max_loops = self.adaptive_config.max_loops
            normalized = (avg_ponder - 1.0) / (max_loops - 1.0) if max_loops > 1 else torch.tensor(0.0, dtype=h.dtype, device=device)
            ponder_loss = (normalized * self.adaptive_config.ponder_penalty_weight).to(logits.dtype)

            return {
                "logits": logits,
                "ponder_loss": ponder_loss,
                "ponder_cost_unweighted": total_ponder_cost,
                "expected_steps": avg_ponder,
                "normalized_steps": normalized,
                "per_layer_ponder_costs": torch.stack(per_layer_ponder_costs) if per_layer_ponder_costs else torch.tensor([], device=device),
                "per_layer_cos_sims": torch.stack(per_layer_cos_sims) if per_layer_cos_sims else torch.tensor([], device=device),
                "loop_scales": torch.stack(per_layer_loop_scales) if per_layer_loop_scales else torch.tensor([], device=device),
                "halt_probs": torch.stack(per_layer_halt_probs) if per_layer_halt_probs else torch.tensor([], device=device),
                "global_mem_scales": torch.stack(per_layer_gate_avgs) if per_layer_gate_avgs else torch.tensor([], device=device),
            }
        return logits


# ==============================================================================
# UTIL (unchanged)
# ==============================================================================


def manual_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
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