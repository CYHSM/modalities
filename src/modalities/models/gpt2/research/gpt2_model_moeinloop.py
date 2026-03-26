import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Optional, overload

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

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
# Metrics Convention
# =============================================================================
#
# MetricsBag = {
#     "scalars":            dict[str, Tensor],   # shape ()    — accumulated & reduced
#     "per_layer_scalars":  dict[str, Tensor],   # shape (L,)  — accumulated & reduced
#     "per_layer_vectors":  dict[str, Tensor],   # shape (L,M) — last-batch snapshot only
# }
# =============================================================================


# =============================================================================
# Configs
# =============================================================================

class AdaptiveComputationConfig(BaseModel):
    max_loops: int = 10
    halt_threshold: float = 0.99
    ponder_penalty_weight: float = 0.00
    scheduler_type: str = "constant"

    # MoE config
    n_experts: int = 8
    top_k: int = 2
    expert_ffn_hidden: Optional[int] = None
    moe_bias_update_speed: float = 0.001
    moe_execution: str = "loop"


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
# QKV Transforms
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
            if param % 64 != 0:
                raise ValueError(f"{param_name} with value {param} should be divisible by 64.")
        return self


# =============================================================================
# Attention
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
# Mixture of Experts
# =============================================================================

class ExpertRouter(nn.Module):
    def __init__(self, n_embd: int, n_experts: int, top_k: int = 2, bias_update_speed: float = 0.001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_update_speed = bias_update_speed

        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("expert_counts", torch.zeros(n_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        logits = self.gate(x)
        biased_logits = logits + self.expert_bias.to(logits.dtype)
        top_k_biased, top_k_indices = biased_logits.topk(self.top_k, dim=-1)
        top_k_logits = logits.gather(-1, top_k_indices)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        if self.training:
            with torch.no_grad():
                flat_indices = top_k_indices.reshape(-1)
                counts = torch.zeros(self.n_experts, device=x.device)
                counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=counts.dtype))
                self.expert_counts += counts
                self.total_tokens += B * T

        return top_k_indices, top_k_weights, logits

    @torch.no_grad()
    def update_bias(self):
        if self.total_tokens == 0:
            return
        avg_count = self.total_tokens * self.top_k / self.n_experts
        relative_usage = self.expert_counts / (avg_count + 1e-8)
        self.expert_bias -= self.bias_update_speed * (relative_usage - 1.0)
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class MoEFFN(nn.Module):
    def __init__(
        self,
        n_embd: int,
        expert_ffn_hidden: int,
        n_experts: int = 8,
        top_k: int = 2,
        bias: bool = False,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
        execution: str = "loop",
        bias_update_speed: float = 0.001,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_embd = n_embd
        self.execution = execution

        self.router = ExpertRouter(
            n_embd=n_embd, n_experts=n_experts, top_k=top_k,
            bias_update_speed=bias_update_speed,
        )

        self.experts = nn.ModuleList([
            SwiGLU(
                n_embd=n_embd, ffn_hidden=expert_ffn_hidden, bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        top_k_indices, top_k_weights, router_logits = self.router(x)

        if self.execution == "loop":
            output = self._forward_loop(x, top_k_indices, top_k_weights)
        else:
            output = self._forward_scatter(x, top_k_indices, top_k_weights)

        with torch.no_grad():
            expert_usage = torch.zeros(self.n_experts, device=x.device)
            expert_usage.scatter_add_(
                0, top_k_indices.reshape(-1),
                torch.ones(B * T * self.top_k, device=x.device),
            )
            expert_usage = expert_usage / (B * T)

            metrics = {
                "expert_usage": expert_usage,
                "router_entropy": -(F.softmax(router_logits, -1) * F.log_softmax(router_logits, -1)).sum(-1).mean(),
                "top1_expert_frac": expert_usage.max() / expert_usage.sum(),
                "expert_bias_std": self.router.expert_bias.std(),
                "top_k_indices": top_k_indices,  # (B, T, top_k) — needed for depth/capacity analysis
            }

        return output, metrics

    def _forward_loop(self, x, top_k_indices, top_k_weights):
        B, T, D = x.shape
        output = torch.zeros_like(x)
        for expert_idx in range(self.n_experts):
            mask = (top_k_indices == expert_idx)
            weight = (top_k_weights * mask.to(x.dtype)).sum(dim=-1, keepdim=True)
            expert_out = self.experts[expert_idx](x)
            output = output + weight * expert_out
        return output

    def _forward_scatter(self, x, top_k_indices, top_k_weights):
        B, T, D = x.shape
        k = self.top_k
        flat_x = x.reshape(B * T, D)
        flat_indices = top_k_indices.reshape(B * T, k)
        flat_weights = top_k_weights.reshape(B * T, k)
        output = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)

        for expert_idx in range(self.n_experts):
            token_mask = (flat_indices == expert_idx).any(dim=-1)
            if not token_mask.any():
                continue
            token_ids = token_mask.nonzero(as_tuple=True)[0]
            expert_input = flat_x[token_ids]
            expert_out = self.experts[expert_idx](expert_input)
            slot_mask = (flat_indices[token_ids] == expert_idx)
            weights = (flat_weights[token_ids] * slot_mask.to(x.dtype)).sum(-1, keepdim=True)
            output.index_add_(0, token_ids, weights * expert_out.to(dtype=x.dtype))

        return output.reshape(B, T, D)


# =============================================================================
# Block: Shared Attention + MoE FFN
# =============================================================================

class MoEBlock(nn.Module):
    """Single transformer block with shared attention and MoE FFN.
    
    Returns attention and MoE outputs separately so the caller can
    measure per-token depth vs capacity contributions.
    """

    def __init__(
        self, n_embd, bias, n_head_q, n_head_kv, attention_impl,
        attention_config, dropout, expert_ffn_hidden, n_experts, top_k,
        attention_norm, ffn_norm, enforce_swiglu_hidden_dim_multiple_of,
        execution="loop", bias_update_speed=0.001,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm

        self.attn = CausalSelfAttention(
            n_head_q=n_head_q, n_head_kv=n_head_kv, n_embd=n_embd,
            attention_config=attention_config, attention_impl=attention_impl,
            bias=bias, dropout=dropout,
        )

        self.mlp = MoEFFN(
            n_embd=n_embd, expert_ffn_hidden=expert_ffn_hidden,
            n_experts=n_experts, top_k=top_k, bias=bias,
            enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            execution=execution, bias_update_speed=bias_update_speed,
        )

    def forward(self, x, scale=1.0):
        """Returns (output, attn_delta, moe_delta, moe_metrics).

        attn_delta: (B, T) norm of the attention residual contribution.
        moe_delta:  (B, T) norm of the MoE FFN residual contribution.
        """
        attn_out = scale * self.attn(self.attention_norm(x))
        h = x + attn_out

        moe_out, moe_metrics = self.mlp(self.ffn_norm(h))
        moe_out = scale * moe_out
        h = h + moe_out

        with torch.no_grad():
            attn_delta = attn_out.detach().norm(dim=-1)  # (B, T)
            moe_delta = moe_out.detach().norm(dim=-1)    # (B, T)

        return h, attn_delta, moe_delta, moe_metrics


# =============================================================================
# Adaptive Computation Components
# =============================================================================

@dataclass
class HaltingState:
    prob_remain: torch.Tensor       # (B, T)
    output: torch.Tensor            # (B, T, D)
    expected_steps: torch.Tensor    # (B, T)

    @staticmethod
    def init(B: int, T: int, D: int, *, device: torch.device, dtype: torch.dtype) -> "HaltingState":
        return HaltingState(
            prob_remain=torch.ones(B, T, device=device, dtype=dtype),
            output=torch.zeros(B, T, D, device=device, dtype=dtype),
            expected_steps=torch.zeros(B, T, device=device, dtype=dtype),
        )

    def update(self, h: torch.Tensor, halt_prob: torch.Tensor, step: int):
        p_stop = self.prob_remain * halt_prob
        self.prob_remain = self.prob_remain * (1.0 - halt_prob)
        self.output = self.output + h * p_stop.unsqueeze(-1)
        self.expected_steps = self.expected_steps + p_stop * (step + 1)

    def finalize(self, h_last: torch.Tensor, last_step: int):
        self.output = self.output + h_last * self.prob_remain.unsqueeze(-1)
        self.expected_steps = self.expected_steps + self.prob_remain * last_step


class StepMetrics:
    def __init__(self, max_loops: int, device: torch.device):
        self.max_loops = max_loops
        self.device = device
        self._buffers: dict[str, list[torch.Tensor]] = {}

    def log(self, key: str, value: torch.Tensor):
        self._buffers.setdefault(key, []).append(value.detach())

    def finalize(self) -> dict[str, torch.Tensor]:
        out = {}
        for key, vals in self._buffers.items():
            stacked = torch.stack(vals)
            if stacked.size(0) < self.max_loops:
                pad = torch.zeros(self.max_loops - stacked.size(0), device=self.device, dtype=stacked.dtype)
                stacked = torch.cat([stacked, pad])
            out[key] = stacked
        return out


class AdaptiveRouter(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, h: torch.Tensor, step_normalized: float) -> torch.Tensor:
        B, T, _ = h.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        logit = self.linear(torch.cat([h, step_feat], dim=-1))
        return torch.sigmoid(logit).squeeze(-1)


# =============================================================================
# Adaptive Recursive Block (Single Path: Looping MoE Block + ACT)
# =============================================================================

class AdaptiveRecursiveBlock(nn.Module):
    _INIT_SCALE_RAW: float = -7.0

    def __init__(
        self,
        block: MoEBlock,
        adaptive_config: AdaptiveComputationConfig,
        n_embd: int,
        layer_idx: int,
        n_layers: int,
    ):
        super().__init__()
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.n_experts = adaptive_config.n_experts
        self.top_k = adaptive_config.top_k
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        self.router = AdaptiveRouter(n_embd)
        self.loop_scales = nn.Parameter(torch.full((self.max_loops,), self._INIT_SCALE_RAW))

    def forward(
        self, x: torch.Tensor, token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = x.shape
        device = x.device

        state = HaltingState.init(B, T, D, device=device, dtype=x.dtype)
        metrics = StepMetrics(self.max_loops, device)

        step_denom = max(1, self.max_loops - 1)
        h = x
        actual_steps = 0

        # Accumulators for depth-vs-capacity analysis
        attn_delta_sum = torch.zeros(B, T, device=device, dtype=x.dtype)
        moe_delta_sum = torch.zeros(B, T, device=device, dtype=x.dtype)
        expert_counts_per_token = torch.zeros(B, T, self.n_experts, device=device, dtype=x.dtype)

        for step in range(self.max_loops):
            actual_steps = step + 1

            scale = F.softplus(self.loop_scales[step])
            metrics.log("loop_scale", scale.detach())

            h_prev = h
            h, attn_delta, moe_delta, moe_metrics = self.block(h, scale=scale)

            # Accumulate depth-vs-capacity per token (weighted by halting probability)
            # We weight by prob_remain so halted tokens stop contributing
            with torch.no_grad():
                weight = state.prob_remain  # (B, T) — how much this step "counts"
                attn_delta_sum += weight * attn_delta
                moe_delta_sum += weight * moe_delta

                # Track expert diversity: which experts each token hits across iterations
                top_k_indices = moe_metrics["top_k_indices"]  # (B, T, top_k)
                expert_counts_per_token.scatter_add_(
                    2,
                    top_k_indices,
                    weight.unsqueeze(-1).expand_as(top_k_indices),
                )

            # Halting
            halt_prob = self.router(h, step_normalized=step / step_denom)
            state.update(h, halt_prob, step)

            rel_change = (h - h_prev).norm(dim=-1) / (h_prev.norm(dim=-1) + 1e-6)

            metrics.log("halt_prob_mean", halt_prob.detach().mean())
            metrics.log("halt_prob_std", halt_prob.detach().std())
            metrics.log("halt_prob_min", halt_prob.detach().min())
            metrics.log("halt_prob_max", halt_prob.detach().max())
            metrics.log("rel_change", rel_change.mean())
            metrics.log("prob_remain_max", state.prob_remain.max().detach())
            metrics.log("prob_remain_mean", state.prob_remain.mean().detach())

            # Per-step MoE scalars
            metrics.log("moe_router_entropy", moe_metrics["router_entropy"])
            metrics.log("moe_top1_expert_frac", moe_metrics["top1_expert_frac"])

        state.finalize(h, actual_steps)
        output = state.output

        # =================================================================
        # Build layer metrics
        # =================================================================
        step_metrics = metrics.finalize()
        es = state.expected_steps.detach()

        # --- Depth vs Capacity metrics (per-token, aggregated to scalars for logging) ---
        with torch.no_grad():
            total_contrib = attn_delta_sum + moe_delta_sum + 1e-8
            depth_ratio = attn_delta_sum / total_contrib  # (B, T) — 1.0 = pure depth, 0.0 = pure capacity

            # Expert diversity: entropy of per-token expert distribution across all iterations
            dist = expert_counts_per_token / (expert_counts_per_token.sum(-1, keepdim=True) + 1e-8)
            expert_diversity = -(dist * (dist + 1e-8).log()).sum(-1)  # (B, T)

        layer_metrics = {
            # ACT core
            "expected_steps": state.expected_steps,
            "actual_steps": torch.tensor(float(actual_steps), device=device),
            "residual_mass": state.prob_remain.mean().detach(),
            "frac_alive": (state.prob_remain.detach() > 0.01).float().mean(),
            "expected_steps_mean": es.mean(),
            "expected_steps_std": es.std(),
            "expected_steps_min": es.min(),
            "expected_steps_max": es.max(),

            # Depth vs Capacity (scalar summaries for W&B)
            "depth_ratio_mean": depth_ratio.mean(),
            "depth_ratio_std": depth_ratio.std(),
            "depth_ratio_min": depth_ratio.min(),
            "depth_ratio_max": depth_ratio.max(),
            "attn_contrib_mean": attn_delta_sum.mean(),
            "moe_contrib_mean": moe_delta_sum.mean(),
            "expert_diversity_mean": expert_diversity.mean(),
            "expert_diversity_std": expert_diversity.std(),

            # MoE aggregate (last step snapshot — representative of overall routing)
            "moe_expert_usage": moe_metrics["expert_usage"],
            "moe_expert_bias_std": moe_metrics["expert_bias_std"],

            # Per-step vectors
            "step_halt_probs": step_metrics.get("halt_prob_mean", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_std": step_metrics.get("halt_prob_std", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_min": step_metrics.get("halt_prob_min", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_max": step_metrics.get("halt_prob_max", torch.zeros(self.max_loops, device=device)),
            "step_changes": step_metrics.get("rel_change", torch.zeros(self.max_loops, device=device)),
            "loop_scales": step_metrics.get("loop_scale", torch.zeros(self.max_loops, device=device)),
            "prob_remain_max": step_metrics.get("prob_remain_max", torch.zeros(self.max_loops, device=device)),
            "prob_remain_mean": step_metrics.get("prob_remain_mean", torch.zeros(self.max_loops, device=device)),
            "step_moe_router_entropy": step_metrics.get("moe_router_entropy", torch.zeros(self.max_loops, device=device)),
            "step_moe_top1_expert_frac": step_metrics.get("moe_top1_expert_frac", torch.zeros(self.max_loops, device=device)),

            # Per-token tensors for eval-time analysis (B, T)
            "depth_ratio_token_probs": depth_ratio,
            "expert_diversity_token_probs": expert_diversity,
        }

        return output, layer_metrics


# =============================================================================
# GPT2LLM — Main Model
# =============================================================================

class GPT2LLM(NNModel):
    def __init__(
        self, sample_key, prediction_key, poe_type, sequence_length, vocab_size,
        n_layer, n_head_q, n_head_kv, n_embd, ffn_hidden, dropout, bias,
        activation_type, attention_implementation, attention_config,
        attention_norm_config, ffn_norm_config, lm_head_norm_config,
        use_weight_tying, seed=None, enforce_swiglu_hidden_dim_multiple_of=32,
        adaptive_config=None,
    ):
        weight_decay_groups = {
            "linear": [
                ".q_attn", ".k_attn", ".v_attn",
                ".attn.c_proj",
                ".mlp",
                ".lm_head.weight",
                ".router.linear.weight",
                ".router.gate.weight",
                ".experts.",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm",
                ".q_norm", ".k_norm",
                ".loop_scales",
                ".router.linear.bias",
                ".expert_bias",
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
        assert adaptive_config is not None, "adaptive_config must be provided"
        self.adaptive_config = adaptive_config

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

        expert_ffn_hidden = (
            adaptive_config.expert_ffn_hidden
            if adaptive_config.expert_ffn_hidden is not None
            else ffn_hidden
        )

        layers = {}
        for layer_idx in range(n_layer):
            block = MoEBlock(
                n_embd=n_embd, bias=bias, n_head_q=n_head_q, n_head_kv=n_head_kv,
                attention_impl=attention_implementation,
                attention_config=attention_config, dropout=dropout,
                expert_ffn_hidden=expert_ffn_hidden,
                n_experts=adaptive_config.n_experts,
                top_k=adaptive_config.top_k,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
                execution=adaptive_config.moe_execution,
                bias_update_speed=adaptive_config.moe_bias_update_speed,
            )
            layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                block=block,
                adaptive_config=adaptive_config,
                n_embd=n_embd,
                layer_idx=layer_idx,
                n_layers=n_layer,
            )

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=wpe,
            drop=nn.Dropout(dropout),
            h=nn.ModuleDict(layers),
            lm_head_norm=lm_head_norm_config.norm_type.value(**dict(lm_head_norm_config.config)),
            lm_head=nn.Linear(n_embd, vocab_size, bias=False),
        ))

        if use_weight_tying:
            self.transformer.wte.weight = self.transformer.lm_head.weight

    # ------------------------------------------------------------------
    # Metrics bag construction
    # ------------------------------------------------------------------

    _VECTOR_KEYS = [
        "step_halt_probs", "step_halt_prob_std", "step_halt_prob_min", "step_halt_prob_max",
        "step_changes", "loop_scales", "prob_remain_max", "prob_remain_mean",
        "step_moe_router_entropy", "step_moe_top1_expert_frac",
    ]

    _PER_LAYER_SCALAR_KEYS = [
        "actual_steps", "residual_mass", "frac_alive",
        "expected_steps_mean", "expected_steps_std", "expected_steps_min", "expected_steps_max",
        "depth_ratio_mean", "depth_ratio_std", "depth_ratio_min", "depth_ratio_max",
        "attn_contrib_mean", "moe_contrib_mean",
        "expert_diversity_mean", "expert_diversity_std",
        "moe_expert_bias_std",
    ]

    def _build_metrics_bag(
        self,
        all_layer_metrics: list[dict[str, torch.Tensor]],
        total_ponder_cost: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict]:
        n_layers = len(all_layer_metrics)
        max_loops = self.adaptive_config.max_loops

        avg_ponder_cost = total_ponder_cost / n_layers
        normalized_steps = (
            (avg_ponder_cost - 1.0) / (max_loops - 1.0)
            if max_loops > 1
            else torch.tensor(0.0, dtype=dtype, device=device)
        )
        weighted_ponder_loss = (normalized_steps * self.adaptive_config.ponder_penalty_weight).to(dtype)

        def stack_key(key: str) -> torch.Tensor:
            return torch.stack([m[key] for m in all_layer_metrics])

        scalars = {
            "ponder_cost_unweighted": total_ponder_cost,
            "expected_steps": avg_ponder_cost,
            "normalized_steps": normalized_steps,
        }

        per_layer_scalars = {
            "ponder_cost": torch.stack([m["expected_steps"].mean() for m in all_layer_metrics]),
        }
        for key in self._PER_LAYER_SCALAR_KEYS:
            per_layer_scalars[key] = stack_key(key)

        per_layer_vectors = {}
        for key in self._VECTOR_KEYS:
            per_layer_vectors[key] = stack_key(key)

        # MoE expert usage is a vector per layer (n_experts,)
        per_layer_vectors["moe_expert_usage"] = stack_key("moe_expert_usage")
        per_layer_scalars["moe_router_entropy"] = torch.stack([
            m.get("step_moe_router_entropy", torch.zeros(max_loops, device=device)).mean()
            for m in all_layer_metrics
        ])

        return weighted_ponder_loss, {
            "scalars": scalars,
            "per_layer_scalars": per_layer_scalars,
            "per_layer_vectors": per_layer_vectors,
        }

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

        h = self.transformer.drop(h) if hasattr(self.transformer, "drop") else h

        all_layer_metrics: list[dict[str, torch.Tensor]] = []
        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer_module = self.transformer.h[layer_key]
            h, layer_metrics = layer_module(h, token_ids=inputs)
            total_ponder_cost = total_ponder_cost + layer_metrics["expected_steps"].mean()
            all_layer_metrics.append(layer_metrics)

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        weighted_ponder_loss, metrics_bag = self._build_metrics_bag(
            all_layer_metrics, total_ponder_cost, device, logits.dtype,
        )

        if not self.training:
            metrics_bag["eval_tokens"] = inputs.detach()
            metrics_bag["eval_depth_ratio"] = torch.stack([
                m.get("depth_ratio_token_probs", torch.zeros_like(inputs, dtype=logits.dtype))
                for m in all_layer_metrics
            ])  # (n_layers, B, T)
            metrics_bag["eval_expert_diversity"] = torch.stack([
                m.get("expert_diversity_token_probs", torch.zeros_like(inputs, dtype=logits.dtype))
                for m in all_layer_metrics
            ])  # (n_layers, B, T)

        return {
            "logits": logits,
            "ponder_loss": weighted_ponder_loss,
            "metrics": metrics_bag,
        }


# =============================================================================
# Manual attention fallback
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