import logging
import math
from abc import abstractmethod
from dataclasses import dataclass, field
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
    enable_adaptive: bool = False
    max_loops: int = 10
    halt_threshold: float = 0.99
    ponder_penalty_weight: float = 0.00
    wide_ffn_hidden: int = 0
    wide_ffn_gate_init_bias: float = 0.0
    deep_gate_init_bias: float = 0.0
    scheduler_type: str = "constant"
    layer_types: Optional[list[str]] = None
    # --- New enhancement flags ---
    loop_input_injection: bool = False
    enrich_router: bool = False
    enrich_gate: bool = False


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


class TransformerMLP(nn.Module):
    def __init__(self, n_embd, ffn_hidden, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ffn_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(
        self, n_embd, bias, n_head_q, n_head_kv, activation_type, attention_impl,
        attention_config, dropout, ffn_hidden, attention_norm, ffn_norm,
        enforce_swiglu_hidden_dim_multiple_of,
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


# =============================================================================
# Adaptive Computation Components
# =============================================================================

@dataclass
class HaltingState:
    """Tracks ACT halting: prob_remain, weighted output, and expected steps."""
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
    """Collects per-step scalars, pads/stacks to (max_loops,) for logging."""

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


# --- Helper: compute scale-invariant summary statistics ---

def _displacement_stats(
    h: torch.Tensor, x: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (relative_norm, cosine_similarity) between h and x.

    relative_norm: ||h - x|| / (||x|| + eps)   — shape (B, T)
    cosine_sim:    cos(h, x)                    — shape (B, T)
    """
    diff = h - x
    x_norm = x.norm(dim=-1).clamp(min=eps)          # (B, T)
    rel_norm = diff.norm(dim=-1) / x_norm            # (B, T)

    h_norm = h.norm(dim=-1).clamp(min=eps)           # (B, T)
    cos_sim = (h * x).sum(dim=-1) / (h_norm * x_norm)  # (B, T)

    return rel_norm, cos_sim


# --- Original router (unchanged) ---

class AdaptiveRouter(nn.Module):
    """Per-token halting: [h; t_normalized] -> sigmoid -> halt_prob."""

    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, h: torch.Tensor, step_normalized: float, x: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = h.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        logit = self.linear(torch.cat([h, step_feat], dim=-1))
        return torch.sigmoid(logit).squeeze(-1)


# --- Enriched router: [h, ||h-x||/||x||, cos(h,x), step_normalized] ---

class EnrichedAdaptiveRouter(nn.Module):
    """Per-token halting with scale-invariant displacement features.

    Input:  [h, rel_norm, cos_sim, step_normalized]  ->  Linear(n_embd + 3, 1)
    """

    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 3, 1, bias=bias)

    def forward(self, h: torch.Tensor, step_normalized: float, x: torch.Tensor = None) -> torch.Tensor:
        assert x is not None, "EnrichedAdaptiveRouter requires x (layer input)"
        B, T, _ = h.shape

        rel_norm, cos_sim = _displacement_stats(h, x)  # each (B, T)

        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        features = torch.cat([
            h,                              # (B, T, D)
            rel_norm.unsqueeze(-1),         # (B, T, 1)
            cos_sim.unsqueeze(-1),          # (B, T, 1)
            step_feat,                      # (B, T, 1)
        ], dim=-1)

        logit = self.linear(features)
        return torch.sigmoid(logit).squeeze(-1)


# --- Original dual gate (unchanged) ---

class DecoupledDualPathGate(nn.Module):
    def __init__(self, n_embd, init_bias_deep=0.0, init_bias_wide=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, 2, bias=True)
        self.proj_w2d = nn.Linear(n_embd, n_embd, bias=False)
        self.proj_d2w = nn.Linear(n_embd, n_embd, bias=False)
        self.cross_scale_wide = nn.Parameter(torch.tensor([-7.0]))
        self.cross_scale_deep = nn.Parameter(torch.tensor([-7.0]))
        self.init_bias_deep = init_bias_deep
        self.init_bias_wide = init_bias_wide
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias[0] = self.init_bias_deep
            self.gate_proj.bias[1] = self.init_bias_wide
        nn.init.zeros_(self.proj_w2d.weight)
        nn.init.zeros_(self.proj_d2w.weight)

    def forward(self, routing_input, h_deep, h_wide, use_cross=False):
        logits = self.gate_proj(routing_input)
        gates = torch.sigmoid(logits)
        gate_deep = gates[..., 0:1]
        gate_wide = gates[..., 1:2]

        if use_cross:
            scale_deep = F.softplus(self.cross_scale_deep)
            scale_wide = F.softplus(self.cross_scale_wide)
            h_deep_out = h_deep + scale_deep * self.proj_w2d(h_wide)
            h_wide_out = h_wide + scale_wide * self.proj_d2w(h_deep)
        else:
            h_deep_out = h_deep
            h_wide_out = h_wide

        blended = gate_deep * h_deep_out + gate_wide * h_wide_out
        return blended, gate_deep, gate_wide


# --- Enriched dual gate: [x, rel_norm_deep, rel_norm_wide, cos_deep, cos_wide] ---

class EnrichedDecoupledDualPathGate(nn.Module):
    """Gate that sees x + scale-invariant summaries of both path outputs (detached).

    Input:  [x, ||h_deep-x||/||x||, ||h_wide-x||/||x||, cos(h_deep,x), cos(h_wide,x)]
            -> Linear(n_embd + 4, 2)
    """

    def __init__(self, n_embd, init_bias_deep=0.0, init_bias_wide=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd + 4, 2, bias=True)
        self.proj_w2d = nn.Linear(n_embd, n_embd, bias=False)
        self.proj_d2w = nn.Linear(n_embd, n_embd, bias=False)
        self.cross_scale_wide = nn.Parameter(torch.tensor([-7.0]))
        self.cross_scale_deep = nn.Parameter(torch.tensor([-7.0]))
        self.init_bias_deep = init_bias_deep
        self.init_bias_wide = init_bias_wide
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias[0] = self.init_bias_deep
            self.gate_proj.bias[1] = self.init_bias_wide
        nn.init.zeros_(self.proj_w2d.weight)
        nn.init.zeros_(self.proj_d2w.weight)

    def forward(self, routing_input, h_deep, h_wide, use_cross=False):
        # routing_input is x (the original layer input)
        x = routing_input

        # Detach both paths: gate evaluates, paths learn to be useful
        h_deep_det = h_deep.detach()
        h_wide_det = h_wide.detach()

        rel_norm_deep, cos_sim_deep = _displacement_stats(h_deep_det, x)
        rel_norm_wide, cos_sim_wide = _displacement_stats(h_wide_det, x)

        gate_input = torch.cat([
            x,                                   # (B, T, D)
            rel_norm_deep.unsqueeze(-1),         # (B, T, 1)
            rel_norm_wide.unsqueeze(-1),         # (B, T, 1)
            cos_sim_deep.unsqueeze(-1),          # (B, T, 1)
            cos_sim_wide.unsqueeze(-1),          # (B, T, 1)
        ], dim=-1)

        logits = self.gate_proj(gate_input)
        gates = torch.sigmoid(logits)
        gate_deep = gates[..., 0:1]
        gate_wide = gates[..., 1:2]

        if use_cross:
            scale_deep = F.softplus(self.cross_scale_deep)
            scale_wide = F.softplus(self.cross_scale_wide)
            h_deep_out = h_deep + scale_deep * self.proj_w2d(h_wide)
            h_wide_out = h_wide + scale_wide * self.proj_d2w(h_deep)
        else:
            h_deep_out = h_deep
            h_wide_out = h_wide

        blended = gate_deep * h_deep_out + gate_wide * h_wide_out
        return blended, gate_deep, gate_wide


# =============================================================================
# Adaptive Recursive Block (Dual Full-Block Paths)
# =============================================================================

class AdaptiveRecursiveBlock(nn.Module):
    _INIT_SCALE_RAW: float = -7.0

    def __init__(
        self,
        block: Optional[GPT2Block],
        adaptive_config: AdaptiveComputationConfig,
        n_embd: int,
        layer_idx: int,
        n_layers: int,
        wide_block: Optional[GPT2Block] = None,
        layer_type: str = "dual",
    ):
        super().__init__()
        self.layer_type = layer_type
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        self.has_loop_path = layer_type in ["loop", "dual"]
        if self.has_loop_path:
            # Select router variant
            if adaptive_config.enrich_router:
                self.router = EnrichedAdaptiveRouter(n_embd)
            else:
                self.router = AdaptiveRouter(n_embd)
            self.loop_scales = nn.Parameter(torch.full((self.max_loops,), self._INIT_SCALE_RAW))

            # Input injection: learnable alpha, init to 0 so softplus(0) ~ 0.69
            # We init the raw param to -7.0 so softplus(-7) ~ 0.0009 ≈ no-op at start
            if adaptive_config.loop_input_injection:
                self.injection_alpha_raw = nn.Parameter(torch.tensor([self._INIT_SCALE_RAW]))

        self.has_wide_path = layer_type in ["wide", "dual"]
        if self.has_wide_path:
            self.wide_block = wide_block
            self.wide_scale = nn.Parameter(torch.tensor([self._INIT_SCALE_RAW]))

        if layer_type == "dual":
            # Select gate variant
            if adaptive_config.enrich_gate:
                self.dual_gate = EnrichedDecoupledDualPathGate(
                    n_embd=n_embd,
                    init_bias_deep=adaptive_config.deep_gate_init_bias,
                    init_bias_wide=adaptive_config.wide_ffn_gate_init_bias,
                )
            else:
                self.dual_gate = DecoupledDualPathGate(
                    n_embd=n_embd,
                    init_bias_deep=adaptive_config.deep_gate_init_bias,
                    init_bias_wide=adaptive_config.wide_ffn_gate_init_bias,
                )

    def forward(
        self, x: torch.Tensor, token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = x.shape
        device = x.device

        use_input_injection = (
            self.has_loop_path and hasattr(self, "injection_alpha_raw")
        )

        # =================================================================
        # 1) Compute path: full block looped with ACT
        # =================================================================
        state = HaltingState.init(B, T, D, device=device, dtype=x.dtype)
        metrics = StepMetrics(self.max_loops, device)

        if self.has_loop_path:
            step_denom = max(1, self.max_loops - 1)
            h_loop = x
            actual_steps = 0

            # Precompute injection alpha once if enabled
            if use_input_injection:
                injection_alpha = F.softplus(self.injection_alpha_raw).squeeze()
                metrics.log("injection_alpha", injection_alpha.detach())

            for step in range(self.max_loops):
                actual_steps = step + 1

                scale = F.softplus(self.loop_scales[step])
                metrics.log("loop_scale", scale.detach())
                h_prev = h_loop

                # Input injection: blend original input back into the loop
                if use_input_injection:
                    h_loop = self.block(h_loop + injection_alpha * x, scale=scale)
                else:
                    h_loop = self.block(h_loop, scale=scale)

                # Pass x to router (enriched router needs it; original ignores it)
                halt_prob = self.router(h_loop, step_normalized=step / step_denom, x=x)
                state.update(h_loop, halt_prob, step)

                rel_change = (h_loop - h_prev).norm(dim=-1) / (h_prev.norm(dim=-1) + 1e-6)

                metrics.log("halt_prob_mean", halt_prob.detach().mean())
                metrics.log("halt_prob_std", halt_prob.detach().std())
                metrics.log("halt_prob_min", halt_prob.detach().min())
                metrics.log("halt_prob_max", halt_prob.detach().max())
                metrics.log("rel_change", rel_change.mean())
                metrics.log("prob_remain_max", state.prob_remain.max().detach())
                metrics.log("prob_remain_mean", state.prob_remain.mean().detach())

            state.finalize(h_loop, actual_steps)
            h_deep = state.output
            frac_alive = (state.prob_remain.detach() > 0.01).float().mean()
        else:
            h_deep = torch.zeros_like(x)
            actual_steps = 0
            frac_alive = torch.tensor(0.0, device=device)

        # =================================================================
        # 2) Capacity path: wide block, single pass
        # =================================================================
        if self.has_wide_path:
            wide_scale_val = F.softplus(self.wide_scale)
            h_wide = self.wide_block(x, scale=wide_scale_val)
        else:
            h_wide = torch.zeros_like(x)
            wide_scale_val = torch.tensor(0.0, device=device)

        # =================================================================
        # 3) Gating and Output
        # =================================================================
        if self.layer_type == "dual":
            # Both gate variants receive x as routing_input.
            # Original gate uses x directly; enriched gate computes
            # displacement stats from x vs detached h_deep/h_wide internally.
            output, gate_deep, gate_wide = self.dual_gate(x, h_deep, h_wide)
        elif self.layer_type == "loop":
            output = h_deep
            gate_deep = torch.ones(B, T, 1, device=device, dtype=x.dtype)
            gate_wide = torch.zeros(B, T, 1, device=device, dtype=x.dtype)
        elif self.layer_type == "wide":
            output = h_wide
            gate_deep = torch.zeros(B, T, 1, device=device, dtype=x.dtype)
            gate_wide = torch.ones(B, T, 1, device=device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # =================================================================
        # 4) Build layer metrics
        # =================================================================
        step_metrics = metrics.finalize()

        es = state.expected_steps.detach()
        gate_deep_d = gate_deep.detach()
        gate_wide_d = gate_wide.detach()

        layer_metrics = {
            "expected_steps": state.expected_steps,
            "actual_steps": torch.tensor(float(actual_steps), device=device),
            "residual_mass": state.prob_remain.mean().detach(),
            "frac_alive": frac_alive,
            "wide_scale": wide_scale_val.squeeze().detach(),
            "expected_steps_mean": es.mean(),
            "expected_steps_std": es.std(),
            "expected_steps_min": es.min(),
            "expected_steps_max": es.max(),

            # Gate Metrics
            "gate_deep_mean": gate_deep_d.mean(),
            "gate_deep_std": gate_deep_d.std(),
            "gate_deep_min": gate_deep_d.min(),
            "gate_deep_max": gate_deep_d.max(),
            "gate_wide_mean": gate_wide_d.mean(),
            "gate_wide_std": gate_wide_d.std(),
            "gate_wide_min": gate_wide_d.min(),
            "gate_wide_max": gate_wide_d.max(),
            "gate_deep_token_probs": gate_deep_d.squeeze(-1),
            "gate_wide_token_probs": gate_wide_d.squeeze(-1),

            "deep_block_norm": h_deep.detach().norm(dim=-1).mean(),
            "wide_block_norm": (
                h_wide.detach().norm(dim=-1).mean()
                if self.has_wide_path else torch.tensor(0.0, device=device)
            ),
            "step_halt_probs": step_metrics.get("halt_prob_mean", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_std": step_metrics.get("halt_prob_std", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_min": step_metrics.get("halt_prob_min", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_max": step_metrics.get("halt_prob_max", torch.zeros(self.max_loops, device=device)),
            "step_changes": step_metrics.get("rel_change", torch.zeros(self.max_loops, device=device)),
            "loop_scales": step_metrics.get("loop_scale", torch.zeros(self.max_loops, device=device)),
            "prob_remain_max": step_metrics.get("prob_remain_max", torch.zeros(self.max_loops, device=device)),
            "prob_remain_mean": step_metrics.get("prob_remain_mean", torch.zeros(self.max_loops, device=device)),
        }

        # Log injection alpha if active
        if use_input_injection:
            layer_metrics["injection_alpha"] = F.softplus(self.injection_alpha_raw).squeeze().detach()
        else:
            layer_metrics["injection_alpha"] = torch.tensor(0.0, device=device)

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
                ".dual_gate.gate_proj.weight",
                ".dual_gate.proj_w2d.weight",
                ".dual_gate.proj_d2w.weight",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm",
                ".q_norm", ".k_norm",
                ".loop_scales", ".wide_scale", ".dual_gate.gate_proj.bias",
                ".router.linear.bias", ".dual_gate.cross_scale",
                ".injection_alpha_raw",
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
            wpe = nn.Embedding(sequence_length, n_embd)
        elif poe_type is PositionTypes.NOPE:
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            c.type_hint.value for c in attention_config.qkv_transforms
        ]:
            raise ValueError('Use "RotaryTransform" together with "NOPE".')

        self.use_adaptive = adaptive_config is not None and adaptive_config.enable_adaptive
        self.adaptive_config = adaptive_config

        def create_block(ffn_hidden_override=None):
            return GPT2Block(
                n_embd=n_embd, bias=bias, n_head_q=n_head_q, n_head_kv=n_head_kv,
                activation_type=activation_type, attention_impl=attention_implementation,
                attention_config=attention_config, dropout=dropout,
                ffn_hidden=ffn_hidden_override if ffn_hidden_override is not None else ffn_hidden,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )

        layer_types_list = []
        if self.use_adaptive:
            assert adaptive_config is not None, "adaptive_config must be provided if enable_adaptive is True"
            if adaptive_config.layer_types is not None:
                layer_types_list = adaptive_config.layer_types
                if len(layer_types_list) != n_layer:
                    raise ValueError(f"layer_types length {len(layer_types_list)} must match n_layer {n_layer}")
            else:
                has_wide = adaptive_config.wide_ffn_hidden > 0
                layer_types_list = ["dual" if has_wide else "loop"] * n_layer

        layers = {}
        for layer_idx in range(n_layer):
            if self.use_adaptive:
                l_type = layer_types_list[layer_idx]
                narrow_block = create_block() if l_type in ["loop", "dual"] else None
                wide_block = create_block(ffn_hidden_override=adaptive_config.wide_ffn_hidden) if l_type in ["wide", "dual"] else None
                layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                    block=narrow_block,
                    adaptive_config=adaptive_config,
                    n_embd=n_embd,
                    layer_idx=layer_idx,
                    n_layers=n_layer,
                    wide_block=wide_block,
                    layer_type=l_type,
                )
            else:
                layers[str(layer_idx)] = create_block()

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
    ]

    _PER_LAYER_SCALAR_KEYS = [
        "actual_steps", "residual_mass", "frac_alive", "wide_scale",
        "expected_steps_mean", "expected_steps_std", "expected_steps_min", "expected_steps_max",
        "gate_deep_mean", "gate_deep_std", "gate_deep_min", "gate_deep_max",
        "gate_wide_mean", "gate_wide_std", "gate_wide_min", "gate_wide_max",
        "deep_block_norm", "wide_block_norm",
        "injection_alpha",
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

            if self.use_adaptive:
                h, layer_metrics = layer_module(h, token_ids=inputs)
                total_ponder_cost = total_ponder_cost + layer_metrics["expected_steps"].mean()
                all_layer_metrics.append(layer_metrics)
            else:
                h = layer_module(h, scale=1.0)

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        if not self.use_adaptive:
            return logits

        weighted_ponder_loss, metrics_bag = self._build_metrics_bag(
            all_layer_metrics, total_ponder_cost, device, logits.dtype,
        )

        if not self.training:
            metrics_bag["eval_tokens"] = inputs.detach()
            metrics_bag["eval_gate_deep_probs"] = torch.stack([m.get("gate_deep_token_probs", torch.ones_like(inputs, dtype=logits.dtype)) for m in all_layer_metrics])
            metrics_bag["eval_gate_wide_probs"] = torch.stack([m.get("gate_wide_token_probs", torch.zeros_like(inputs, dtype=logits.dtype)) for m in all_layer_metrics])

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