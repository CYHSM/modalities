import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Optional, Union, overload

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
#     "scalars":              dict[str, Tensor],   # shape ()    — accumulated & reduced
#     "per_layer_scalars":    dict[str, Tensor],   # shape (L,)  — accumulated & reduced
#     "per_layer_vectors":    dict[str, Tensor],   # shape (L,max_loops) — ACT step-wise
# }
#
# Eval-time per-token attachments (only when not self.training):
#     "eval_tokens":              (B, T)
#     "eval_expected_steps":      (L, B, T)
#     "eval_delta_deep_norm":     (L, B, T)
#     "eval_delta_wide_norm":     (L, B, T)
#
#   In gate_mode = "convex":
#     "eval_gate":                (L, B, T)   single convex gate, [0,1]
#
#   In gate_mode = "two_gates":
#     "eval_gate_deep":           (L, B, T)   deep gate, [0,1] (independent)
#     "eval_gate_wide":           (L, B, T)   wide gate, [0,1] (independent)
#
#   In either mode, when use_cross is True:
#     "eval_cross_w2d_norm":      (L, B, T)
#     "eval_cross_d2w_norm":      (L, B, T)
# =============================================================================


# =============================================================================
# Configs
# =============================================================================

class AdaptiveComputationConfig(BaseModel):
    enable_adaptive: bool = False
    max_loops: int = 3
    halt_threshold: float = 1.00
    ponder_penalty_weight: float = 0.00
    wide_ffn_hidden: int = 0

    # ---- Gate mode ---------------------------------------------------------
    # "convex":     output = g * h_deep + (1-g) * h_wide  (single gate)
    # "two_gates":  output = g_d * h_deep_eff + g_w * h_wide_eff
    gate_mode: Literal["convex", "two_gates", "softmax", "fixed"] = "two_gates"
    gate_init_bias: float = 0.0
    deep_gate_init_bias: float = 0.0
    wide_gate_init_bias: float = 0.0
    loop_scale_init: float = -7
    wide_scale_init: float = -7
    scheduler_type: str = "constant"
    layer_types: Optional[list[str]] = None
    use_cross: bool = True
    cross_scale_deep_init: float = -7.0
    cross_scale_wide_init: float = -7.0

    @model_validator(mode="after")
    def _check_two_gate_field_use(self) -> "AdaptiveComputationConfig":
        # Soft sanity check: warn (not error) when fields don't match the mode,
        # since the *_init_bias fields default to 0.0 and harm nothing.
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


def _displacement_stats(
    h: torch.Tensor, x: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (relative_norm, cosine_similarity) between h and x."""
    diff = h - x
    x_norm = x.norm(dim=-1).clamp(min=eps)
    rel_norm = diff.norm(dim=-1) / x_norm

    h_norm = h.norm(dim=-1).clamp(min=eps)
    cos_sim = (h * x).sum(dim=-1) / (h_norm * x_norm)

    return rel_norm, cos_sim


def _batch_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation between two (B, T) tensors, flattened over all tokens."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=eps)
    return (a * b).sum() / denom


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


# =============================================================================
# Dual Path Gate — convex (single gate) variant
# =============================================================================

class DualPathGateConvex(nn.Module):
    """Single-gate convex combination, optionally with cross-path mixing.

    Without cross-path (use_cross=False):
        g      = sigmoid(gate_proj(x))        shape (B, T, 1)
        output = g * h_deep + (1 - g) * h_wide

    With cross-path (use_cross=True):
        s_d = softplus(cross_scale_deep)
        s_w = softplus(cross_scale_wide)
        h_deep_eff = h_deep + s_d * proj_w2d(h_wide)
        h_wide_eff = h_wide + s_w * proj_d2w(h_deep)
        output     = g * h_deep_eff + (1 - g) * h_wide_eff
    """

    def __init__(
        self,
        n_embd: int,
        gate_init_bias: float = 0.0,
        use_cross: bool = False,
        cross_scale_deep_init: float = -7.0,
        cross_scale_wide_init: float = -7.0,
    ):
        super().__init__()
        self.gate_init_bias = gate_init_bias
        self.use_cross = use_cross
        self.cross_scale_deep_init = cross_scale_deep_init
        self.cross_scale_wide_init = cross_scale_wide_init

        self.gate_proj = nn.Linear(n_embd, 1, bias=True)

        if use_cross:
            self.proj_w2d = nn.Linear(n_embd, n_embd, bias=False)
            self.proj_d2w = nn.Linear(n_embd, n_embd, bias=False)
            self.cross_scale_deep = nn.Parameter(torch.empty(1))
            self.cross_scale_wide = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias.fill_(self.gate_init_bias)

        if self.use_cross:
            nn.init.zeros_(self.proj_w2d.weight)
            nn.init.zeros_(self.proj_d2w.weight)
            nn.init.constant_(self.cross_scale_deep, self.cross_scale_deep_init)
            nn.init.constant_(self.cross_scale_wide, self.cross_scale_wide_init)

    def forward(
        self,
        x: torch.Tensor,
        h_deep: torch.Tensor,
        h_wide: torch.Tensor,
        gate_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logit = self.gate_proj(x)
        gate_raw = torch.sigmoid(logit)

        if gate_override is not None:
            if gate_override.dim() == 2:
                gate_override = gate_override.unsqueeze(-1)
            gate = gate_override.to(dtype=gate_raw.dtype, device=gate_raw.device)
        else:
            gate = gate_raw

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            cross_w2d = s_d * self.proj_w2d(h_wide)
            cross_d2w = s_w * self.proj_d2w(h_deep)
            h_deep_eff = h_deep + cross_w2d
            h_wide_eff = h_wide + cross_d2w
        else:
            cross_w2d = None
            cross_d2w = None
            h_deep_eff = h_deep
            h_wide_eff = h_wide

        output = gate * h_deep_eff + (1.0 - gate) * h_wide_eff

        with torch.no_grad():
            aux = {
                "gate_logit_mean": logit.mean(),
                "gate_logit_std":  logit.std(),
            }
            if self.use_cross:
                aux["cross_w2d_norm_per_token"] = cross_w2d.norm(dim=-1)
                aux["cross_d2w_norm_per_token"] = cross_d2w.norm(dim=-1)
                aux["cross_scale_deep"] = s_d.detach().squeeze()
                aux["cross_scale_wide"] = s_w.detach().squeeze()
        return output, gate, gate_raw, aux


# =============================================================================
# Dual Path Gate — two-gates variant (original old-code formulation)
# =============================================================================

class DualPathGateTwoGates(nn.Module):
    """Two independent sigmoid gates, optionally with cross-path mixing.

    Without cross-path:
        logits = gate_proj(x)                        # (B, T, 2)
        g_d, g_w = sigmoid(logits[..., 0:1]), sigmoid(logits[..., 1:2])
        output   = g_d * h_deep + g_w * h_wide

    With cross-path (faithful to the old code):
        cross_w2d = scale_d * proj_w2d(h_wide)       # gated by g_d
        cross_d2w = scale_w * proj_d2w(h_deep)       # gated by g_w
        contam_w2d = g_d * cross_w2d
        contam_d2w = g_w * cross_d2w
        output     = (g_d * h_deep + contam_w2d) + (g_w * h_wide + contam_d2w)

    The gates are independent: g_d + g_w is unconstrained, so the layer can
    lean hard on both paths or near-zero out either independently. Compared
    with convex mode, this gives strictly more expressive capacity but loses
    the "fraction routed to deep" reading of g.
    """

    def __init__(
        self,
        n_embd: int,
        deep_gate_init_bias: float = 0.0,
        wide_gate_init_bias: float = 0.0,
        use_cross: bool = False,
        cross_scale_deep_init: float = -7.0,
        cross_scale_wide_init: float = -7.0,
    ):
        super().__init__()
        self.deep_gate_init_bias = deep_gate_init_bias
        self.wide_gate_init_bias = wide_gate_init_bias
        self.use_cross = use_cross
        self.cross_scale_deep_init = cross_scale_deep_init
        self.cross_scale_wide_init = cross_scale_wide_init

        # Single Linear projecting to 2 logits — matches old code's gate_proj.
        self.gate_proj = nn.Linear(n_embd, 2, bias=True)

        if use_cross:
            self.proj_w2d = nn.Linear(n_embd, n_embd, bias=False)
            self.proj_d2w = nn.Linear(n_embd, n_embd, bias=False)
            self.cross_scale_deep = nn.Parameter(torch.empty(1))
            self.cross_scale_wide = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        # Zero-init weight (gate is driven only by bias at init), match old code.
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias[0] = self.deep_gate_init_bias
            self.gate_proj.bias[1] = self.wide_gate_init_bias

        if self.use_cross:
            nn.init.zeros_(self.proj_w2d.weight)
            nn.init.zeros_(self.proj_d2w.weight)
            nn.init.constant_(self.cross_scale_deep, self.cross_scale_deep_init)
            nn.init.constant_(self.cross_scale_wide, self.cross_scale_wide_init)

    def forward(
        self,
        x: torch.Tensor,
        h_deep: torch.Tensor,
        h_wide: torch.Tensor,
        gate_override: Optional[
            Union[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            gate_override: one of:
                - None: use computed gates (default)
                - Tensor of shape (B, T) or (B, T, 1): overrides gate_deep ONLY,
                  gate_wide is left as-computed. (Most common intervention:
                  "force deep path on/off".)
                - tuple (deep_override, wide_override) of (B, T) | (B, T, 1) | None:
                  selectively override either or both. Use None for "leave
                  this gate alone".

        Returns:
            output:        (B, T, D)
            gate_deep:     (B, T, 1) post-override
            gate_wide:     (B, T, 1) post-override
            gate_deep_raw: (B, T, 1) pre-override
            gate_wide_raw: (B, T, 1) pre-override
            aux:           dict
        """
        logits = self.gate_proj(x)                       # (B, T, 2)
        gates_raw = torch.sigmoid(logits)                # (B, T, 2)
        gate_deep_raw = gates_raw[..., 0:1]
        gate_wide_raw = gates_raw[..., 1:2]

        # Resolve overrides
        deep_ovr, wide_ovr = self._unpack_override(gate_override)
        gate_deep = self._apply_override(gate_deep_raw, deep_ovr)
        gate_wide = self._apply_override(gate_wide_raw, wide_ovr)

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            cross_w2d_full = s_d * self.proj_w2d(h_wide)
            cross_d2w_full = s_w * self.proj_d2w(h_deep)
            contam_w2d = gate_deep * cross_w2d_full
            contam_d2w = gate_wide * cross_d2w_full
            h_deep_branch = gate_deep * h_deep + contam_w2d
            h_wide_branch = gate_wide * h_wide + contam_d2w
        else:
            cross_w2d_full = None
            cross_d2w_full = None
            contam_w2d = None
            contam_d2w = None
            h_deep_branch = gate_deep * h_deep
            h_wide_branch = gate_wide * h_wide

        output = h_deep_branch + h_wide_branch

        with torch.no_grad():
            aux = {
                "gate_logit_deep_mean": logits[..., 0].mean(),
                "gate_logit_deep_std":  logits[..., 0].std(),
                "gate_logit_wide_mean": logits[..., 1].mean(),
                "gate_logit_wide_std":  logits[..., 1].std(),
            }
            if self.use_cross:
                # Use post-gate contamination magnitudes — that's the actual
                # signal the next layer sees. Matches old-code convention.
                aux["cross_w2d_norm_per_token"] = contam_w2d.norm(dim=-1)
                aux["cross_d2w_norm_per_token"] = contam_d2w.norm(dim=-1)
                aux["cross_scale_deep"] = s_d.detach().squeeze()
                aux["cross_scale_wide"] = s_w.detach().squeeze()

        return output, gate_deep, gate_wide, gate_deep_raw, gate_wide_raw, aux

    @staticmethod
    def _unpack_override(
        ovr,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ovr is None:
            return None, None
        if isinstance(ovr, tuple):
            assert len(ovr) == 2, "tuple override must be (deep, wide)"
            return ovr[0], ovr[1]
        # Bare tensor → applies to deep only (most common intervention).
        return ovr, None

    @staticmethod
    def _apply_override(
        gate: torch.Tensor, override: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if override is None:
            return gate
        if override.dim() == 2:
            override = override.unsqueeze(-1)
        return override.to(dtype=gate.dtype, device=gate.device)


# =============================================================================
# Dual Path Gate — softmax variant
# =============================================================================

class DualPathGateSoftmax(DualPathGateTwoGates):
    """Softmax over the two paths (deep vs wide)."""
    def forward(
        self,
        x: torch.Tensor,
        h_deep: torch.Tensor,
        h_wide: torch.Tensor,
        gate_override: Optional[
            Union[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.gate_proj(x)                       # (B, T, 2)
        gates_raw = F.softmax(logits, dim=-1)            # (B, T, 2)
        gate_deep_raw = gates_raw[..., 0:1]
        gate_wide_raw = gates_raw[..., 1:2]

        deep_ovr, wide_ovr = self._unpack_override(gate_override)
        gate_deep = self._apply_override(gate_deep_raw, deep_ovr)
        gate_wide = self._apply_override(gate_wide_raw, wide_ovr)

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            cross_w2d_full = s_d * self.proj_w2d(h_wide)
            cross_d2w_full = s_w * self.proj_d2w(h_deep)
            contam_w2d = gate_deep * cross_w2d_full
            contam_d2w = gate_wide * cross_d2w_full
            h_deep_branch = gate_deep * h_deep + contam_w2d
            h_wide_branch = gate_wide * h_wide + contam_d2w
        else:
            cross_w2d_full = None
            cross_d2w_full = None
            contam_w2d = None
            contam_d2w = None
            h_deep_branch = gate_deep * h_deep
            h_wide_branch = gate_wide * h_wide

        output = h_deep_branch + h_wide_branch

        with torch.no_grad():
            aux = {
                "gate_logit_deep_mean": logits[..., 0].mean(),
                "gate_logit_deep_std":  logits[..., 0].std(),
                "gate_logit_wide_mean": logits[..., 1].mean(),
                "gate_logit_wide_std":  logits[..., 1].std(),
            }
            if self.use_cross:
                aux["cross_w2d_norm_per_token"] = contam_w2d.norm(dim=-1)
                aux["cross_d2w_norm_per_token"] = contam_d2w.norm(dim=-1)
                aux["cross_scale_deep"] = s_d.detach().squeeze()
                aux["cross_scale_wide"] = s_w.detach().squeeze()

        return output, gate_deep, gate_wide, gate_deep_raw, gate_wide_raw, aux

# =============================================================================
# Dual Path Gate — fixed 0.5 variant
# =============================================================================

class DualPathGateFixed(DualPathGateTwoGates):
    """Fixed gate at 0.5 for both paths (ignores input)."""
    def forward(
        self,
        x: torch.Tensor,
        h_deep: torch.Tensor,
        h_wide: torch.Tensor,
        gate_override: Optional[
            Union[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Fixed 0.5
        gates_raw = torch.full((x.shape[0], x.shape[1], 2), 0.5, device=x.device, dtype=x.dtype)
        gate_deep_raw = gates_raw[..., 0:1]
        gate_wide_raw = gates_raw[..., 1:2]

        deep_ovr, wide_ovr = self._unpack_override(gate_override)
        gate_deep = self._apply_override(gate_deep_raw, deep_ovr)
        gate_wide = self._apply_override(gate_wide_raw, wide_ovr)

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            cross_w2d_full = s_d * self.proj_w2d(h_wide)
            cross_d2w_full = s_w * self.proj_d2w(h_deep)
            contam_w2d = gate_deep * cross_w2d_full
            contam_d2w = gate_wide * cross_d2w_full
            h_deep_branch = gate_deep * h_deep + contam_w2d
            h_wide_branch = gate_wide * h_wide + contam_d2w
        else:
            cross_w2d_full = None
            cross_d2w_full = None
            contam_w2d = None
            contam_d2w = None
            h_deep_branch = gate_deep * h_deep
            h_wide_branch = gate_wide * h_wide

        output = h_deep_branch + h_wide_branch

        with torch.no_grad():
            aux = {}
            if self.use_cross:
                aux["cross_w2d_norm_per_token"] = contam_w2d.norm(dim=-1)
                aux["cross_d2w_norm_per_token"] = contam_d2w.norm(dim=-1)
                aux["cross_scale_deep"] = s_d.detach().squeeze()
                aux["cross_scale_wide"] = s_w.detach().squeeze()

        return output, gate_deep, gate_wide, gate_deep_raw, gate_wide_raw, aux


# =============================================================================
# Adaptive Recursive Block
# =============================================================================

class AdaptiveRecursiveBlock(nn.Module):
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
        self.gate_mode = adaptive_config.gate_mode

        self.has_loop_path = layer_type in ["loop", "dual"]
        if self.has_loop_path:
            self.router = AdaptiveRouter(n_embd)
            self.loop_scales = nn.Parameter(
                torch.full((self.max_loops,), adaptive_config.loop_scale_init)
            )

        self.has_wide_path = layer_type in ["wide", "dual"]
        if self.has_wide_path:
            self.wide_block = wide_block
            self.wide_scale = nn.Parameter(
                torch.tensor([adaptive_config.wide_scale_init])
            )

        if layer_type == "dual":
            if self.gate_mode == "convex":
                self.dual_gate = DualPathGateConvex(
                    n_embd=n_embd,
                    gate_init_bias=adaptive_config.gate_init_bias,
                    use_cross=adaptive_config.use_cross,
                    cross_scale_deep_init=adaptive_config.cross_scale_deep_init,
                    cross_scale_wide_init=adaptive_config.cross_scale_wide_init,
                )
            elif self.gate_mode == "two_gates":
                self.dual_gate = DualPathGateTwoGates(
                    n_embd=n_embd,
                    deep_gate_init_bias=adaptive_config.deep_gate_init_bias,
                    wide_gate_init_bias=adaptive_config.wide_gate_init_bias,
                    use_cross=adaptive_config.use_cross,
                    cross_scale_deep_init=adaptive_config.cross_scale_deep_init,
                    cross_scale_wide_init=adaptive_config.cross_scale_wide_init,
                )
            elif self.gate_mode == "softmax":
                self.dual_gate = DualPathGateSoftmax(
                    n_embd=n_embd,
                    deep_gate_init_bias=adaptive_config.deep_gate_init_bias,
                    wide_gate_init_bias=adaptive_config.wide_gate_init_bias,
                    use_cross=adaptive_config.use_cross,
                    cross_scale_deep_init=adaptive_config.cross_scale_deep_init,
                    cross_scale_wide_init=adaptive_config.cross_scale_wide_init,
                )
            elif self.gate_mode == "fixed":
                self.dual_gate = DualPathGateFixed(
                    n_embd=n_embd,
                    deep_gate_init_bias=adaptive_config.deep_gate_init_bias,
                    wide_gate_init_bias=adaptive_config.wide_gate_init_bias,
                    use_cross=adaptive_config.use_cross,
                    cross_scale_deep_init=adaptive_config.cross_scale_deep_init,
                    cross_scale_wide_init=adaptive_config.cross_scale_wide_init,
                )
            else:
                raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor = None,
        gate_override=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x:             (B, T, D) input to the layer
            token_ids:     (B, T) unused here, kept for interface compatibility
            gate_override: see DualPathGateConvex/TwoGates for signature.
                           In convex mode: a tensor in [0,1] overrides g.
                           In two_gates mode: tensor (overrides deep) or
                           tuple (deep_override, wide_override).
        """
        B, T, D = x.shape
        device = x.device

        state = HaltingState.init(B, T, D, device=device, dtype=x.dtype)
        metrics = StepMetrics(self.max_loops, device)

        # ---------------------------------------------------------------
        # Deep path: recursive block, ACT-halted
        # ---------------------------------------------------------------
        if self.has_loop_path:
            step_denom = max(1, self.max_loops - 1)
            h_loop = x
            actual_steps = 0

            for step in range(self.max_loops):
                actual_steps = step + 1

                scale = F.softplus(self.loop_scales[step])
                h_prev = h_loop

                h_loop = self.block(h_loop, scale=scale)

                halt_prob = self.router(h_loop, step_normalized=step / step_denom, x=x)
                state.update(h_loop, halt_prob, step)

                with torch.no_grad():
                    metrics.log("loop_scale", scale.detach())

                    rel_change = (h_loop - h_prev).norm(dim=-1) / (h_prev.norm(dim=-1) + 1e-6)
                    metrics.log("step_h_norm", h_loop.norm(dim=-1).mean())

                    rel_norm_to_input, cos_sim_to_input = _displacement_stats(h_loop, x)
                    metrics.log("step_cos_sim_to_input", cos_sim_to_input.mean())
                    metrics.log("step_rel_norm_to_input", rel_norm_to_input.mean())

                    metrics.log("halt_prob_mean", halt_prob.mean())
                    metrics.log("halt_prob_std", halt_prob.std())
                    metrics.log("halt_prob_min", halt_prob.min())
                    metrics.log("halt_prob_max", halt_prob.max())
                    metrics.log("rel_change", rel_change.mean())
                    metrics.log("prob_remain_max", state.prob_remain.max())
                    metrics.log("prob_remain_mean", state.prob_remain.mean())

            state.finalize(h_loop, actual_steps)
            h_deep = state.output

            with torch.no_grad():
                frac_alive = (state.prob_remain > 0.01).float().mean()
        else:
            h_deep = torch.zeros_like(x)
            actual_steps = 0
            frac_alive = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------
        # Wide path: single pass
        # ---------------------------------------------------------------
        if self.has_wide_path:
            wide_scale_val = F.softplus(self.wide_scale)
            h_wide = self.wide_block(x, scale=wide_scale_val)
        else:
            h_wide = torch.zeros_like(x)
            wide_scale_val = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------
        # Gate / combine
        # ---------------------------------------------------------------
        gate_aux: dict[str, torch.Tensor] = {}
        # gate_deep_flat / gate_wide_flat are the per-token gate tensors used
        # for downstream logging. In convex mode, gate_wide_flat = 1 - gate_deep_flat.
        if self.layer_type == "dual":
            if self.gate_mode == "convex":
                output, gate, gate_raw, gate_aux = self.dual_gate(
                    x, h_deep, h_wide, gate_override=gate_override
                )
                gate_deep_flat = gate.detach().squeeze(-1)
                gate_wide_flat = (1.0 - gate).detach().squeeze(-1)
                gate_deep_raw_flat = gate_raw.detach().squeeze(-1)
                gate_wide_raw_flat = (1.0 - gate_raw).detach().squeeze(-1)
            else:  # two_gates
                output, gate_deep, gate_wide, gate_deep_raw, gate_wide_raw, gate_aux = (
                    self.dual_gate(x, h_deep, h_wide, gate_override=gate_override)
                )
                gate_deep_flat = gate_deep.detach().squeeze(-1)
                gate_wide_flat = gate_wide.detach().squeeze(-1)
                gate_deep_raw_flat = gate_deep_raw.detach().squeeze(-1)
                gate_wide_raw_flat = gate_wide_raw.detach().squeeze(-1)
        elif self.layer_type == "loop":
            output = h_deep
            gate_deep_flat = torch.ones(B, T, device=device, dtype=x.dtype)
            gate_wide_flat = torch.zeros(B, T, device=device, dtype=x.dtype)
            gate_deep_raw_flat = gate_deep_flat
            gate_wide_raw_flat = gate_wide_flat
        elif self.layer_type == "wide":
            output = h_wide
            gate_deep_flat = torch.zeros(B, T, device=device, dtype=x.dtype)
            gate_wide_flat = torch.ones(B, T, device=device, dtype=x.dtype)
            gate_deep_raw_flat = gate_deep_flat
            gate_wide_raw_flat = gate_wide_flat
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # ---------------------------------------------------------------
        # Diagnostics — Dynamo-traceable, no_grad-wrapped.
        # ---------------------------------------------------------------
        step_metrics = metrics.finalize()

        with torch.no_grad():
            h_deep_norm = h_deep.norm(dim=-1)
            h_wide_norm = h_wide.norm(dim=-1)
            delta_deep  = h_deep - x
            delta_wide  = h_wide - x
            delta_deep_norm = delta_deep.norm(dim=-1)
            delta_wide_norm = delta_wide.norm(dim=-1)

            dd_n = delta_deep_norm.clamp(min=1e-6)
            dw_n = delta_wide_norm.clamp(min=1e-6)
            delta_cos_sim = (delta_deep * delta_wide).sum(dim=-1) / (dd_n * dw_n)

            residual_norm = x.norm(dim=-1)

            # Confound diagnostics use gate_deep (in either mode).
            corr_gate_delta_deep = _batch_corr(gate_deep_flat, delta_deep_norm)
            corr_gate_delta_wide = _batch_corr(gate_deep_flat, delta_wide_norm)

            zero_bt = torch.zeros(B, T, device=device, dtype=x.dtype)
            cross_w2d_norm_per_token = gate_aux.get("cross_w2d_norm_per_token", zero_bt)
            cross_d2w_norm_per_token = gate_aux.get("cross_d2w_norm_per_token", zero_bt)
            zero_scalar = torch.tensor(0.0, device=device)
            cross_scale_deep_val = gate_aux.get("cross_scale_deep", zero_scalar)
            cross_scale_wide_val = gate_aux.get("cross_scale_wide", zero_scalar)

            es = state.expected_steps.detach()

        layer_metrics: dict[str, torch.Tensor] = {
            # ACT
            "expected_steps": state.expected_steps,
            "actual_steps": torch.tensor(float(actual_steps), device=device),
            "residual_mass": state.prob_remain.mean().detach(),
            "frac_alive": frac_alive,
            "wide_scale": wide_scale_val.squeeze().detach(),
            "expected_steps_mean": es.mean(),
            "expected_steps_std":  es.std(),
            "expected_steps_min":  es.min(),
            "expected_steps_max":  es.max(),

            # Gates — ALWAYS log both deep and wide stats, regardless of mode.
            # Convex mode reports gate_wide as (1 - gate_deep), so plots stay
            # consistent across modes.
            "gate_deep_mean": gate_deep_flat.mean(),
            "gate_deep_std":  gate_deep_flat.std(),
            "gate_deep_min":  gate_deep_flat.min(),
            "gate_deep_max":  gate_deep_flat.max(),
            "gate_wide_mean": gate_wide_flat.mean(),
            "gate_wide_std":  gate_wide_flat.std(),
            "gate_wide_min":  gate_wide_flat.min(),
            "gate_wide_max":  gate_wide_flat.max(),
            "gate_deep_raw_mean": gate_deep_raw_flat.mean(),
            "gate_wide_raw_mean": gate_wide_raw_flat.mean(),

            # Logits — convex mode has 1 logit (deep), two_gates has 2.
            # We log both keys in either mode; absent ones default to 0.
            "gate_logit_deep_mean": gate_aux.get(
                "gate_logit_deep_mean",
                gate_aux.get("gate_logit_mean", torch.tensor(0.0, device=device)),
            ),
            "gate_logit_deep_std": gate_aux.get(
                "gate_logit_deep_std",
                gate_aux.get("gate_logit_std",  torch.tensor(0.0, device=device)),
            ),
            "gate_logit_wide_mean": gate_aux.get("gate_logit_wide_mean", torch.tensor(0.0, device=device)),
            "gate_logit_wide_std":  gate_aux.get("gate_logit_wide_std",  torch.tensor(0.0, device=device)),

            # Branch magnitudes
            "h_deep_norm_mean": h_deep_norm.mean(),
            "h_wide_norm_mean": h_wide_norm.mean() if self.has_wide_path else torch.tensor(0.0, device=device),
            "delta_deep_norm_mean": delta_deep_norm.mean(),
            "delta_deep_norm_std":  delta_deep_norm.std(),
            "delta_wide_norm_mean": delta_wide_norm.mean() if self.has_wide_path else torch.tensor(0.0, device=device),
            "delta_wide_norm_std":  delta_wide_norm.std()  if self.has_wide_path else torch.tensor(0.0, device=device),
            "delta_cos_sim_mean": delta_cos_sim.mean() if (self.has_loop_path and self.has_wide_path) else torch.tensor(0.0, device=device),
            "delta_cos_sim_std":  delta_cos_sim.std()  if (self.has_loop_path and self.has_wide_path) else torch.tensor(0.0, device=device),
            "residual_norm_mean": residual_norm.mean(),

            # Confound diagnostics
            "corr_gate_delta_deep": corr_gate_delta_deep,
            "corr_gate_delta_wide": corr_gate_delta_wide,

            # Cross-path
            "cross_scale_deep": cross_scale_deep_val,
            "cross_scale_wide": cross_scale_wide_val,
            "cross_w2d_norm_mean": cross_w2d_norm_per_token.mean(),
            "cross_d2w_norm_mean": cross_d2w_norm_per_token.mean(),

            # Per-step ACT scalars (max_loops,)
            "step_halt_probs":        step_metrics.get("halt_prob_mean",         torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_std":     step_metrics.get("halt_prob_std",          torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_min":     step_metrics.get("halt_prob_min",          torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_max":     step_metrics.get("halt_prob_max",          torch.zeros(self.max_loops, device=device)),
            "step_changes":           step_metrics.get("rel_change",             torch.zeros(self.max_loops, device=device)),
            "loop_scales":            step_metrics.get("loop_scale",             torch.zeros(self.max_loops, device=device)),
            "prob_remain_max":        step_metrics.get("prob_remain_max",        torch.zeros(self.max_loops, device=device)),
            "prob_remain_mean":       step_metrics.get("prob_remain_mean",       torch.zeros(self.max_loops, device=device)),
            "step_h_norm":            step_metrics.get("step_h_norm",            torch.zeros(self.max_loops, device=device)),
            "step_cos_sim_to_input":  step_metrics.get("step_cos_sim_to_input",  torch.zeros(self.max_loops, device=device)),
            "step_rel_norm_to_input": step_metrics.get("step_rel_norm_to_input", torch.zeros(self.max_loops, device=device)),

            # Per-token (B, T) — stacked at top-level for eval-time bag.
            "gate_deep_token_probs":     gate_deep_flat,
            "gate_wide_token_probs":     gate_wide_flat,
            "delta_deep_norm_per_token": delta_deep_norm,
            "delta_wide_norm_per_token": delta_wide_norm if self.has_wide_path else torch.zeros_like(delta_deep_norm),
            "cross_w2d_norm_per_token":  cross_w2d_norm_per_token,
            "cross_d2w_norm_per_token":  cross_d2w_norm_per_token,
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
                ".dual_gate.gate_proj.weight",
                ".dual_gate.proj_w2d.weight",
                ".dual_gate.proj_d2w.weight",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm",
                ".q_norm", ".k_norm",
                ".loop_scales", ".wide_scale",
                ".dual_gate.gate_proj.bias",
                ".router.linear.bias",
                ".dual_gate.cross_scale_deep",
                ".dual_gate.cross_scale_wide",
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
            assert adaptive_config is not None
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

        self._layer_order = sorted(layers.keys(), key=int)

    # ------------------------------------------------------------------
    # Metrics bag construction
    # ------------------------------------------------------------------

    _LOOP_VECTOR_KEYS = [
        "step_halt_probs", "step_halt_prob_std", "step_halt_prob_min", "step_halt_prob_max",
        "step_changes", "loop_scales", "prob_remain_max", "prob_remain_mean",
        "step_h_norm", "step_cos_sim_to_input", "step_rel_norm_to_input",
    ]

    _PER_LAYER_SCALAR_KEYS = [
        "actual_steps", "residual_mass", "frac_alive", "wide_scale",
        "expected_steps_mean", "expected_steps_std", "expected_steps_min", "expected_steps_max",
        # Gates — both deep and wide always logged (in convex mode wide = 1-deep)
        "gate_deep_mean", "gate_deep_std", "gate_deep_min", "gate_deep_max",
        "gate_wide_mean", "gate_wide_std", "gate_wide_min", "gate_wide_max",
        "gate_deep_raw_mean", "gate_wide_raw_mean",
        "gate_logit_deep_mean", "gate_logit_deep_std",
        "gate_logit_wide_mean", "gate_logit_wide_std",
        # Branch magnitudes
        "h_deep_norm_mean", "h_wide_norm_mean",
        "delta_deep_norm_mean", "delta_deep_norm_std",
        "delta_wide_norm_mean", "delta_wide_norm_std",
        "delta_cos_sim_mean", "delta_cos_sim_std",
        "residual_norm_mean",
        # Confound diagnostics
        "corr_gate_delta_deep", "corr_gate_delta_wide",
        # Cross-path
        "cross_scale_deep", "cross_scale_wide",
        "cross_w2d_norm_mean", "cross_d2w_norm_mean",
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
            "ponder_cost_unweighted": total_ponder_cost.detach(),
            "expected_steps": avg_ponder_cost.detach(),
            "normalized_steps": normalized_steps.detach(),
        }

        per_layer_scalars = {
            "ponder_cost": torch.stack([m["expected_steps"].mean().detach() for m in all_layer_metrics]),
        }
        for key in self._PER_LAYER_SCALAR_KEYS:
            per_layer_scalars[key] = stack_key(key)

        per_layer_vectors: dict[str, torch.Tensor] = {}
        for key in self._LOOP_VECTOR_KEYS:
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

    def forward(self, inputs, gate_overrides: Optional[dict[int, object]] = None):
        """
        Args:
            inputs: either a dict {sample_key: tensor} or a bare tensor.
            gate_overrides: optional dict mapping layer_idx -> override.
                Convex mode: override is a (B, T) tensor in [0,1].
                Two-gates mode: override is either a (B, T) tensor (overrides
                deep gate only) or a (deep, wide) tuple of (B, T) | None.
        """
        if isinstance(inputs, dict):
            result = self.forward_impl(inputs[self.sample_key], gate_overrides=gate_overrides)
            return {self.prediction_key: result} if isinstance(result, dict) else {self.prediction_key: result}
        return self.forward_impl(inputs, gate_overrides=gate_overrides)

    def forward_impl(
        self,
        inputs: torch.Tensor,
        gate_overrides: Optional[dict[int, object]] = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
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

        for layer_key in self._layer_order:
            layer_idx = int(layer_key)
            layer_module = self.transformer.h[layer_key]

            if self.use_adaptive:
                layer_override = (
                    gate_overrides.get(layer_idx) if gate_overrides is not None else None
                )
                h, layer_metrics = layer_module(
                    h, token_ids=inputs, gate_override=layer_override,
                )
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
            with torch.no_grad():
                metrics_bag["eval_tokens"] = inputs

                eval_gate_deep = torch.stack([
                    m["gate_deep_token_probs"] for m in all_layer_metrics
                ])
                eval_gate_wide = torch.stack([
                    m["gate_wide_token_probs"] for m in all_layer_metrics
                ])
                metrics_bag["eval_gate_deep"] = eval_gate_deep
                metrics_bag["eval_gate_wide"] = eval_gate_wide

                denom = (eval_gate_deep + eval_gate_wide).clamp(min=1e-6)
                metrics_bag["eval_gate"] = eval_gate_deep / denom

                metrics_bag["eval_expected_steps"] = torch.stack([
                    m["expected_steps"].detach() for m in all_layer_metrics
                ])
                metrics_bag["eval_delta_deep_norm"] = torch.stack([
                    m["delta_deep_norm_per_token"] for m in all_layer_metrics
                ])
                metrics_bag["eval_delta_wide_norm"] = torch.stack([
                    m["delta_wide_norm_per_token"] for m in all_layer_metrics
                ])
                metrics_bag["eval_cross_w2d_norm"] = torch.stack([
                    m["cross_w2d_norm_per_token"] for m in all_layer_metrics
                ])
                metrics_bag["eval_cross_d2w_norm"] = torch.stack([
                    m["cross_d2w_norm_per_token"] for m in all_layer_metrics
                ])

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