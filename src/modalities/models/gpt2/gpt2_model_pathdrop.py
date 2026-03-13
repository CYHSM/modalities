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
    scheduler_type: str = "constant"
    layer_types: Optional[list[str]] = None
    # --- Path dropout ---
    path_dropout_prob: float = 0.0          # probability of dropping a path (0 = disabled)
    path_dropout_mode: str = "uniform"      # "uniform" | "gate_biased"
    # --- Predictive coding capacity module ---
    capacity_conv_kernel: int = 5           # causal conv kernel size for local context
    capacity_ffn_hidden: int = 0            # 0 = use wide_ffn_hidden; >0 = override


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


class AdaptiveRouter(nn.Module):
    """Per-token halting: [h; t_normalized] -> sigmoid -> halt_prob."""

    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, h: torch.Tensor, step_normalized: float) -> torch.Tensor:
        B, T, _ = h.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        logit = self.linear(torch.cat([h, step_feat], dim=-1))
        return torch.sigmoid(logit).squeeze(-1)


class DualPathGate(nn.Module):
    """Per-element gate blending compute path (loop) and capacity path (wide block)."""

    def __init__(self, n_embd: int, init_bias: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.init_bias = init_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, self.init_bias)

    def forward(
        self, routing_input: torch.Tensor, h_deep: torch.Tensor, h_wide: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_proj(routing_input))   # (B, T, D)
        blended = gate * h_deep + (1.0 - gate) * h_wide
        return blended, gate


# =============================================================================
# Path Dropout
# =============================================================================

class PathDropout(nn.Module):
    """
    During training, randomly drops one of the two paths (deep / wide) before
    the dual gate blends them.  This forces each path to be independently
    useful and closes the train-soft / test-hard gap.

    Modes
    -----
    "uniform"      – 50/50 chance of dropping either path.
    "gate_biased"  – preferentially drop the path the gate is already
                     down-weighting (reinforces specialization).

    At eval time this module is a no-op.
    """

    def __init__(self, p: float = 0.0, mode: str = "uniform"):
        super().__init__()
        assert 0.0 <= p <= 1.0, f"path_dropout_prob must be in [0, 1], got {p}"
        assert mode in ("uniform", "gate_biased"), f"Unknown mode {mode}"
        self.p = p
        self.mode = mode

    def forward(
        self,
        h_deep: torch.Tensor,
        h_wide: torch.Tensor,
        gate: torch.Tensor,           # (B, T, D) — gate values before blending
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns (h_deep, h_wide, info).  Fully dynamo-traceable — no .item()
        calls, no Python-level branching on tensor values.

        All decisions are made with torch.where so the graph is static.

        info keys:
            "path_dropped"  : 0 = none, 1 = deep dropped, 2 = wide dropped
            "drop_applied"  : 1.0 if a drop happened this call, else 0.0
        """
        device = h_deep.device
        zero = torch.tensor(0.0, device=device)

        # No-op at eval or when disabled
        if not self.training or self.p == 0.0:
            return h_deep, h_wide, {
                "path_dropped": torch.tensor(0, device=device),
                "drop_applied": zero,
            }

        # ---- All randomness as tensors, no .item() ----

        # 1) Should we drop at all?  (single Bernoulli)
        do_drop = torch.rand((), device=device) < self.p          # scalar bool tensor

        # 2) Which path to drop?
        if self.mode == "uniform":
            choose_deep = torch.rand((), device=device) < 0.5
        elif self.mode == "gate_biased":
            gate_mean = gate.detach().mean()
            choose_deep = torch.rand((), device=device) < gate_mean
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        drop_deep = do_drop & choose_deep
        drop_wide = do_drop & (~choose_deep)

        h_deep = torch.where(drop_deep, torch.zeros_like(h_deep), h_deep)
        h_wide = torch.where(drop_wide, torch.zeros_like(h_wide), h_wide)

        path_dropped = torch.where(drop_deep, torch.tensor(1, device=device),
                       torch.where(drop_wide, torch.tensor(2, device=device),
                                              torch.tensor(0, device=device)))
        drop_applied = torch.where(do_drop, torch.tensor(1.0, device=device), zero)

        return h_deep, h_wide, {
            "path_dropped": path_dropped,
            "drop_applied": drop_applied,
        }


# =============================================================================
# Capacity Module (Predictive Coding — capacity path)
# =============================================================================

class CapacityModule(nn.Module):
    """
    Cheap capacity path for predictive coding architecture.

    Produces a predicted residual update Δ for the current layer position.
    Uses causal depthwise convolution for local context (replaces attention)
    followed by a wide FFN for knowledge retrieval.

    No self-attention — cross-token interaction is local and O(T·D·k),
    not O(T²·D).  The expensive global attention lives in the loop path.

    Architecture:
        norm(x) → causal_depthwise_conv → SiLU → ffn_norm → wide_FFN → Δ
    """

    def __init__(
        self,
        n_embd: int,
        ffn_hidden: int,
        bias: bool,
        dropout: float,
        activation_type: ActivationType,
        conv_kernel: int = 5,
        pre_norm: nn.Module = None,
        ffn_norm: nn.Module = None,
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.ffn_norm = ffn_norm
        self.conv_kernel = conv_kernel

        # Causal depthwise convolution: each channel sees its local neighborhood
        # Channel mixing is delegated to the FFN that follows
        self.causal_conv = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=conv_kernel,
            groups=n_embd,      # depthwise — no cross-channel mixing
            bias=bias,
        )
        self.conv_act = nn.SiLU()

        # Wide FFN: the actual "knowledge store"
        # This is where most of the capacity module's parameters live
        if activation_type == ActivationType.GELU:
            self.ffn = TransformerMLP(
                n_embd=n_embd, ffn_hidden=ffn_hidden,
                bias=bias, dropout=dropout,
            )
        elif activation_type == ActivationType.SWIGLU:
            self.ffn = SwiGLU(
                n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
        else:
            raise NotImplementedError(f"Unsupported activation: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input hidden states

        Returns:
            delta: (B, T, D) predicted residual update
        """
        # Pre-norm
        h = self.pre_norm(x)

        # Causal depthwise convolution for local context
        # Conv1d expects (B, C, T), we have (B, T, C)
        h_t = h.transpose(1, 2)                                    # (B, D, T)
        h_t = F.pad(h_t, (self.conv_kernel - 1, 0))                # left-pad for causality
        h_t = self.causal_conv(h_t)                                 # (B, D, T)
        h_t = self.conv_act(h_t)
        h_t = h_t.transpose(1, 2)                                  # (B, T, D)

        # FFN norm → wide FFN (knowledge retrieval)
        h_t = self.ffn_norm(h_t)
        delta = self.ffn(h_t)

        return delta


# =============================================================================
# Adaptive Recursive Block (supports loop, wide, dual, AND predictive)
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
        capacity_module: Optional[CapacityModule] = None,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        # --- Loop path (shared for loop, dual, predictive) ---
        self.has_loop_path = layer_type in ["loop", "dual", "predictive"]
        if self.has_loop_path:
            self.router = AdaptiveRouter(n_embd)
            self.loop_scales = nn.Parameter(torch.full((self.max_loops,), self._INIT_SCALE_RAW))

        # --- Wide path (for wide, dual) ---
        self.has_wide_path = layer_type in ["wide", "dual"]
        if self.has_wide_path:
            self.wide_block = wide_block
            self.wide_scale = nn.Parameter(torch.tensor([self._INIT_SCALE_RAW]))

        # --- Dual gate + path dropout (for dual only) ---
        if layer_type == "dual":
            self.dual_gate = DualPathGate(n_embd=n_embd, init_bias=adaptive_config.wide_ffn_gate_init_bias)
            self.path_dropout = PathDropout(
                p=adaptive_config.path_dropout_prob,
                mode=adaptive_config.path_dropout_mode,
            )

        # --- Predictive coding path ---
        self.has_capacity_path = layer_type == "predictive"
        if self.has_capacity_path:
            assert capacity_module is not None, "capacity_module required for predictive layer type"
            self.capacity_module = capacity_module
            self.capacity_scale = nn.Parameter(torch.tensor([self._INIT_SCALE_RAW]))

    # ------------------------------------------------------------------
    # Predictive coding forward
    # ------------------------------------------------------------------

    def _forward_predictive(
        self, x: torch.Tensor, token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Predictive coding forward pass:
          1. Capacity module predicts the residual update Δ
          2. Draft = x + scale * Δ  (capacity's best guess at layer output)
          3. Loop iteratively refines the draft via shared block + ACT
          4. Output = ACT-weighted refinement

        The loop starts from a better initial condition when the prediction
        is good, so ACT halts earlier → adaptive compute for free.
        """
        B, T, D = x.shape
        device = x.device

        # ---- 1. Capacity prediction ----
        delta_predicted = self.capacity_module(x)                   # (B, T, D)
        cap_scale = F.softplus(self.capacity_scale)
        draft = x + cap_scale * delta_predicted                     # (B, T, D)

        # ---- 2. Iterative refinement from draft ----
        state = HaltingState.init(B, T, D, device=device, dtype=x.dtype)
        metrics = StepMetrics(self.max_loops, device)

        step_denom = max(1, self.max_loops - 1)
        h_loop = draft
        actual_steps = 0

        for step in range(self.max_loops):
            actual_steps = step + 1

            scale = F.softplus(self.loop_scales[step])
            metrics.log("loop_scale", scale.detach())

            h_prev = h_loop
            h_loop = self.block(h_loop, scale=scale)

            halt_prob = self.router(h_loop, step_normalized=step / step_denom)
            state.update(h_loop, halt_prob, step)

            # Per-step diagnostics
            rel_change = (h_loop - h_prev).norm(dim=-1) / (h_prev.norm(dim=-1) + 1e-6)

            metrics.log("halt_prob_mean", halt_prob.detach().mean())
            metrics.log("halt_prob_std", halt_prob.detach().std())
            metrics.log("halt_prob_min", halt_prob.detach().min())
            metrics.log("halt_prob_max", halt_prob.detach().max())
            metrics.log("rel_change", rel_change.mean())
            metrics.log("prob_remain_max", state.prob_remain.max().detach())
            metrics.log("prob_remain_mean", state.prob_remain.mean().detach())

        state.finalize(h_loop, actual_steps)
        output = state.output
        frac_alive = (state.prob_remain.detach() > 0.01).float().mean()

        # ---- 3. Predictive coding diagnostics ----
        prediction_norm = delta_predicted.detach().norm(dim=-1).mean()
        loop_correction = output - draft                            # what the loop changed
        loop_correction_norm = loop_correction.detach().norm(dim=-1).mean()

        # Prediction quality: how much of the final update came from capacity vs loop
        total_update_norm = (output - x).detach().norm(dim=-1).mean() + 1e-8
        prediction_ratio = (cap_scale.detach() * prediction_norm) / total_update_norm

        # ---- 4. Build metrics dict ----
        step_metrics = metrics.finalize()
        es = state.expected_steps.detach()

        layer_metrics = {
            # Per-token (B, T) — used for ponder cost
            "expected_steps": state.expected_steps,
            # Scalars
            "actual_steps": torch.tensor(float(actual_steps), device=device),
            "residual_mass": state.prob_remain.mean().detach(),
            "frac_alive": frac_alive,
            "wide_scale": cap_scale.squeeze().detach(),  # reuse slot for capacity scale
            # Expected steps distribution
            "expected_steps_mean": es.mean(),
            "expected_steps_std": es.std(),
            "expected_steps_min": es.min(),
            "expected_steps_max": es.max(),
            # Dual gate — not used in predictive, fill with meaningful defaults
            "dual_gate_mean": torch.tensor(1.0, device=device),
            "dual_gate_std": torch.tensor(0.0, device=device),
            "dual_gate_min": torch.tensor(1.0, device=device),
            "dual_gate_max": torch.tensor(1.0, device=device),
            "dual_gate_token_probs": torch.ones(B, T, device=device, dtype=x.dtype),
            # Path output magnitudes
            "deep_block_norm": loop_correction_norm,
            "wide_block_norm": prediction_norm,         # reuse: "wide" = capacity prediction
            # Per-step vectors
            "step_halt_probs": step_metrics.get("halt_prob_mean", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_std": step_metrics.get("halt_prob_std", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_min": step_metrics.get("halt_prob_min", torch.zeros(self.max_loops, device=device)),
            "step_halt_prob_max": step_metrics.get("halt_prob_max", torch.zeros(self.max_loops, device=device)),
            "step_changes": step_metrics.get("rel_change", torch.zeros(self.max_loops, device=device)),
            "loop_scales": step_metrics.get("loop_scale", torch.zeros(self.max_loops, device=device)),
            "prob_remain_max": step_metrics.get("prob_remain_max", torch.zeros(self.max_loops, device=device)),
            "prob_remain_mean": step_metrics.get("prob_remain_mean", torch.zeros(self.max_loops, device=device)),
            # Path dropout — not applicable for predictive
            "path_dropped": torch.tensor(0, device=device),
            "path_drop_applied": torch.tensor(0.0, device=device),
            # === NEW: Predictive coding specific metrics ===
            "prediction_norm": prediction_norm,
            "loop_correction_norm": loop_correction_norm,
            "prediction_ratio": prediction_ratio,
            "capacity_scale": cap_scale.squeeze().detach(),
        }

        return output, layer_metrics

    # ------------------------------------------------------------------
    # Main forward (dispatches to predictive or original paths)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # --- Predictive coding path: entirely separate flow ---
        if self.layer_type == "predictive":
            return self._forward_predictive(x, token_ids)

        # === Original dual / loop / wide paths (unchanged) ===
        B, T, D = x.shape
        device = x.device

        # =================================================================
        # 1) Compute path: full block looped with ACT
        # =================================================================
        state = HaltingState.init(B, T, D, device=device, dtype=x.dtype)
        metrics = StepMetrics(self.max_loops, device)

        if self.has_loop_path:
            step_denom = max(1, self.max_loops - 1)
            h_loop = x
            actual_steps = 0

            for step in range(self.max_loops):
                actual_steps = step + 1

                scale = F.softplus(self.loop_scales[step])
                metrics.log("loop_scale", scale.detach())
                h_prev = h_loop
                h_loop = self.block(h_loop, scale=scale)

                halt_prob = self.router(h_loop, step_normalized=step / step_denom)
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
        # 2b) Path dropout + gating (dual only)
        # =================================================================
        if self.layer_type == "dual":
            raw_gate = torch.sigmoid(self.dual_gate.gate_proj(x))
            h_deep, h_wide, drop_info = self.path_dropout(h_deep, h_wide, raw_gate)
            output = raw_gate * h_deep + (1.0 - raw_gate) * h_wide
            gate = raw_gate
        elif self.layer_type == "loop":
            output = h_deep
            gate = None
            drop_info = {"path_dropped": torch.tensor(0, device=device),
                         "drop_applied": torch.tensor(0.0, device=device)}
        elif self.layer_type == "wide":
            output = h_wide
            gate = None
            drop_info = {"path_dropped": torch.tensor(0, device=device),
                         "drop_applied": torch.tensor(0.0, device=device)}
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        # =================================================================
        # 3) Build layer metrics
        # =================================================================
        step_metrics = metrics.finalize()
        es = state.expected_steps.detach()

        if gate is not None:
            gate_d = gate.detach()
            gate_mean = gate_d.mean()
            gate_std = gate_d.std()
            gate_min = gate_d.min()
            gate_max = gate_d.max()
            dual_gate_token_probs = gate_d.mean(dim=-1)
        else:
            gate_mean = torch.tensor(1.0, device=device)
            gate_std = torch.tensor(0.0, device=device)
            gate_min = torch.tensor(1.0, device=device)
            gate_max = torch.tensor(1.0, device=device)
            dual_gate_token_probs = torch.ones(B, T, device=device, dtype=x.dtype)

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
            "dual_gate_mean": gate_mean,
            "dual_gate_std": gate_std,
            "dual_gate_min": gate_min,
            "dual_gate_max": gate_max,
            "dual_gate_token_probs": dual_gate_token_probs,
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
            "path_dropped": drop_info["path_dropped"],
            "path_drop_applied": drop_info["drop_applied"],
            # Predictive coding metrics — zeros for non-predictive layers
            "prediction_norm": torch.tensor(0.0, device=device),
            "loop_correction_norm": torch.tensor(0.0, device=device),
            "prediction_ratio": torch.tensor(0.0, device=device),
            "capacity_scale": torch.tensor(0.0, device=device),
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
                # Capacity module linear weights
                ".capacity_module.ffn",
                ".capacity_module.causal_conv.weight",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm",
                ".q_norm", ".k_norm",
                ".loop_scales", ".wide_scale", ".dual_gate.gate_proj.bias",
                ".router.linear.bias",
                # Capacity module norms and scales
                ".capacity_scale",
                ".capacity_module.pre_norm",
                ".capacity_module.ffn_norm",
                ".capacity_module.causal_conv.bias",
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

        def create_capacity_module(cap_ffn_hidden):
            """Create a CapacityModule for predictive layer types."""
            return CapacityModule(
                n_embd=n_embd,
                ffn_hidden=cap_ffn_hidden,
                bias=bias,
                dropout=dropout,
                activation_type=activation_type,
                conv_kernel=adaptive_config.capacity_conv_kernel,
                pre_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
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

                if l_type == "predictive":
                    # Predictive coding: narrow block (loop) + capacity module (no attention)
                    narrow_block = create_block()
                    cap_hidden = (
                        adaptive_config.capacity_ffn_hidden
                        if adaptive_config.capacity_ffn_hidden > 0
                        else adaptive_config.wide_ffn_hidden
                    )
                    cap_module = create_capacity_module(cap_hidden)
                    layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                        block=narrow_block,
                        adaptive_config=adaptive_config,
                        n_embd=n_embd,
                        layer_idx=layer_idx,
                        n_layers=n_layer,
                        wide_block=None,
                        layer_type="predictive",
                        capacity_module=cap_module,
                    )
                elif l_type in ["loop", "dual", "wide"]:
                    # Original layer types
                    narrow_block = create_block() if l_type in ["loop", "dual"] else None
                    wide_block = (
                        create_block(ffn_hidden_override=adaptive_config.wide_ffn_hidden)
                        if l_type in ["wide", "dual"]
                        else None
                    )
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
                    raise ValueError(f"Unknown layer type: {l_type}")
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

    # Keys in layer_metrics that are per-step vectors (max_loops,)
    _VECTOR_KEYS = [
        "step_halt_probs", "step_halt_prob_std", "step_halt_prob_min", "step_halt_prob_max",
        "step_changes", "loop_scales", "prob_remain_max", "prob_remain_mean",
    ]

    # Keys that are per-layer scalars (stacked to (n_layers,))
    _PER_LAYER_SCALAR_KEYS = [
        "actual_steps", "residual_mass", "frac_alive", "wide_scale",
        "expected_steps_mean", "expected_steps_std", "expected_steps_min", "expected_steps_max",
        "dual_gate_mean", "dual_gate_std", "dual_gate_min", "dual_gate_max",
        "deep_block_norm", "wide_block_norm",
        # Predictive coding metrics
        "prediction_norm", "loop_correction_norm", "prediction_ratio", "capacity_scale",
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

        # Path dropout diagnostics (per-layer scalars)
        per_layer_scalars["path_dropped"] = stack_key("path_dropped").float()
        per_layer_scalars["path_drop_applied"] = stack_key("path_drop_applied")

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
                h = layer_module(h, scale=1.0 / (int(layer_key) + 1))

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        if not self.use_adaptive:
            return logits

        weighted_ponder_loss, metrics_bag = self._build_metrics_bag(
            all_layer_metrics, total_ponder_cost, device, logits.dtype,
        )

        if not self.training:
            metrics_bag["eval_tokens"] = inputs.detach()
            metrics_bag["eval_gate_probs"] = torch.stack([m.get("dual_gate_token_probs", torch.ones_like(inputs, dtype=logits.dtype)) for m in all_layer_metrics])

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