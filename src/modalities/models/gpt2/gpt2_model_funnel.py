import logging
import math
from abc import abstractmethod
from enum import Enum
from typing import Annotated, Optional, overload, Literal

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


# ==========================================
# Capacity Curve Scheduling
# ==========================================

class CapacityCurveType(str, Enum):
    """Available capacity curve shapes."""
    NONE = "none"           # All layers same size
    LINEAR = "linear"       # Triangle shape
    GAUSSIAN = "gaussian"   # Bell curve
    EXPONENTIAL = "exponential"  # Sharp peak
    COSINE = "cosine"       # Smooth cosine


class CapacityCurveConfig(BaseModel):
    """
    Configuration for layer-wise capacity scaling.
    
    This creates an "inverted bottleneck" architecture where capacity
    varies across depth - small at edges, large in the middle (or wherever peak is).
    """
    curve_type: CapacityCurveType = CapacityCurveType.NONE
    peak_layer: Optional[int] = None  # None = auto (middle)
    min_ratio: float = Field(default=0.25, ge=0.1, le=1.0)  # Minimum capacity ratio
    sharpness: float = Field(default=3.0, ge=0.5, le=10.0)  # For exponential curve
    sigma: float = Field(default=0.4, ge=0.1, le=1.0)  # For gaussian curve
    
    # What to scale
    scale_ffn: bool = True
    scale_heads: bool = False  # More experimental, can cause issues


def compute_capacity_schedule(
    n_layers: int,
    config: CapacityCurveConfig,
) -> list[float]:
    """
    Compute capacity multipliers for each layer.
    
    Returns:
        List of floats in [min_ratio, 1.0], one per layer.
    """
    if config.curve_type == CapacityCurveType.NONE:
        return [1.0] * n_layers
    
    # Default peak to middle
    peak_layer = config.peak_layer if config.peak_layer is not None else n_layers // 2
    peak_layer = max(0, min(n_layers - 1, peak_layer))  # Clamp
    
    # Max distance from peak (for normalization)
    max_dist = max(peak_layer, n_layers - 1 - peak_layer)
    if max_dist == 0:
        max_dist = 1  # Edge case: single layer
    
    multipliers = []
    
    for i in range(n_layers):
        # Normalized distance from peak: 0 = at peak, 1 = furthest
        dist = abs(i - peak_layer) / max_dist
        
        if config.curve_type == CapacityCurveType.LINEAR:
            mult = 1.0 - dist
            
        elif config.curve_type == CapacityCurveType.GAUSSIAN:
            mult = math.exp(-0.5 * (dist / config.sigma) ** 2)
            
        elif config.curve_type == CapacityCurveType.EXPONENTIAL:
            mult = math.exp(-config.sharpness * dist)
            
        elif config.curve_type == CapacityCurveType.COSINE:
            mult = 0.5 * (1 + math.cos(math.pi * dist))
        
        else:
            mult = 1.0
        
        # Scale to [min_ratio, 1.0]
        mult = config.min_ratio + mult * (1.0 - config.min_ratio)
        multipliers.append(mult)
    
    return multipliers


def apply_capacity_to_dim(base_dim: int, multiplier: float, multiple_of: int = 128) -> int:
    """
    Apply capacity multiplier to a dimension, ensuring alignment.
    
    Args:
        base_dim: Original dimension (e.g., ffn_hidden=3072)
        multiplier: Capacity multiplier in [0, 1]
        multiple_of: Ensure result is divisible by this (for efficiency)
    
    Returns:
        Scaled dimension, aligned to multiple_of
    """
    scaled = int(base_dim * multiplier)
    # Round to nearest multiple
    aligned = max(multiple_of, (scaled // multiple_of) * multiple_of)
    return aligned


def apply_capacity_to_heads(base_heads: int, multiplier: float, min_heads: int = 1) -> int:
    """
    Apply capacity multiplier to number of heads.
    
    Args:
        base_heads: Original number of heads
        multiplier: Capacity multiplier in [0, 1]
        min_heads: Minimum number of heads
    
    Returns:
        Scaled number of heads (at least min_heads)
    """
    scaled = int(base_heads * multiplier)
    return max(min_heads, scaled)


def visualize_capacity_schedule(multipliers: list[float], bar_width: int = 20) -> str:
    """Pretty print the capacity schedule."""
    lines = ["Layer | Capacity | Visual"]
    lines.append("-" * 45)
    
    for i, mult in enumerate(multipliers):
        bar_len = int(mult * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        lines.append(f"  {i:2d}  |  {mult:.3f}  | {bar}")
    
    return "\n".join(lines)


# ==========================================
# Existing Components (unchanged)
# ==========================================

class AdaptiveComputationConfig(BaseModel):
    enable_adaptive: bool = False
    max_loops: int = 10
    halt_threshold: float = 0.99
    ponder_penalty_weight: float = 0.01


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


class QueryKeyValueTransform(nn.Module):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class IdentityTransform(QueryKeyValueTransform):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# ==========================================
# Updated Config with Capacity Curve
# ==========================================

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
    
    # NEW: Capacity curve configuration
    capacity_curve_config: Optional[CapacityCurveConfig] = None

    @model_validator(mode="after")
    def check_divisibility(self) -> "GPT2LLMConfig":
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError("n_head_q must be divisible by n_head_kv")
        return self

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


# ==========================================
# Attention & MLP (unchanged)
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        attention_config: AttentionConfig,
        attention_impl: AttentionImplementation,
        bias: bool,
        dropout: float,
    ):
        super().__init__()
        assert n_embd % n_head_q == 0, "`n_embd` needs to be divisible by `n_head_q`."
        assert n_head_q % n_head_kv == 0, "`n_head_q` needs to be divisible by `n_head_kv`."

        self.n_rep = n_head_q // n_head_kv
        self.attention_impl = attention_impl

        self.q_attn = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.k_attn = nn.Linear(in_features=n_embd, out_features=n_embd // self.n_rep, bias=bias)
        self.v_attn = nn.Linear(in_features=n_embd, out_features=n_embd // self.n_rep, bias=bias)
        self.c_proj = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)

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
            self.q_norm = attention_config.qk_norm_config.norm_type.value(
                **dict(attention_config.qk_norm_config.config)
            )
            self.k_norm = attention_config.qk_norm_config.norm_type.value(
                **dict(attention_config.qk_norm_config.config)
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q_attn(x), self.k_attn(x), self.v_attn(x)

    @staticmethod
    def execute_qkv_transforms(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qkv_transforms: nn.ModuleList, n_head_q: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, embedding_dim = q.size()
        n_head_dim = embedding_dim // n_head_q

        q = q.view(batch_size, sequence_length, n_head_q, n_head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()

        for transform in qkv_transforms:
            q, k, v = transform(q, k, v)

        return q, k, v

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        B, nh_kv, T, hs = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    @classmethod
    def repeat_kv_heads(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        n_head_q = q.shape[1]
        n_head_kv = k.shape[1]
        if n_head_q != n_head_kv:
            n_rep = n_head_q // n_head_kv
            k = cls._repeat_kv(k, n_rep)
            v = cls._repeat_kv(v, n_rep)
        return k, v

    @classmethod
    def execute_attention(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: float,
        attention_impl: AttentionImplementation,
    ) -> torch.Tensor:
        if attention_impl == AttentionImplementation.MANUAL:
            k, v = cls.repeat_kv_heads(q, k, v)
            y = manual_scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=None, dropout_p=dropout, is_causal=True,
            )
            y = y.transpose(1, 2).contiguous()
        elif attention_impl == AttentionImplementation.PYTORCH_FLASH:
            k, v = cls.repeat_kv_heads(q, k, v)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=None, dropout_p=dropout, is_causal=True,
            )
            y = y.transpose(1, 2).contiguous()
        elif attention_impl == AttentionImplementation.DAO_FLASH:
            if flash_attn_func is None:
                raise NotImplementedError("ERROR! Dao Flash Attention is not installed.")
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            y = flash_attn_func(q, k, v, dropout_p=dropout, causal=True, softmax_scale=None, window_size=(-1, -1))
        else:
            raise NotImplementedError(f"Attention implementation {attention_impl} not supported")
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(in_features=n_embd, out_features=ffn_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(in_features=ffn_hidden, out_features=n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        n_head_q: int,
        n_head_kv: int,
        activation_type: ActivationType,
        attention_impl: AttentionImplementation,
        attention_config: AttentionConfig,
        dropout: float,
        ffn_hidden: int,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        enforce_swiglu_hidden_dim_multiple_of: int,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self._check_ffn_hidden_dim(n_embd=n_embd, ffn_hidden=ffn_hidden)
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_config=attention_config,
            attention_impl=attention_impl,
            bias=bias,
            dropout=dropout,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(
                n_embd=n_embd,
                ffn_hidden=ffn_hidden,
                bias=bias,
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
        else:
            raise NotImplementedError("unimplemented activation")

    def _check_ffn_hidden_dim(self, n_embd: int, ffn_hidden: int) -> None:
        expected_hidden_dim = 4 * n_embd
        if ffn_hidden != expected_hidden_dim:
            logger.warning(
                f"Expected `ffn_hidden` to be 4 * `n_embd` ({expected_hidden_dim}), "
                f"but got `n_embd = {n_embd}` and `ffn_hidden = {ffn_hidden}`."
            )

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        x = x + scale * self.attn(self.attention_norm(x))
        x = x + scale * self.mlp(self.ffn_norm(x))
        return x


# ==========================================
# Adaptive Components (unchanged)
# ==========================================

class AdaptiveRouter(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.net = nn.Linear(n_embd + 2, 1, bias=bias)

    def forward(self, x: torch.Tensor, step_normalized: float, cos_sim: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=x.device, dtype=x.dtype)
        logits = self.net(torch.cat([x, step_feat, cos_sim], dim=-1))
        return torch.sigmoid(logits).squeeze(-1)


class AdaptiveRecursiveBlock(nn.Module):
    def __init__(
        self,
        block: GPT2Block,
        adaptive_config: AdaptiveComputationConfig,
        n_embd: int,
        layer_idx: int,
    ):
        super().__init__()
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.router = AdaptiveRouter(n_embd)
        self.step_gate = nn.Parameter(torch.tensor([0.01]))
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        
        h = x
        accumulated_output = torch.zeros_like(h)
        
        prob_remain = torch.ones(B, T, device=x.device, dtype=x.dtype)
        expected_steps = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        total_cos_sim = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        
        denom = max(1, self.max_loops - 1)

        for step in range(self.max_loops):
            current_depth = (self.layer_idx * self.max_loops) + step + 1
            lns_scale = 1.0 / current_depth
            
            h_prev = h
            h = self.block(h, scale=lns_scale)
            
            cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8).unsqueeze(-1)
            
            step_norm = step / denom
            halt_prob = self.router(h, step_norm, cos_sim)

            if step == self.max_loops - 1:
                p_stop_here = prob_remain
                prob_remain = torch.zeros_like(prob_remain)
            else:
                p_stop_here = prob_remain * halt_prob
                prob_remain = prob_remain * (1.0 - halt_prob)

            accumulated_output = accumulated_output + (h * p_stop_here.unsqueeze(-1))
            expected_steps = expected_steps + p_stop_here * (step + 1)
            total_cos_sim = total_cos_sim + (cos_sim.squeeze(-1) * p_stop_here)
            
            if not self.training:
                if prob_remain.max() < (1.0 - self.config.halt_threshold):
                    break

        if not self.training and prob_remain.sum() > 0:
            accumulated_output = accumulated_output + (h * prob_remain.unsqueeze(-1))
            final_cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8)
            total_cos_sim = total_cos_sim + (prob_remain * final_cos_sim)

        return accumulated_output, expected_steps, total_cos_sim


# ==========================================
# Main Model with Capacity Curve Support
# ==========================================

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
        capacity_curve_config: Optional[CapacityCurveConfig] = None,
    ):
        weight_decay_groups = {
            "linear": [".attn", ".mlp", ".lm_head.weight", ".router"],
            "embedding": [".wte", ".wpe", ".step_emb"],
            "layernorm": [".attention_norm", ".ffn_norm", ".lm_head_norm", ".step_gate"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.poe_type = poe_type

        # Compute capacity schedule
        if capacity_curve_config is None:
            capacity_curve_config = CapacityCurveConfig()  # Default: no scaling
        
        capacity_multipliers = compute_capacity_schedule(n_layer, capacity_curve_config)
        self.capacity_multipliers = capacity_multipliers  # Store for inspection
        
        # Log the capacity schedule
        if capacity_curve_config.curve_type != CapacityCurveType.NONE:
            logger.info(f"Using capacity curve: {capacity_curve_config.curve_type.value}")
            logger.info(f"\n{visualize_capacity_schedule(capacity_multipliers)}")

        # Position embeddings
        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=sequence_length, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention_config.qkv_transforms
        ]:
            raise ValueError('It is expected to use "RotaryTransform" together with "NOPE".')

        self.use_adaptive = adaptive_config is not None and adaptive_config.enable_adaptive
        self.adaptive_config = adaptive_config

        # Build layers with capacity scaling
        layers = {}
        total_params_info = []
        
        for layer_idx in range(n_layer):
            mult = capacity_multipliers[layer_idx]
            
            # Scale FFN hidden dimension
            if capacity_curve_config.scale_ffn:
                layer_ffn_hidden = apply_capacity_to_dim(
                    ffn_hidden, mult, 
                    multiple_of=128 if activation_type == ActivationType.GELU else enforce_swiglu_hidden_dim_multiple_of
                )
            else:
                layer_ffn_hidden = ffn_hidden
            
            # Scale heads (optional, more experimental)
            if capacity_curve_config.scale_heads:
                layer_n_head_q = apply_capacity_to_heads(n_head_q, mult, min_heads=1)
                layer_n_head_kv = apply_capacity_to_heads(n_head_kv, mult, min_heads=1)
                # Ensure divisibility
                while layer_n_head_q % layer_n_head_kv != 0:
                    layer_n_head_kv = max(1, layer_n_head_kv - 1)
            else:
                layer_n_head_q = n_head_q
                layer_n_head_kv = n_head_kv
            
            total_params_info.append({
                "layer": layer_idx,
                "mult": mult,
                "ffn_hidden": layer_ffn_hidden,
                "n_head_q": layer_n_head_q,
            })
            
            block = GPT2Block(
                n_embd=n_embd,
                bias=bias,
                n_head_q=layer_n_head_q,
                n_head_kv=layer_n_head_kv,
                activation_type=activation_type,
                attention_impl=attention_implementation,
                attention_config=attention_config,
                dropout=dropout,
                ffn_hidden=layer_ffn_hidden,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )
            
            if self.use_adaptive:
                layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                    block=block,
                    adaptive_config=adaptive_config,
                    n_embd=n_embd,
                    layer_idx=layer_idx,
                )
            else:
                layers[str(layer_idx)] = block

        # Log layer configurations
        if capacity_curve_config.curve_type != CapacityCurveType.NONE:
            logger.info("Layer configurations:")
            for info in total_params_info:
                logger.info(f"  Layer {info['layer']}: mult={info['mult']:.3f}, ffn_hidden={info['ffn_hidden']}, n_head_q={info['n_head_q']}")

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

    def get_capacity_info(self) -> str:
        """Return a string describing the capacity schedule."""
        return visualize_capacity_schedule(self.capacity_multipliers)

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(inputs, dict):
            result = self.forward_impl(inputs[self.sample_key])
            if isinstance(result, dict):
                return {self.prediction_key: result}
            else:
                return {self.prediction_key: result}
        else:
            return self.forward_impl(inputs)

    def forward_impl(self, inputs: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        device = inputs.device
        seq_len = inputs.size(1)
        assert seq_len <= self.sequence_length, f"Cannot forward sequence of length {seq_len}, max is {self.sequence_length}."

        h = self.transformer.wte(inputs) if hasattr(self.transformer, "wte") else inputs

        if self.poe_type is PositionTypes.ABSOLUTE and hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            h = h + pos_emb

        h = self.transformer.drop(h) if hasattr(self.transformer, "drop") else h

        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)
        num_adaptive_layers = 0
        step_gate_values = []
        per_layer_ponder_costs = []
        per_layer_cos_sims = []

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer_module = self.transformer.h[layer_key]
            layer_idx = int(layer_key)

            if self.use_adaptive:
                h, cost, cos_sim = layer_module(h)
                
                cost_mean = cost.mean()
                sim_mean = cos_sim.mean()
                total_ponder_cost = total_ponder_cost + cost_mean
                per_layer_ponder_costs.append(cost_mean)
                per_layer_cos_sims.append(sim_mean)
                num_adaptive_layers += 1
                step_gate_values.append(torch.tanh(layer_module.step_gate))
            else:
                lns_scale = 1.0 / (layer_idx + 1)
                h = layer_module(h, scale=lns_scale)

        h = self.transformer.lm_head_norm(h) if hasattr(self.transformer, "lm_head_norm") else h
        logits = self.transformer.lm_head(h) if hasattr(self.transformer, "lm_head") else h

        if self.use_adaptive:
            avg_ponder_cost = total_ponder_cost / num_adaptive_layers if num_adaptive_layers > 0 else torch.tensor(0.0, dtype=h.dtype, device=device)
            normalized_steps = (avg_ponder_cost - 1.0) / (self.adaptive_config.max_loops - 1.0) if self.adaptive_config.max_loops > 1 else torch.tensor(0.0, dtype=h.dtype, device=device)
            weighted_ponder_loss = (normalized_steps * self.adaptive_config.ponder_penalty_weight).to(logits.dtype)
            avg_step_gate = torch.mean(torch.stack(step_gate_values)) if step_gate_values else torch.tensor(0.0, device=device)
            layer_costs_tensor = torch.stack(per_layer_ponder_costs) if per_layer_ponder_costs else torch.tensor([], device=device)
            layer_sims_tensor = torch.stack(per_layer_cos_sims) if per_layer_cos_sims else torch.tensor([], device=device)

            return {
                "logits": logits,
                "ponder_loss": weighted_ponder_loss,
                "ponder_cost_unweighted": total_ponder_cost,
                "expected_steps": avg_ponder_cost,
                "normalized_steps": normalized_steps,
                "step_gate_mean": avg_step_gate,
                "per_layer_ponder_costs": layer_costs_tensor,
                "per_layer_cos_sims": layer_sims_tensor,
            }
        
        return logits


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
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


# ==========================================
# Convenience functions for quick testing
# ==========================================

def print_capacity_curves(n_layers: int = 12):
    """Print all available capacity curves for comparison."""
    print(f"\n{'='*60}")
    print(f"Capacity Curves Comparison (n_layers={n_layers}, peak=middle)")
    print(f"{'='*60}\n")
    
    for curve_type in CapacityCurveType:
        if curve_type == CapacityCurveType.NONE:
            continue
        
        config = CapacityCurveConfig(
            curve_type=curve_type,
            peak_layer=n_layers // 2,
            min_ratio=0.25,
        )
        multipliers = compute_capacity_schedule(n_layers, config)
        
        print(f"\n{curve_type.value.upper()}:")
        print(visualize_capacity_schedule(multipliers))


def estimate_param_reduction(
    n_layer: int,
    n_embd: int,
    ffn_hidden: int,
    capacity_config: CapacityCurveConfig,
) -> dict:
    """Estimate parameter count with and without capacity curve."""
    
    # Baseline: all layers same size
    baseline_ffn_params_per_layer = 2 * n_embd * ffn_hidden  # up + down projection
    baseline_total_ffn = baseline_ffn_params_per_layer * n_layer
    
    # With capacity curve
    multipliers = compute_capacity_schedule(n_layer, capacity_config)
    curved_total_ffn = 0
    for mult in multipliers:
        layer_ffn = apply_capacity_to_dim(ffn_hidden, mult, multiple_of=128)
        curved_total_ffn += 2 * n_embd * layer_ffn
    
    reduction = (baseline_total_ffn - curved_total_ffn) / baseline_total_ffn * 100
    
    return {
        "baseline_ffn_params": baseline_total_ffn,
        "curved_ffn_params": curved_total_ffn,
        "reduction_percent": reduction,
        "multipliers": multipliers,
    }


if __name__ == "__main__":
    # Demo: show all curves
    print_capacity_curves(12)
    
    # Demo: estimate savings
    print("\n" + "="*60)
    print("Parameter Reduction Estimates")
    print("="*60)
    
    for curve_type in [CapacityCurveType.LINEAR, CapacityCurveType.GAUSSIAN, CapacityCurveType.EXPONENTIAL]:
        config = CapacityCurveConfig(
            curve_type=curve_type,
            min_ratio=0.25,
        )
        stats = estimate_param_reduction(
            n_layer=12,
            n_embd=768,
            ffn_hidden=3072,
            capacity_config=config,
        )
        print(f"\n{curve_type.value}: {stats['reduction_percent']:.1f}% FFN param reduction")
        print(f"  Baseline: {stats['baseline_ffn_params']:,} -> Curved: {stats['curved_ffn_params']:,}")

# # Add to your GPT2LLMConfig
# capacity_curve_config:
#   curve_type: gaussian  # or linear, exponential, cosine
#   peak_layer: 6         # null = auto middle
#   min_ratio: 0.25       # smallest layers = 25% of full size
#   scale_ffn: true       # scale FFN width
#   scale_heads: false    # scaling heads is experimental
  
#   # Curve-specific:
#   sigma: 0.4            # gaussian width (smaller = sharper)
#   sharpness: 3.0        # exponential sharpness (higher = sharper)