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

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class AdaptiveComputationConfig(BaseModel):
    """
    Configuration for Adaptive Computation Time (PonderNet-style).
    """
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
            ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


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
        assert n_embd % n_head_q == 0
        assert n_head_q % n_head_kv == 0

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


# ==============================================================================
# ADAPTIVE COMPUTATION COMPONENTS
# ==============================================================================

class AdaptiveRouter(nn.Module):
    """Halting router for PonderNet-style adaptive computation."""
    
    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.net = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, x: torch.Tensor, step_normalized: float) -> torch.Tensor:
        B, T, _ = x.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=x.device, dtype=x.dtype)
        logits = self.net(torch.cat([x, step_feat], dim=-1))
        return torch.sigmoid(logits).squeeze(-1)

class MemoryBankRegistry(nn.Module):
    """
    Memory Bank with both local (per-layer) and global (shared) memory.
    """
    
    def __init__(self, n_layer: int, n_embd: int, num_local_slots: int = 1024, num_global_slots: int = 512):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.scale = n_embd ** -0.5
        
        # Local (per-layer) memory
        self.local_keys = nn.Parameter(torch.randn(n_layer, num_local_slots, n_embd) * 0.02)
        self.local_values = nn.Parameter(torch.randn(n_layer, num_local_slots, n_embd) * 0.02)
        
        # Global (shared across all layers) memory
        self.global_keys = nn.Parameter(torch.randn(num_global_slots, n_embd) * 0.02)
        self.global_values = nn.Parameter(torch.randn(num_global_slots, n_embd) * 0.02)
        
        # QK-norm for stable attention
        self.q_norm = nn.LayerNorm(n_embd)
        self.k_norm = nn.LayerNorm(n_embd)

    def forward(self, query: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (local_retrieved, global_retrieved).
        Args:
            query: [B, T, D]
            layer_idx: which layer's local memory to access
        Returns:
            local_out: [B, T, D] - from layer-specific memory
            global_out: [B, T, D] - from shared global memory
        """
        q = self.q_norm(query)
        
        # Local lookup (layer-specific)
        local_k = self.k_norm(self.local_keys[layer_idx])
        local_v = self.local_values[layer_idx]
        local_attn = F.softmax(torch.einsum('btd,sd->bts', q, local_k) * self.scale, dim=-1)
        local_out = torch.einsum('bts,sd->btd', local_attn, local_v)
        
        # Global lookup (shared)
        global_k = self.k_norm(self.global_keys)
        global_v = self.global_values
        global_attn = F.softmax(torch.einsum('btd,sd->bts', q, global_k) * self.scale, dim=-1)
        global_out = torch.einsum('bts,sd->btd', global_attn, global_v)
        
        return local_out, global_out
    

class AdaptiveRecursiveBlock(nn.Module):
    """
    Adaptive computation with both local and global memory.
    
    Key design:
    - local_mem_scale: controls contribution from layer-specific memory
    - global_mem_scale: controls contribution from shared global memory
    - Both initialized small; model learns to use what it needs
    """
    
    def __init__(
        self,
        block: GPT2Block,
        adaptive_config: AdaptiveComputationConfig,
        n_embd: int,
        layer_idx: int,
        memory_registry: MemoryBankRegistry,
    ):
        super().__init__()
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.layer_idx = layer_idx
        self.memory_registry = memory_registry
        
        # Halting mechanism
        self.halt_router = AdaptiveRouter(n_embd)
        self.loop_scales = nn.Parameter(torch.full((self.max_loops,), -7.0))
        
        # Memory injection - LOCAL
        self.mem_norm = nn.LayerNorm(n_embd)
        self.local_mem_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.local_mem_scale = nn.Parameter(torch.tensor([0.1]))
        
        # Memory injection - GLOBAL
        self.global_mem_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.global_mem_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        device = x.device
        
        h = x
        accumulated_output = torch.zeros_like(h)
        prob_remain = torch.ones(B, T, device=device, dtype=x.dtype)
        expected_steps = torch.zeros(B, T, device=device, dtype=x.dtype)
        total_cos_sim = torch.zeros(B, T, device=device, dtype=x.dtype)
        halt_probs_list = []
        
        denom = max(1, self.max_loops - 1)

        for step in range(self.max_loops):
            # ========== 1. MEMORY RETRIEVAL ==========
            query = self.mem_norm(h)
            local_mem, global_mem = self.memory_registry(query, self.layer_idx)
            
            # ========== 2. DUAL MEMORY INJECTION ==========
            h_enriched = (
                h 
                + self.local_mem_scale * self.local_mem_proj(local_mem)
                + self.global_mem_scale * self.global_mem_proj(global_mem)
            )
            
            # ========== 3. TRANSFORMER BLOCK ==========
            scale = F.softplus(self.loop_scales[step])
            h_prev = h
            h_new = self.block(h_enriched, scale=scale)
            
            # ========== 4. HALTING (PonderNet) ==========
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
            
            # Early exit during inference
            if not self.training and prob_remain.max() < 0.01:
                break
        
        # Handle remainder for early exits
        if not self.training and prob_remain.sum() > 0:
            accumulated_output = accumulated_output + h * prob_remain.unsqueeze(-1)

        # Pack diagnostics
        halt_probs_stacked = self._pad_to_max(halt_probs_list, device)
        
        return (
            accumulated_output,
            expected_steps,
            total_cos_sim,
            self.loop_scales.detach(),
            halt_probs_stacked,
            self.local_mem_scale.detach().item(),
            self.global_mem_scale.detach().item(),  # NEW: track global scale too
        )
    
    def _pad_to_max(self, lst, device):
        stacked = torch.stack(lst)
        if stacked.size(0) < self.max_loops:
            pad = torch.zeros(self.max_loops - stacked.size(0), device=device, dtype=stacked.dtype)
            stacked = torch.cat([stacked, pad])
        return stacked


# class MemoryBankRegistry(nn.Module):
#     """
#     Simplified Memory Bank with QK-norm for stability.
#     """
    
#     def __init__(self, n_layer: int, n_embd: int, num_slots: int = 1024):
#         super().__init__()
#         self.n_layer = n_layer
#         self.n_embd = n_embd
#         self.num_slots = num_slots
#         self.scale = n_embd ** -0.5
        
#         # Learnable memory
#         self.keys = nn.Parameter(torch.randn(n_layer, num_slots, n_embd) * 0.02)
#         self.values = nn.Parameter(torch.randn(n_layer, num_slots, n_embd) * 0.02)
        
#         # QK-norm for stable attention
#         self.q_norm = nn.LayerNorm(n_embd)
#         self.k_norm = nn.LayerNorm(n_embd)

#     def forward(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
#         """
#         Simple single-layer lookup (no cross-layer routing).
#         Args:
#             query: [B, T, D]
#             layer_idx: which layer's memory to access
#         Returns:
#             retrieved: [B, T, D]
#         """
#         q = self.q_norm(query)  # [B, T, D]
#         k = self.k_norm(self.keys[layer_idx])  # [num_slots, D]
#         v = self.values[layer_idx]  # [num_slots, D]
        
#         # Standard attention
#         attn_logits = torch.einsum('btd,sd->bts', q, k) * self.scale
#         attn_weights = F.softmax(attn_logits, dim=-1)
#         retrieved = torch.einsum('bts,sd->btd', attn_weights, v)
        
#         return retrieved


# class AdaptiveRecursiveBlock(nn.Module):
#     """
#     Simplified adaptive computation with memory INSIDE the loop.
    
#     Key design choices:
#     - Memory accessed every iteration (enables multi-hop reasoning)
#     - Simple additive injection with learnable scale
#     - No complex gating - let the model learn what it needs
#     """
    
#     def __init__(
#         self,
#         block: GPT2Block,
#         adaptive_config: AdaptiveComputationConfig,
#         n_embd: int,
#         layer_idx: int,
#         memory_registry: MemoryBankRegistry,
#     ):
#         super().__init__()
#         self.block = block
#         self.config = adaptive_config
#         self.max_loops = adaptive_config.max_loops
#         self.layer_idx = layer_idx
#         self.memory_registry = memory_registry
        
#         # Halting mechanism
#         self.halt_router = AdaptiveRouter(n_embd)
#         self.loop_scales = nn.Parameter(torch.full((self.max_loops,), -7.0))
        
#         # Memory injection (SIMPLE: just norm + projection + scale)
#         self.mem_norm = nn.LayerNorm(n_embd)
#         self.mem_proj = nn.Linear(n_embd, n_embd, bias=False)
#         self.mem_scale = nn.Parameter(torch.tensor([0.01]))

#     def forward(self, x: torch.Tensor) -> tuple:
#         B, T, D = x.shape
#         device = x.device
        
#         h = x
#         accumulated_output = torch.zeros_like(h)
#         prob_remain = torch.ones(B, T, device=device, dtype=x.dtype)
#         expected_steps = torch.zeros(B, T, device=device, dtype=x.dtype)
#         total_cos_sim = torch.zeros(B, T, device=device, dtype=x.dtype)
#         halt_probs_list = []
        
#         denom = max(1, self.max_loops - 1)

#         for step in range(self.max_loops):
#             # ========== 1. MEMORY RETRIEVAL ==========
#             query = self.mem_norm(h)
#             mem_out = self.memory_registry(query, self.layer_idx)
            
#             # ========== 2. SIMPLE ADDITIVE INJECTION ==========
#             # h_enriched = h + scale * proj(memory)
#             h_enriched = h + self.mem_scale * self.mem_proj(mem_out)
            
#             # ========== 3. TRANSFORMER BLOCK ==========
#             scale = F.softplus(self.loop_scales[step])
#             h_prev = h
#             h_new = self.block(h_enriched, scale=scale)
            
#             # ========== 4. HALTING (PonderNet) ==========
#             cos_sim = F.cosine_similarity(h_new, h_prev, dim=-1, eps=1e-8)
#             halt_prob = self.halt_router(h_new, step / denom)
#             halt_probs_list.append(halt_prob.detach().mean())

#             if step == self.max_loops - 1:
#                 p_stop = prob_remain
#                 prob_remain = torch.zeros_like(prob_remain)
#             else:
#                 p_stop = prob_remain * halt_prob
#                 prob_remain = prob_remain * (1.0 - halt_prob)

#             accumulated_output = accumulated_output + h_new * p_stop.unsqueeze(-1)
#             expected_steps = expected_steps + p_stop * (step + 1)
#             total_cos_sim = total_cos_sim + cos_sim * p_stop
            
#             h = h_new
            
#             # Early exit during inference
#             if not self.training and prob_remain.max() < 0.01:
#                 break
        
#         # Handle remainder for early exits
#         if not self.training and prob_remain.sum() > 0:
#             accumulated_output = accumulated_output + h * prob_remain.unsqueeze(-1)

#         # Pack diagnostics
#         halt_probs_stacked = self._pad_to_max(halt_probs_list, device)
        
#         return (
#             accumulated_output,
#             expected_steps,
#             total_cos_sim,
#             self.loop_scales.detach(),
#             halt_probs_stacked,
#             self.mem_scale.detach().item(),  # Track memory scale
#         )
    
#     def _pad_to_max(self, lst, device):
#         stacked = torch.stack(lst)
#         if stacked.size(0) < self.max_loops:
#             pad = torch.zeros(self.max_loops - stacked.size(0), device=device, dtype=stacked.dtype)
#             stacked = torch.cat([stacked, pad])
#         return stacked


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
                ".local_mem_proj.weight",
                ".global_mem_proj.weight",
            ],
            "embedding": [
                ".wte",
                ".wpe",
            ],
            "layernorm": [
                ".attention_norm",
                ".ffn_norm",
                ".lm_head_norm",
                ".mem_norm",
                ".loop_scales",
                ".halt_router.net.bias",
                "memory_registry.local_keys",
                "memory_registry.local_values",
                "memory_registry.global_keys",
                "memory_registry.global_values",
                "memory_registry.q_norm",
                "memory_registry.k_norm",
                ".local_mem_scale",
                ".global_mem_scale",
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
            raise ValueError('It is expected to use "RotaryTransform" together with "NOPE".')

        self.use_adaptive = adaptive_config is not None and adaptive_config.enable_adaptive
        self.adaptive_config = adaptive_config
        
        def create_block():
            return GPT2Block(
                n_embd=n_embd,
                bias=bias,
                n_head_q=n_head_q,
                n_head_kv=n_head_kv,
                activation_type=activation_type,
                attention_impl=attention_implementation,
                attention_config=attention_config,
                dropout=dropout,
                ffn_hidden=ffn_hidden,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )

        # Create memory registry if adaptive mode is enabled
        if adaptive_config and adaptive_config.enable_adaptive:
            self.memory_registry = MemoryBankRegistry(
                n_layer=n_layer,
                n_embd=n_embd,
                num_local_slots=1024,
                num_global_slots=512,
            )
        else:
            self.memory_registry = None

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
                    memory_registry=self.memory_registry,
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
        assert seq_len <= self.sequence_length

        h = self.transformer.wte(inputs)

        if self.poe_type is PositionTypes.ABSOLUTE and hasattr(self.transformer, "wpe"):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            h = h + pos_emb

        h = self.transformer.drop(h) if hasattr(self.transformer, "drop") else h

        # Tracking
        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)
        num_adaptive_layers = 0
        per_layer_ponder_costs = []
        per_layer_cos_sims = []
        per_layer_loop_scales = []
        per_layer_halt_probs = []
        per_layer_mem_scales = []  # NEW: track mem_scale per layer

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer_module = self.transformer.h[layer_key]
            layer_idx = int(layer_key)

            if self.use_adaptive:
                # Now returns 7 values (added global_mem_scale)
                h, cost, cos_sim, scales, halt_probs, local_mem_scale, global_mem_scale = layer_module(h)
                
                cost_mean = cost.mean()
                sim_mean = cos_sim.mean()
                total_ponder_cost = total_ponder_cost + cost_mean
                per_layer_ponder_costs.append(cost_mean)
                per_layer_cos_sims.append(sim_mean)
                per_layer_loop_scales.append(scales)
                per_layer_halt_probs.append(halt_probs)
                per_layer_mem_scales.append((local_mem_scale, global_mem_scale))
                num_adaptive_layers += 1
            else:
                lns_scale = 1.0 / (layer_idx + 1)
                h = layer_module(h, scale=lns_scale)

        h = self.transformer.lm_head_norm(h) if hasattr(self.transformer, "lm_head_norm") else h
        logits = self.transformer.lm_head(h) if hasattr(self.transformer, "lm_head") else h

        if self.use_adaptive:
            avg_ponder_cost = total_ponder_cost / num_adaptive_layers if num_adaptive_layers > 0 else torch.tensor(0.0, dtype=h.dtype, device=device)
            
            normalized_steps = (avg_ponder_cost - 1.0) / (self.adaptive_config.max_loops - 1.0) if self.adaptive_config.max_loops > 1 else torch.tensor(0.0, dtype=h.dtype, device=device)
            
            weighted_ponder_loss = (normalized_steps * self.adaptive_config.ponder_penalty_weight).to(logits.dtype)
            
            layer_costs_tensor = torch.stack(per_layer_ponder_costs) if per_layer_ponder_costs else torch.tensor([], device=device)
            layer_sims_tensor = torch.stack(per_layer_cos_sims) if per_layer_cos_sims else torch.tensor([], device=device)
            local_scales = torch.tensor([s[0] for s in per_layer_mem_scales], device=device)
            global_scales = torch.tensor([s[1] for s in per_layer_mem_scales], device=device)

            return {
                "logits": logits,
                "ponder_loss": weighted_ponder_loss,
                "ponder_cost_unweighted": total_ponder_cost,
                "expected_steps": avg_ponder_cost,
                "normalized_steps": normalized_steps,
                "per_layer_ponder_costs": layer_costs_tensor,
                "per_layer_cos_sims": layer_sims_tensor,
                "loop_scales": torch.stack(per_layer_loop_scales) if per_layer_loop_scales else torch.tensor([], device=device),
                "halt_probs": torch.stack(per_layer_halt_probs) if per_layer_halt_probs else torch.tensor([], device=device),
                "local_mem_scales": local_scales,
                "global_mem_scales": global_scales,
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


# def estimate_flops(self):
#     """
#     Estimated FLOPs per token (forward + backward).
#     Each matmul parameter contributes 6 FLOPs (2 fwd, 4 bwd).
#     Attention QK matmul adds 12 * n_head * head_dim * effective_seq per layer.

#     For adaptive layers the narrow block is looped max_loops times and the
#     wide block (if present) runs once, so we multiply accordingly.
#     """
#     h = self.transformer.wte.weight.shape[1]  # n_embd
#     head_dim = h // self.n_head_q if hasattr(self, "n_head_q") else h // 12
#     n_heads = h // head_dim
#     seq_len = self.sequence_length

#     # Attention flops for one full-context layer (no sliding window in your setup)
#     attn_flops_per_layer = 12 * n_heads * head_dim * seq_len

#     def matmul_params(module):
#         """Count parameters that participate in matmuls (exclude embeddings, scalars, norms)."""
#         total = 0
#         for name, p in module.named_parameters():
#             if p.ndim < 2:
#                 continue  # skip biases, scalars, norm weights
#             total += p.numel()
#         return total

#     total_flops = 0

#     sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
#     for key in sorted_keys:
#         layer = self.transformer.h[key]

#         if self.use_adaptive:
#             # Narrow block: looped max_loops times
#             narrow_matmul = matmul_params(layer.block)
#             narrow_flops = 6 * narrow_matmul + attn_flops_per_layer
#             total_flops += layer.max_loops * narrow_flops

#             # Wide block: single pass (if present)
#             if layer.has_wide_path:
#                 wide_matmul = matmul_params(layer.wide_block)
#                 wide_flops = 6 * wide_matmul + attn_flops_per_layer
#                 total_flops += wide_flops

#             # Router + gate are negligible but count them anyway
#             router_params = matmul_params(layer.router)
#             gate_params = matmul_params(layer.dual_gate) if layer.has_wide_path else 0
#             total_flops += 6 * (router_params + gate_params) * layer.max_loops
#         else:
#             # Standard block: single pass
#             block_matmul = matmul_params(layer)
#             total_flops += 6 * block_matmul + attn_flops_per_layer

#     # lm_head (weight-tied or not)
#     lm_head_params = self.transformer.lm_head.weight.numel()
#     total_flops += 6 * lm_head_params

#     return total_flops