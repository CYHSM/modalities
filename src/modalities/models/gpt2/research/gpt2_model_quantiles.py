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


# =============================================================================
# Metrics Convention
# =============================================================================
# The model returns a structured dict alongside logits and ponder_loss.
# This is the ONLY place you need to touch when adding new logged metrics.
#
# MetricsBag = {
#     "scalars":            dict[str, Tensor],   # shape ()    — accumulated & reduced
#     "per_layer_scalars":  dict[str, Tensor],   # shape (L,)  — accumulated & reduced
#     "per_layer_vectors":  dict[str, Tensor],   # shape (L,D) — last-batch snapshot only
# }
# =============================================================================


# =============================================================================
# Configs
# =============================================================================

class AdaptiveComputationConfig(BaseModel):
    enable_adaptive: bool = False
    max_loops: int = 10
    halt_threshold: float = 0.99
    ponder_penalty_weight: float = 0.01

    use_memory_bank: bool = True
    num_local_slots: int = 1024
    num_global_slots: int = 512
    scheduler_type: str = "constant"
    frozen_gate: Optional[float] = None
    uses_new_names: Optional[bool] = True

    # Progressive exit: quantile increases from base_exit_quantile to 1.0 across layers
    base_exit_quantile: float = 0.75


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

class MemoryBankRegistry(nn.Module):
    def __init__(self, n_layer, n_embd, num_local_slots, num_global_slots):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.scale = n_embd ** -0.5

        self.local_keys = nn.Parameter(torch.randn(n_layer, num_local_slots, n_embd) * 0.02)
        self.local_values = nn.Parameter(torch.randn(n_layer, num_local_slots, n_embd) * 0.02)
        self.global_keys = nn.Parameter(torch.randn(num_global_slots, n_embd) * 0.02)
        self.global_values = nn.Parameter(torch.randn(num_global_slots, n_embd) * 0.02)

        self.q_norm = nn.LayerNorm(n_embd)
        self.k_norm = nn.LayerNorm(n_embd)

    def forward(self, query, layer_idx):
        q = self.q_norm(query)

        local_k = self.k_norm(self.local_keys[layer_idx])
        local_v = self.local_values[layer_idx]
        local_attn = F.softmax(torch.einsum('btd,sd->bts', q, local_k) * self.scale, dim=-1)
        local_out = torch.einsum('bts,sd->btd', local_attn, local_v)

        global_k = self.k_norm(self.global_keys)
        global_v = self.global_values
        global_attn = F.softmax(torch.einsum('btd,sd->bts', q, global_k) * self.scale, dim=-1)
        global_out = torch.einsum('bts,sd->btd', global_attn, global_v)

        return local_out, global_out


class GatedMemoryUnit(nn.Module):
    def __init__(self, n_embd, init_bias=-3.0, frozen_gate=None):
        super().__init__()
        self.n_embd = n_embd
        self.init_bias = init_bias
        self.frozen_gate = frozen_gate

        self.mem_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.gate_net = nn.Linear(n_embd, n_embd, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mem_proj.weight)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, self.init_bias)

    def forward(self, h, memory):
        mem_feat = self.mem_proj(memory)
        if self.frozen_gate is not None:
            gate = torch.full_like(h, self.frozen_gate)
        else:
            gate = torch.sigmoid(self.gate_net(h))
        return gate * mem_feat, gate.mean()


# =============================================================================
# Adaptive Recursive Block
# =============================================================================

class AdaptiveRecursiveBlock(nn.Module):
    def __init__(self, block, adaptive_config, n_embd, layer_idx, n_layers, memory_registry):
        super().__init__()
        self.block = block
        self.config = adaptive_config
        self.max_loops = adaptive_config.max_loops
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.memory_registry = memory_registry

        self.loop_scales = nn.Parameter(torch.full((self.max_loops,), -7.0))
        self.halt_temperature = nn.Parameter(torch.tensor([0.0]))

        # Progressive exit quantile: base_quantile at layer 0 → 1.0 at last layer
        progress = layer_idx / max(n_layers - 1, 1)
        self.exit_quantile = adaptive_config.base_exit_quantile + (1.0 - adaptive_config.base_exit_quantile) * progress

        if self.memory_registry is not None:
            self.mem_norm = nn.LayerNorm(n_embd)
            self.local_mem_gate = GatedMemoryUnit(n_embd, init_bias=-3.0, frozen_gate=adaptive_config.frozen_gate)
            self.global_mem_gate = GatedMemoryUnit(n_embd, init_bias=-3.0, frozen_gate=adaptive_config.frozen_gate)
        else:
            self.mem_norm = None
            self.local_mem_gate = None
            self.global_mem_gate = None

    def _pad_to_max(self, lst, device):
        if not lst:
            return torch.zeros(self.max_loops, device=device)
        stacked = torch.stack(lst)
        if stacked.size(0) < self.max_loops:
            pad = torch.zeros(self.max_loops - stacked.size(0), device=device, dtype=stacked.dtype)
            stacked = torch.cat([stacked, pad])
        return stacked

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = x.shape
        device = x.device
        done_threshold = 1.0 - self.config.halt_threshold

        h = x
        accumulated_output = torch.zeros_like(h)
        prob_remain = torch.ones(B, T, device=device, dtype=x.dtype)
        expected_steps = torch.zeros(B, T, device=device, dtype=x.dtype)
        total_relative_change = torch.zeros(B, T, device=device, dtype=x.dtype)

        halt_probs_list, rel_changes_list = [], []
        local_gate_avgs, global_gate_avgs = [], []

        # prob_remain decay curve traces
        prob_remain_max_trace, prob_remain_mean_trace = [], []

        # Second-order effect traces
        acceleration_trace, cosine_sim_trace = [], []
        delta_prev = None

        # Exit diagnostics (detached)
        done_frac_trace = []

        temperature = F.softplus(self.halt_temperature)
        actual_steps = 0

        for step in range(self.max_loops):
            actual_steps = step + 1

            # Memory enrichment
            if self.memory_registry is not None:
                query = self.mem_norm(h)
                local_retrieved, global_retrieved = self.memory_registry(query, self.layer_idx)
                local_contribution, l_gate_avg = self.local_mem_gate(h, local_retrieved)
                global_contribution, g_gate_avg = self.global_mem_gate(h, global_retrieved)
                h_enriched = h + local_contribution + global_contribution
                local_gate_avgs.append(l_gate_avg)
                global_gate_avgs.append(g_gate_avg)
            else:
                h_enriched = h

            # Transformer block
            scale = F.softplus(self.loop_scales[step])
            h_prev = h
            h_new = self.block(h_enriched, scale=scale)

            # Halt from relative change
            delta = h_new - h_prev
            raw_change = delta.norm(dim=-1) / h_prev.norm(dim=-1)
            step_relative_change = torch.nan_to_num(raw_change, nan=0.0, posinf=0.0)
            rel_changes_list.append(step_relative_change.mean())

            # Second-order effects (detached — logging only)
            with torch.no_grad():
                if delta_prev is not None:
                    acceleration_trace.append((delta - delta_prev).norm(dim=-1).mean())
                    cosine_sim_trace.append(F.cosine_similarity(delta, delta_prev, dim=-1).mean())
                else:
                    zero = torch.tensor(0.0, device=device)
                    acceleration_trace.append(zero)
                    cosine_sim_trace.append(zero)
            delta_prev = delta.detach()

            halt_prob = torch.exp(-step_relative_change / temperature)
            halt_probs_list.append(halt_prob.detach().mean())

            # Accumulate
            p_stop = prob_remain * halt_prob
            prob_remain = prob_remain * (1.0 - halt_prob)

            # prob_remain decay curve (detached — logging only)
            with torch.no_grad():
                prob_remain_max_trace.append(prob_remain.max())
                prob_remain_mean_trace.append(prob_remain.mean())

            accumulated_output = accumulated_output + h_new * p_stop.unsqueeze(-1)
            expected_steps = expected_steps + p_stop * (step + 1)
            total_relative_change = total_relative_change + (step_relative_change * p_stop)

            h = h_new

            # ---- Progressive done_frac exit ----
            with torch.no_grad():
                done_frac = (prob_remain < done_threshold).float().mean()
                done_frac_trace.append(done_frac)

            if self.training and done_frac >= self.exit_quantile:
                break

        # Dump remaining mass
        if prob_remain.sum() > 0:
            accumulated_output = accumulated_output + h * prob_remain.unsqueeze(-1)
            expected_steps = expected_steps + prob_remain * actual_steps
            final_change = (h - h_prev).norm(dim=-1) / h_prev.norm(dim=-1)
            final_change = torch.nan_to_num(final_change, nan=0.0, posinf=0.0)
            total_relative_change = total_relative_change + (prob_remain * final_change)

        # Memory gate stats
        local_s = torch.stack(local_gate_avgs).mean() if local_gate_avgs else torch.tensor(0.0, device=device)
        global_s = torch.stack(global_gate_avgs).mean() if global_gate_avgs else torch.tensor(0.0, device=device)

        # ---- Exit diagnostics (all detached) ----
        with torch.no_grad():
            # Residual mass: how much prob_remain was left when we exited
            residual_mass = prob_remain.mean()

            # Hard token fraction: tokens with significant remaining mass at exit
            hard_token_frac = (prob_remain > 0.1).float().mean()

            # Truncated mass: total prob_remain on tokens that were "done"
            # (low mass = healthy, the dumped remainder is small)
            done_mask = prob_remain < done_threshold
            # Mass on NOT-done tokens that got force-dumped (the actual waste)
            truncated_mass = prob_remain[~done_mask].sum() / max(prob_remain.numel(), 1)

            # Step distribution entropy: how spread out is the exit distribution?
            # Approximate from p_stop at each step
            # Collect per-step total p_stop mass (averaged over B,T)
            # We can reconstruct from halt_probs and the accumulation logic,
            # but it's simpler to just track it:
            _actual_steps_tensor = torch.tensor(float(actual_steps), device=device)
            _exit_quantile_tensor = torch.tensor(self.exit_quantile, device=device)
            _done_frac_at_exit = done_frac_trace[-1] if done_frac_trace else torch.tensor(0.0, device=device)

        # =====================================================================
        # LAYER METRICS
        # =====================================================================
        layer_metrics = {
            # --- existing metrics ---
            "expected_steps": expected_steps,                       # (B, T)
            "total_relative_change": total_relative_change,         # (B, T)
            "loop_scales": self.loop_scales.detach(),               # (max_loops,)
            "halt_temperature": self.halt_temperature.detach(),     # (1,)
            "local_mem_scale": local_s,                             # scalar
            "global_mem_scale": global_s,                           # scalar
            "step_halt_probs": self._pad_to_max(halt_probs_list, device),   # (max_loops,)
            "step_changes": self._pad_to_max(rel_changes_list, device),     # (max_loops,)
            # prob_remain decay curve
            "prob_remain_max": self._pad_to_max(prob_remain_max_trace, device),     # (max_loops,)
            "prob_remain_mean": self._pad_to_max(prob_remain_mean_trace, device),   # (max_loops,)
            # Second-order effects
            "acceleration": self._pad_to_max(acceleration_trace, device),   # (max_loops,)
            "cosine_sim": self._pad_to_max(cosine_sim_trace, device),       # (max_loops,)
            # done_frac per step (how fast tokens "finish" within this layer)
            "done_frac": self._pad_to_max(done_frac_trace, device),         # (max_loops,)

            # --- NEW: exit diagnostics (scalars) ---
            "actual_steps": _actual_steps_tensor,                   # scalar: how many loops ran
            "exit_quantile": _exit_quantile_tensor,                 # scalar: the threshold used
            "done_frac_at_exit": _done_frac_at_exit,                # scalar: done_frac when loop broke
            "residual_mass": residual_mass,                         # scalar: mean prob_remain at exit
            "hard_token_frac": hard_token_frac,                     # scalar: fraction with prob_remain > 0.1
            "truncated_mass": truncated_mass,                       # scalar: wasted mass on not-done tokens
        }

        return accumulated_output, layer_metrics


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
                ".mem_proj.weight", ".gate_net.weight",
            ],
            "embedding": [".wte", ".wpe"],
            "layernorm": [
                ".attention_norm", ".ffn_norm", ".lm_head_norm", ".mem_norm",
                ".loop_scales", ".halt_temperature", ".gate_net.bias",
                "memory_registry",
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

        def create_block():
            return GPT2Block(
                n_embd=n_embd, bias=bias, n_head_q=n_head_q, n_head_kv=n_head_kv,
                activation_type=activation_type, attention_impl=attention_implementation,
                attention_config=attention_config, dropout=dropout, ffn_hidden=ffn_hidden,
                attention_norm=attention_norm_config.norm_type.value(**dict(attention_norm_config.config)),
                ffn_norm=ffn_norm_config.norm_type.value(**dict(ffn_norm_config.config)),
                enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of,
            )

        if adaptive_config and adaptive_config.enable_adaptive and adaptive_config.use_memory_bank:
            self.memory_registry = MemoryBankRegistry(
                n_layer=n_layer, n_embd=n_embd,
                num_local_slots=adaptive_config.num_local_slots,
                num_global_slots=adaptive_config.num_global_slots,
            )
        else:
            self.memory_registry = None

        layers = {}
        for layer_idx in range(n_layer):
            block = create_block()
            if self.use_adaptive:
                layers[str(layer_idx)] = AdaptiveRecursiveBlock(
                    block=block, adaptive_config=adaptive_config,
                    n_embd=n_embd, layer_idx=layer_idx,
                    n_layers=n_layer,
                    memory_registry=self.memory_registry,
                )
            else:
                layers[str(layer_idx)] = block

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

        # ------------------------------------------------------------------
        # Collect per-layer metrics during forward pass
        # ------------------------------------------------------------------
        all_layer_metrics: list[dict[str, torch.Tensor]] = []
        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)

        sorted_keys = sorted(self.transformer.h.keys(), key=lambda k: int(k))
        for layer_key in sorted_keys:
            layer_module = self.transformer.h[layer_key]

            if self.use_adaptive:
                h, layer_metrics = layer_module(h)
                cost_mean = layer_metrics["expected_steps"].mean()
                total_ponder_cost = total_ponder_cost + cost_mean
                all_layer_metrics.append(layer_metrics)
            else:
                h = layer_module(h, scale=1.0 / (int(layer_key) + 1))

        h = self.transformer.lm_head_norm(h)
        logits = self.transformer.lm_head(h)

        if not self.use_adaptive:
            return logits

        # ==================================================================
        # BUILD METRICS BAG
        # ==================================================================
        n_layers = len(all_layer_metrics)
        max_loops = self.adaptive_config.max_loops

        avg_ponder_cost = total_ponder_cost / n_layers
        normalized_steps = (
            (avg_ponder_cost - 1.0) / (max_loops - 1.0)
            if max_loops > 1
            else torch.tensor(0.0, dtype=h.dtype, device=device)
        )
        weighted_ponder_loss = (normalized_steps * self.adaptive_config.ponder_penalty_weight).to(logits.dtype)

        # --- Helper: stack a key from all layer metrics ---
        def stack_key(key: str) -> torch.Tensor:
            return torch.stack([m[key] for m in all_layer_metrics])

        # --- Scalars: accumulated across microbatches, then all-reduced ---
        scalars = {
            "ponder_cost_unweighted": total_ponder_cost,
            "expected_steps": avg_ponder_cost,
            "normalized_steps": normalized_steps,
        }

        # --- Per-layer scalars: shape (n_layer,), accumulated & reduced ---
        per_layer_scalars = {
            "ponder_cost": torch.stack([m["expected_steps"].mean() for m in all_layer_metrics]),
            "weighted_change": torch.stack([m["total_relative_change"].mean() for m in all_layer_metrics]),
            "temperature": stack_key("halt_temperature").squeeze(-1),
            "local_mem_scale": stack_key("local_mem_scale"),
            "global_mem_scale": stack_key("global_mem_scale"),
            # NEW: exit diagnostics per layer
            "actual_steps": stack_key("actual_steps"),
            "exit_quantile": stack_key("exit_quantile"),
            "done_frac_at_exit": stack_key("done_frac_at_exit"),
            "residual_mass": stack_key("residual_mass"),
            "hard_token_frac": stack_key("hard_token_frac"),
            "truncated_mass": stack_key("truncated_mass"),
        }

        # --- Per-layer vectors: shape (n_layer, max_loops), last-batch snapshot ---
        per_layer_vectors = {
            "step_halt_prob": stack_key("step_halt_probs"),    # (n_layer, max_loops)
            "step_change": stack_key("step_changes"),          # (n_layer, max_loops)
            "loop_scale": stack_key("loop_scales"),            # (n_layer, max_loops)
            # prob_remain decay curve
            "prob_remain_max": stack_key("prob_remain_max"),       # (n_layer, max_loops)
            "prob_remain_mean": stack_key("prob_remain_mean"),     # (n_layer, max_loops)
            # Second-order effects
            "acceleration": stack_key("acceleration"),             # (n_layer, max_loops)
            "cosine_sim": stack_key("cosine_sim"),                 # (n_layer, max_loops)
            # NEW: done_frac per step per layer (how fast tokens converge)
            "done_frac": stack_key("done_frac"),                   # (n_layer, max_loops)
        }

        return {
            "logits": logits,
            "ponder_loss": weighted_ponder_loss,
            "metrics": {
                "scalars": scalars,
                "per_layer_scalars": per_layer_scalars,
                "per_layer_vectors": per_layer_vectors,
            },
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