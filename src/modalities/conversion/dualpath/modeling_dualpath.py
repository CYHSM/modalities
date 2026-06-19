# coding=utf-8
"""
HF-compatible AdaptiveGPT with optional per-token diagnostics recording.

Diagnostics design
------------------
Recording is OFF by default. Enable it from outside with:

    model.set_record_diagnostics(True)

After each forward pass, when recording is on AND the model is in eval mode,
the most recent batch's diagnostics are stashed on:

    model.last_diagnostics: dict

The dict contains CPU tensors (so GPU memory isn't held), shape conventions:

    "tokens":           (B, T)      int32     input_ids
    "loss":             (B, T)      fp16      per-token CE loss for predicting
                                              token at position i (so loss[..., 0]
                                              corresponds to predicting tokens[..., 1])
                                              The last position is NaN (no next token).
    "gate_deep":        (L, B, T)   fp16      per-layer post-override gate, deep path
    "gate_wide":        (L, B, T)   fp16      per-layer post-override gate, wide path
                                              For "loop" layers: gate_deep=1, gate_wide=0
                                              For "wide" layers: gate_deep=0, gate_wide=1
                                              In convex mode: gate_wide = 1 - gate_deep
    "expected_steps":   (L, B, T)   fp16      ACT expected steps. 0 for "wide" layers.
    "delta_deep_norm":  (L, B, T)   fp16      ||h_deep - x|| per token
    "delta_wide_norm":  (L, B, T)   fp16      ||h_wide - x|| per token
    "delta_cos_sim":    (L, B, T)   fp16      cos(h_deep-x, h_wide-x) per token,
                                              0 when one path is missing
    "cross_w2d_norm":   (L, B, T)   fp16      ||post-gate wide-to-deep contamination||
    "cross_d2w_norm":   (L, B, T)   fp16      ||post-gate deep-to-wide contamination||

    "step_halt_probs":  (L, max_loops)  fp16  mean halt prob per (layer, step)
    "step_loop_scales": (L, max_loops)  fp16  softplus(loop_scales[step]) per (layer, step)

The flag is plumbed via a single attribute on the top-level model that is
read by each AdaptiveRecursiveBlock during forward. Training and inference
paths without the flag set are unchanged.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class AdaptiveGPTConfig(PretrainedConfig):
    model_type = "adaptive_gpt"

    def __init__(
        self,
        vocab_size: int = 50304,
        sequence_length: int = 2048,
        n_layer: int = 12,
        n_head_q: int = 12,
        n_head_kv: int = 12,
        n_embd: int = 768,
        ffn_hidden: int = 3072,
        dropout: float = 0.0,
        bias: bool = False,
        activation_type: str = "swiglu",
        enforce_swiglu_hidden_dim_multiple_of: int = 256,
        poe_type: str = "NOPE",
        use_rotary: bool = True,
        rotary_base_freq: int = 10000,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        norm_elementwise_affine: bool = True,
        use_weight_tying: bool = False,
        use_qk_norm: bool = False,
        qk_norm_dim: Optional[int] = None,
        enable_adaptive: bool = False,
        max_loops: int = 10,
        ponder_penalty_weight: float = 0.0,
        wide_ffn_hidden: int = 0,
        gate_mode: str = "two_gates",
        gate_init_bias: float = 0.0,
        deep_gate_init_bias: float = 0.0,
        wide_gate_init_bias: float = 0.0,
        loop_scale_init: float = -7,
        wide_scale_init: float = -7,
        use_cross: bool = True,
        cross_scale_deep_init: float = -7.0,
        cross_scale_wide_init: float = -7.0,
        adaptive_layer_types: Optional[list] = None,
        **kwargs,
    ):
        kwargs["tie_word_embeddings"] = use_weight_tying
        legacy = kwargs.pop("layer_types", None)
        if legacy is not None and adaptive_layer_types is None:
            adaptive_layer_types = legacy
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_layer = n_layer
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv
        self.n_embd = n_embd
        self.ffn_hidden = ffn_hidden
        self.dropout = dropout
        self.bias = bias
        self.activation_type = activation_type.lower()
        self.enforce_swiglu_hidden_dim_multiple_of = enforce_swiglu_hidden_dim_multiple_of
        self.poe_type = poe_type
        self.use_rotary = use_rotary
        self.rotary_base_freq = rotary_base_freq
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_weight_tying = use_weight_tying
        self.use_qk_norm = use_qk_norm
        self.qk_norm_dim = qk_norm_dim
        self.enable_adaptive = enable_adaptive
        self.max_loops = max_loops
        self.ponder_penalty_weight = ponder_penalty_weight
        self.wide_ffn_hidden = wide_ffn_hidden
        self.gate_mode = gate_mode
        self.gate_init_bias = gate_init_bias
        self.deep_gate_init_bias = deep_gate_init_bias
        self.wide_gate_init_bias = wide_gate_init_bias
        self.loop_scale_init = loop_scale_init
        self.wide_scale_init = wide_scale_init
        self.use_cross = use_cross
        self.cross_scale_deep_init = cross_scale_deep_init
        self.cross_scale_wide_init = cross_scale_wide_init
        self.adaptive_layer_types = adaptive_layer_types
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head_q
        self.hidden_size = n_embd

        if gate_mode not in ("convex", "two_gates"):
            raise ValueError(f"gate_mode must be 'convex' or 'two_gates', got {gate_mode}")


# =============================================================================
# Building blocks (unchanged from original)
# =============================================================================

class RMSLayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight + self.bias if self.bias is not None else out * self.weight


def _build_norm(config, dim=None):
    d = dim if dim is not None else config.n_embd
    if config.norm_type == "layer_norm":
        return nn.LayerNorm(d, eps=config.norm_eps, elementwise_affine=config.norm_elementwise_affine, bias=config.norm_bias)
    elif config.norm_type == "rms_norm":
        return RMSLayerNorm(ndim=d, bias=config.norm_bias, epsilon=config.norm_eps)
    elif config.norm_type == "pytorch_rms_norm":
        return nn.RMSNorm(d, eps=config.norm_eps)
    raise ValueError(f"Unknown norm_type: {config.norm_type}")


class SwiGLU(nn.Module):
    def __init__(self, n_embd, ffn_hidden, bias, enforce_swiglu_hidden_dim_multiple_of=256):
        super().__init__()
        hidden = self._hidden(ffn_hidden, enforce_swiglu_hidden_dim_multiple_of)
        self.W = nn.Linear(n_embd, hidden, bias=bias)
        self.silu = nn.SiLU()
        self.V = nn.Linear(n_embd, hidden, bias=bias)
        self.W_2 = nn.Linear(hidden, n_embd, bias=bias)

    @staticmethod
    def _hidden(ffn_hidden, mult):
        adj = int(2 * ffn_hidden / 3)
        return ((adj + mult - 1) // mult) * mult

    def forward(self, x):
        return self.W_2(self.silu(self.W(x)) * self.V(x))


class RotaryTransform(nn.Module):
    def __init__(self, n_embd, n_head, seq_length_dim=-2, base_freq=10000):
        super().__init__()
        self.dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        self.base_freq = base_freq
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, self.dim_model, 2).float() / self.dim_model))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update(self, x):
        seq_len = x.shape[self.seq_length_dim]
        if (seq_len != self._seq_len_cached or self._cos_cached is None
            or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)
        return self._cos_cached, self._sin_cached

    def _apply_rotary(self, x, cos, sin):
        cos = cos[:, :, : x.shape[self.seq_length_dim], :]
        sin = sin[:, :, : x.shape[self.seq_length_dim], :]
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, q, k, v):
        cos, sin = self._update(k)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin), v


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head_q = config.n_head_q
        self.n_head_kv = config.n_head_kv
        self.n_embd = config.n_embd
        self.n_rep = config.n_head_q // config.n_head_kv
        self.dropout = config.dropout

        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd // self.n_rep, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd // self.n_rep, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        transforms = []
        if config.use_rotary:
            transforms.append(RotaryTransform(config.n_embd, config.n_head_q, base_freq=config.rotary_base_freq))
        self.qkv_transforms = nn.ModuleList(transforms)

        if config.use_qk_norm:
            q_dim = config.qk_norm_dim or (config.n_embd // config.n_head_q)
            self.q_norm = _build_norm(config, dim=q_dim)
            self.k_norm = _build_norm(config, dim=q_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    @staticmethod
    def _repeat_kv(x, n_rep):
        B, nh, T, hs = x.shape
        if n_rep == 1: return x
        return x[:, :, None, :, :].expand(B, nh, n_rep, T, hs).reshape(B, nh * n_rep, T, hs)

    def forward(self, x):
        B, T, D = x.size()
        hd = D // self.n_head_q
        q = self.q_attn(x).view(B, T, self.n_head_q, hd).transpose(1, 2).contiguous()
        k = self.k_attn(x).view(B, T, self.n_head_kv, hd).transpose(1, 2).contiguous()
        v = self.v_attn(x).view(B, T, self.n_head_kv, hd).transpose(1, 2).contiguous()
        for tr in self.qkv_transforms:
            q, k, v = tr(q, k, v)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().reshape(B, T, D)))


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
    def __init__(self, config, ffn_hidden_override=None):
        super().__init__()
        self.attention_norm = _build_norm(config)
        self.ffn_norm = _build_norm(config)
        self.attn = CausalSelfAttention(config)
        ffn_hidden = ffn_hidden_override if ffn_hidden_override is not None else config.ffn_hidden
        if config.activation_type == "gelu":
            self.mlp = TransformerMLP(config.n_embd, ffn_hidden, config.bias, config.dropout)
        elif config.activation_type == "swiglu":
            self.mlp = SwiGLU(config.n_embd, ffn_hidden, config.bias, config.enforce_swiglu_hidden_dim_multiple_of)
        else:
            raise ValueError(f"Unknown activation: {config.activation_type}")

    def forward(self, x, scale=1.0):
        x = x + scale * self.attn(self.attention_norm(x))
        x = x + scale * self.mlp(self.ffn_norm(x))
        return x


class AdaptiveRouter(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, h, step_normalized, x=None):
        B, T, _ = h.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        return torch.sigmoid(self.linear(torch.cat([h, step_feat], dim=-1))).squeeze(-1)


# =============================================================================
# Dual-path gates — now return cross norms when recording, for diagnostics
# =============================================================================

class DualPathGateConvex(nn.Module):
    def __init__(self, n_embd, gate_init_bias=0.0, use_cross=False,
                 cross_scale_deep_init=-7.0, cross_scale_wide_init=-7.0):
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

    def forward(self, x, h_deep, h_wide, record=False):
        gate = torch.sigmoid(self.gate_proj(x))
        cross_w2d_norm = cross_d2w_norm = None
        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            c_w2d = s_d * self.proj_w2d(h_wide)
            c_d2w = s_w * self.proj_d2w(h_deep)
            h_deep_eff = h_deep + c_w2d
            h_wide_eff = h_wide + c_d2w
            if record:
                # In convex mode, contamination is pre-gate (added into the
                # effective path before the convex combination), so report
                # the raw per-token magnitudes of the contamination terms.
                cross_w2d_norm = c_w2d.norm(dim=-1)
                cross_d2w_norm = c_d2w.norm(dim=-1)
        else:
            h_deep_eff, h_wide_eff = h_deep, h_wide
        out = gate * h_deep_eff + (1.0 - gate) * h_wide_eff
        return out, gate.squeeze(-1), cross_w2d_norm, cross_d2w_norm


class DualPathGateTwoGates(nn.Module):
    def __init__(self, n_embd, deep_gate_init_bias=0.0, wide_gate_init_bias=0.0,
                 use_cross=False, cross_scale_deep_init=-7.0, cross_scale_wide_init=-7.0):
        super().__init__()
        self.deep_gate_init_bias = deep_gate_init_bias
        self.wide_gate_init_bias = wide_gate_init_bias
        self.use_cross = use_cross
        self.cross_scale_deep_init = cross_scale_deep_init
        self.cross_scale_wide_init = cross_scale_wide_init
        self.gate_proj = nn.Linear(n_embd, 2, bias=True)
        if use_cross:
            self.proj_w2d = nn.Linear(n_embd, n_embd, bias=False)
            self.proj_d2w = nn.Linear(n_embd, n_embd, bias=False)
            self.cross_scale_deep = nn.Parameter(torch.empty(1))
            self.cross_scale_wide = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias[0] = self.deep_gate_init_bias
            self.gate_proj.bias[1] = self.wide_gate_init_bias
        if self.use_cross:
            nn.init.zeros_(self.proj_w2d.weight)
            nn.init.zeros_(self.proj_d2w.weight)
            nn.init.constant_(self.cross_scale_deep, self.cross_scale_deep_init)
            nn.init.constant_(self.cross_scale_wide, self.cross_scale_wide_init)

    def forward(self, x, h_deep, h_wide, record=False):
        gates = torch.sigmoid(self.gate_proj(x))  # (B, T, 2)
        gd = gates[..., 0:1]
        gw = gates[..., 1:2]
        cross_w2d_norm = cross_d2w_norm = None
        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            c_w2d = s_d * self.proj_w2d(h_wide)
            c_d2w = s_w * self.proj_d2w(h_deep)
            contam_w2d = gd * c_w2d
            contam_d2w = gw * c_d2w
            h_deep_branch = gd * h_deep + contam_w2d
            h_wide_branch = gw * h_wide + contam_d2w
            if record:
                # Post-gate contamination — what the next layer actually sees.
                cross_w2d_norm = contam_w2d.norm(dim=-1)
                cross_d2w_norm = contam_d2w.norm(dim=-1)
        else:
            h_deep_branch = gd * h_deep
            h_wide_branch = gw * h_wide
        out = h_deep_branch + h_wide_branch
        return out, gd.squeeze(-1), gw.squeeze(-1), cross_w2d_norm, cross_d2w_norm


def _build_dual_gate(config):
    if config.gate_mode == "convex":
        return DualPathGateConvex(
            n_embd=config.n_embd,
            gate_init_bias=config.gate_init_bias,
            use_cross=config.use_cross,
            cross_scale_deep_init=config.cross_scale_deep_init,
            cross_scale_wide_init=config.cross_scale_wide_init,
        )
    return DualPathGateTwoGates(
        n_embd=config.n_embd,
        deep_gate_init_bias=config.deep_gate_init_bias,
        wide_gate_init_bias=config.wide_gate_init_bias,
        use_cross=config.use_cross,
        cross_scale_deep_init=config.cross_scale_deep_init,
        cross_scale_wide_init=config.cross_scale_wide_init,
    )


# =============================================================================
# AdaptiveRecursiveBlock — augmented with diagnostics
# =============================================================================

class AdaptiveRecursiveBlock(nn.Module):
    def __init__(self, config, layer_type="dual"):
        super().__init__()
        self.layer_type = layer_type
        self.max_loops = config.max_loops
        self.gate_mode = config.gate_mode
        n_embd = config.n_embd

        self.has_loop_path = layer_type in ("loop", "dual")
        if self.has_loop_path:
            self.block = GPT2Block(config)
            self.router = AdaptiveRouter(n_embd)
            self.loop_scales = nn.Parameter(torch.full((self.max_loops,), float(config.loop_scale_init)))

        self.has_wide_path = layer_type in ("wide", "dual")
        if self.has_wide_path:
            self.wide_block = GPT2Block(config, ffn_hidden_override=config.wide_ffn_hidden)
            self.wide_scale = nn.Parameter(torch.tensor([float(config.wide_scale_init)]))

        if layer_type == "dual":
            self.dual_gate = _build_dual_gate(config)

        # Filled by enclosing model when recording is on.
        self._record_diagnostics = False

    def forward(self, x):
        B, T, D = x.shape
        device, dtype = x.device, x.dtype
        record = self._record_diagnostics and not self.training

        # ---- Deep path (ACT) -------------------------------------------------
        step_halt_probs = None
        step_loop_scales = None
        step_displacement = None  # mean ||h[s+1]-h[s]|| over tokens, per step
        h_after_step_1 = None     # snapshot for per-token loop_displacement
        loop_displacement = None  # (B, T) — ||h_final - h_after_step_1|| / ||h_after_step_1||
        if self.has_loop_path:
            prob_remain = torch.ones(B, T, device=device, dtype=dtype)
            output_acc = torch.zeros(B, T, D, device=device, dtype=dtype)
            expected_steps = torch.zeros(B, T, device=device, dtype=dtype)
            step_denom = max(1, self.max_loops - 1)
            h_loop = x
            h_prev_step = None  # tracks previous-step h for step-to-step displacement
            actual_steps = 0
            if record:
                step_halt_probs = torch.zeros(self.max_loops, device=device, dtype=dtype)
                step_loop_scales = torch.zeros(self.max_loops, device=device, dtype=dtype)
                step_displacement = torch.zeros(self.max_loops, device=device, dtype=dtype)
            for step in range(self.max_loops):
                actual_steps = step + 1
                scale = F.softplus(self.loop_scales[step])
                if record:
                    h_prev_step = h_loop  # snapshot before this step's transformer block
                h_loop = self.block(h_loop, scale=scale)
                halt_prob = self.router(h_loop, step_normalized=step / step_denom, x=x)
                p_stop = prob_remain * halt_prob
                prob_remain = prob_remain * (1.0 - halt_prob)
                output_acc = output_acc + h_loop * p_stop.unsqueeze(-1)
                expected_steps = expected_steps + p_stop * (step + 1)
                if record:
                    step_halt_probs[step] = halt_prob.mean().detach()
                    step_loop_scales[step] = scale.detach()
                    # mean ||h[s+1] - h[s]|| per step, in absolute terms
                    step_displacement[step] = (h_loop - h_prev_step).norm(dim=-1).mean().detach()
                    if step == 0:
                        # Snapshot for per-token "did the loop do real work beyond step 1"
                        h_after_step_1 = h_loop.detach()
            output_acc = output_acc + h_loop * prob_remain.unsqueeze(-1)
            expected_steps = expected_steps + prob_remain * actual_steps
            h_deep = output_acc
            if record and h_after_step_1 is not None:
                # Per-token relative displacement of final loop output from step-1 output.
                # Near zero -> loop did nothing past step 1 for this token.
                with torch.no_grad():
                    base_norm = h_after_step_1.norm(dim=-1).clamp(min=1e-6)
                    loop_displacement = (h_loop - h_after_step_1).norm(dim=-1) / base_norm
        else:
            h_deep = torch.zeros_like(x)
            expected_steps = torch.zeros((B, T), device=device, dtype=dtype)

        # ---- Wide path -------------------------------------------------------
        if self.has_wide_path:
            wide_scale_val = F.softplus(self.wide_scale)
            h_wide = self.wide_block(x, scale=wide_scale_val)
        else:
            h_wide = torch.zeros_like(x)

        # ---- Gate / combine --------------------------------------------------
        cross_w2d_norm = None
        cross_d2w_norm = None
        if self.layer_type == "dual":
            if self.gate_mode == "convex":
                output, gate, c_w2d_n, c_d2w_n = self.dual_gate(x, h_deep, h_wide, record=record)
                gate_deep_flat = gate
                gate_wide_flat = 1.0 - gate
            else:
                output, gd, gw, c_w2d_n, c_d2w_n = self.dual_gate(x, h_deep, h_wide, record=record)
                gate_deep_flat = gd
                gate_wide_flat = gw
            cross_w2d_norm = c_w2d_n
            cross_d2w_norm = c_d2w_n
        elif self.layer_type == "loop":
            output = h_deep
            gate_deep_flat = torch.ones(B, T, device=device, dtype=dtype)
            gate_wide_flat = torch.zeros(B, T, device=device, dtype=dtype)
        else:  # wide
            output = h_wide
            gate_deep_flat = torch.zeros(B, T, device=device, dtype=dtype)
            gate_wide_flat = torch.ones(B, T, device=device, dtype=dtype)

        # ---- Per-token diagnostics ------------------------------------------
        diag = None
        if record:
            with torch.no_grad():
                delta_deep = h_deep - x
                delta_wide = h_wide - x
                delta_deep_norm = delta_deep.norm(dim=-1)
                delta_wide_norm = delta_wide.norm(dim=-1)
                # Cosine sim between deep and wide update directions.
                # 0 when either path is missing (norm is exactly 0).
                if self.has_loop_path and self.has_wide_path:
                    denom = (delta_deep_norm.clamp(min=1e-6) * delta_wide_norm.clamp(min=1e-6))
                    delta_cos_sim = (delta_deep * delta_wide).sum(dim=-1) / denom
                else:
                    delta_cos_sim = torch.zeros(B, T, device=device, dtype=dtype)

                zero_bt = torch.zeros(B, T, device=device, dtype=dtype)
                diag = {
                    "gate_deep": gate_deep_flat.detach(),
                    "gate_wide": gate_wide_flat.detach(),
                    "expected_steps": expected_steps.detach(),
                    "delta_deep_norm": delta_deep_norm,
                    "delta_wide_norm": delta_wide_norm,
                    "delta_cos_sim": delta_cos_sim,
                    "cross_w2d_norm": cross_w2d_norm if cross_w2d_norm is not None else zero_bt,
                    "cross_d2w_norm": cross_d2w_norm if cross_d2w_norm is not None else zero_bt,
                    "loop_displacement": loop_displacement if loop_displacement is not None else zero_bt,
                    "step_halt_probs": step_halt_probs if step_halt_probs is not None
                                       else torch.zeros(self.max_loops, device=device, dtype=dtype),
                    "step_loop_scales": step_loop_scales if step_loop_scales is not None
                                        else torch.zeros(self.max_loops, device=device, dtype=dtype),
                    "step_displacement": step_displacement if step_displacement is not None
                                         else torch.zeros(self.max_loops, device=device, dtype=dtype),
                }

        return output, expected_steps, diag


# =============================================================================
# Pretrained / model / for-causal-LM
# =============================================================================

class AdaptiveGPTPreTrainedModel(PreTrainedModel):
    config_class = AdaptiveGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class AdaptiveGPTModel(AdaptiveGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd) if config.poe_type == "ABSOLUTE" else nn.Identity()
        self.drop = nn.Dropout(config.dropout)

        if config.enable_adaptive:
            layer_types = config.adaptive_layer_types
            if not layer_types:
                has_wide = config.wide_ffn_hidden > 0
                layer_types = ["dual" if has_wide else "loop"] * config.n_layer
            elif len(layer_types) != config.n_layer:
                raise ValueError(f"adaptive_layer_types length {len(layer_types)} must match n_layer {config.n_layer}")
        else:
            layer_types = [None] * config.n_layer

        layers = {}
        for i in range(config.n_layer):
            if config.enable_adaptive:
                layers[str(i)] = AdaptiveRecursiveBlock(config, layer_type=layer_types[i])
            else:
                layers[str(i)] = GPT2Block(config)
        self.h = nn.ModuleDict(layers)
        self.lm_head_norm = _build_norm(config)
        self._layer_order = sorted(layers.keys(), key=int)

    def forward(self, input_ids, record_diagnostics=False):
        device = input_ids.device
        seq_len = input_ids.size(1)
        h = self.wte(input_ids)
        if isinstance(self.wpe, nn.Embedding):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            h = h + self.wpe(pos)
        h = self.drop(h)

        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)
        per_layer_diags = [] if record_diagnostics else None

        for key in self._layer_order:
            layer = self.h[key]
            if self.config.enable_adaptive:
                layer._record_diagnostics = record_diagnostics
                h, expected_steps, diag = layer(h)
                total_ponder_cost = total_ponder_cost + expected_steps.mean()
                if record_diagnostics and diag is not None:
                    per_layer_diags.append(diag)
            else:
                h = layer(h, scale=1.0)

        h = self.lm_head_norm(h)
        return h, total_ponder_cost, per_layer_diags


class AdaptiveGPTForCausalLM(AdaptiveGPTPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = AdaptiveGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.use_weight_tying:
            self.lm_head.weight = self.transformer.wte.weight
        self.post_init()
        for module in self.modules():
            if isinstance(module, (DualPathGateConvex, DualPathGateTwoGates)):
                module.reset_parameters()

        # Diagnostics state — OFF by default. Toggle via set_record_diagnostics().
        self._record_diagnostics = False
        self.last_diagnostics: Optional[dict] = None

    # ------------- Diagnostics public API -------------
    def set_record_diagnostics(self, on: bool):
        """Toggle per-token diagnostics recording. Recording also requires
        the model to be in eval() mode; training-mode forwards never record."""
        self._record_diagnostics = bool(on)

    # ------------- HF plumbing -------------
    def get_input_embeddings(self): return self.transformer.wte
    def set_input_embeddings(self, e): self.transformer.wte = e
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, e): self.lm_head = e
    def can_generate(self): return True
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kw):
        return {"input_ids": input_ids}

    def forward(self, input_ids, labels=None, return_dict=True, **kwargs):
        record = self._record_diagnostics and not self.training
        hidden, total_ponder_cost, per_layer_diags = self.transformer(
            input_ids, record_diagnostics=record
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if self.config.enable_adaptive and self.config.ponder_penalty_weight > 0:
                avg_ponder = total_ponder_cost / self.config.n_layer
                max_loops = self.config.max_loops
                if max_loops > 1:
                    loss = loss + ((avg_ponder - 1.0) / (max_loops - 1.0)) * self.config.ponder_penalty_weight

        # Build diagnostics bag (CPU tensors, fp16) if recording.
        if record:
            with torch.no_grad():
                # Per-token loss against input_ids (always computed, even if
                # labels=None — that's what we want for the Paloma analysis).
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                tok_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(input_ids.size(0), -1)
                # Pad to (B, T) with NaN at last position (no next token).
                pad = torch.full((input_ids.size(0), 1), float("nan"),
                                 device=tok_loss.device, dtype=tok_loss.dtype)
                tok_loss_full = torch.cat([tok_loss, pad], dim=1)

                def stack(name):
                    return torch.stack([d[name] for d in per_layer_diags]).to(torch.float16).cpu()

                # Step-wise tensors are per-layer (L, max_loops), already small.
                self.last_diagnostics = {
                    "tokens": input_ids.detach().to(torch.int32).cpu(),
                    "loss": tok_loss_full.to(torch.float16).cpu(),
                    "gate_deep": stack("gate_deep"),
                    "gate_wide": stack("gate_wide"),
                    "expected_steps": stack("expected_steps"),
                    "delta_deep_norm": stack("delta_deep_norm"),
                    "delta_wide_norm": stack("delta_wide_norm"),
                    "delta_cos_sim": stack("delta_cos_sim"),
                    "cross_w2d_norm": stack("cross_w2d_norm"),
                    "cross_d2w_norm": stack("cross_d2w_norm"),
                    "loop_displacement": stack("loop_displacement"),
                    "step_halt_probs": stack("step_halt_probs"),
                    "step_loop_scales": stack("step_loop_scales"),
                    "step_displacement": stack("step_displacement"),
                }

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits)


AdaptiveGPTConfig.register_for_auto_class()
AdaptiveGPTForCausalLM.register_for_auto_class("AutoModelForCausalLM")