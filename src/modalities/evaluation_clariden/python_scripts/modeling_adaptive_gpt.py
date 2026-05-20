# coding=utf-8
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
        # ---- Adaptive computation ------------------------------------------
        enable_adaptive: bool = False,
        max_loops: int = 10,
        ponder_penalty_weight: float = 0.0,
        wide_ffn_hidden: int = 0,
        # ---- Gate mode -----------------------------------------------------
        # "convex":    output = g * h_deep + (1 - g) * h_wide  (single gate)
        # "two_gates": output = g_d * h_deep_eff + g_w * h_wide_eff
        #              with g_d, g_w independent sigmoids on x.
        gate_mode: str = "two_gates",
        # ---- Convex-mode gate init ----------------------------------------
        gate_init_bias: float = 0.0,
        # ---- Two-gates-mode gate inits ------------------------------------
        deep_gate_init_bias: float = 0.0,
        wide_gate_init_bias: float = 0.0,
        # ---- Per-iteration loop / wide scales -----------------------------
        loop_scale_init: float = -7,
        wide_scale_init: float = -7,
        # ---- Cross-path mixing --------------------------------------------
        # Learnable, zero-initialized leak between branches BEFORE gating.
        # In convex mode:
        #     h_deep_eff = h_deep + softplus(s_d) * proj_w2d(h_wide)
        #     h_wide_eff = h_wide + softplus(s_w) * proj_d2w(h_deep)
        #     output     = g * h_deep_eff + (1-g) * h_wide_eff
        # In two_gates mode (post-gate contamination, original formulation):
        #     contam_w2d = g_d * (softplus(s_d) * proj_w2d(h_wide))
        #     contam_d2w = g_w * (softplus(s_w) * proj_d2w(h_deep))
        #     output     = (g_d * h_deep + contam_w2d) + (g_w * h_wide + contam_d2w)
        use_cross: bool = True,
        cross_scale_deep_init: float = -7.0,
        cross_scale_wide_init: float = -7.0,
        # ---- Layer-type schedule ------------------------------------------
        # Per-layer choice of {"loop", "wide", "dual"}. Note: this is renamed
        # from "layer_types" to avoid colliding with HF's PretrainedConfig,
        # which validates a same-named field against an attention-type enum.
        adaptive_layer_types: Optional[list] = None,
        **kwargs,
    ):
        kwargs["tie_word_embeddings"] = use_weight_tying
        # Back-compat: accept the old "layer_types" kwarg if someone passes it
        # (e.g. from a serialized config saved before the rename), but route
        # it to our own attribute so HF's validator never sees it.
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
        # HF compatibility aliases
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head_q
        self.hidden_size = n_embd

        if gate_mode not in ("convex", "two_gates"):
            raise ValueError(f"gate_mode must be 'convex' or 'two_gates', got {gate_mode}")


class RMSLayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.bias is None:
            return output * self.weight
        return output * self.weight + self.bias


def _build_norm(config: AdaptiveGPTConfig, dim: Optional[int] = None) -> nn.Module:
    d = dim if dim is not None else config.n_embd
    if config.norm_type == "layer_norm":
        return nn.LayerNorm(d, eps=config.norm_eps, elementwise_affine=config.norm_elementwise_affine, bias=config.norm_bias)
    elif config.norm_type == "rms_norm":
        return RMSLayerNorm(ndim=d, bias=config.norm_bias, epsilon=config.norm_eps)
    elif config.norm_type == "pytorch_rms_norm":
        return nn.RMSNorm(d, eps=config.norm_eps)
    else:
        raise ValueError(f"Unknown norm_type: {config.norm_type}")


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, enforce_swiglu_hidden_dim_multiple_of: int = 256):
        super().__init__()
        hidden_dim = self._get_hidden_dim(ffn_hidden, enforce_swiglu_hidden_dim_multiple_of)
        self.W = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.V = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.W_2 = nn.Linear(hidden_dim, n_embd, bias=bias)

    @staticmethod
    def _get_hidden_dim(ffn_hidden: int, multiple: int) -> int:
        adjusted = int(2 * ffn_hidden / 3)
        return ((adjusted + multiple - 1) // multiple) * multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(self.silu(self.W(x)) * self.V(x))


class RotaryTransform(nn.Module):
    def __init__(self, n_embd: int, n_head: int, seq_length_dim: int = -2, base_freq: int = 10000):
        super().__init__()
        self.dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        self.base_freq = base_freq
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, self.dim_model, 2).float() / self.dim_model))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached: Optional[int] = None
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update_cos_sin_tables(self, x: torch.Tensor):
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
        cos, sin = self._update_cos_sin_tables(k)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AdaptiveGPTConfig):
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

        transforms: list = []
        if config.use_rotary:
            transforms.append(RotaryTransform(n_embd=config.n_embd, n_head=config.n_head_q, base_freq=config.rotary_base_freq))
        self.qkv_transforms = nn.ModuleList(transforms)

        if config.use_qk_norm:
            q_dim = config.qk_norm_dim if config.qk_norm_dim else (config.n_embd // config.n_head_q)
            self.q_norm = _build_norm(config, dim=q_dim)
            self.k_norm = _build_norm(config, dim=q_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        B, nh_kv, T, hs = x.shape
        if n_rep == 1: return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        n_head_dim = D // self.n_head_q

        q = self.q_attn(x).view(B, T, self.n_head_q, n_head_dim).transpose(1, 2).contiguous()
        k = self.k_attn(x).view(B, T, self.n_head_kv, n_head_dim).transpose(1, 2).contiguous()
        v = self.v_attn(x).view(B, T, self.n_head_kv, n_head_dim).transpose(1, 2).contiguous()

        for transform in self.qkv_transforms:
            q, k, v = transform(q, k, v)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.resid_dropout(self.c_proj(y))


class TransformerMLP(nn.Module):
    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, ffn_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ffn_hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, config: AdaptiveGPTConfig, ffn_hidden_override: Optional[int] = None):
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

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        x = x + scale * self.attn(self.attention_norm(x))
        x = x + scale * self.mlp(self.ffn_norm(x))
        return x


class AdaptiveRouter(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd + 1, 1, bias=bias)

    def forward(self, h: torch.Tensor, step_normalized: float, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = h.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=h.device, dtype=h.dtype)
        logit = self.linear(torch.cat([h, step_feat], dim=-1))
        return torch.sigmoid(logit).squeeze(-1)


class DualPathGateConvex(nn.Module):
    """Single-gate convex combination, optionally with cross-path mixing.

    Without cross-path:
        g      = sigmoid(gate_proj(x))
        output = g * h_deep + (1 - g) * h_wide

    With cross-path:
        h_deep_eff = h_deep + softplus(s_d) * proj_w2d(h_wide)
        h_wide_eff = h_wide + softplus(s_w) * proj_d2w(h_deep)
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

    def forward(self, x: torch.Tensor, h_deep: torch.Tensor, h_wide: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            h_deep_eff = h_deep + s_d * self.proj_w2d(h_wide)
            h_wide_eff = h_wide + s_w * self.proj_d2w(h_deep)
        else:
            h_deep_eff = h_deep
            h_wide_eff = h_wide

        return gate * h_deep_eff + (1.0 - gate) * h_wide_eff


class DualPathGateTwoGates(nn.Module):
    """Two independent sigmoid gates, optionally with cross-path mixing.

    Without cross-path:
        logits   = gate_proj(x)                  # (B, T, 2)
        g_d, g_w = sigmoid(logits[..., 0:1]), sigmoid(logits[..., 1:2])
        output   = g_d * h_deep + g_w * h_wide

    With cross-path (post-gate contamination):
        cross_w2d = softplus(s_d) * proj_w2d(h_wide)   # gated by g_d
        cross_d2w = softplus(s_w) * proj_d2w(h_deep)   # gated by g_w
        output    = (g_d * h_deep + g_d * cross_w2d) + (g_w * h_wide + g_w * cross_d2w)

    The gates are independent: g_d + g_w is unconstrained.
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

    def forward(self, x: torch.Tensor, h_deep: torch.Tensor, h_wide: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_proj(x))           # (B, T, 2)
        gate_deep = gates[..., 0:1]
        gate_wide = gates[..., 1:2]

        if self.use_cross:
            s_d = F.softplus(self.cross_scale_deep)
            s_w = F.softplus(self.cross_scale_wide)
            cross_w2d = s_d * self.proj_w2d(h_wide)
            cross_d2w = s_w * self.proj_d2w(h_deep)
            h_deep_branch = gate_deep * h_deep + gate_deep * cross_w2d
            h_wide_branch = gate_wide * h_wide + gate_wide * cross_d2w
        else:
            h_deep_branch = gate_deep * h_deep
            h_wide_branch = gate_wide * h_wide

        return h_deep_branch + h_wide_branch


def _build_dual_gate(config: AdaptiveGPTConfig) -> nn.Module:
    if config.gate_mode == "convex":
        return DualPathGateConvex(
            n_embd=config.n_embd,
            gate_init_bias=config.gate_init_bias,
            use_cross=config.use_cross,
            cross_scale_deep_init=config.cross_scale_deep_init,
            cross_scale_wide_init=config.cross_scale_wide_init,
        )
    elif config.gate_mode == "two_gates":
        return DualPathGateTwoGates(
            n_embd=config.n_embd,
            deep_gate_init_bias=config.deep_gate_init_bias,
            wide_gate_init_bias=config.wide_gate_init_bias,
            use_cross=config.use_cross,
            cross_scale_deep_init=config.cross_scale_deep_init,
            cross_scale_wide_init=config.cross_scale_wide_init,
        )
    else:
        raise ValueError(f"Unknown gate_mode: {config.gate_mode}")


class AdaptiveRecursiveBlock(nn.Module):
    def __init__(self, config: AdaptiveGPTConfig, layer_type: str = "dual"):
        super().__init__()
        self.layer_type = layer_type
        self.max_loops = config.max_loops
        n_embd = config.n_embd

        self.has_loop_path = layer_type in ("loop", "dual")
        if self.has_loop_path:
            self.block = GPT2Block(config)
            self.router = AdaptiveRouter(n_embd)
            self.loop_scales = nn.Parameter(
                torch.full((self.max_loops,), config.loop_scale_init)
            )

        self.has_wide_path = layer_type in ("wide", "dual")
        if self.has_wide_path:
            self.wide_block = GPT2Block(config, ffn_hidden_override=config.wide_ffn_hidden)
            self.wide_scale = nn.Parameter(torch.tensor([config.wide_scale_init]))

        if layer_type == "dual":
            self.dual_gate = _build_dual_gate(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if self.has_loop_path:
            prob_remain = torch.ones(B, T, device=device, dtype=dtype)
            output_acc = torch.zeros(B, T, D, device=device, dtype=dtype)
            expected_steps = torch.zeros(B, T, device=device, dtype=dtype)
            step_denom = max(1, self.max_loops - 1)
            h_loop = x
            actual_steps = 0

            for step in range(self.max_loops):
                actual_steps = step + 1

                scale = F.softplus(self.loop_scales[step])
                h_loop = self.block(h_loop, scale=scale)

                halt_prob = self.router(h_loop, step_normalized=step / step_denom, x=x)
                p_stop = prob_remain * halt_prob
                prob_remain = prob_remain * (1.0 - halt_prob)

                output_acc = output_acc + h_loop * p_stop.unsqueeze(-1)
                expected_steps = expected_steps + p_stop * (step + 1)

            output_acc = output_acc + h_loop * prob_remain.unsqueeze(-1)
            expected_steps = expected_steps + prob_remain * actual_steps
            h_deep = output_acc
        else:
            h_deep = torch.zeros_like(x)
            expected_steps = torch.zeros((B, T), device=device, dtype=dtype)

        if self.has_wide_path:
            wide_scale_val = F.softplus(self.wide_scale)
            h_wide = self.wide_block(x, scale=wide_scale_val)
        else:
            h_wide = torch.zeros_like(x)

        if self.layer_type == "dual":
            output = self.dual_gate(x, h_deep, h_wide)
        elif self.layer_type == "loop":
            output = h_deep
        elif self.layer_type == "wide":
            output = h_wide
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        return output, expected_steps


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
        # NOTE: We do NOT dispatch DualPathGate* here. HF's smart-apply may
        # skip parent modules whose parameters were already initialized via
        # their child Linears. We instead call `reset_parameters()` on the
        # gate modules explicitly after `post_init()` (see _reset_dual_gates).


class AdaptiveGPTModel(AdaptiveGPTPreTrainedModel):
    def __init__(self, config: AdaptiveGPTConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.poe_type == "ABSOLUTE":
            self.wpe = nn.Embedding(config.sequence_length, config.n_embd)
        else:
            self.wpe = nn.Identity()

        self.drop = nn.Dropout(config.dropout)

        if config.enable_adaptive:
            layer_types = config.adaptive_layer_types
            if not layer_types:
                has_wide = config.wide_ffn_hidden > 0
                layer_types = ["dual" if has_wide else "loop"] * config.n_layer
            elif len(layer_types) != config.n_layer:
                raise ValueError(
                    f"adaptive_layer_types length {len(layer_types)} must match n_layer {config.n_layer}"
                )

        layers: dict = {}
        for i in range(config.n_layer):
            if config.enable_adaptive:
                layers[str(i)] = AdaptiveRecursiveBlock(config, layer_type=layer_types[i])
            else:
                layers[str(i)] = GPT2Block(config)
        self.h = nn.ModuleDict(layers)

        self.lm_head_norm = _build_norm(config)
        self._layer_order = sorted(layers.keys(), key=int)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        seq_len = input_ids.size(1)

        h = self.wte(input_ids)
        if isinstance(self.wpe, nn.Embedding):
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            h = h + self.wpe(pos)

        h = self.drop(h)
        total_ponder_cost = torch.tensor(0.0, device=device, dtype=h.dtype)

        for key in self._layer_order:
            layer = self.h[key]
            if self.config.enable_adaptive:
                h, expected_steps = layer(h)
                total_ponder_cost = total_ponder_cost + expected_steps.mean()
            else:
                h = layer(h, scale=1.0)

        h = self.lm_head_norm(h)
        return h, total_ponder_cost


class AdaptiveGPTForCausalLM(AdaptiveGPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: AdaptiveGPTConfig):
        super().__init__(config)
        self.transformer = AdaptiveGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.use_weight_tying:
            self.lm_head.weight = self.transformer.wte.weight

        self.post_init()

        # Re-run reset_parameters() on every dual-path gate. The HF init
        # path may skip these modules (their direct parameters get marked
        # initialized via their child Linears first), so the carefully-chosen
        # zero-init for gate_proj weights and bias values would otherwise be
        # overwritten by the default Linear init.
        for module in self.modules():
            if isinstance(module, (DualPathGateConvex, DualPathGateTwoGates)):
                module.reset_parameters()

    def get_input_embeddings(self): return self.transformer.wte
    def set_input_embeddings(self, new_embeddings): self.transformer.wte = new_embeddings
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        hidden, total_ponder_cost = self.transformer(input_ids)
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

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutput(loss=loss, logits=logits)


AdaptiveGPTConfig.register_for_auto_class()
AdaptiveGPTForCausalLM.register_for_auto_class("AutoModelForCausalLM")