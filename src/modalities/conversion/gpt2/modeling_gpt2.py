# coding=utf-8
# This code was copied and modified from the Llama implementation of the Hugging Face Transformers library.
# The original code can be found at:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Original license information:
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)


@dataclass
class AdaptiveCausalLMOutputWithPast(ModelOutput):
    """
    Output type for causal language models with adaptive computation.
    
    Args:
        loss: Language modeling loss (if labels provided).
        logits: Prediction scores of the language modeling head.
        past_key_values: Pre-computed key/value pairs for efficient generation.
        hidden_states: Hidden states at each layer output.
        attentions: Attention weights at each layer.
        ponder_loss: Weighted ponder cost penalty for adaptive computation.
        ponder_cost_unweighted: Raw sum of expected steps across layers.
        expected_steps: Average expected computation steps per layer.
        normalized_steps: Expected steps normalized to [0, 1] range.
        per_layer_ponder_costs: Expected steps for each layer.
        per_layer_cos_sims: Cosine similarity metrics for each layer.
        loop_scales: Learned scaling factors for each loop iteration.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    # Adaptive computation outputs
    ponder_loss: Optional[torch.FloatTensor] = None
    ponder_cost_unweighted: Optional[torch.FloatTensor] = None
    expected_steps: Optional[torch.FloatTensor] = None
    normalized_steps: Optional[torch.FloatTensor] = None
    per_layer_ponder_costs: Optional[torch.FloatTensor] = None
    per_layer_cos_sims: Optional[torch.FloatTensor] = None
    loop_scales: Optional[torch.FloatTensor] = None


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: GPT2Config, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        
        # Compute head_dim the same way as modalities: n_embd // n_head
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Compute inv_freq exactly like modalities RotaryTransform.reset_parameters()
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        
        # For default rope, attention_scaling is 1.0
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Match modalities behavior: compute in model dtype (e.g., bfloat16)
        # Convert inv_freq to x.dtype before computation
        inv_freq_in_dtype = self.inv_freq.to(x.dtype)
        
        # Create position indices in float32. 
        # FIX: Do NOT cast 't' to x.dtype here. Keep it as float32 to ensure
        # the einsum and subsequent cos/sin calculations retain precision.
        seq_len = position_ids.shape[-1]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32) 
        
        # Compute freqs using einsum. 
        # PyTorch will handle the mixed precision (Float32 't' vs BFloat16 'inv_freq').
        # This matches the Modalities implementation exactly.
        freqs = torch.einsum("i,j->ij", t, inv_freq_in_dtype)  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        
        # Shape for broadcasting: (1, 1, seq_len, head_dim)
        # We compute cos/sin on the result of the einsum (likely Float32), 
        # then cast the final result to x.dtype (BFloat16).
        cos = emb.cos()[None, None, :, :].to(x.dtype) * self.attention_scaling
        sin = emb.sin()[None, None, :, :].to(x.dtype) * self.attention_scaling

        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor of shape (B, nh, T, hd).
        k (`torch.Tensor`): The key tensor of shape (B, nh, T, hd).
        cos (`torch.Tensor`): The cosine part of the rotary embedding, shape (1, 1, seq_len, hd).
        sin (`torch.Tensor`): The sine part of the rotary embedding, shape (1, 1, seq_len, hd).
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            Unused, kept for API compatibility.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Slice cos/sin to match sequence length (like modalities does)
    seq_len = q.shape[-2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    
    # Apply rotary embedding: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        # QK Normalization
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.qk_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Apply QK normalization after RoPE (matching modalities implementation)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GPT2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        # Use RMSNorm instead of LayerNorm
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        scale: float = 1.0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + scale * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + scale * hidden_states
        return hidden_states


class AdaptiveRouter(nn.Module):
    """Linear router that predicts halt probability based on hidden state and step."""
    
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.net = nn.Linear(hidden_size + 1, 1, bias=bias)

    def forward(self, x: torch.Tensor, step_normalized: float) -> torch.Tensor:
        B, T, _ = x.shape
        step_feat = torch.full((B, T, 1), step_normalized, device=x.device, dtype=x.dtype)
        logits = self.net(torch.cat([x, step_feat], dim=-1))
        return torch.sigmoid(logits).squeeze(-1)


class AdaptiveDecoderLayer(nn.Module):
    """
    Decoder layer with adaptive computation (PonderNet-style looping).
    
    Wraps a GPT2DecoderLayer and implements learned halting mechanism where
    each token can halt at different iterations based on difficulty.
    """
    
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.max_loops = config.max_loops
        self.halt_threshold = config.halt_threshold
        
        # Core transformer block
        self.block = GPT2DecoderLayer(config, layer_idx)
        
        # Adaptive computation components
        self.router = AdaptiveRouter(config.hidden_size)
        self.loop_scales = nn.Parameter(torch.full((self.max_loops,), -7.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive looping.
        
        Returns:
            hidden_states: Output hidden states (B, T, D)
            expected_steps: Expected number of steps per token (B, T)
            total_cos_sim: Accumulated cosine similarity (B, T)
            loop_scales: Detached loop scaling factors
        """
        B, T, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        h = hidden_states
        accumulated_output = torch.zeros_like(h)
        
        # State tracking
        prob_remain = torch.ones(B, T, device=device, dtype=dtype)
        expected_steps = torch.zeros(B, T, device=device, dtype=dtype)
        total_cos_sim = torch.zeros(B, T, device=device, dtype=dtype)
        
        # Denominator for normalization
        denom = max(1, self.max_loops - 1)
        
        for step in range(self.max_loops):
            # Learned scaling factor per step
            learnable = F.softplus(self.loop_scales[step])
            
            # LayerNorm scaling factor
            current_depth = (self.layer_idx * self.max_loops) + step + 1
            lns_scale = 1.0
            
            current_scale = learnable * lns_scale
            
            h_prev = h
            
            # Apply block with scaling
            h = self.block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                scale=current_scale,
                **kwargs,
            )
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8)
            
            step_norm = step / denom
            halt_prob = self.router(h, step_norm)
            
            if step == self.max_loops - 1:
                # Last step: halt all remaining probability
                p_stop_here = prob_remain
                prob_remain = torch.zeros_like(prob_remain)
            else:
                p_stop_here = prob_remain * halt_prob
                prob_remain = prob_remain * (1.0 - halt_prob)
            
            accumulated_output = accumulated_output + (h * p_stop_here.unsqueeze(-1))
            expected_steps = expected_steps + p_stop_here * (step + 1)
            total_cos_sim = total_cos_sim + (cos_sim * p_stop_here)
            
            if not self.training:
                # Early exit if all tokens have halted
                if prob_remain.max() < (1.0 - self.halt_threshold):
                    break
        
        # Handle early exit: add remaining mass
        if not self.training and prob_remain.sum() > 0:
            accumulated_output = accumulated_output + (h * prob_remain.unsqueeze(-1))
            final_cos_sim = F.cosine_similarity(h, h_prev, dim=-1, eps=1e-8)
            total_cos_sim = total_cos_sim + (prob_remain * final_cos_sim)
        
        return accumulated_output, expected_steps, total_cos_sim, self.loop_scales.detach()


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2DecoderLayer", "AdaptiveDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GPT2DecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.enable_adaptive = config.enable_adaptive

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Build layers - use adaptive layers if enabled
        if self.enable_adaptive:
            self.layers = nn.ModuleList(
                [AdaptiveDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [GPT2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
        
        # Use RMSNorm for final norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | dict:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Track adaptive computation metrics
        if self.enable_adaptive:
            device = hidden_states.device
            dtype = hidden_states.dtype
            total_ponder_cost = torch.tensor(0.0, device=device, dtype=dtype)
            per_layer_ponder_costs = []
            per_layer_cos_sims = []
            per_layer_loop_scales = []
            
            for decoder_layer in self.layers:
                hidden_states, cost, cos_sim, scales = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                cost_mean = cost.mean()
                sim_mean = cos_sim.mean()
                total_ponder_cost = total_ponder_cost + cost_mean
                per_layer_ponder_costs.append(cost_mean)
                per_layer_cos_sims.append(sim_mean)
                per_layer_loop_scales.append(scales)
            
            hidden_states = self.norm(hidden_states)
            
            num_layers = len(self.layers)
            avg_ponder_cost = total_ponder_cost / num_layers
            normalized_steps = (avg_ponder_cost - 1.0) / (self.config.max_loops - 1.0) if self.config.max_loops > 1 else torch.tensor(0.0, dtype=dtype, device=device)
            weighted_ponder_loss = normalized_steps * self.config.ponder_penalty_weight
            
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": past_key_values,
                "ponder_loss": weighted_ponder_loss.to(hidden_states.dtype),
                "ponder_cost_unweighted": total_ponder_cost,
                "expected_steps": avg_ponder_cost,
                "normalized_steps": normalized_steps,
                "per_layer_ponder_costs": torch.stack(per_layer_ponder_costs),
                "per_layer_cos_sims": torch.stack(per_layer_cos_sims),
                "loop_scales": torch.stack(per_layer_loop_scales),
            }
        else:
            # Standard forward pass
            for layer_idx, decoder_layer in enumerate(self.layers):
                lns_scale = 1.0 / (layer_idx + 1)
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    scale=lns_scale,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )


@auto_docstring
class GPT2ForCausalLM(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.enable_adaptive = config.enable_adaptive

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> AdaptiveCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPT2ForCausalLM

        >>> model = GPT2ForCausalLM.from_pretrained("path/to/model")
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")

        >>> prompt = "Hello, how are you?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        if self.enable_adaptive:
            hidden_states = outputs["last_hidden_state"]
        else:
            hidden_states = outputs.last_hidden_state
            
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if self.enable_adaptive:
            return AdaptiveCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs["past_key_values"],
                hidden_states=None,
                attentions=None,
                ponder_loss=outputs["ponder_loss"],
                ponder_cost_unweighted=outputs["ponder_cost_unweighted"],
                expected_steps=outputs["expected_steps"],
                normalized_steps=outputs["normalized_steps"],
                per_layer_ponder_costs=outputs["per_layer_ponder_costs"],
                per_layer_cos_sims=outputs["per_layer_cos_sims"],
                loop_scales=outputs["loop_scales"],
            )
        else:
            return AdaptiveCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class GPT2ForSequenceClassification(GenericForSequenceClassification, GPT2PreTrainedModel):
    ...


class GPT2ForQuestionAnswering(GenericForQuestionAnswering, GPT2PreTrainedModel):
    base_model_prefix = "transformer"


class GPT2ForTokenClassification(GenericForTokenClassification, GPT2PreTrainedModel):
    ...


__all__ = [
    "GPT2ForCausalLM",
    "GPT2Model",
    "GPT2PreTrainedModel",
    "GPT2ForSequenceClassification",
    "GPT2ForQuestionAnswering",
    "GPT2ForTokenClassification",
    "AdaptiveDecoderLayer",
    "AdaptiveRouter",
    "AdaptiveCausalLMOutputWithPast",
]