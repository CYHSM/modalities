# coding=utf-8
# This code was copied and modified from the Llama implementation of the Hugging Face Transformers library.
# The original code can be found at:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
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
"""LLaMA-like GPT2 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class GPT2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPT2Model`]. It is used to instantiate an GPT2
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the GPT2 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            Number of key_value heads for Grouped Query Attention.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply QK normalization in attention.
        qk_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon for QK normalization.
        enable_adaptive (`bool`, *optional*, defaults to `False`):
            Whether to enable adaptive computation (PonderNet-style looping).
        max_loops (`int`, *optional*, defaults to 10):
            Maximum number of loops per layer when adaptive computation is enabled.
        halt_threshold (`float`, *optional*, defaults to 0.99):
            Probability threshold for early halting during inference.
        ponder_penalty_weight (`float`, *optional*, defaults to 0.01):
            Weight for the ponder cost penalty in the loss.
    """

    model_type = "modalities-gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `GPT2Model`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps: float = 1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        # QK normalization
        use_qk_norm=False,
        qk_norm_eps=1e-05,
        # Adaptive computation parameters
        enable_adaptive=False,
        max_loops=10,
        halt_threshold=0.99,
        ponder_penalty_weight=0.01,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        
        # QK normalization
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        
        # Adaptive computation parameters
        self.enable_adaptive = enable_adaptive
        self.max_loops = max_loops
        self.halt_threshold = halt_threshold
        self.ponder_penalty_weight = ponder_penalty_weight
        
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )