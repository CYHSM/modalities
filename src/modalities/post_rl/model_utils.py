"""Model loading utilities with memory optimization options."""
import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format

from config import LoRAConfig, ModelConfig

logger = logging.getLogger(__name__)


def create_bnb_config(load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Create BitsAndBytes config for quantization."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    return None


def create_lora_config(lora_config: LoRAConfig) -> LoraConfig:
    """Create PEFT LoRA configuration."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias="none",
    )


def setup_custom_chat_format(model, tokenizer):
    """Setup custom chat format that starts with <think>."""

    # Custom chat template that forces <think> at start of assistant responses
    chat_template = """{% for message in messages %}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
<think>
{{ message['content'] }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
<think>\n
{%- endif -%}"""

    tokenizer.chat_template = chat_template

    # Set special tokens if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model_and_tokenizer(model_config: ModelConfig):
    """Load model with memory optimization options."""
    logger.info(f"Loading model: {model_config.model_path}")

    # Quantization config
    quantization_config = None
    if model_config.lora.use_qlora or model_config.load_in_4bit:
        quantization_config = create_bnb_config(load_in_4bit=True)
        logger.info("Using 4-bit quantization (QLoRA)")
    elif model_config.load_in_8bit:
        quantization_config = create_bnb_config(load_in_8bit=True)
        logger.info("Using 8-bit quantization")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=model_config.get_torch_dtype(),
        device_map=model_config.device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

    # Setup chat format if needed
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
        logger.info("Chat format applied")
    else:  # use custom one
        model, tokenizer = setup_custom_chat_format(model, tokenizer)
        logger.info("Custom chat format applied")

    # Apply LoRA if configured
    if model_config.lora.use_lora or model_config.lora.use_qlora:
        if quantization_config:
            model = prepare_model_for_kbit_training(model)

        peft_config = create_lora_config(model_config.lora)
        model = get_peft_model(model, peft_config)

        # Log trainable parameters
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model, tokenizer
