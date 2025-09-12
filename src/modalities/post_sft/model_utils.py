"""Model loading, saving, and utility functions."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

from modalities.post_sft.config import LoRAConfig

logger = logging.getLogger(__name__)


def create_peft_config(lora_config: LoRAConfig) -> LoraConfig:
    """Create PEFT LoRA configuration from our config."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_dora=lora_config.use_dora,
        use_rslora=lora_config.use_rslora,
    )


def freeze_model_layers(model, trainable_layers):
    """Freeze all layers except specified ones."""
    if trainable_layers is None:
        return

    total_layers = len(model.model.layers)
    logger.info(f"Model has {total_layers} layers")

    # Parse trainable layers
    if trainable_layers == "all":
        return  # Don't freeze anything
    elif isinstance(trainable_layers, str) and trainable_layers.startswith("last_"):
        n = int(trainable_layers.split("_")[1])
        trainable_indices = list(range(total_layers - n, total_layers))
    else:
        trainable_indices = trainable_layers if isinstance(trainable_layers, list) else [trainable_layers]

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    for i in trainable_indices:
        for param in model.model.layers[i].parameters():
            param.requires_grad = True

    # Always keep LM head trainable
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Log stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable layers: {trainable_indices}")
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    trainable_layers=None,
    lora_config: Optional[LoRAConfig] = None,
):
    """Load model and tokenizer with optional LoRA and chat format setup."""
    logger.info(f"Loading model from: {model_path}")

    # Try optimized attention implementations with fallbacks
    attention_implementations = ["flash_attention_2", "sdpa", None]
    model = None
    
    for attn_impl in attention_implementations:
        try:
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            }
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
            
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            if attn_impl == "flash_attention_2":
                logger.info("Using Flash Attention 2")
            elif attn_impl == "sdpa":
                logger.info("Using PyTorch SDPA")
            else:
                logger.info("Using default attention")
            break
            
        except (ImportError, RuntimeError, ValueError) as e:
            if attn_impl:
                logger.warning(f"{attn_impl} not available: {e}")
                continue
            else:
                raise

    # model = torch.compile(model)
    # logger.info("Model compiled with torch.compile")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Setup chat format
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        logger.info("Chat template already exists")
    else:
        logger.info("Setting up chat format")
        model, tokenizer = setup_chat_format(model, tokenizer)
    
    # Special handling for Teuken model
    if "opengpt-x/teuken-7b-instruct-v0.6" in model_path.lower():
        logger.info("Detected Teuken model, resetting chat template")
        tokenizer.chat_template = None
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Apply LoRA if configured
    if lora_config and lora_config.use_lora:
        logger.info(f"Applying LoRA: r={lora_config.lora_r}, alpha={lora_config.lora_alpha}")
        peft_config = create_peft_config(lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif trainable_layers is not None:
        # Apply layer freezing if not using LoRA
        freeze_model_layers(model, trainable_layers)

    logger.info(f"Model loaded successfully. Device: {model.device}")
    return model, tokenizer


def save_model_with_custom_code(model, save_path: str, source_model_path: str):
    """Save model and copy custom code files."""
    save_path = Path(save_path)
    source_path = Path(source_model_path)

    # Save the model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")

    # Copy custom code files
    for file_name in ["modeling_gpt2.py", "configuration_gpt2.py"]:
        source_file = source_path / file_name
        dest_file = save_path / file_name

        if source_file.exists():
            shutil.copy(source_file, dest_file)
            logger.info(f"✅ Copied {file_name}")


def estimate_model_size(model) -> str:
    """Estimate model size in parameters."""
    total_params = sum(p.numel() for p in model.parameters())

    if total_params >= 1e9:
        return f"{total_params/1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params/1e6:.1f}M"
    else:
        return f"{total_params/1e3:.1f}K"


def get_model_info(model, tokenizer) -> Dict[str, Any]:
    """Get comprehensive model information."""
    info = {
        "model_size": estimate_model_size(model),
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type if hasattr(model, "config") else "unknown",
        "device": str(model.device),
        "dtype": str(model.dtype) if hasattr(model, "dtype") else "unknown",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Add LoRA specific info if it's a PEFT model
    if hasattr(model, "peft_config"):
        info["is_peft_model"] = True
        peft_config = model.peft_config
        if hasattr(peft_config, "default"):
            config = peft_config["default"]
            info["lora_r"] = config.r
            info["lora_alpha"] = config.lora_alpha
            info["lora_dropout"] = config.lora_dropout
            info["target_modules"] = config.target_modules
    else:
        info["is_peft_model"] = False

    return info