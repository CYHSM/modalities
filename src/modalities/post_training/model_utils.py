"""Model loading, saving, and utility functions."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

logger = logging.getLogger(__name__)


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

    # also embedding
    # if hasattr(model.model, 'embed_tokens'):
    #     for param in model.model.embed_tokens.parameters():
    #         param.requires_grad = True

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
):
    """Load model and tokenizer with chat format setup."""
    logger.info(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, device_map=device_map, torch_dtype=torch_dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Setup chat format
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        logger.info("Chat template already exists, skipping setup_chat_format")
    else:
        logger.info("Setting up chat format")
        model, tokenizer = setup_chat_format(model, tokenizer)

    if trainable_layers is not None:
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
    custom_files = ["modeling_gpt2.py", "configuration_gpt2.py"]

    for file_name in custom_files:
        source_file = source_path / file_name
        dest_file = save_path / file_name

        if source_file.exists():
            shutil.copy(source_file, dest_file)
            logger.info(f"✅ Copied {file_name} to {save_path}")
        else:
            logger.warning(f"⚠️  Warning: {file_name} not found in {source_path}")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    pad_token_id: Optional[int] = None,
) -> str:
    """Generate a response from the model."""
    # Format as chat if needed
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set pad_token_id if not provided
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def batch_generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    batch_size: int = 4,
) -> List[str]:
    """Generate responses for multiple prompts."""
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = []

        for prompt in batch_prompts:
            try:
                response = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )
                batch_responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for prompt: {e}")
                batch_responses.append(f"ERROR: {str(e)}")

        responses.extend(batch_responses)
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")

    return responses


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
    return {
        "model_size": estimate_model_size(model),
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type if hasattr(model, "config") else "unknown",
        "device": str(model.device),
        "dtype": str(model.dtype) if hasattr(model, "dtype") else "unknown",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
