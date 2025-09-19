"""Model loading and utility functions for GKD training."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

from modalities.post_gkd.config import LoRAConfig

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

    if trainable_layers == "all":
        return
    elif isinstance(trainable_layers, str) and trainable_layers.startswith("last_"):
        n = int(trainable_layers.split("_")[1])
        trainable_indices = list(range(total_layers - n, total_layers))
    else:
        trainable_indices = trainable_layers if isinstance(trainable_layers, list) else [trainable_layers]

    for param in model.parameters():
        param.requires_grad = False

    for i in trainable_indices:
        for param in model.model.layers[i].parameters():
            param.requires_grad = True

    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable layers: {trainable_indices}")
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")


def load_model_with_attention(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    is_teacher: bool = False,
):
    """Load a model with optimized attention implementation."""
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
                logger.info(f"{'Teacher' if is_teacher else 'Student'} using Flash Attention 2")
            elif attn_impl == "sdpa":
                logger.info(f"{'Teacher' if is_teacher else 'Student'} using PyTorch SDPA")
            else:
                logger.info(f"{'Teacher' if is_teacher else 'Student'} using default attention")
            break
            
        except (ImportError, RuntimeError, ValueError) as e:
            if attn_impl:
                logger.warning(f"{attn_impl} not available: {e}")
                continue
            else:
                raise

    return model


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    trainable_layers=None,
    lora_config: Optional[LoRAConfig] = None,
):
    """Load student model and tokenizer with optional LoRA."""
    logger.info(f"Loading student model from: {model_path}")

    model = load_model_with_attention(
        model_path, torch_dtype, trust_remote_code, device_map, is_teacher=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    #     logger.info("Chat template already exists")
    # else:
    #     logger.info("Setting up chat format for student")
    #     model, tokenizer = setup_chat_format(model, tokenizer)
    
    # Special handling for Qwen models - ensure proper chat template
    # if "qwen" in model_path.lower():
    #     logger.info("Detected Qwen model, ensuring proper chat template")
    #     if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
    #         tokenizer.chat_template = None
    #         model, tokenizer = setup_chat_format(model, tokenizer)

    if lora_config and lora_config.use_lora:
        logger.info(f"Applying LoRA to student: r={lora_config.lora_r}, alpha={lora_config.lora_alpha}")
        peft_config = create_peft_config(lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif trainable_layers is not None:
        freeze_model_layers(model, trainable_layers)

    logger.info(f"Student model loaded successfully. Device: {model.device}")
    return model, tokenizer


def load_teacher_model(
    teacher_model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
):
    """Load teacher model for distillation."""
    logger.info(f"Loading teacher model from: {teacher_model_path}")

    teacher_model = load_model_with_attention(
        teacher_model_path, torch_dtype, trust_remote_code, device_map, is_teacher=True
    )

    # Set teacher to eval mode for distillation
    teacher_model.eval()
    
    # Freeze teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False

    logger.info(f"Teacher model loaded and frozen. Device: {teacher_model.device}")
    return teacher_model


def create_vocabulary_mapping(student_tokenizer, teacher_tokenizer):
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    
    token_mapping = {}
    matched_tokens = 0
    unmatched_tokens = []
    
    for token, teacher_id in teacher_vocab.items():
        if token in student_vocab:
            student_id = student_vocab[token]
            token_mapping[teacher_id] = student_id
            matched_tokens += 1
        else:
            unmatched_tokens.append((token, teacher_id))
    
    logger.info(f"Matched {matched_tokens}/{len(teacher_vocab)} tokens")
    logger.info(f"Unmatched tokens: {len(unmatched_tokens)}")
    
    return token_mapping, unmatched_tokens


def smart_resize_embeddings(
    student_model,
    teacher_model, 
    student_tokenizer,
    teacher_tokenizer,
    initialization_method="mean"
):
    student_vocab_size = student_model.get_input_embeddings().num_embeddings
    teacher_vocab_size = teacher_model.get_input_embeddings().num_embeddings
    embedding_dim = student_model.get_input_embeddings().embedding_dim
    
    logger.info(f"Resizing student embeddings: {student_vocab_size} -> {teacher_vocab_size}")
    
    old_embeddings = student_model.get_input_embeddings().weight.data.clone()
    old_lm_head = student_model.get_output_embeddings().weight.data.clone()
    
    model_dtype = old_embeddings.dtype
    model_device = old_embeddings.device
    
    new_embed_tokens = nn.Embedding(teacher_vocab_size, embedding_dim, padding_idx=0)
    new_lm_head = nn.Linear(embedding_dim, teacher_vocab_size, bias=False)
    
    new_embed_tokens = new_embed_tokens.to(device=model_device, dtype=model_dtype)
    new_lm_head = new_lm_head.to(device=model_device, dtype=model_dtype)
    
    token_mapping, unmatched_tokens = create_vocabulary_mapping(student_tokenizer, teacher_tokenizer)
    
    with torch.no_grad():
        if initialization_method == "mean":
            mean_embedding = old_embeddings.mean(dim=0)
            mean_lm_head = old_lm_head.mean(dim=0)
            new_embed_tokens.weight.data.fill_(0)
            new_embed_tokens.weight.data += mean_embedding.unsqueeze(0)
            new_lm_head.weight.data.fill_(0)  
            new_lm_head.weight.data += mean_lm_head.unsqueeze(0)
        elif initialization_method == "random":
            nn.init.normal_(new_embed_tokens.weight, mean=0.0, std=0.02)
            nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
        
        for teacher_id, student_id in token_mapping.items():
            if student_id < student_vocab_size and teacher_id < teacher_vocab_size:
                new_embed_tokens.weight.data[teacher_id] = old_embeddings[student_id]
                new_lm_head.weight.data[teacher_id] = old_lm_head[student_id]
        
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<bos>", "<eos>"]
        for token in special_tokens:
            if token in student_tokenizer.get_vocab() and token in teacher_tokenizer.get_vocab():
                student_id = student_tokenizer.get_vocab()[token]
                teacher_id = teacher_tokenizer.get_vocab()[token]
                if student_id < student_vocab_size and teacher_id < teacher_vocab_size:
                    new_embed_tokens.weight.data[teacher_id] = old_embeddings[student_id]
                    new_lm_head.weight.data[teacher_id] = old_lm_head[student_id]
    
    student_model.model.embed_tokens = new_embed_tokens
    student_model.lm_head = new_lm_head
    
    final_size = student_model.get_input_embeddings().num_embeddings
    logger.info(f"Embedding layer resized to {final_size}")
    
    return student_model


def initialize_unmatched_tokens_from_subwords(
    student_model,
    teacher_tokenizer,
    student_tokenizer,
    unmatched_tokens
):
    embed_layer = student_model.get_input_embeddings()
    lm_head = student_model.get_output_embeddings()
    
    student_vocab = student_tokenizer.get_vocab()
    
    with torch.no_grad():
        for token, teacher_id in unmatched_tokens[:1000]:
            subwords = teacher_tokenizer.tokenize(token)
            
            if len(subwords) > 1:
                embeddings = []
                lm_heads = []
                
                for subword in subwords:
                    if subword in student_vocab:
                        student_id = student_vocab[subword]
                        if student_id < len(student_tokenizer):
                            embeddings.append(embed_layer.weight.data[teacher_id].clone())
                            lm_heads.append(lm_head.weight.data[teacher_id].clone())
                
                if embeddings:
                    embed_layer.weight.data[teacher_id] = torch.stack(embeddings).mean(dim=0)
                    lm_head.weight.data[teacher_id] = torch.stack(lm_heads).mean(dim=0)
    
    logger.info("Initialized unmatched tokens using subword averaging")


def freeze_all_except_embeddings(model):
    """Freeze all parameters except embedding and lm_head layers."""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            logger.info(f"  Trainable: {name}")
        else:
            param.requires_grad = False
    
    logger.info(f"Embedding-only training:")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")


def load_student_and_teacher(
    student_path: str,
    teacher_path: str,
    student_device: str = "cuda:0",
    teacher_device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    initialization_method: str = "mean",
    align_vocabularies: bool = True,
    **kwargs
) -> Tuple:
    student_device_map = {"": student_device} if student_device != "auto" else "auto"
    teacher_device_map = {"": teacher_device} if teacher_device != "auto" else "auto"
    
    student_model = AutoModelForCausalLM.from_pretrained(
        student_path,
        trust_remote_code=trust_remote_code,
        device_map=student_device_map,
        torch_dtype=torch_dtype
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        trust_remote_code=trust_remote_code,
        device_map=teacher_device_map,
        torch_dtype=torch_dtype
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    student_model_size = student_model.get_input_embeddings().num_embeddings
    teacher_model_size = teacher_model.get_input_embeddings().num_embeddings
    
    logger.info(f"Student vocab size: {student_model_size}")
    logger.info(f"Teacher vocab size: {teacher_model_size}")
    
    if align_vocabularies and student_model_size != teacher_model_size:
        logger.info("Aligning vocabularies using smart mapping...")
        
        student_model = smart_resize_embeddings(
            student_model,
            teacher_model,
            student_tokenizer, 
            teacher_tokenizer,
            initialization_method=initialization_method
        )
        
        token_mapping, unmatched_tokens = create_vocabulary_mapping(student_tokenizer, teacher_tokenizer)
        
        if initialization_method == "subword" and unmatched_tokens:
            initialize_unmatched_tokens_from_subwords(
                student_model,
                teacher_tokenizer,
                student_tokenizer,
                unmatched_tokens
            )
        
        tokenizer_to_use = teacher_tokenizer
        logger.info("Using teacher tokenizer for training")
    else:
        tokenizer_to_use = student_tokenizer
        logger.info("Using student tokenizer for training")
    
    # only embedding training
    # freeze_all_except_embeddings(student_model)

    return student_model, teacher_model, tokenizer_to_use


# def load_student_and_teacher(
#     student_path: str,
#     teacher_path: str,
#     device: str = "cuda",
#     teacher_device: str = "cuda:0", 
#     student_device: str = "cuda:0",
#     torch_dtype: torch.dtype = torch.bfloat16,
#     trust_remote_code: bool = True,
#     device_map: str = "auto",
#     trainable_layers=None,
#     lora_config: Optional[LoRAConfig] = None,
#     vocab_alignment_method: str = "teacher_tokenizer",
# ) -> Tuple[Any, Any, Any]:
#     """Load both student and teacher models for GKD with vocabulary alignment."""
    
#     # Load models first to get ACTUAL embedding sizes
#     logger.info("Loading models to check actual embedding sizes...")
    
#     # CHANGE: Use specific device maps for each model
#     student_device_map = {"": student_device} if student_device != "auto" else device_map
#     teacher_device_map = {"": teacher_device} if teacher_device != "auto" else device_map
    
#     print(f"Student device map: {student_device_map}")
#     print(f"Teacher device map: {teacher_device_map}")

#     student_model, student_tokenizer = load_model_and_tokenizer(
#         student_path, device=student_device, torch_dtype=torch_dtype,
#         trust_remote_code=trust_remote_code, device_map=student_device_map,
#         trainable_layers=trainable_layers, lora_config=lora_config,
#     )
    
#     teacher_model = load_teacher_model(
#         teacher_path, torch_dtype=torch_dtype,
#         trust_remote_code=trust_remote_code, device_map=teacher_device_map,
#     )
    
#     teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
#     # Get ACTUAL sizes from models, not tokenizers
#     student_tokenizer_size = len(student_tokenizer)
#     teacher_tokenizer_size = len(teacher_tokenizer)
#     student_model_size = student_model.get_input_embeddings().num_embeddings
#     teacher_model_size = teacher_model.get_input_embeddings().num_embeddings
    
#     logger.info(f"=== VOCAB SIZE DEBUG ===")
#     logger.info(f"Student tokenizer: {student_tokenizer_size}")
#     logger.info(f"Student model embeddings: {student_model_size}")
#     logger.info(f"Teacher tokenizer: {teacher_tokenizer_size}")  
#     logger.info(f"Teacher model embeddings: {teacher_model_size}")
#     logger.info(f"========================")
    
#     # Check if model embedding sizes match (this is what matters for GKD)
#     if student_model_size != teacher_model_size:
#         logger.warning(f"Model embedding mismatch: Student={student_model_size}, Teacher={teacher_model_size}")
        
#         if vocab_alignment_method == "teacher_tokenizer":
#             logger.info(f"Resizing student embeddings: {student_model_size} -> {teacher_model_size}")
            
#             # Manual resize based on ACTUAL teacher model size
#             old_embeddings = student_model.model.embed_tokens.weight.data
#             old_lm_head = student_model.lm_head.weight.data
            
#             # Get the correct dtype and device from the existing model
#             model_dtype = old_embeddings.dtype
#             model_device = old_embeddings.device
            
#             logger.info(f"Using model dtype: {model_dtype}, device: {model_device}")
            
#             # Create new layers with teacher MODEL size, matching dtype and device
#             new_embed_tokens = torch.nn.Embedding(teacher_model_size, old_embeddings.size(1), padding_idx=0)
#             new_lm_head = torch.nn.Linear(old_embeddings.size(1), teacher_model_size, bias=False)
            
#             # Move to correct device and dtype BEFORE initialization
#             new_embed_tokens = new_embed_tokens.to(device=model_device, dtype=model_dtype)
#             new_lm_head = new_lm_head.to(device=model_device, dtype=model_dtype)
            
#             # Initialize with random values in the correct dtype
#             with torch.no_grad():
#                 torch.nn.init.normal_(new_embed_tokens.weight, mean=0.0, std=0.02)
#                 torch.nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
            
#             # Copy overlapping embeddings
#             min_vocab = min(student_model_size, teacher_model_size)
#             with torch.no_grad():
#                 new_embed_tokens.weight.data[:min_vocab] = old_embeddings[:min_vocab]
#                 new_lm_head.weight.data[:min_vocab] = old_lm_head[:min_vocab]
            
#             # Replace layers
#             student_model.model.embed_tokens = new_embed_tokens
#             student_model.lm_head = new_lm_head
            
#             # Verify alignment worked
#             final_student_size = student_model.get_input_embeddings().num_embeddings
#             if final_student_size != teacher_model_size:
#                 raise RuntimeError(f"Alignment failed: {final_student_size} != {teacher_model_size}")
            
#             logger.info(f"Models aligned: {final_student_size} embeddings with dtype {model_dtype}")
            
#         else:
#             raise ValueError(f"Model embedding mismatch: {student_model_size} != {teacher_model_size}")
#     else:
#         logger.info(f"Model embeddings already match: {student_model_size}")
    
#     return student_model, teacher_model, teacher_tokenizer


def save_model_with_custom_code(model, save_path: str, source_model_path: str):
    """Save model and copy custom code files."""
    save_path = Path(save_path)
    source_path = Path(source_model_path)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")

    for file_name in ["modeling_qwen3.py", "configuration_qwen3.py"]:
        source_file = source_path / file_name
        dest_file = save_path / file_name

        if source_file.exists():
            shutil.copy(source_file, dest_file)
            logger.info(f"Copied {file_name}")


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
        "vocab_size": len(tokenizer),
        "model_type": model.config.model_type if hasattr(model, "config") else "unknown",
        "device": str(model.device),
        "dtype": str(model.dtype) if hasattr(model, "dtype") else "unknown",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

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


def load_model_with_teacher_tokenizer(
    student_path: str,
    teacher_tokenizer,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    trainable_layers=None,
    lora_config: Optional[LoRAConfig] = None,
):
    """Load student model but adapt it to use teacher tokenizer."""
    logger.info("Loading student model with teacher tokenizer adaptation...")
    
    # Load original student model
    student_model = load_model_with_attention(
        student_path, torch_dtype, trust_remote_code, device_map, is_teacher=False
    )
    
    original_student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    
    # Get vocabulary sizes
    original_vocab_size = len(original_student_tokenizer)
    teacher_vocab_size = len(teacher_tokenizer)
    
    logger.info(f"Adapting embedding layer: {original_vocab_size} -> {teacher_vocab_size}")
    
    # Save original embedding and lm_head weights
    original_embeddings = student_model.get_input_embeddings().weight.data.clone()
    original_lm_head = student_model.get_output_embeddings().weight.data.clone()
    
    # Resize model embeddings to match teacher vocab size
    student_model.resize_token_embeddings(teacher_vocab_size)
    
    actual_student_vocab = student_model.get_input_embeddings().num_embeddings
    if actual_student_vocab != teacher_vocab_size:
        raise RuntimeError(f"Embedding resize failed! Student: {actual_student_vocab}, Teacher: {teacher_vocab_size}")

    # Initialize new embeddings intelligently
    new_embeddings = student_model.get_input_embeddings().weight.data
    new_lm_head = student_model.get_output_embeddings().weight.data
    
    # Map overlapping tokens
    student_vocab = original_student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    
    mapped_count = 0
    for token, teacher_id in teacher_vocab.items():
        if token in student_vocab:
            student_id = student_vocab[token]
            if student_id < original_vocab_size and teacher_id < teacher_vocab_size:
                new_embeddings[teacher_id] = original_embeddings[student_id]
                new_lm_head[teacher_id] = original_lm_head[student_id]
                mapped_count += 1
    
    logger.info(f"Mapped {mapped_count}/{teacher_vocab_size} tokens from original embeddings")
    
    # Apply LoRA or layer freezing if configured
    if lora_config and lora_config.use_lora:
        logger.info(f"Applying LoRA to adapted student: r={lora_config.lora_r}")
        peft_config = create_peft_config(lora_config)
        student_model = get_peft_model(student_model, peft_config)
        student_model.print_trainable_parameters()
    elif trainable_layers is not None:
        freeze_model_layers(student_model, trainable_layers)
    
    logger.info("Student model successfully adapted to teacher tokenizer")
    return student_model, teacher_tokenizer


def get_gkd_model_info(student_model, teacher_model, tokenizer) -> Dict[str, Any]:
    """Get comprehensive information for both student and teacher models."""
    student_info = get_model_info(student_model, tokenizer)
    teacher_info = get_model_info(teacher_model, tokenizer)
    
    return {
        "student": student_info,
        "teacher": teacher_info,
        "compression_ratio": teacher_info["num_parameters"] / student_info["num_parameters"],
    }