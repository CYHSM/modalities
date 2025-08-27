"""Configuration classes for GRPO training."""
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration for memory-efficient training."""

    use_lora: bool = False
    use_qlora: bool = False  # 4-bit quantization
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str
    torch_dtype: str = "bfloat16"  # Options: float16, bfloat16, float32
    load_in_4bit: bool = False  # For QLoRA
    load_in_8bit: bool = False  # Alternative quantization
    gradient_checkpointing: bool = True  # Save memory
    device_map: str = "auto"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def get_torch_dtype(self):
        """Convert string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_size: Optional[int] = None  # Limit training samples (for testing)
    eval_size: int = 500  # Number of test samples for evaluation
    seed: int = 42


@dataclass
class TrainingConfig:
    """GRPO training configuration."""

    output_dir: str
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # GRPO specific
    num_generations: int = 4  # Number of completions per prompt
    max_completion_length: int = 256
    temperature: float = 0.7
    beta: float = 0.0  # KL penalty (0 = no penalty)

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    optim: str = "paged_adamw_8bit"  # Memory efficient optimizer


@dataclass
class Config:
    """Main configuration."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
