"""Configuration classes and default settings."""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass
class LoRAConfig:
    """LoRA configuration for parameter-efficient fine-tuning."""

    # Enable LoRA
    use_lora: bool = False

    # LoRA rank - higher for better performance (64-128 for full-FT comparable results)
    lora_r: int = 128

    # LoRA alpha - typically 2x rank for aggressive adaptation
    lora_alpha: int = 256

    # LoRA dropout for regularization
    lora_dropout: float = 0.05

    # Target modules - targeting all linear layers for maximum adaptation
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention layers
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP layers
            "lm_head",  # Output head
        ]
    )

    # LoRA bias - can help with performance
    bias: str = "none"  # Options: "none", "all", "lora_only"

    # Task type
    task_type: str = "CAUSAL_LM"

    # Use DoRA (Weight-Decomposed Low-Rank Adaptation) - can improve performance
    use_dora: bool = False

    # Use RSLoRA (Rank-Stabilized LoRA) - helps with higher ranks
    use_rslora: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"
    trainable_layers: Optional[Union[List[int], str]] = None
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

    dataset_name: str = "nvidia/OpenMathInstruct-2"
    dataset_split: str = "train"
    test_size: float = 0.01
    seed: int = 42
    max_length: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = True
    push_to_hub: bool = False
    report_to: str = "wandb"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    source_model_path: str = "/raid/s3/opengptx/mfrey/instruct/checkpoints/checkpoint-70000"
    gpu_id: int = 7
    max_samples: int = 100
    max_workers: int = 1
    tasks: str = "leaderboard|hellaswag|10|1,leaderboard|gsm8k|8|1"
    timeout: int = 3600


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "trl-training"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    wandb: WandBConfig
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"

    def setup_environment(self):
        os.environ["HF_HOME"] = self.hf_home
