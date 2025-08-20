"""Configuration classes and default settings."""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"

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

    dataset_name: str = "meta-math/MetaMathQA"
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
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    push_to_hub: bool = False
    report_to: str = "wandb"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    gpu_id: int = 7
    max_samples: int = 500
    max_workers: int = 2
    tasks: str = "leaderboard|hellaswag|10|1,leaderboard|gsm8k|8|1"
    batch_size: int = 16
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
