"""Configuration classes and default settings."""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass
class LoRAConfig:
    """LoRA configuration for parameter-efficient fine-tuning."""
    use_lora: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False
    use_rslora: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str = ""
    device: str = "cuda"
    trust_remote_code: bool = True
    device_map: str = "auto"
    trainable_layers: Optional[Union[List[int], str]] = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @property
    def torch_dtype(self):
        return torch.bfloat16


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = "nvidia/OpenMathInstruct-2:train:1.0"  # format: "dataset:subset:weight,dataset2:subset2:weight2"
    test_size: float = 0.0001
    seed: int = 42
    max_length: Optional[int] = None
    total_samples: Optional[int] = 21e6


@dataclass
class TrainingConfig:
    """Training configuration."""
    resume_from_checkpoint: Optional[str] = None
    output_dir: str = "" 
    num_train_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    gradient_checkpointing: bool = False
    push_to_hub: bool = False
    report_to: str = "wandb"
    max_grad_norm: float = 0.1
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "constant_with_warmup"
    lr_decay_from_step: Optional[int] = None
    lr_decay_steps: int = 30000
    dataloader_num_workers: int = 4
    use_liger_kernel: bool = False
    packing: bool = True
    completion_only_loss: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_enabled: bool = True
    source_model_path: str = ""
    eval_gpu: int = 7
    eval_max_samples: int = 50
    eval_max_workers: int = 1
    eval_tasks: str = "leaderboard|arc:challenge|3|1,leaderboard|hellaswag|10|1,helm|mmlu|5|1,leaderboard|gsm8k|8|1,leaderboard|truthfulqa:mc|0|1"


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    project: str = "sft-Aug2025"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"

    def setup_environment(self):
        os.environ["HF_HOME"] = self.hf_home