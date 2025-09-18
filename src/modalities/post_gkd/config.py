"""Configuration classes for GKD training."""
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
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False
    use_rslora: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    model_path: str = "/raid/s3/opengptx/mfrey/instruct/hf_model"
    teacher_model_path: str = ""
    device: str = "cuda"
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:0"
    trust_remote_code: bool = True
    device_map: str = "auto"
    trainable_layers: Optional[Union[List[int], str]] = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    vocab_alignment_method: str = "teacher_tokenizer"  # "teacher_tokenizer", "error"

    @property
    def torch_dtype(self):
        return torch.bfloat16


@dataclass
class GKDConfig:
    """GKD-specific configuration."""

    temperature: float = 0.9
    lmbda: float = 1.0  # Student data fraction (1.0 = pure on-policy)
    beta: float = 0.5   # JSD interpolation (0.0=forward KL, 1.0=reverse KL)
    max_new_tokens: int = 128
    disable_dropout: bool = True
    seq_kd: bool = False  # Sequence-level KD
    log_completions: bool = True 
    num_completions_to_print: int = 5 

@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset: str = "nvidia/OpenMathInstruct-2:train_1M:1.0"
    test_size: float = 0.0001
    seed: int = 42
    total_samples: Optional[int] = 21000  # Reduced for GKD experiments


@dataclass
class TrainingConfig:
    """Training configuration."""

    resume_from_checkpoint: Optional[str] = None
    output_dir: str = ""
    num_train_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Increased for GKD
    learning_rate: float = 1e-5  # Lower LR for distillation
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    max_steps: int = -1
    gradient_checkpointing: bool = False
    push_to_hub: bool = False
    report_to: str = "wandb"
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = 4
    use_liger_kernel: bool = False
    max_length: int = 1024


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    eval_enabled: bool = True
    source_model_path: str = ""
    eval_gpu: int = 7
    eval_max_samples: int = 50
    eval_max_workers: int = 1
    eval_tasks: str = "leaderboard|arc:challenge|3|1,leaderboard|hellaswag|10|1,helm|mmlu|5|1,leaderboard|gsm8k|8|1"


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "gkd-distillation-Jan2025"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["gkd", "distillation"])
    notes: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    gkd: GKDConfig = field(default_factory=GKDConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"

    def setup_environment(self):
        os.environ["HF_HOME"] = self.hf_home