"""Configuration for GOLD training."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    student_model: str = ""
    teacher_model: str = ""
    student_gpu: int = 0
    teacher_gpu: int = 7
    student_trust_remote_code: bool = True
    teacher_trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset: str = "HuggingFaceTB/OpenR1-Math-220k-default-verified"
    subset: str = "all"
    split: str = "train[:1024]"
    eval_ratio: float = 0.001
    seed: int = 42


@dataclass
class GOLDTrainingConfig:
    output_dir: str = ""
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    max_steps: int = -1
    
    temperature: float = 0.9
    lmbda: float = 0.5
    beta: float = 0.5
    max_length: int = 512
    max_completion_length: int = 256
    
    use_uld_loss: bool = True
    uld_use_hybrid_loss: bool = True
    uld_student_temperature: float = 1.0
    uld_teacher_temperature: float = 1.0
    
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    
    report_to: str = "wandb"
    push_to_hub: bool = False


@dataclass
class EvaluationConfig:
    eval_enabled: bool = True
    eval_gpu: int = 6
    eval_max_samples: int = 50
    eval_max_workers: int = 1
    eval_tasks: str = "leaderboard|arc:challenge|3|1,leaderboard|hellaswag|10|1,helm|mmlu|5|1,leaderboard|gsm8k|8|1"


@dataclass
class WandBConfig:
    project: str = "gold-distillation"
    name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: GOLDTrainingConfig = field(default_factory=GOLDTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"
    
    def setup_environment(self):
        os.environ["HF_HOME"] = self.hf_home