"""Main GOLD training script."""
import logging
import os
import sys

import wandb
from simple_parsing import parse
from transformers import TrainerCallback
from trl.experimental.gold import GOLDConfig, GOLDTrainer

from config import Config
from data import load_gold_dataset
from evaluation import AsyncEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvalCallback(TrainerCallback):
    def __init__(self, evaluator: AsyncEvaluator, initial_model_path: str):
        self.evaluator = evaluator
        self.initial_model_path = initial_model_path
    
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("🔍 Running initial evaluation on student model at step 0...")
        self.evaluator.submit_evaluation(self.initial_model_path, step=0)
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path):
            logger.info(f"💾 Checkpoint saved at step {state.global_step}, triggering evaluation...")
            self.evaluator.submit_evaluation(checkpoint_path, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.evaluator.wait_for_completion()


def setup_wandb(config: Config):
    if config.training.report_to == "wandb":
        wandb_name = config.wandb.name
        if not wandb_name:
            student_name = config.model.student_model.split("/")[-1]
            teacher_name = config.model.teacher_model.split("/")[-1]
            wandb_name = f"{student_name}-from-{teacher_name}-lr{config.training.learning_rate}"
        
        wandb.init(
            project=config.wandb.project,
            name=wandb_name,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            config={
                "model": config.model.__dict__,
                "data": config.data.__dict__,
                "training": config.training.__dict__,
                "evaluation": config.evaluation.__dict__,
            },
        )
        logger.info(f"WandB initialized: {wandb.run.name}")


def create_gold_config(config: Config) -> GOLDConfig:
    return GOLDConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy="steps" if config.evaluation.eval_enabled else "no",
        save_total_limit=config.training.save_total_limit,
        max_steps=config.training.max_steps,
        
        temperature=config.training.temperature,
        lmbda=config.training.lmbda,
        beta=config.training.beta,
        max_length=config.training.max_length,
        max_completion_length=config.training.max_completion_length,
        
        use_uld_loss=config.training.use_uld_loss,
        uld_use_hybrid_loss=config.training.uld_use_hybrid_loss,
        uld_student_temperature=config.training.uld_student_temperature,
        uld_teacher_temperature=config.training.uld_teacher_temperature,
        teacher_tokenizer_name_or_path=config.model.teacher_model,
        
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.training.dataloader_num_workers,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        
        report_to=config.training.report_to,
        push_to_hub=config.training.push_to_hub,
        
        bf16=True,
        fp16=False,

        torch_compile=True,
    )


def main():
    config: Config = parse(Config, description="GOLD distillation training")
    
    if not config.model.student_model:
        logger.error("--model.student-model is required")
        sys.exit(1)
    if not config.model.teacher_model:
        logger.error("--model.teacher-model is required")
        sys.exit(1)
    if not config.training.output_dir:
        logger.error("--training.output-dir is required")
        sys.exit(1)
    
    config.setup_environment()
    
    logger.info(f"Student: {config.model.student_model} (GPU {config.model.student_gpu})")
    logger.info(f"Teacher: {config.model.teacher_model} (GPU {config.model.teacher_gpu})")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Output: {config.training.output_dir}")
    
    try:
        setup_wandb(config)
        
        logger.info("Loading datasets...")
        train_dataset, eval_dataset = load_gold_dataset(
            config.data.dataset,
            subset=config.data.subset,
            split=config.data.split,
            eval_ratio=config.data.eval_ratio,
            seed=config.data.seed
        )
        
        logger.info("Creating GOLD config...")
        gold_config = create_gold_config(config)
        
        callbacks = []
        if config.evaluation.eval_enabled:
            evaluator = AsyncEvaluator(
                eval_gpu=config.evaluation.eval_gpu,
                eval_tasks=config.evaluation.eval_tasks,
                eval_max_samples=config.evaluation.eval_max_samples,
                eval_max_workers=config.evaluation.eval_max_workers,
                hf_home=config.hf_home,
            )
            callbacks.append(EvalCallback(evaluator, config.model.student_model))
            logger.info("✅ Evaluation callback enabled")
        
        logger.info("Loading models...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load student model on specific GPU
        logger.info(f"Loading student model on GPU {config.model.student_gpu}...")
        student_model = AutoModelForCausalLM.from_pretrained(
            config.model.student_model,
            device_map={"": f"cuda:{config.model.student_gpu}"},
            trust_remote_code=config.model.student_trust_remote_code,
            torch_dtype="auto",
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            config.model.student_model,
            trust_remote_code=config.model.student_trust_remote_code,
        )
        
        # Setup chat format for student
        from trl import setup_chat_format
        if hasattr(student_tokenizer, "chat_template") and student_tokenizer.chat_template is not None:
            logger.info("Student tokenizer already has chat template")
        else:
            logger.info("Setting up chat format for student model")
            student_model, student_tokenizer = setup_chat_format(student_model, student_tokenizer)
        
        # Special handling for Teuken model
        if "opengpt-x/teuken-7b-instruct" in config.model.student_model.lower():
            logger.info("Detected Teuken student model, resetting chat template")
            student_tokenizer.chat_template = None
            student_model, student_tokenizer = setup_chat_format(student_model, student_tokenizer)
        
        # Load teacher model on specific GPU
        logger.info(f"Loading teacher model on GPU {config.model.teacher_gpu}...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.model.teacher_model,
            device_map={"": f"cuda:{config.model.teacher_gpu}"},
            trust_remote_code=config.model.teacher_trust_remote_code,
            torch_dtype="auto",
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.model.teacher_model,
            trust_remote_code=config.model.teacher_trust_remote_code,
        )
        
        # Setup chat format for teacher
        if hasattr(teacher_tokenizer, "chat_template") and teacher_tokenizer.chat_template is not None:
            logger.info("Teacher tokenizer already has chat template")
        else:
            logger.info("Setting up chat format for teacher model")
            teacher_model, teacher_tokenizer = setup_chat_format(teacher_model, teacher_tokenizer)
        
        logger.info("Initializing GOLD trainer...")
        trainer = GOLDTrainer(
            model=student_model,
            teacher_model=teacher_model,
            args=gold_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=student_tokenizer,
            callbacks=callbacks,
        )
        
        logger.info("🚀 Starting GOLD training...")
        trainer.train()
        
        logger.info("Saving final model...")
        trainer.save_model()
        
        logger.info("✨ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()