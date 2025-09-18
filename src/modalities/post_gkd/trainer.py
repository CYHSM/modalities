"""Custom GKD trainer implementation with evaluation callbacks."""
import logging
import os
from pathlib import Path

from modalities.post_gkd.evaluation import AsyncEvaluator
from modalities.post_gkd.model_utils import save_model_with_custom_code
from transformers import TrainerCallback
from trl import GKDConfig, GKDTrainer

from modalities.post_gkd.config import EvaluationConfig, TrainingConfig, GKDConfig as CustomGKDConfig

logger = logging.getLogger(__name__)


class EvalCallback(TrainerCallback):
    """Callback to trigger async LightEval CLI on checkpoint saves and at training start."""

    def __init__(self, eval_config: EvaluationConfig, source_model_path: str, hf_home: str):
        self.eval_config = eval_config
        self.source_model_path = source_model_path
        self.hf_home = hf_home
        self.evaluator = AsyncEvaluator(eval_config, hf_home)

    def on_train_begin(self, args, state, control, **kwargs):
        """Run initial evaluation on the base model at step 0."""
        logger.info("Running initial evaluation on base student model at step 0...")
        self.evaluator.submit_evaluation(self.source_model_path, step=0)

    def on_save(self, args, state, control, **kwargs):
        """Trigger evaluation when checkpoint is saved."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint saved at step {state.global_step}, triggering evaluation...")
            self.evaluator.submit_evaluation(checkpoint_path, state.global_step)
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist, skipping evaluation")

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all evaluations to complete."""
        self.evaluator.wait_for_completion()


class CustomGKDTrainer(GKDTrainer):
    """Custom GKD Trainer with model saving enhancements."""

    def __init__(self, source_model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_model_path = source_model_path

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with custom code files."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        save_model_with_custom_code(self.model, output_dir, self.source_model_path)
        logger.info(f"GKD model saved with custom code to: {output_dir}")


def create_gkd_config(
    training_config: TrainingConfig, 
    gkd_config: CustomGKDConfig, 
    eval_config: EvaluationConfig
) -> GKDConfig:
    """Create GKDConfig from our configuration classes."""
    print("GKD Config:")
    print(gkd_config)
    print("Training Config:")
    print(training_config)
    print("Evaluation Config:")
    print(eval_config)
    config = GKDConfig(
        # Basic training args
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy="steps" if eval_config.eval_enabled else "no",
        save_total_limit=training_config.save_total_limit,
        save_only_model=training_config.save_total_limit > 10,
        max_steps=training_config.max_steps,
        load_best_model_at_end=True if eval_config.eval_enabled else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=training_config.report_to,
        gradient_checkpointing=training_config.gradient_checkpointing,
        push_to_hub=training_config.push_to_hub,
        max_grad_norm=training_config.max_grad_norm,
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        dataloader_num_workers=training_config.dataloader_num_workers,
        use_liger_kernel=training_config.use_liger_kernel,
        max_length=training_config.max_length,

        # Log
        log_completions=gkd_config.log_completions,
        num_completions_to_print=gkd_config.num_completions_to_print,
        
        # GKD-specific parameters
        temperature=gkd_config.temperature,
        lmbda=gkd_config.lmbda,
        beta=gkd_config.beta,
        max_new_tokens=gkd_config.max_new_tokens,
        disable_dropout=gkd_config.disable_dropout,
        seq_kd=gkd_config.seq_kd,
        
        # Optimization settings
        bf16=True,
        fp16=False,
        remove_unused_columns=False,  # Important for GKD        
    )

    logger.info(f"GKD Config created:")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Lambda (student data fraction): {config.lmbda}")
    logger.info(f"  Beta (JSD interpolation): {config.beta}")
    logger.info(f"  Max new tokens: {config.max_new_tokens}")
    logger.info(f"  Sequence KD: {config.seq_kd}")

    return config


def setup_gkd_trainer(
    student_model,
    teacher_model,
    tokenizer,
    train_dataset,
    eval_dataset,
    source_model_path: str,
    training_config: TrainingConfig,
    gkd_config: CustomGKDConfig,
    eval_config: EvaluationConfig = None,
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
) -> CustomGKDTrainer:
    """Set up the custom GKD trainer with all configurations."""

    # Create GKD config
    config = create_gkd_config(training_config, gkd_config, eval_config)

    # Create callbacks
    callbacks = []
    if eval_config and eval_config.eval_enabled:
        callbacks.append(
            EvalCallback(
                eval_config,
                source_model_path
                if training_config.resume_from_checkpoint is None
                else training_config.resume_from_checkpoint,
                hf_home,
            )
        )
        logger.info("Evaluation callback enabled for GKD")
    else:
        logger.info("Evaluation callback disabled")

    # Create GKD trainer
    trainer = CustomGKDTrainer(
        source_model_path=source_model_path,
        model=student_model,
        teacher_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("GKD Trainer setup completed")
    logger.info(f"Student model: {student_model.__class__.__name__}")
    logger.info(f"Teacher model: {teacher_model.__class__.__name__}")
    
    return trainer