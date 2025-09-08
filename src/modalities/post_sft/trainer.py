"""Custom trainer implementation with evaluation callbacks."""
import logging
import math
import os

from evaluation import AsyncEvaluator
from model_utils import save_model_with_custom_code
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from config import EvaluationConfig, TrainingConfig

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
        logger.info("🔍 Running initial evaluation on base model at step 0...")
        self.evaluator.submit_evaluation(self.source_model_path, step=0)

    def on_save(self, args, state, control, **kwargs):
        """Trigger evaluation when checkpoint is saved."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path):
            logger.info(f"💾 Checkpoint saved at step {state.global_step}, triggering evaluation...")
            self.evaluator.submit_evaluation(checkpoint_path, state.global_step)
        else:
            logger.warning(f"⚠️ Checkpoint path {checkpoint_path} does not exist, skipping evaluation")

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all evaluations to complete."""
        self.evaluator.wait_for_completion()


class CustomSFTTrainer(SFTTrainer):
    """Custom SFT Trainer with model saving enhancements."""

    def __init__(self, source_model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_model_path = source_model_path

    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with custom code files."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        save_model_with_custom_code(self.model, output_dir, self.source_model_path)
        logger.info(f"Model saved with custom code to: {output_dir}")

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Override scheduler creation for custom LR decay from checkpoint."""
        if optimizer is None:
            optimizer = self.optimizer

        # Check if custom decay is configured
        if hasattr(self.args, "lr_decay_from_step") and self.args.lr_decay_from_step is not None:
            decay_start = self.args.lr_decay_from_step
            decay_steps = self.args.lr_decay_steps

            def lr_lambda(current_step):
                if current_step < decay_start:
                    return 1.0  # Full LR before decay

                # Cosine decay to 10% of original
                progress = min((current_step - decay_start) / decay_steps, 1.0)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return 0.1 + 0.9 * cosine_factor  # From 100% to 10%

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            logger.info(f"✅ Custom LR: cosine decay from step {decay_start} " f"over {decay_steps} steps (100% → 10%)")
            return self.lr_scheduler

        # Default scheduler
        return super().create_scheduler(num_training_steps, optimizer)


def create_sft_config(training_config: TrainingConfig, eval_config) -> SFTConfig:
    """Create SFTConfig from TrainingConfig."""
    print(training_config)
    config = SFTConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size * training_config.gradient_accumulation_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy="steps" if eval_config.eval_enabled else "no",
        save_total_limit=training_config.save_total_limit,
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
        packing=training_config.packing,
        completion_only_loss=training_config.completion_only_loss,
        bf16=True,
        fp16=False,
        torch_compile=True,
    )

    config.lr_decay_from_step = training_config.lr_decay_from_step
    config.lr_decay_steps = training_config.lr_decay_steps

    return config


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    source_model_path: str,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig = None,
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
) -> CustomSFTTrainer:
    """Set up the custom SFT trainer with all configurations."""

    # Create SFT config
    sft_config = create_sft_config(training_config, eval_config)

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
        logger.info("✅ Evaluation callback enabled")
    else:
        logger.info("⚠️ Evaluation callback disabled")

    # Create trainer
    trainer = CustomSFTTrainer(
        source_model_path=source_model_path,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Trainer setup completed")
    return trainer
