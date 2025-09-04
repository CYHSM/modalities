"""Custom trainer implementation with evaluation callbacks."""
import logging
import os

from evaluation import AsyncEvaluator
from model_utils import save_model_with_custom_code
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from config import EvaluationConfig, TrainingConfig

logger = logging.getLogger(__name__)


class EvalCallback(TrainerCallback):
    """Callback to trigger async LightEval CLI on checkpoint saves and at training start."""

    def __init__(
        self,
        eval_config: EvaluationConfig,
        source_model_path: str,
        hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
    ):
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

        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_path):
            logger.warning(f"⚠️  Checkpoint path {checkpoint_path} does not exist, skipping evaluation")
            return

        logger.info(f"💾 Checkpoint saved at step {state.global_step}, triggering evaluation...")
        self.evaluator.submit_evaluation(checkpoint_path, state.global_step)

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


def create_sft_config(training_config: TrainingConfig) -> SFTConfig:
    """Create SFTConfig from TrainingConfig."""
    return SFTConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy="steps",
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=training_config.report_to,
        gradient_checkpointing=training_config.gradient_checkpointing,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        push_to_hub=training_config.push_to_hub,
        use_liger_kernel=True,
        packing=True,
        completion_only_loss=False,
        max_grad_norm=0.1,
        optim="paged_adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        lr_scheduler_type="cosine",
    )


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    source_model_path: str,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig,
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
) -> CustomSFTTrainer:
    """Set up the custom SFT trainer with all configurations."""

    # Create SFT config
    sft_config = create_sft_config(training_config)

    # Create callbacks
    callbacks = [EvalCallback(eval_config, source_model_path, hf_home)]

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
