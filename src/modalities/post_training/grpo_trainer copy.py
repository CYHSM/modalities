"""GRPO trainer implementation with separate GPU for inference."""
import logging
import re
from typing import List

from evaluation import AsyncEvaluator
from model_utils import save_model_with_custom_code
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from config import EvaluationConfig, TrainingConfig

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1) if match else ""


def math_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """
    Reward function for math problems.
    Returns 1.0 if the boxed answer matches expected, 0.0 if boxed but wrong, -0.5 if no boxed answer.
    """
    rewards = []
    for completion, expected in zip(completions, expected_answer):
        extracted = extract_boxed_answer(completion)
        if not extracted:
            reward = -0.5
        elif extracted == expected:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards


class GRPOEvalCallback(TrainerCallback):
    """Evaluation callback for GRPO training."""

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
        """Run initial evaluation on base model."""
        logger.info("🔍 Running initial evaluation on base model...")
        self.evaluator.submit_evaluation(self.source_model_path, step=0)

    def on_save(self, args, state, control, **kwargs):
        """Trigger evaluation when checkpoint is saved."""
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
        logger.info(f"💾 Checkpoint saved at step {state.global_step}, triggering evaluation...")
        self.evaluator.submit_evaluation(checkpoint_path, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for all evaluations to complete."""
        self.evaluator.wait_for_completion()


class CustomGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer with model saving enhancements."""

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


def create_grpo_config(
    training_config: TrainingConfig,
    num_generations: int = 4,
    use_vllm: bool = False,
    vllm_server_host: str = None,
    vllm_server_port: int = 8000,
) -> GRPOConfig:
    """Create GRPOConfig from TrainingConfig."""
    config = GRPOConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy="steps",
        save_total_limit=training_config.save_total_limit,
        report_to=training_config.report_to,
        gradient_checkpointing=training_config.gradient_checkpointing,
        bf16=training_config.bf16,
        # GRPO specific settings
        num_generations=num_generations,
        max_completion_length=512,
        temperature=0.8,
        beta=0.01,  # No KL penalty by default (following recent practices)
        scale_rewards=True,
        log_completions=True,
        num_completions_to_print=1,
        # vLLM settings
        use_vllm=use_vllm,
        vllm_mode="server" if use_vllm else None,
        vllm_server_host=vllm_server_host or "localhost",
        vllm_server_port=vllm_server_port,
    )

    if use_vllm:
        logger.info(f"Using vLLM server at {config.vllm_server_host}:{config.vllm_server_port}")

    return config


def setup_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    source_model_path: str,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig,
    num_generations: int = 4,
    use_vllm: bool = False,
    vllm_server_host: str = None,
    vllm_server_port: int = 8000,
    hf_home: str = "/raid/s3/opengptx/mfrey/huggingface",
) -> CustomGRPOTrainer:
    """Set up the GRPO trainer."""

    # Create GRPO config
    grpo_config = create_grpo_config(
        training_config,
        num_generations=num_generations,
        use_vllm=use_vllm,
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
    )

    # Create callbacks
    callbacks = [GRPOEvalCallback(eval_config, source_model_path, hf_home)]

    # Create trainer
    trainer = CustomGRPOTrainer(
        source_model_path=source_model_path,
        model=model,
        args=grpo_config,
        reward_funcs=math_reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("GRPO trainer setup completed")
    return trainer
