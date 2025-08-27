"""GRPO trainer with binary reward for GSM8K."""
import logging
import re
from typing import List

from trl import GRPOConfig, GRPOTrainer

from config import TrainingConfig

logger = logging.getLogger(__name__)


def extract_answer_from_completion(text: str) -> str:
    """Extract answer from model completion."""
    # Look for #### followed by answer
    match = re.search(r"####\s*(\S+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: look for final number
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        return numbers[-1]

    return ""


def binary_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """Binary reward: 1.0 for correct answer, 0.0 otherwise."""
    if len(expected_answer) == 1:
        expected_answer = expected_answer * len(completions)

    rewards = []
    correct_count = 0

    for completion, expected in zip(completions, expected_answer):
        extracted = extract_answer_from_completion(completion)

        # Normalize and compare
        is_correct = extracted.strip() == expected.strip()
        reward = 1.0 if is_correct else 0.0

        rewards.append(reward)
        if is_correct:
            correct_count += 1

    # Log accuracy
    accuracy = correct_count / len(completions) if completions else 0
    logger.debug(f"Batch accuracy: {accuracy:.2%} ({correct_count}/{len(completions)})")

    return rewards


def create_grpo_config(training_config: TrainingConfig) -> GRPOConfig:
    """Create GRPO configuration."""
    return GRPOConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.num_generations,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        eval_strategy="steps",
        save_total_limit=training_config.save_total_limit,
        # GRPO specific
        num_generations=training_config.num_generations,
        max_completion_length=training_config.max_completion_length,
        temperature=training_config.temperature,
        beta=training_config.beta,
        # Memory optimization
        gradient_checkpointing=training_config.gradient_checkpointing,
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        optim=training_config.optim,
        # Other settings
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=2,
        dataloader_num_workers=4,
    )


def setup_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config: TrainingConfig,
) -> GRPOTrainer:
    """Setup GRPO trainer."""

    grpo_config = create_grpo_config(training_config)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=binary_reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("GRPO trainer initialized")
    return trainer
