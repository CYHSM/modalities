"""GRPO trainer with binary reward for GSM8K."""
import logging
import re
from typing import List, Optional

from trl import GRPOConfig, GRPOTrainer

from config import TrainingConfig

logger = logging.getLogger(__name__)


def extract_answer_from_completion(text: str) -> str:
    """Extract answer from model completion."""
    # Look for \boxed{} format first (matching their approach)
    import re

    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()

    return ""


def binary_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """Binary reward: 1.0 for correct answer, 0.0 otherwise."""
    if len(expected_answer) == 1:
        expected_answer = expected_answer * len(completions)

    rewards = []
    correct_count = 0

    for completion, expected in zip(completions, expected_answer):
        extracted = extract_answer_from_completion(completion)

        # Try to convert both to integers for numerical comparison
        try:
            extracted_int = int(extracted)
            expected_int = int(expected)
            is_correct = extracted_int == expected_int
        except (ValueError, TypeError):
            # Fallback to string comparison
            is_correct = extracted.strip() == expected.strip()

        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)
        if is_correct:
            correct_count += 1

    # Log accuracy
    accuracy = correct_count / len(completions) if completions else 0
    logger.debug(f"Batch accuracy: {accuracy:.2%} ({correct_count}/{len(completions)})")

    return rewards


def accuracy_reward_advanced(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """
    Advanced accuracy reward function using math_verify for robust mathematical expression evaluation.

    Args:
        completions: List of completions, each containing content
        solution: List of ground truth solutions
        **kwargs: Additional keyword arguments

    Returns:
        List of accuracy scores (1.0 for correct, 0.0 for incorrect, None for unparseable)
    """
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            try:
                accuracy_score = float(verify(gold_parsed, answer_parsed))
                rewards.append(accuracy_score)
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                rewards.append(None)
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            print("Failed to parse gold solution: ", sol)
            rewards.append(None)

    return rewards


def format_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """
    Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags.

    Args:
        completions: List of completions, each containing content
        solution: List of ground truth solutions (not used in this function)
        **kwargs: Additional keyword arguments

    Returns:
        List of format scores (1.0 for correct format, 0.0 for incorrect format)
    """
    import re

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"

    for content in contents:
        format_match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        format_score = 1.0 if format_match else 0.0
        rewards.append(format_score)

    return rewards


def tag_count_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    """
    Reward function that checks if we produce the desired number of think and answer tags.
    Gives partial credit for each correctly formatted tag.

    Args:
        completions: List of completions, each containing content
        solution: List of ground truth solutions (not used in this function)
        **kwargs: Additional keyword arguments

    Returns:
        List of tag count scores (0.0 to 1.0 based on correct tag usage)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    for content in contents:
        tag_score = count_tags(content)
        rewards.append(tag_score)

    return rewards


def accuracy_reward(completions, **kwargs):
    """Reward function that extracts answer from \\boxed{} format."""
    solutions = kwargs["expected_answer"]  # Ground truth answers
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, expected in zip(completion_contents, solutions):
        # Extract from \boxed{} format
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", content)
        if boxed_match:
            model_answer = boxed_match.group(1).strip().replace(",", "")
        else:
            # Fallback: no boxed format found
            model_answer = ""

        # Compare with expected answer
        if model_answer == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

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
        num_completions_to_print=1,
        dataloader_num_workers=4,
        # max_steps=1000,
        # Stability
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        adam_beta1=0.9,
        adam_beta2=0.99,
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
        reward_funcs=[binary_reward_func],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("GRPO trainer initialized")
    return trainer
