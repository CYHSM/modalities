"""GRPO trainer implementation with separate GPU for inference."""
import logging
import re
from fractions import Fraction
from typing import Dict, List, Union

from evaluation import AsyncEvaluator
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from config import EvaluationConfig, TrainingConfig

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else ""


def normalize_math_expression(expr: str) -> str:
    """Normalize mathematical expressions for comparison."""
    if not expr:
        return ""

    # Remove whitespace and convert to lowercase
    expr = expr.replace(" ", "").lower()

    # Handle common mathematical notation
    replacements = {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\frac": "",
        "{": "",
        "}": "",
        "\\": "",
        "$": "",
        ",": "",  # Remove dollar signs and commas
    }

    for old, new in replacements.items():
        expr = expr.replace(old, new)

    return expr


def try_parse_number(text: str) -> Union[float, None]:
    """Try to parse text as a number or fraction."""
    if not text:
        return None

    text = normalize_math_expression(text)

    # Try direct float conversion
    try:
        return float(text)
    except ValueError:
        pass

    # Try fraction conversion
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass

    # Try parsing fractions in different formats
    fraction_patterns = [r"^(\d+)/(\d+)$", r"^(\d+)over(\d+)$", r"^(\d+)÷(\d+)$"]

    for pattern in fraction_patterns:
        match = re.match(pattern, text)
        if match:
            try:
                num, den = float(match.group(1)), float(match.group(2))
                if den != 0:
                    return num / den
            except ValueError:
                continue

    return None


def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Binary reward for proper formatting (\\boxed{} answer).
    Returns 1.0 if properly formatted, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        if extract_boxed_answer(completion):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def correctness_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """
    Binary reward for correctness.
    Returns 1.0 if answer is correct, 0.0 otherwise.
    Uses exact matching after normalization.
    """
    if len(expected_answer) == 1:
        expected_answer = expected_answer * len(completions)

    rewards = []
    for completion, expected in zip(completions, expected_answer):
        extracted = extract_boxed_answer(completion)

        if not extracted or not expected:
            rewards.append(0.0)
            continue

        # Try exact string match after normalization
        if normalize_math_expression(extracted) == normalize_math_expression(expected):
            rewards.append(1.0)
            continue

        # Try numerical comparison with tight tolerance
        extracted_num = try_parse_number(extracted)
        expected_num = try_parse_number(expected)

        if extracted_num is not None and expected_num is not None:
            # Very tight tolerance for "correct" - must be nearly exact
            if abs(extracted_num - expected_num) < 1e-8:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def reasoning_quality_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Binary reward for reasoning quality.
    Returns 1.0 if shows clear step-by-step reasoning, 0.0 otherwise.
    """
    rewards = []

    for completion in completions:
        score = 0.0
        text_lower = completion.lower()

        # Check for step indicators
        step_indicators = ["step", "first", "next", "then", "therefore", "thus", "hence", "so"]
        has_steps = sum(1 for indicator in step_indicators if indicator in text_lower) >= 2

        # Check for mathematical operations
        math_operations = ["=", "+", "-", "*", "/", "substitute", "solve", "calculate"]
        has_math = sum(1 for op in math_operations if op in completion) >= 3

        # Check for explanatory text (not just equations)
        words = text_lower.split()
        explanatory_words = ["because", "since", "we", "need", "can", "will", "this", "the"]
        has_explanation = sum(1 for word in explanatory_words if word in words) >= 5

        # Must have all three components for full reasoning credit
        if has_steps and has_math and has_explanation:
            score = 1.0

        rewards.append(score)

    return rewards


def length_penalty_reward_func(completions: List[str], target_length: int = 200, **kwargs) -> List[float]:
    """
    Length penalty to discourage extremely long responses.
    Returns 1.0 for reasonable length, penalty for excessive length.
    """
    rewards = []

    for completion in completions:
        length = len(completion.split())

        if length <= target_length:
            rewards.append(1.0)
        elif length <= target_length * 2:
            # Linear penalty for moderate excess
            penalty = 1.0 - (length - target_length) / target_length
            rewards.append(max(0.0, penalty))
        else:
            # Heavy penalty for very long responses
            rewards.append(0.0)

    return rewards


def combined_math_reward_func(
    completions: List[str], expected_answer: List[str], weights: Dict[str, float] = None, **kwargs
) -> List[float]:
    """
    Combined reward function with multiple binary components.

    Args:
        completions: List of model completions
        expected_answer: List of expected answers
        weights: Dictionary of component weights

    Returns:
        List of combined rewards
    """
    if weights is None:
        weights = {
            "correctness": 0.6,  # Most important: getting the right answer
            "format": 0.2,  # Important: proper format
            "reasoning": 0.15,  # Good: showing reasoning
            "length_penalty": 0.05,  # Small: don't be too verbose
        }

    # Get individual reward components
    correctness_rewards = correctness_reward_func(completions, expected_answer, **kwargs)
    format_rewards = format_reward_func(completions, **kwargs)
    reasoning_rewards = reasoning_quality_reward_func(completions, **kwargs)
    length_rewards = length_penalty_reward_func(completions, **kwargs)

    # Combine rewards
    combined_rewards = []
    for i in range(len(completions)):
        combined_reward = (
            weights["correctness"] * correctness_rewards[i]
            + weights["format"] * format_rewards[i]
            + weights["reasoning"] * reasoning_rewards[i]
            + weights["length_penalty"] * length_rewards[i]
        )
        combined_rewards.append(combined_reward)

    # Log statistics for first few examples
    if len(completions) > 0:
        logger.info("Reward breakdown for first completion:")
        logger.info(f"  Correctness: {correctness_rewards[0]:.2f}")
        logger.info(f"  Format: {format_rewards[0]:.2f}")
        logger.info(f"  Reasoning: {reasoning_rewards[0]:.2f}")
        logger.info(f"  Length: {length_rewards[0]:.2f}")
        logger.info(f"  Combined: {combined_rewards[0]:.2f}")

    return combined_rewards


# Alternative: Pure binary reward function (most effective according to recent research)
def binary_math_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """
    Pure binary reward function - 1.0 for correct and properly formatted, 0.0 otherwise.
    Based on recent research showing binary rewards work best for GRPO.
    """
    if len(expected_answer) == 1:
        expected_answer = expected_answer * len(completions)

    rewards = []
    correct_count = 0

    for completion, expected in zip(completions, expected_answer):
        reward = 0.0
        extracted = extract_boxed_answer(completion)

        if extracted and expected:
            # Check exact match after normalization
            if normalize_math_expression(extracted) == normalize_math_expression(expected):
                reward = 1.0
                correct_count += 1
            else:
                # Check numerical match with tight tolerance
                extracted_num = try_parse_number(extracted)
                expected_num = try_parse_number(expected)

                if extracted_num is not None and expected_num is not None and abs(extracted_num - expected_num) < 1e-8:
                    reward = 1.0
                    correct_count += 1

        rewards.append(reward)

    # Log success rate
    success_rate = correct_count / len(completions) if completions else 0
    logger.debug(f"Batch success rate: {success_rate:.2%} ({correct_count}/{len(completions)})")

    return rewards


def math_length_reward(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """
    Reward function similar to TRL's len_reward but for math problems.
    - Correct answers get higher reward for being shorter
    - Incorrect answers get penalized more for being longer
    """
    rewards = []
    lengths = [len(comp.split()) for comp in completions]  # Word count
    min_len = min(lengths) if lengths else 1
    max_len = max(lengths) if lengths else 1

    if len(expected_answer) == 1:
        expected_answer = expected_answer * len(completions)

    for completion, expected, length in zip(completions, expected_answer, lengths):
        # Check correctness
        extracted = extract_boxed_answer(completion)
        is_correct = False
        if extracted and expected:
            extracted_norm = extracted.replace(" ", "").lower().replace(",", "")
            expected_norm = expected.replace(" ", "").lower().replace(",", "")
            is_correct = extracted_norm == expected_norm

        # Length-based reward calculation
        if max_len == min_len:
            # All same length, just use correctness
            reward = 1.0 if is_correct else 0.0
        else:
            # Length penalty/bonus
            length_factor = 0.5 - (length - min_len) / (max_len - min_len)

            if is_correct:
                # Correct answers: reward decreases with length (0.5 to -0.5 range)
                # But ensure positive reward for correct answers
                reward = max(0.1, length_factor + 0.5)  # Range: 0.1 to 1.0
            else:
                # Incorrect answers: more penalty for longer responses
                reward = min(0.0, length_factor)  # Range: -0.5 to 0.0

        rewards.append(float(reward))

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
        # self.evaluator.submit_evaluation(self.source_model_path, step=0)

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

        # if output_dir is None:
        #     output_dir = self.args.output_dir

        # save_model_with_custom_code(self.model, output_dir, self.source_model_path)
        # logger.info(f"Model saved with custom code to: {output_dir}")


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
        max_grad_norm=1.0,
        weight_decay=0.1,
        num_generations=num_generations,
        max_completion_length=512,
        temperature=0.7,
        beta=0.0,  # No KL penalty by default (following recent practices)
        scale_rewards=False,
        log_completions=True,
        num_completions_to_print=1,
        # vLLM settings
        use_vllm=use_vllm,
        vllm_mode="server" if use_vllm else None,
        vllm_server_host=vllm_server_host or "localhost",
        vllm_server_port=vllm_server_port,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
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
        reward_funcs=binary_math_reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("GRPO trainer setup completed")
    return trainer
