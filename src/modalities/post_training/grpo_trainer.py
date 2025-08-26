"""GRPO trainer implementation with separate GPU for inference."""
import logging
import re
from fractions import Fraction
from typing import List, Union

from evaluation import AsyncEvaluator
from model_utils import save_model_with_custom_code
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
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\frac", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("\\", "")

    return expr


def try_parse_number(text: str) -> Union[float, Fraction, None]:
    """Try to parse text as a number, fraction, or mathematical expression."""
    if not text:
        return None

    # Clean up the text
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
    fraction_patterns = [
        r"^(\d+)/(\d+)$",  # Simple fraction like 3/4
        r"^(\d+)\\div(\d+)$",  # Division notation
        r"^(\d+)\s*over\s*(\d+)$",  # "over" notation
    ]

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


def calculate_numerical_similarity(val1: float, val2: float, tolerance: float = 1e-6) -> float:
    """Calculate similarity between two numerical values."""
    if abs(val1 - val2) <= tolerance:
        return 1.0

    # Use relative error for large numbers, absolute error for small numbers
    if max(abs(val1), abs(val2)) > 1:
        relative_error = abs(val1 - val2) / max(abs(val1), abs(val2))
        # Give partial credit based on how close we are
        if relative_error < 0.01:  # Within 1%
            return 0.8
        elif relative_error < 0.05:  # Within 5%
            return 0.6
        elif relative_error < 0.1:  # Within 10%
            return 0.4
        elif relative_error < 0.2:  # Within 20%
            return 0.2
    else:
        absolute_error = abs(val1 - val2)
        if absolute_error < 0.01:
            return 0.8
        elif absolute_error < 0.05:
            return 0.6
        elif absolute_error < 0.1:
            return 0.4
        elif absolute_error < 0.2:
            return 0.2

    return 0.0


def has_mathematical_reasoning(text: str) -> float:
    """Check if the text shows mathematical reasoning and give partial credit."""
    reasoning_indicators = [
        "therefore",
        "thus",
        "hence",
        "because",
        "since",
        "so",
        "step",
        "first",
        "next",
        "then",
        "finally",
        "calculate",
        "compute",
        "solve",
        "find",
        "substitute",
        "simplify",
        "factor",
        "equation",
        "formula",
        "method",
    ]

    math_symbols = ["=", "+", "-", "*", "/", "^", "(", ")", "sqrt", "log"]

    text_lower = text.lower()
    reasoning_score = sum(1 for indicator in reasoning_indicators if indicator in text_lower)
    math_score = sum(1 for symbol in math_symbols if symbol in text)

    # Normalize scores (cap at reasonable values)
    reasoning_score = min(reasoning_score / 3, 1.0)  # Max 1.0 for reasoning
    math_score = min(math_score / 5, 1.0)  # Max 1.0 for math symbols

    return (reasoning_score + math_score) / 2


def math_reward_func(completions: List[str], expected_answer: List[str], **kwargs) -> List[float]:
    """
    Improved reward function for math problems with partial credit.

    Reward structure:
    - 1.0: Exact match (after normalization)
    - 0.8-0.9: Very close numerically (within 1-5%)
    - 0.4-0.6: Somewhat close numerically (within 10-20%)
    - 0.1-0.3: Has boxed answer format but wrong + shows reasoning
    - 0.05-0.15: Shows good mathematical reasoning but no boxed answer
    - -0.2: Has boxed format but completely wrong
    - -0.5: No boxed answer and no mathematical reasoning
    """
    logger.debug(
        f"Reward function called with {len(completions)} completions and {len(expected_answer)} expected answers"
    )

    if len(completions) != len(expected_answer):
        logger.warning(f"Length mismatch: {len(completions)} completions vs {len(expected_answer)} expected answers")
        # Use the first expected answer for all completions as fallback
        expected_answer = [expected_answer[0]] * len(completions) if expected_answer else [""] * len(completions)

    rewards = []

    for i, (completion, expected) in enumerate(zip(completions, expected_answer)):
        reward = 0.0
        extracted = extract_boxed_answer(completion)

        # Base reward components
        has_boxed = bool(extracted)
        reasoning_score = has_mathematical_reasoning(completion)

        if has_boxed and expected:
            # Try exact string match first (after normalization)
            if normalize_math_expression(extracted) == normalize_math_expression(expected):
                reward = 1.0
            else:
                # Try numerical comparison
                extracted_num = try_parse_number(extracted)
                expected_num = try_parse_number(expected)

                if extracted_num is not None and expected_num is not None:
                    numerical_similarity = calculate_numerical_similarity(extracted_num, expected_num)
                    if numerical_similarity > 0:
                        # Boost reward if numerical match
                        reward = max(0.9 * numerical_similarity, 0.4)
                    else:
                        # Wrong answer but has format
                        reward = max(0.1 + 0.2 * reasoning_score, -0.2)
                else:
                    # Could not parse as numbers - give small credit for format + reasoning
                    reward = max(0.05 + 0.25 * reasoning_score, -0.2)

        elif has_boxed and not expected:
            # Has boxed format but no expected answer to compare
            reward = 0.3 + 0.2 * reasoning_score

        elif not has_boxed and expected:
            # No boxed format but shows reasoning
            reasoning_reward = 0.15 * reasoning_score
            # Check if the expected answer appears anywhere in the text
            if expected.lower() in completion.lower():
                reasoning_reward += 0.1
            reward = max(reasoning_reward, -0.5)

        else:
            # No boxed format and no expected answer
            reward = max(0.1 * reasoning_score, -0.5)

        # Cap rewards to reasonable range
        reward = max(-0.5, min(1.0, reward))
        rewards.append(reward)

        # Log first few examples for debugging
        if i < 3:
            logger.debug(f"Sample {i}:")
            logger.debug(f"  Extracted: '{extracted}'")
            logger.debug(f"  Expected: '{expected}'")
            logger.debug(f"  Has boxed: {has_boxed}")
            logger.debug(f"  Reasoning score: {reasoning_score:.3f}")
            logger.debug(f"  Final reward: {reward:.3f}")

    # Log statistics
    reward_mean = sum(rewards) / len(rewards)
    reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    logger.debug(
        f"Reward statistics: mean={reward_mean:.3f}, std={reward_std:.3f}, min={min(rewards):.3f}, max={max(rewards):.3f}"  # noqa E501
    )

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
        beta=0.001,  # No KL penalty by default (following recent practices)
        scale_rewards=False,
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
