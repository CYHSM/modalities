"""GSM8K dataset handling for GRPO."""
import logging
import re
from typing import Any, Dict, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


def extract_answer(text: str) -> str:
    """Extract numerical answer from GSM8K format."""
    # GSM8K answers are after ####
    match = re.search(r"####\s*(\S+)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def format_openmathinstruct2_grpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format OpenMathInstruct-2 dataset for GRPO training."""
    instruction = (
        "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."  # noqa: E501
    )
    formatted_problem = f"{instruction}\nQuestion:{example['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}\nAnswer:"  # noqa: E501

    return {
        "prompt": formatted_problem,
        "expected_answer": example.get("expected_answer", ""),
    }


def format_gsm8k_for_grpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format GSM8K for GRPO training."""
    
    instruction = (
        "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."  # noqa: E501
    )
    formatted_problem = f"{instruction}\nQuestion:{example['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}\nAnswer:"  # noqa: E501
    expected_answer = extract_answer(example["answer"])

    return {
        "prompt": formatted_problem,
        "expected_answer": expected_answer,
    }

def load_openmathinstruct2_dataset(train_size: Optional[int] = None, eval_size: int = 500, seed: int = 42):
    logger.info("Loading OpenMathInstruct-2 dataset...")

    # Load dataset
    dataset_gsm8k = load_dataset("openai/gsm8k", "main")
    dataset_omi2 = load_dataset("nvidia/OpenMathInstruct-2", "default")

    # Prepare train split
    train_ds = dataset_omi2["train_1M"]
    if train_size:
        train_ds = train_ds.select(range(min(train_size, len(train_ds))))

    # Prepare eval split (from test)
    eval_ds = dataset_gsm8k["test"].select(range(min(eval_size, len(dataset_gsm8k["test"]))))

    # Format for GRPO
    train_ds = train_ds.map(format_openmathinstruct2_grpo, remove_columns=train_ds.column_names, load_from_cache_file=False)
    eval_ds = eval_ds.map(format_gsm8k_for_grpo, remove_columns=eval_ds.column_names, load_from_cache_file=False)

    logger.info(f"Dataset loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    return {"train": train_ds, "test": eval_ds}


def load_gsm8k_dataset(train_size: Optional[int] = None, eval_size: int = 500, seed: int = 42):
    """Load and prepare GSM8K dataset for GRPO."""
    logger.info("Loading GSM8K dataset...")

    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Prepare train split
    train_ds = dataset["train"]
    if train_size:
        train_ds = train_ds.select(range(min(train_size, len(train_ds))))

    # Prepare eval split (from test)
    eval_ds = dataset["test"].select(range(min(eval_size, len(dataset["test"]))))

    # Format for GRPO
    train_ds = train_ds.map(format_gsm8k_for_grpo, remove_columns=train_ds.column_names, load_from_cache_file=False)
    eval_ds = eval_ds.map(format_gsm8k_for_grpo, remove_columns=eval_ds.column_names, load_from_cache_file=False)

    logger.info(f"Dataset loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    return {"train": train_ds, "test": eval_ds}
