"""Dataset loading and formatting utilities."""
import logging
from typing import Any, Dict

from datasets import load_dataset

logger = logging.getLogger(__name__)


def format_alpaca(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Alpaca format to chat format."""
    if example.get("input"):
        content = f"{example['instruction']}\n\nInput: {example['input']}"
    else:
        content = example["instruction"]

    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": example["output"]}]}


def format_metamath(example: Dict[str, Any]) -> Dict[str, Any]:
    if "word problem" in example["query"].lower() or any(
        word in example["query"].lower() for word in ["cost", "price", "buy", "sell", "money"]
    ):
        instruction = "Solve this word problem step by step. Show your work clearly and put the final numerical answer in \\boxed{}."  # noqa: E501
    else:
        instruction = (
            "Solve this mathematical problem. Show each step of your solution and put the final answer in \\boxed{}."
        )

    formatted_query = f"{instruction}\n\n{example['query']}"

    return {
        "messages": [
            {"role": "user", "content": formatted_query},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def format_openmathinstruct2(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format OpenMathInstruct-2 dataset to chat format with instruction template."""
    instruction = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
    formatted_problem = f"{instruction}\n\n{example['problem']}"

    return {
        "messages": [
            {"role": "user", "content": formatted_problem},
            {"role": "assistant", "content": example["generated_solution"]},
        ]
    }


def format_tulu3_sft(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format Tulu 3 SFT dataset to chat format."""
    return example


def format_dataset(dataset_name: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset based on its name/type."""
    formatters = {
        "tatsu-lab/alpaca": format_alpaca,
        "meta-math/MetaMathQA": format_metamath,
        "nvidia/OpenMathInstruct-2": format_openmathinstruct2,
        "allenai/tulu-3-sft-mixture": format_tulu3_sft,
    }

    formatter = formatters.get(dataset_name)
    if formatter is None:
        # Default formatting for unknown datasets
        logger.warning(f"No specific formatter for {dataset_name}, using default")
        return example

    return formatter(example)


def load_and_format_dataset(
    dataset_name: str, dataset_split: str = "train", test_size: float = 0.01, seed: int = 42, max_length: int = None
):
    """Load and format dataset for training."""
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    ds = load_dataset(dataset_name, split=dataset_split)

    # Format dataset
    def format_fn(example):
        return format_dataset(dataset_name, example)

    # Change this line:
    if dataset_name == "allenai/tulu-3-sft-mixture":
        ds = ds.map(format_fn)
    else:
        ds = ds.map(format_fn, remove_columns=ds.column_names)

    # Split dataset
    if test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    else:
        ds = {"train": ds, "test": None}
        logger.info(f"Using full dataset for training: {len(ds['train'])} samples")
    return ds


def format_openmathinstruct2_grpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format OpenMathInstruct-2 dataset for GRPO training."""
    instruction = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
    formatted_problem = f"{instruction}\nProblem:{example['problem']}\nSolution:"

    return {
        "prompt": formatted_problem,
        "expected_answer": example.get("expected_answer", ""),
        # Keep original solution for reference if needed
        "reference_solution": example.get("generated_solution", ""),
    }


def load_grpo_dataset(
    dataset_name: str, dataset_split: str = "train", test_size: float = 0.01, seed: int = 42, max_samples: int = None
):
    """Load and format dataset for GRPO training."""
    logger.info(f"Loading dataset for GRPO: {dataset_name}")

    # Load dataset
    ds = load_dataset(dataset_name, split=dataset_split)

    # Limit samples if specified (useful for testing)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Format for GRPO - only keep prompt and expected_answer
    if dataset_name == "nvidia/OpenMathInstruct-2":
        ds = ds.map(format_openmathinstruct2_grpo)
        # Remove columns not needed for GRPO
        columns_to_keep = ["prompt", "expected_answer"]
        columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
        ds = ds.remove_columns(columns_to_remove)
    else:
        raise ValueError(f"GRPO formatting not implemented for {dataset_name}")

    # Split dataset
    if test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    else:
        ds = {"train": ds, "test": None}
        logger.info(f"Using full dataset for training: {len(ds['train'])} samples")

    return ds
