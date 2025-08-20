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
    """Format MetaMath dataset to chat format."""
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def format_dataset(dataset_name: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset based on its name/type."""
    formatters = {
        "tatsu-lab/alpaca": format_alpaca,
        "meta-math/MetaMathQA": format_metamath,
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

    ds = ds.map(format_fn, remove_columns=ds.column_names)

    # Filter by length if specified
    if max_length:

        def filter_length(example):
            total_length = sum(len(msg["content"]) for msg in example["messages"])
            return total_length <= max_length

        original_size = len(ds)
        ds = ds.filter(filter_length)
        logger.info(f"Filtered dataset from {original_size} to {len(ds)} samples")

    # Split dataset
    if test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    else:
        ds = {"train": ds, "test": None}
        logger.info(f"Using full dataset for training: {len(ds['train'])} samples")

    return ds
