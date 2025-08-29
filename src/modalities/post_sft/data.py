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


def format_infinity_instruct(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format Infinity-Instruct dataset to chat format."""
    messages = []
    
    for turn in example["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    
    return {"messages": messages}


def format_dataset(dataset_name: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """Format dataset based on its name/type."""
    formatters = {
        "tatsu-lab/alpaca": format_alpaca,
        "meta-math/MetaMathQA": format_metamath,
        "nvidia/OpenMathInstruct-2": format_openmathinstruct2,
        "allenai/tulu-3-sft-mixture": format_tulu3_sft,
        "BAAI/Infinity-Instruct": format_infinity_instruct,
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
    ds = ds.map(format_fn)

    # Split dataset
    if test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    else:
        ds = {"train": ds, "test": None}
        logger.info(f"Using full dataset for training: {len(ds['train'])} samples")
    return ds