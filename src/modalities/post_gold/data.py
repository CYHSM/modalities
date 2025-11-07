"""Dataset loading for GOLD training."""
import logging
from typing import Any, Dict

from datasets import load_dataset

logger = logging.getLogger(__name__)


def format_openmathinstruct2(example: Dict[str, Any]) -> Dict[str, Any]:
    instruction = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
    formatted_problem = f"{instruction}\n\n{example['problem']}"
    return {
        "messages": [
            {"role": "user", "content": formatted_problem},
            {"role": "assistant", "content": example["generated_solution"]},
        ]
    }


def load_gold_dataset(dataset_name: str, subset: str = None, split: str = "train", eval_ratio: float = 0.001, seed: int = 42):
    logger.info(f"Loading dataset: {dataset_name}")
    
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    
    if "nvidia/OpenMathInstruct-2" in dataset_name:
        logger.info("Formatting OpenMathInstruct-2 dataset")
        ds = ds.map(format_openmathinstruct2)
    
    if "messages" not in ds.column_names:
        raise ValueError(f"Dataset must have 'messages' column. Found: {ds.column_names}")
    
    split_dataset = ds.train_test_split(test_size=eval_ratio, seed=seed)
    
    logger.info(f"Train: {len(split_dataset['train'])} samples")
    logger.info(f"Eval: {len(split_dataset['test'])} samples")
    
    return split_dataset["train"], split_dataset["test"]