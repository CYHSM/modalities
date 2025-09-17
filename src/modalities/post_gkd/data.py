"""Dataset loading and formatting utilities for GKD training."""
import logging
import random
from typing import Any, Dict, List
from datasets import load_dataset, concatenate_datasets
import numpy as np

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
        instruction = "Solve this word problem step by step. Show your work clearly and put the final numerical answer in \\boxed{}."
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

def format_nemotron_sft(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format Nemotron SFT dataset to chat format."""
    return example

def format_dataset(dataset_name: str, example: Dict[str, Any], subset: str = None) -> Dict[str, Any]:
    """Format dataset based on its name/type."""
    formatters = {
        "tatsu-lab/alpaca": format_alpaca,
        "meta-math/MetaMathQA": format_metamath,
        "nvidia/OpenMathInstruct-2": format_openmathinstruct2,
        "allenai/tulu-3-sft-mixture": format_tulu3_sft,
        "BAAI/Infinity-Instruct": format_infinity_instruct,
        "nvidia/Nemotron-Pretraining-SFT-v1": format_nemotron_sft,
    }
    
    formatter = formatters.get(dataset_name)
    if formatter is None:
        logger.warning(f"No specific formatter for {dataset_name}, using default")
        return example
    return formatter(example)

def load_single_dataset(
    dataset_name: str, 
    subset: str = None, 
    split: str = "train", 
    max_samples: int = None,
    seed: int = 42
):
    """Load a single dataset with optional subset and sampling."""
    logger.info(f"Loading dataset: {dataset_name}" + (f", subset: {subset}" if subset else ""))
    
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Sampled {max_samples} examples from {dataset_name}")
    
    def format_fn(example):
        return format_dataset(dataset_name, example, subset)
    
    ds = ds.map(format_fn)
    logger.info(f"Loaded and formatted {len(ds)} examples from {dataset_name}")
    return ds

def load_datasets(
    dataset_string: str,
    total_samples: int = None,
    eval_ratio: float = 0.001,
    seed: int = 42
):    
    dataset_configs = []
    for dataset_spec in dataset_string.split(','):
        dataset_spec = dataset_spec.strip()
        parts = dataset_spec.split(':')
        
        if len(parts) != 3:
            raise ValueError(f"Invalid dataset spec: {dataset_spec}")
        
        full_name, subset, weight_str = parts
        weight = float(weight_str)
        
        dataset_configs.append({
            "name": full_name,
            "subset": subset,
            "weight": weight
        })
    
    datasets = []
    total_weight = sum(config['weight'] for config in dataset_configs)
    
    for config in dataset_configs:
        logger.info(f"Loading {config['name']} - {config['subset']}")
        
        ds = load_dataset(config['name'], split=config['subset'])
        
        def format_fn(example):
            return format_dataset(config['name'], example, config['subset'])
        
        ds = ds.map(format_fn)
        logger.info(f"Formatted dataset: {config['name']}")

        if total_samples:
            max_samples = int(total_samples * (config['weight'] / total_weight))
            if len(ds) > max_samples:
                ds = ds.shuffle(seed=seed).select(range(max_samples))
        
        datasets.append(ds)
    
    combined_dataset = concatenate_datasets(datasets)
    shuffled_dataset = combined_dataset.shuffle(seed=seed)
    
    split_dataset = shuffled_dataset.train_test_split(
        test_size=eval_ratio, 
        seed=seed
    )
    
    logger.info(f"Train: {len(split_dataset['train'])} samples")
    logger.info(f"Eval: {len(split_dataset['test'])} samples")
    
    return {
        'train': split_dataset['train'],
        'test': split_dataset['test']
    }

def display_dataset_samples(dataset, num_samples: int = 3):
    """Display sample entries from the dataset."""
    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")
    
    print(f"\n=== Sample Entries (showing first {num_samples}) ===")
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        sample = dataset[i]
        
        if 'messages' in sample:
            messages = sample['messages']
            print(f"Conversation with {len(messages)} messages:")
            for j, msg in enumerate(messages[:2]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200] + '...' if len(msg.get('content', '')) > 200 else msg.get('content', '')
                print(f"  {role}: {content}")
            if len(messages) > 2:
                print(f"  ... and {len(messages) - 2} more messages")
        else:
            for key, value in list(sample.items())[:3]:
                if isinstance(value, str):
                    value_str = value[:100] + '...' if len(value) > 100 else value
                else:
                    value_str = str(value)
                print(f"  {key}: {value_str}")