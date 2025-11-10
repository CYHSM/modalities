from datasets import load_dataset
from typing import List, Dict


def load_gsm8k(subset_size: int = None) -> List[Dict]:
    """
    Load GSM8K test set.
    
    Args:
        subset_size: If specified, only load this many problems
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    problems = []
    for idx, item in enumerate(dataset):
        if subset_size and idx >= subset_size:
            break
        
        problems.append({
            'idx': idx,
            'question': item['question'],
            'answer': item['answer'].split('####')[1].strip()
        })
    
    print(f"Loaded {len(problems)} problems from GSM8K")
    return problems
