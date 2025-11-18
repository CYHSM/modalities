from datasets import load_dataset
from typing import List, Dict


def load_gsm8k(subset_size: int = None) -> List[Dict]:
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


def load_gsm8k_fewshot(n_examples: int = 8) -> List[Dict]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    examples = []
    for idx, item in enumerate(dataset):
        if idx >= n_examples:
            break
        
        full_answer = item['answer']
        reasoning = full_answer.split('####')[0].strip()
        numeric_answer = full_answer.split('####')[1].strip()
        
        examples.append({
            'question': item['question'],
            'answer': f"{reasoning} \\boxed{{{numeric_answer}}}"
        })
    
    return examples


def load_aime25(subset_size: int = None) -> List[Dict]:
    print("Loading AIME25 dataset...")
    dataset = load_dataset("math-ai/aime25", split="test")
    
    problems = []
    for idx, item in enumerate(dataset):
        if subset_size and idx >= subset_size:
            break
        
        problems.append({
            'idx': idx,
            'question': item['problem'],
            'answer': item['answer']
        })
    
    print(f"Loaded {len(problems)} problems from AIME25")
    return problems