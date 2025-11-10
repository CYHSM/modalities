import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text as fallback."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize answer for comparison."""
    if answer is None:
        return None
    
    answer = answer.strip()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace('%', '')
    answer = answer.strip()
    
    try:
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    answer = str(num / denom)
        
        num = float(answer)
        if num == int(num):
            return str(int(num))
        else:
            return f"{num:.6f}".rstrip('0').rstrip('.')
    except:
        return answer.lower().strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    
    last_num = extract_last_number(text)
    return normalize_answer(last_num)


def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm is None or gt_norm is None:
        return False
    
    if pred_norm == gt_norm:
        return True
    
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-4
    except:
        return pred_norm == gt_norm
