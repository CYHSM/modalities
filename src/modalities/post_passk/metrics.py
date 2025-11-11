import numpy as np
from scipy.special import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    
    return 1.0 - (comb(n - c, k) / comb(n, k))

def compute_pass_at_k(results: list, k_values: list) -> dict:
    pass_k_results = {}
    for k in k_values:
        scores = []
        for result in results:
            n = result['total_samples']
            c = result['correct_count']
            scores.append(pass_at_k(n, c, k))
        pass_k_results[k] = np.mean(scores)
    
    return pass_k_results