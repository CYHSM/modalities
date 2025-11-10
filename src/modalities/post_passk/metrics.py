import numpy as np
from scipy.special import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k metric.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider
    
    Returns:
        pass@k value for this problem
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0
    
    return 1.0 - (comb(n - c, k) / comb(n, k))


def compute_pass_at_k(results: list, k_values: list) -> dict:
    """
    Compute pass@k for multiple k values.
    
    Args:
        results: List of dicts with 'correct_count' and 'total_samples'
        k_values: List of k values to compute
    
    Returns:
        Dict mapping k to pass@k value
    """
    pass_k_results = {}
    
    for k in k_values:
        scores = []
        for result in results:
            n = result['total_samples']
            c = result['correct_count']
            scores.append(pass_at_k(n, c, k))
        
        pass_k_results[k] = np.mean(scores)
    
    return pass_k_results
