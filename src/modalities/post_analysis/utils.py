"""
Common utilities for model comparison analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def calculate_distribution_stats(diff_tensor: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive distribution statistics for parameter differences"""
    diff_flat = diff_tensor.flatten()
    
    return {
        "mean_diff": float(torch.mean(diff_flat)),
        "median_diff": float(torch.median(diff_flat)),
        "std_diff": float(torch.std(diff_flat)),
        "min_diff": float(torch.min(diff_flat)),
        "max_diff": float(torch.max(diff_flat)),
        "abs_mean_diff": float(torch.mean(torch.abs(diff_flat))),
        "num_positive": int(torch.sum(diff_flat > 0)),
        "num_negative": int(torch.sum(diff_flat < 0)),
        "num_unchanged": int(torch.sum(diff_flat == 0)),
    }


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate cosine similarity between two tensors"""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    cosine_sim = torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0), 
        flat2.unsqueeze(0)
    ).item()
    
    return cosine_sim


def shorten_layer_name(name: str) -> str:
    """Create shorter, readable layer names"""
    # Extract key parts
    parts = name.split('.')
    
    # Find layer number
    layer_num = None
    for part in parts:
        if part.isdigit():
            layer_num = part
            break
    
    # Identify component type
    if 'self_attn' in name:
        if 'q_proj' in name:
            component = 'Q'
        elif 'k_proj' in name:
            component = 'K'
        elif 'v_proj' in name:
            component = 'V'
        elif 'o_proj' in name:
            component = 'O'
        else:
            component = 'Attn'
    elif 'mlp' in name:
        if 'gate_proj' in name:
            component = 'Gate'
        elif 'up_proj' in name:
            component = 'Up'
        elif 'down_proj' in name:
            component = 'Down'
        else:
            component = 'MLP'
    elif any(norm in name for norm in ['norm', 'layernorm']):
        component = 'Norm'
    elif 'embed' in name:
        component = 'Embed'
    elif 'lm_head' in name:
        component = 'Head'
    else:
        component = parts[-1][:8]  # Last part, truncated
    
    if layer_num:
        return f"L{layer_num}.{component}"
    else:
        return component


def categorize_layers(layer_names: List[str]) -> List[str]:
    """Categorize layers by type"""
    layer_types = []
    for name in layer_names:
        if 'self_attn' in name:
            layer_types.append('Attention')
        elif 'mlp' in name:
            layer_types.append('MLP')
        elif any(norm in name for norm in ['norm', 'layernorm']):
            layer_types.append('Normalization')
        elif 'embed' in name:
            layer_types.append('Embedding')
        elif 'lm_head' in name:
            layer_types.append('LM Head')
        else:
            layer_types.append('Other')
    return layer_types


def get_layer_number(name: str) -> int:
    """Extract layer number from layer name"""
    parts = name.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1


def get_layer_type_colors():
    """Get consistent colors for layer types"""
    return {
        'Attention': '#FF6B6B',
        'MLP': '#4ECDC4', 
        'Normalization': '#45B7D1',
        'Embedding': '#96CEB4',
        'LM Head': '#FFEAA7',
        'Other': '#DDA0DD'
    }


def group_layers_by_actual_layer(layer_data: List[Dict]) -> Tuple[List[int], List[str], List[str]]:
    """Group layers by actual layer number for cleaner x-axis"""
    # Filter out layers without layer numbers and sort
    numbered_layers = [l for l in layer_data if l['layer_number'] >= 0]
    numbered_layers.sort(key=lambda x: x['layer_number'])
    
    # Group by layer number
    layer_groups = {}
    for layer in numbered_layers:
        layer_num = layer['layer_number']
        if layer_num not in layer_groups:
            layer_groups[layer_num] = []
        layer_groups[layer_num].append(layer)
    
    # Create x-axis positions and labels
    x_positions = []
    x_labels = []
    layer_types = []
    
    for i, (layer_num, layers) in enumerate(sorted(layer_groups.items())):
        x_positions.extend([i] * len(layers))
        x_labels.extend([f"L{layer_num}"] * len(layers))
        layer_types.extend([categorize_layers([l['layer']])[0] for l in layers])
    
    return x_positions, x_labels, layer_types