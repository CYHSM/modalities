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
    """Categorize layers by type with granular separation"""
    layer_types = []
    for name in layer_names:
        if any(norm in name for norm in ['norm', 'layernorm', 'layer_norm']):
            if 'input' in name or 'pre' in name:
                if 'weight' in name:
                    layer_types.append('LayerNorm (W) Pre')
                elif 'bias' in name:
                    layer_types.append('LayerNorm (B) Pre')
                else:
                    layer_types.append('LayerNorm Pre')
            elif 'post' in name or 'final' in name:
                if 'weight' in name:
                    layer_types.append('LayerNorm (W) Post')
                elif 'bias' in name:
                    layer_types.append('LayerNorm (B) Post')
                else:
                    layer_types.append('LayerNorm Post')
            else:
                if 'weight' in name:
                    layer_types.append('LayerNorm (W)')
                elif 'bias' in name:
                    layer_types.append('LayerNorm (B)')
                else:
                    layer_types.append('LayerNorm')
        elif 'self_attn' in name or 'attention' in name.lower():
            if 'q_proj' in name:
                layer_types.append('Att Q')
            elif 'k_proj' in name:
                layer_types.append('Att K')
            elif 'v_proj' in name:
                layer_types.append('Att V')
            elif 'o_proj' in name:
                layer_types.append('Att Out')
            else:
                print(f"Warning: Unrecognized attention layer name '{name}'")
                layer_types.append('Attention Other')
        # MLP components - separate each type
        elif 'mlp' in name or 'feed_forward' in name:
            if 'gate_proj' in name:
                layer_types.append('MLP Gate')
            elif 'up_proj' in name:
                layer_types.append('MLP Up')
            elif 'down_proj' in name:
                layer_types.append('MLP Down')
            else:
                print(f"Warning: Unrecognized MLP layer name '{name}'")
                layer_types.append('MLP Other')
        elif 'embed' in name or 'emb' in name:
            if 'token' in name:
                layer_types.append('Token Embedding')
            elif 'position' in name or 'pos' in name:
                layer_types.append('Position Embedding')
            elif 'rotary' in name:
                layer_types.append('Position Embedding')
            else:
                print(f"Warning: Unrecognized embedding layer name '{name}'")
                layer_types.append('Embedding')
        # Output/head layers
        elif 'lm_head' in name or 'output' in name:
            layer_types.append('LM Head')
        else:
            print(f"Warning: Unrecognized layer name '{name}'")
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
    """Get consistent colors for layer types with distinct groups and subcolors"""
    return {
        # Attention components (RED FAMILY) - High contrast reds
        'Att Q': '#FF4444',      # Bright red
        'Att K': '#CC0000',      # Pure red  
        'Att V': '#FF8888',      # Light red
        'Att Out': '#880000',    # Dark red
        'Attention Other': '#FFCCCC',  # Very light red
        
        # MLP components (BLUE FAMILY) - High contrast blues
        'MLP Gate': '#4444FF',   # Bright blue
        'MLP Up': '#0000CC',     # Dark blue
        'MLP Down': '#8888FF',   # Light blue
        'MLP Other': '#CCCCFF',  # Very light blue
        
        # Normalization layers (GREEN FAMILY) - High contrast greens
        'LayerNorm (W) Pre': '#00CC00',     # Bright green
        'LayerNorm (B) Pre': '#44FF44',     # Light green  
        'LayerNorm (W) Post': '#008800',    # Dark green
        'LayerNorm (B) Post': '#006600',    # Very dark green
        'Input LayerNorm': '#88FF88',       # Very light green
        'LayerNorm Post': "#10B810",       # Light green variant
        'Final LayerNorm (W)': '#004400',   # Darkest green
        'Final LayerNorm (B)': '#66AA66',   # Medium green
        'Final LayerNorm': '#CCFFCC',       # Palest green
        'LayerNorm (W)': '#22AA22',         # Medium-dark green
        'LayerNorm (B)': '#66CC66',         # Medium-light green
        'LayerNorm': '#AAFFAA',             # Light green
        
        # Embedding layers (ORANGE FAMILY)
        'Token Embedding': '#FF8800',       # Dark orange
        'Position Embedding': '#FFAA00',    # Orange
        'Rotary Position Embedding': '#FFCC44',  # Light orange
        'Embedding': '#FFDDAA',             # Very light orange
        
        # Output layers (PURPLE FAMILY)
        'LM Head': '#8844AA',               # Purple
        
        # Other (GRAY FAMILY)
        'Other': '#888888'                  # Gray
    }


def group_layers_by_actual_layer(layer_data: List[Dict]) -> Tuple[List[int], List[str], List[str]]:
    """Group layers by actual layer number for cleaner x-axis"""
    # Filter out layers without layer numbers and sort
    numbered_layers = [l for l in layer_data if l['layer_number'] >= 0]
    numbered_layers.sort(key=lambda x: (x['layer_number'], x['layer_type']))
    
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
    
    for layer_num, layers in sorted(layer_groups.items()):
        for i, layer in enumerate(layers):
            # Spread components within each layer with more spacing for better visibility
            x_pos = layer_num + (i - len(layers)/2 + 0.5) * 0.15  # Increased spacing from 0.1 to 0.15
            x_positions.append(x_pos)
            x_labels.append(f"L{layer_num}")
            layer_types.append(layer['layer_type'])
    
    return x_positions, x_labels, layer_types