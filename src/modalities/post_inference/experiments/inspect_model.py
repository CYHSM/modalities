import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from modalities.post_inference.utils.h5_utils import H5Store


def inspect_model_architecture(model_path):
    print(f"\nLoading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"\n{'='*80}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*80}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"\n{'LAYER NAME':<60} | {'SHAPE':<20} | {'PARAMS':>12}")
    print("-" * 95)
    
    special_layers = {}
    
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['embed', 'norm', 'lm_head', 'rope', 'rotary']):
            if 'layers.' not in name:
                print(f"{name:<60} | {str(list(param.shape)):<20} | {param.numel():>12,}")
                special_layers[name] = param.shape
    
    return special_layers


def inspect_statistics(h5_path, layer_name_pattern):
    stats_path = Path(h5_path).parent / "statistics.h5"
    
    if not stats_path.exists():
        print(f"Statistics file not found: {stats_path}")
        return
    
    print(f"\nLoading statistics from {stats_path}...")
    statistics = H5Store.load_statistics(stats_path)
    
    matching_layers = {}
    for layer_name, stats in statistics.items():
        if layer_name_pattern.lower() in layer_name.lower():
            matching_layers[layer_name] = stats
    
    if not matching_layers:
        print(f"No layers matching pattern '{layer_name_pattern}' found")
        return
    
    print(f"\n{'='*80}")
    print(f"STATISTICS FOR LAYERS MATCHING: '{layer_name_pattern}'")
    print(f"{'='*80}")
    
    for layer_name, stats in matching_layers.items():
        print(f"\nLayer: {layer_name}")
        print("-" * 80)
        
        for key in ['mean_diff', 't_stats', 'p_values', 'cohens_d', 'pooled_std']:
            if key in stats:
                arr = stats[key]
                print(f"\n{key}:")
                print(f"  Shape: {arr.shape}")
                print(f"  Min: {np.min(arr):.6f}")
                print(f"  Max: {np.max(arr):.6f}")
                print(f"  Mean: {np.mean(arr):.6f}")
                print(f"  Std: {np.std(arr):.6f}")
                
                if key == 'cohens_d':
                    top_indices = np.argsort(np.abs(arr))[-10:][::-1]
                    print(f"\n  Top 10 neurons by |Cohen's d|:")
                    for i, idx in enumerate(top_indices, 1):
                        print(f"    {i}. Neuron {idx}: d={arr[idx]:.4f}, "
                              f"p={stats['p_values'][idx]:.4e}, "
                              f"mean_diff={stats['mean_diff'][idx]:.4f}")
        
        if 'n_math' in stats and 'n_nonmath' in stats:
            print(f"\nSample sizes:")
            print(f"  Math: {stats['n_math']}")
            print(f"  Non-math: {stats['n_nonmath']}")


def analyze_significant_neurons(h5_path, d_threshold, std_threshold):
    stats_path = Path(h5_path).parent / "statistics.h5"
    statistics = H5Store.load_statistics(stats_path)
    
    print(f"\n{'='*80}")
    print(f"SIGNIFICANT NEURONS (|d|>{d_threshold}, std>{std_threshold})")
    print(f"{'='*80}")
    
    results = []
    
    for layer_name, stats in statistics.items():
        cohens_d = stats['cohens_d']
        pooled_std = stats['pooled_std']
        p_values = stats.get('p_corrected', stats.get('p_values'))
        
        significant_mask = (np.abs(cohens_d) > d_threshold) & (pooled_std > std_threshold)
        n_significant = np.sum(significant_mask)
        
        if n_significant > 0:
            results.append({
                'layer': layer_name,
                'n_significant': n_significant,
                'total': len(cohens_d),
                'pct': 100 * n_significant / len(cohens_d),
                'max_d': np.max(np.abs(cohens_d)),
                'mean_d_significant': np.mean(np.abs(cohens_d[significant_mask]))
            })
    
    results.sort(key=lambda x: x['n_significant'], reverse=True)
    
    print(f"\n{'LAYER':<50} | {'SIGNIFICANT':>12} | {'TOTAL':>10} | {'%':>7} | {'MAX |d|':>10} | {'MEAN |d|':>10}")
    print("-" * 110)
    
    for r in results[:20]:
        print(f"{r['layer']:<50} | {r['n_significant']:>12,} | {r['total']:>10,} | "
              f"{r['pct']:>6.2f}% | {r['max_d']:>10.4f} | {r['mean_d_significant']:>10.4f}")
    
    return results


def compare_layer_stats(h5_path, layer_pattern1, layer_pattern2):
    stats_path = Path(h5_path).parent / "statistics.h5"
    statistics = H5Store.load_statistics(stats_path)
    
    layers1 = {k: v for k, v in statistics.items() if layer_pattern1.lower() in k.lower()}
    layers2 = {k: v for k, v in statistics.items() if layer_pattern2.lower() in k.lower()}
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: '{layer_pattern1}' vs '{layer_pattern2}'")
    print(f"{'='*80}")
    
    if not layers1:
        print(f"No layers found matching '{layer_pattern1}'")
    if not layers2:
        print(f"No layers found matching '{layer_pattern2}'")
    
    print(f"\nLayers matching '{layer_pattern1}': {len(layers1)}")
    for name in layers1.keys():
        print(f"  - {name}")
    
    print(f"\nLayers matching '{layer_pattern2}': {len(layers2)}")
    for name in layers2.keys():
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--h5_path", type=str, help="Path to H5 activations file")
    parser.add_argument("--layer_pattern", type=str, default="norm", 
                       help="Layer name pattern to inspect")
    parser.add_argument("--d_threshold", type=float, default=1.0)
    parser.add_argument("--std_threshold", type=float, default=0.1)
    parser.add_argument("--compare", type=str, nargs=2, 
                       help="Compare two layer patterns, e.g., 'norm' 'lm_head'")
    
    args = parser.parse_args()
    
    if args.model:
        special_layers = inspect_model_architecture(args.model)
    
    if args.h5_path:
        if args.compare:
            compare_layer_stats(args.h5_path, args.compare[0], args.compare[1])
        else:
            inspect_statistics(args.h5_path, args.layer_pattern)
        
        print("\n")
        analyze_significant_neurons(args.h5_path, args.d_threshold, args.std_threshold)


if __name__ == "__main__":
    main()