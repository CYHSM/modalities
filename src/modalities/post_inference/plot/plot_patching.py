import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def extract_data(results):
    baseline_gsm8k = results["results"]["baseline"]["gsm8k"]
    baseline_hellaswag = results["results"]["baseline"]["hellaswag"]
    
    data = {}
    for exp in results["results"]["experiments"]:
        d_threshold = exp["d_threshold"]
        
        for scale_result in exp["scale_results"]:
            scale = scale_result["scale_factor"]
            
            if scale not in data:
                data[scale] = {
                    "d_thresholds": [],
                    "gsm8k": [],
                    "hellaswag": [],
                    "gsm8k_delta": [],
                    "hellaswag_delta": []
                }
            
            data[scale]["d_thresholds"].append(d_threshold)
            data[scale]["gsm8k"].append(scale_result["gsm8k"])
            data[scale]["hellaswag"].append(scale_result["hellaswag"])
            data[scale]["gsm8k_delta"].append(scale_result["gsm8k_delta"])
            data[scale]["hellaswag_delta"].append(scale_result["hellaswag_delta"])
    
    return data, baseline_gsm8k, baseline_hellaswag


def plot_performance(*, data, baseline, metric, ylabel, output_path, filename):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
    
    sorted_scales = sorted(data.keys())
    
    for scale, color in zip(sorted_scales, colors):
        scale_data = data[scale]
        d_thresholds = scale_data["d_thresholds"]
        values = scale_data[metric]
        
        sort_idx = np.argsort(d_thresholds)
        t_sorted = np.array(d_thresholds)[sort_idx]
        v_sorted = np.array(values)[sort_idx]
        
        label = f"scale={scale:.2f}"
        if scale == 0.0:
            label = "scale=0.0 (ablate)"
        elif scale == 1.0:
            label = "scale=1.0 (baseline)"
        
        plt.plot(t_sorted, v_sorted, marker='o', linewidth=2, 
                markersize=6, label=label, color=color)
    
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, 
                label='Original baseline', alpha=0.7)
    
    plt.xlabel('T-statistic threshold', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    ax = plt.gca()
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_delta(*, data, metric, ylabel, output_path, filename):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
    
    sorted_scales = sorted(data.keys())
    
    for scale, color in zip(sorted_scales, colors):
        if scale == 1.0:
            continue
            
        scale_data = data[scale]
        d_thresholds = scale_data["d_thresholds"]
        values = scale_data[metric]
        
        sort_idx = np.argsort(d_thresholds)
        t_sorted = np.array(d_thresholds)[sort_idx]
        v_sorted = np.array(values)[sort_idx]
        
        label = f"scale={scale:.2f}"
        if scale == 0.0:
            label = "scale=0.0 (ablate)"
        
        plt.plot(t_sorted, v_sorted, marker='o', linewidth=2,
                markersize=6, label=label, color=color)
    
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.xlabel('T-statistic threshold', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    ax = plt.gca()
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_neuron_count(*, results, output_path):
    plt.figure(figsize=(10, 6))
    
    d_thresholds = []
    n_neurons = []
    n_layers = []
    
    for exp in results["results"]["experiments"]:
        d_thresholds.append(exp["d_threshold"])
        n_neurons.append(exp["n_neurons"])
        n_layers.append(exp["n_layers"])
    
    sort_idx = np.argsort(d_thresholds)
    t_sorted = np.array(d_thresholds)[sort_idx]
    n_sorted = np.array(n_neurons)[sort_idx]
    l_sorted = np.array(n_layers)[sort_idx]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('T-statistic threshold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of neurons', color=color, fontsize=14, fontweight='bold')
    ax1.plot(t_sorted, n_sorted, marker='o', linewidth=2, markersize=6, 
            color=color, label='Neurons')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Number of layers', color=color, fontsize=14, fontweight='bold')
    ax2.plot(t_sorted, l_sorted, marker='s', linewidth=2, markersize=6,
            color=color, label='Layers')
    ax2.tick_params(axis='y', labelcolor=color)
    
    despine(ax1)
    
    fig.tight_layout()
    plt.savefig(output_path / 'neuron_layer_counts.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_plot(*, data, baseline_gsm8k, baseline_hellaswag, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
    sorted_scales = sorted(data.keys())
    
    for scale, color in zip(sorted_scales, colors):
        scale_data = data[scale]
        d_thresholds = scale_data["d_thresholds"]
        
        sort_idx = np.argsort(d_thresholds)
        t_sorted = np.array(d_thresholds)[sort_idx]
        
        gsm8k_sorted = np.array(scale_data["gsm8k"])[sort_idx]
        hellaswag_sorted = np.array(scale_data["hellaswag"])[sort_idx]
        
        label = f"scale={scale:.2f}"
        if scale == 0.0:
            label = "scale=0.0 (ablate)"
        elif scale == 1.0:
            label = "scale=1.0 (baseline)"
        
        ax1.plot(t_sorted, gsm8k_sorted, marker='o', linewidth=2,
                markersize=6, label=label, color=color)
        ax2.plot(t_sorted, hellaswag_sorted, marker='o', linewidth=2,
                markersize=6, label=label, color=color)
    
    ax1.axhline(y=baseline_gsm8k, color='gray', linestyle='--', 
               linewidth=2, label='Original baseline', alpha=0.7)
    ax2.axhline(y=baseline_hellaswag, color='gray', linestyle='--',
               linewidth=2, label='Original baseline', alpha=0.7)
    
    ax1.set_xlabel('T-statistic threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GSM8K Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('GSM8K Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2.set_xlabel('T-statistic threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('HellaSwag Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('HellaSwag Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    despine(ax1)
    despine(ax2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'combined_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_tradeoff(*, data, baseline_gsm8k, baseline_hellaswag, output_path):
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'X']
    
    scales_below = [s for s in sorted(data.keys()) if s < 1.0]
    scales_above = [s for s in sorted(data.keys()) if s > 1.0]
    
    all_d_thresholds = set()
    for scale_data in data.values():
        all_d_thresholds.update(scale_data["d_thresholds"])
    unique_d_thresholds = sorted(all_d_thresholds)
    
    t_to_color = {}
    cmap = plt.cm.turbo
    for i, t in enumerate(unique_d_thresholds):
        t_to_color[t] = cmap(i / max(1, len(unique_d_thresholds) - 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for scale_idx, scale in enumerate(scales_below):
        scale_data = data[scale]
        gsm8k_vals = scale_data["gsm8k"]
        hellaswag_vals = scale_data["hellaswag"]
        d_thresholds = scale_data["d_thresholds"]
        
        marker = markers[scale_idx % len(markers)]
        
        label = f"scale={scale:.2f}"
        if scale == 0.0:
            label = "scale=0.0 (ablate)"
        
        colors = [t_to_color[t] for t in d_thresholds]
        
        ax1.scatter(gsm8k_vals, hellaswag_vals, s=100, c=colors,
                   marker=marker, label=label, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax1.scatter([baseline_gsm8k], [baseline_hellaswag], s=250, c='red',
               marker='*', label='Baseline', edgecolors='black', 
               linewidth=1.5, zorder=10)
    
    ax1.set_xlabel('GSM8K Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('HellaSwag Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Ablation (scale < 1.0)', fontsize=16, fontweight='bold')
    legend1 = ax1.legend(loc='best', fontsize=10, framealpha=0.9, markerscale=0.7)
    ax1.grid(True, alpha=0.3, linestyle='--')
    despine(ax1)
    
    for scale_idx, scale in enumerate(scales_above):
        scale_data = data[scale]
        gsm8k_vals = scale_data["gsm8k"]
        hellaswag_vals = scale_data["hellaswag"]
        d_thresholds = scale_data["d_thresholds"]
        
        marker = markers[scale_idx % len(markers)]
        
        label = f"scale={scale:.2f}"
        
        colors = [t_to_color[t] for t in d_thresholds]
        
        ax2.scatter(gsm8k_vals, hellaswag_vals, s=100, c=colors,
                   marker=marker, label=label, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax2.scatter([baseline_gsm8k], [baseline_hellaswag], s=250, c='red',
               marker='*', label='Baseline', edgecolors='black', 
               linewidth=1.5, zorder=10)
    
    ax2.set_xlabel('GSM8K Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('HellaSwag Accuracy', fontsize=14, fontweight='bold')
    ax2.set_title('Amplification (scale > 1.0)', fontsize=16, fontweight='bold')
    legend2 = ax2.legend(loc='best', fontsize=10, framealpha=0.9, markerscale=0.7)
    ax2.grid(True, alpha=0.3, linestyle='--')
    despine(ax2)
    
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(unique_d_thresholds), vmax=max(unique_d_thresholds)))
    #sm.set_array([])
    #cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.08, aspect=40)
    #cbar.set_label('T-statistic threshold', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {args.json_path}")
    results = load_results(args.json_path)
    
    print("Extracting data...")
    data, baseline_gsm8k, baseline_hellaswag = extract_data(results)
    
    print("Creating plots...")
    
    plot_performance(
        data=data,
        baseline=baseline_gsm8k,
        metric="gsm8k",
        ylabel="GSM8K Accuracy",
        output_path=output_path,
        filename="gsm8k_performance.png"
    )
    
    plot_performance(
        data=data,
        baseline=baseline_hellaswag,
        metric="hellaswag",
        ylabel="HellaSwag Accuracy",
        output_path=output_path,
        filename="hellaswag_performance.png"
    )
    
    plot_delta(
        data=data,
        metric="gsm8k_delta",
        ylabel="GSM8K Accuracy Change (Δ)",
        output_path=output_path,
        filename="gsm8k_delta.png"
    )
    
    plot_delta(
        data=data,
        metric="hellaswag_delta",
        ylabel="HellaSwag Accuracy Change (Δ)",
        output_path=output_path,
        filename="hellaswag_delta.png"
    )
    
    plot_neuron_count(results=results, output_path=output_path)
    
    create_combined_plot(
        data=data,
        baseline_gsm8k=baseline_gsm8k,
        baseline_hellaswag=baseline_hellaswag,
        output_path=output_path
    )
    
    plot_accuracy_tradeoff(
        data=data,
        baseline_gsm8k=baseline_gsm8k,
        baseline_hellaswag=baseline_hellaswag,
        output_path=output_path
    )
    
    print(f"\n✓ All plots saved to {output_path}")
    print(f"  - gsm8k_performance.png")
    print(f"  - hellaswag_performance.png")
    print(f"  - gsm8k_delta.png")
    print(f"  - hellaswag_delta.png")
    print(f"  - neuron_layer_counts.png")
    print(f"  - combined_performance.png")
    print(f"  - accuracy_tradeoff.png")


if __name__ == "__main__":
    main()