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


def plot_layer_impact(*, results, output_path, top_n=20, metric='gsm8k'):
    if "layer_results" not in results:
        print("No layer_results found, skipping layer impact plot")
        return
    
    layer_results = results["layer_results"][:top_n]
    
    names = [r["group_name"] for r in layer_results]
    deltas = [r[f"{metric}_delta"] for r in layer_results]
    n_neurons = [r["n_neurons"] for r in layer_results]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))
    
    colors = ['#d73027' if d < 0 else '#1a9850' for d in deltas]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (bar, neurons) in enumerate(zip(bars, n_neurons)):
        width = bar.get_width()
        label_x = width + (0.002 if width >= 0 else -0.002)
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{neurons}n', ha=ha, va='center', fontsize=8, style='italic', color='gray')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f'{metric.upper()} Accuracy Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Layers by {metric.upper()} Impact', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / f'layer_impact_{metric}_top{top_n}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_dual_impact(*, results, output_path, top_n=20):
    if "layer_results" not in results:
        print("No layer_results found, skipping dual impact plot")
        return
    
    layer_results = results["layer_results"][:top_n]
    
    names = [r["group_name"] for r in layer_results]
    gsm8k_deltas = [r["gsm8k_delta"] for r in layer_results]
    hellaswag_deltas = [r["hellaswag_delta"] for r in layer_results]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(names) * 0.35)))
    
    y_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.barh(y_pos - width/2, gsm8k_deltas, width, 
                    label='GSM8K', color='#4575b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_pos + width/2, hellaswag_deltas, width,
                    label='HellaSwag', color='#d73027', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Accuracy Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Layers: Impact on Both Benchmarks', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / f'dual_impact_top{top_n}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_tradeoff(*, results, output_path):
    if "layer_results" not in results:
        print("No layer_results found, skipping scatter plot")
        return
    
    layer_results = results["layer_results"]
    
    gsm8k_deltas = [r["gsm8k_delta"] for r in layer_results]
    hellaswag_deltas = [r["hellaswag_delta"] for r in layer_results]
    n_neurons = [r["n_neurons"] for r in layer_results]
    names = [r["group_name"] for r in layer_results]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [max(20, min(500, n * 5)) for n in n_neurons]
    
    scatter = ax.scatter(gsm8k_deltas, hellaswag_deltas, s=sizes, 
                        c=n_neurons, cmap='viridis', alpha=0.6, 
                        edgecolors='black', linewidth=0.5)
    
    for i, name in enumerate(names[:10]):
        ax.annotate(name, (gsm8k_deltas[i], hellaswag_deltas[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('GSM8K Accuracy Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('HellaSwag Accuracy Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Trade-off Between Benchmarks', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Neurons', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scatter_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_performance(*, results, output_path):
    if "cumulative_results" not in results:
        print("No cumulative_results found, skipping cumulative plot")
        return
    
    cumulative_results = results["cumulative_results"]
    
    n_layers = [r["n_layers"] for r in cumulative_results]
    gsm8k = [r["gsm8k"] for r in cumulative_results]
    hellaswag = [r["hellaswag"] for r in cumulative_results]
    
    baseline_gsm8k = results["baseline"]["gsm8k"]
    baseline_hellaswag = results["baseline"]["hellaswag"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(n_layers, gsm8k, marker='o', linewidth=2, markersize=4, 
            label='GSM8K', color='#4575b4')
    ax1.axhline(y=baseline_gsm8k, color='gray', linestyle='--', 
               linewidth=2, label='Baseline', alpha=0.7)
    ax1.set_xlabel('Number of Layers Patched', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GSM8K Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('GSM8K: Cumulative Effect', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    despine(ax1)
    
    ax2.plot(n_layers, hellaswag, marker='o', linewidth=2, markersize=4,
            label='HellaSwag', color='#d73027')
    ax2.axhline(y=baseline_hellaswag, color='gray', linestyle='--',
               linewidth=2, label='Baseline', alpha=0.7)
    ax2.set_xlabel('Number of Layers Patched', fontsize=12, fontweight='bold')
    ax2.set_ylabel('HellaSwag Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('HellaSwag: Cumulative Effect', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    despine(ax2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cumulative_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_deltas(*, results, output_path):
    if "cumulative_results" not in results:
        return
    
    cumulative_results = results["cumulative_results"]
    
    n_layers = [r["n_layers"] for r in cumulative_results]
    gsm8k_deltas = [r["gsm8k_delta"] for r in cumulative_results]
    hellaswag_deltas = [r["hellaswag_delta"] for r in cumulative_results]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(n_layers, gsm8k_deltas, marker='o', linewidth=2, markersize=5,
            label='GSM8K Δ', color='#4575b4')
    ax.plot(n_layers, hellaswag_deltas, marker='s', linewidth=2, markersize=5,
            label='HellaSwag Δ', color='#d73027')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Number of Layers Patched', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Change from Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    despine(ax)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cumulative_deltas.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_neuron_distribution(*, results, output_path):
    if "layer_results" not in results:
        return
    
    layer_results = results["layer_results"]
    
    names = [r["group_name"] for r in layer_results[:30]]
    n_neurons = [r["n_neurons"] for r in layer_results[:30]]
    gsm8k_deltas = [r["gsm8k_delta"] for r in layer_results[:30]]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    y_pos = np.arange(len(names))
    
    bars = ax1.barh(y_pos, n_neurons, color='#74add1', alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Number of Neurons', fontsize=11, fontweight='bold')
    ax1.set_title('Neuron Count per Layer', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    despine(ax1)
    
    colors = ['#d73027' if d < 0 else '#1a9850' for d in gsm8k_deltas]
    bars = ax2.barh(y_pos, gsm8k_deltas, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('GSM8K Accuracy Change (Δ)', fontsize=11, fontweight='bold')
    ax2.set_title('Impact per Layer', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    despine(ax2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'neuron_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(*, results, output_path, top_n=30):
    if "layer_results" not in results:
        return
    
    layer_results = results["layer_results"][:top_n]
    
    names = [r["group_name"] for r in layer_results]
    
    data = np.array([
        [r["gsm8k_delta"] for r in layer_results],
        [r["hellaswag_delta"] for r in layer_results]
    ])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['GSM8K Δ', 'HellaSwag Δ'], fontsize=11)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, ha='right', fontsize=8)
    ax.set_title(f'Impact Heatmap: Top {top_n} Layers', fontsize=14, fontweight='bold')
    
    for i in range(2):
        for j in range(len(names)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black" if abs(data[i, j]) < vmax*0.6 else "white",
                          fontsize=7)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Change (Δ)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / f'heatmap_top{top_n}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_figure(*, results, output_path):
    if "layer_results" not in results:
        return
    
    layer_results = results["layer_results"][:15]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    names = [r["group_name"][:30] for r in layer_results]
    gsm8k_deltas = [r["gsm8k_delta"] for r in layer_results]
    colors = ['#d73027' if d < 0 else '#1a9850' for d in gsm8k_deltas]
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, gsm8k_deltas, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('GSM8K Δ', fontsize=10, fontweight='bold')
    ax1.set_title('Top 15 Layers (GSM8K)', fontsize=11, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    despine(ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    gsm8k_deltas_all = [r["gsm8k_delta"] for r in results["layer_results"]]
    hellaswag_deltas_all = [r["hellaswag_delta"] for r in results["layer_results"]]
    n_neurons_all = [r["n_neurons"] for r in results["layer_results"]]
    sizes = [max(20, min(300, n * 5)) for n in n_neurons_all]
    scatter = ax2.scatter(gsm8k_deltas_all, hellaswag_deltas_all, s=sizes,
                         c=n_neurons_all, cmap='viridis', alpha=0.6,
                         edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('GSM8K Δ', fontsize=10, fontweight='bold')
    ax2.set_ylabel('HellaSwag Δ', fontsize=10, fontweight='bold')
    ax2.set_title('Performance Trade-off', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Neurons', fontsize=8)
    despine(ax2)
    
    ax3 = fig.add_subplot(gs[1, :])
    layer_results_sorted = sorted(results["layer_results"], 
                                  key=lambda x: x["n_neurons"], reverse=True)[:20]
    names_sorted = [r["group_name"][:35] for r in layer_results_sorted]
    neurons_sorted = [r["n_neurons"] for r in layer_results_sorted]
    gsm8k_sorted = [r["gsm8k_delta"] for r in layer_results_sorted]
    y_pos = np.arange(len(names_sorted))
    width = 0.35
    ax3.barh(y_pos - width/2, neurons_sorted, width, label='Neurons', 
            color='#74add1', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3_twin = ax3.twiny()
    colors_impact = ['#d73027' if d < 0 else '#1a9850' for d in gsm8k_sorted]
    ax3_twin.barh(y_pos + width/2, gsm8k_sorted, width, label='GSM8K Δ',
                 color=colors_impact, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names_sorted, fontsize=8)
    ax3.set_xlabel('Number of Neurons', fontsize=10, fontweight='bold', color='#74add1')
    ax3_twin.set_xlabel('GSM8K Δ', fontsize=10, fontweight='bold', color='#1a9850')
    ax3.set_title('Top 20 Layers by Neuron Count + Impact', fontsize=11, fontweight='bold')
    ax3_twin.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    despine(ax3)
    
    config = results["config"]
    fig.suptitle(
        f'Layer Attribution Summary | Scale={config["scale_factor"]} | '
        f'd>{config["d_threshold"]} | std>{config["std_threshold"]}',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.savefig(output_path / 'summary_figure.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    output_path = json_path.parent / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {json_path}")
    results = load_results(json_path)
    
    print(f"Saving plots to: {output_path}")
    
    mode = "cumulative" if "cumulative_results" in results else "layer_by_layer"
    print(f"Detected mode: {mode}")
    
    if mode == "layer_by_layer":
        print("Creating layer-by-layer plots...")
        plot_layer_impact(results=results, output_path=output_path, 
                         top_n=args.top_n, metric='gsm8k')
        plot_layer_impact(results=results, output_path=output_path,
                         top_n=args.top_n, metric='hellaswag')
        plot_dual_impact(results=results, output_path=output_path, top_n=args.top_n)
        plot_scatter_tradeoff(results=results, output_path=output_path)
        plot_neuron_distribution(results=results, output_path=output_path)
        plot_heatmap(results=results, output_path=output_path, top_n=min(30, args.top_n))
        create_summary_figure(results=results, output_path=output_path)
    else:
        print("Creating cumulative plots...")
        plot_cumulative_performance(results=results, output_path=output_path)
        plot_cumulative_deltas(results=results, output_path=output_path)
    
    print(f"\n✓ All plots saved to {output_path}")


if __name__ == "__main__":
    main()