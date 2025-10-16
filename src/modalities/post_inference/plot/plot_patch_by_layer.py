import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def parse_layer_order(group_name):
    if group_name == "embed":
        return -2, 0
    elif group_name == "rope":
        return -1, 0
    elif group_name == "final_norm":
        return 1000, 0
    elif group_name == "other":
        return 1001, 0
    elif group_name.startswith("layer_"):
        parts = group_name.split("_")
        layer_num = int(parts[1])

        component_order = {"input": 0, "self": 1, "post": 2, "mlp": 3}

        if len(parts) > 2:
            component = parts[2]
            return layer_num, component_order.get(component, 4)
        return layer_num, 4

    return 999, 0


def plot_layer_sequence(*, results, output_path):
    if "layer_results" not in results:
        print("No layer_results found")
        return

    layer_results = results["layer_results"]

    sorted_results = sorted(layer_results, key=lambda x: parse_layer_order(x["group_name"]))

    x_pos = list(range(len(sorted_results)))
    names = [r["group_name"] for r in sorted_results]
    gsm8k = [r["gsm8k"] for r in sorted_results]
    hellaswag = [r["hellaswag"] for r in sorted_results]

    baseline_gsm8k = results["baseline"]["gsm8k"]
    baseline_hellaswag = results["baseline"]["hellaswag"]

    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(x_pos, gsm8k, marker="o", linewidth=2, markersize=3, label="GSM8K", color="#4575b4", alpha=0.8)
    ax.plot(x_pos, hellaswag, marker="s", linewidth=2, markersize=3, label="HellaSwag", color="#d73027", alpha=0.8)

    ax.axhline(y=baseline_gsm8k, color="#4575b4", linestyle="--", linewidth=1.5, label="GSM8K Baseline", alpha=0.5)
    ax.axhline(
        y=baseline_hellaswag, color="#d73027", linestyle="--", linewidth=1.5, label="HellaSwag Baseline", alpha=0.5
    )

    ax.set_xlabel("Model Position (Input → Output)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Performance When Patching Each Layer/Component", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    ax.set_xticks(x_pos[:: max(1, len(x_pos) // 30)])
    ax.set_xticklabels(
        [names[i] for i in range(0, len(names), max(1, len(names) // 30))], rotation=45, ha="right", fontsize=8
    )

    despine(ax)

    plt.tight_layout()
    plt.savefig(output_path / "layer_sequence_all.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plotted {len(sorted_results)} layers in sequence")


def plot_layer_sequence_deltas(*, results, output_path):
    if "layer_results" not in results:
        print("No layer_results found")
        return

    layer_results = results["layer_results"]

    sorted_results = sorted(layer_results, key=lambda x: parse_layer_order(x["group_name"]))

    x_pos = list(range(len(sorted_results)))
    names = [r["group_name"] for r in sorted_results]
    gsm8k_deltas = [r["gsm8k_delta"] for r in sorted_results]
    hellaswag_deltas = [r["hellaswag_delta"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(x_pos, gsm8k_deltas, marker="o", linewidth=2, markersize=3, label="GSM8K Δ", color="#4575b4", alpha=0.8)
    ax.plot(
        x_pos, hellaswag_deltas, marker="s", linewidth=2, markersize=3, label="HellaSwag Δ", color="#d73027", alpha=0.8
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    ax.set_xlabel("Model Position (Input → Output)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy Change (Δ)", fontsize=14, fontweight="bold")
    ax.set_title("Impact When Patching Each Layer/Component", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    ax.set_xticks(x_pos[:: max(1, len(x_pos) // 30)])
    ax.set_xticklabels(
        [names[i] for i in range(0, len(names), max(1, len(names) // 30))], rotation=45, ha="right", fontsize=8
    )

    despine(ax)

    plt.tight_layout()
    plt.savefig(output_path / "layer_sequence_deltas_all.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plotted deltas for {len(sorted_results)} layers in sequence")


def filter_results(layer_results, *, metric="gsm8k", positive_only=False):
    if not positive_only:
        return layer_results

    delta_key = f"{metric}_delta"
    return [r for r in layer_results if r[delta_key] > 0]


def plot_layer_impact(*, results, output_path, top_n=20, metric="gsm8k", positive_only=False):
    if "layer_results" not in results:
        print("No layer_results found, skipping layer impact plot")
        return

    filtered_results = filter_results(results["layer_results"], metric=metric, positive_only=positive_only)
    layer_results = filtered_results[:top_n]

    if not layer_results:
        print(f"No {'positive' if positive_only else ''} results found for {metric}")
        return

    names = [r["group_name"] for r in layer_results]
    deltas = [r[f"{metric}_delta"] for r in layer_results]
    n_neurons = [r["n_neurons"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    colors = ["#d73027" if d < 0 else "#1a9850" for d in deltas]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (bar, neurons) in enumerate(zip(bars, n_neurons)):
        width = bar.get_width()
        label_x = width + (0.002 if width >= 0 else -0.002)
        ha = "left" if width >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{neurons}n",
            ha=ha,
            va="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"{metric.upper()} Accuracy Change (Δ)", fontsize=12, fontweight="bold")

    title_suffix = " (Positive Only)" if positive_only else ""
    ax.set_title(f"Top {len(names)} Layers by {metric.upper()} Impact{title_suffix}", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    despine(ax)

    plt.tight_layout()
    suffix = "_positive" if positive_only else ""
    plt.savefig(output_path / f"layer_impact_{metric}_top{len(names)}{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--positive_only", action="store_true", help="Only show layers with positive GSM8K delta")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    output_path = json_path.parent / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {json_path}")
    results = load_results(json_path)

    print(f"Saving plots to: {output_path}")

    if args.positive_only:
        print("Filtering for positive GSM8K effects only")

    print("Creating sequence plots...")
    plot_layer_sequence(results=results, output_path=output_path)
    plot_layer_sequence_deltas(results=results, output_path=output_path)

    print("Creating top-N plots...")
    plot_layer_impact(
        results=results, output_path=output_path, top_n=args.top_n, metric="gsm8k", positive_only=args.positive_only
    )
    plot_layer_impact(
        results=results, output_path=output_path, top_n=args.top_n, metric="hellaswag", positive_only=args.positive_only
    )

    print(f"\n✓ All plots saved to {output_path}")


if __name__ == "__main__":
    main()
