from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

DATASET_COLORS = {"full_code": "#FF6B6B", "full_general": "#A12FCE", "full_math": "#45B7D1", "full_mix": "#34C983"}


def load_data(data_dir, method="pca+tsne", window=1000):
    data_dir = Path(data_dir)
    coords = np.load(data_dir / f"coords_2d_{method}_window{window}_normFalse.npy")

    benchmark_scores = {}
    for metric in ["gsm8k_math", "hellaswag_reasoning", "humaneval_coding", "wmt16_translation"]:
        scores_path = data_dir / f"{metric}_scores.npy"
        if scores_path.exists():
            benchmark_scores[metric] = np.load(scores_path)

    return coords, benchmark_scores


def plot_trajectories(coords, dataset_names, checkpoint_numbers, dims=(0, 1), figsize=(12, 8), arrow_stride=2):
    fig, ax = plt.subplots(figsize=figsize)
    coords_2d = coords[:, dims]

    for dataset in dataset_names:
        start_idx = dataset_names.index(dataset) * len(checkpoint_numbers)
        end_idx = start_idx + len(checkpoint_numbers)
        trajectory = coords_2d[start_idx:end_idx]
        color = DATASET_COLORS.get(dataset, "gray")

        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.5, linewidth=2.5, zorder=1)

        X = trajectory[:-1, 0]
        Y = trajectory[:-1, 1]
        U = np.diff(trajectory[:, 0])
        V = np.diff(trajectory[:, 1])

        ax.quiver(
            X[::arrow_stride],
            Y[::arrow_stride],
            U[::arrow_stride],
            V[::arrow_stride],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.9,
            width=0.004,
            zorder=2,
        )

        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color=color,
            s=100,
            marker="o",
            alpha=0.8,
            edgecolors="white",
            linewidth=2,
            zorder=3,
        )

        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color=color,
            s=400,
            marker="*",
            alpha=1.0,
            edgecolors="white",
            linewidth=2,
            zorder=0,
            label=dataset.replace("full_", "").title(),
        )

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.set_xlabel(f"Dim {dims[0]}", fontsize=12)
    ax.set_ylabel(f"Dim {dims[1]}", fontsize=12)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig


def plot_contour_with_trajectories(
    coords, benchmark_scores, metric_name, dataset_names, checkpoint_numbers, dims=(0, 1), figsize=(14, 10)
):
    if metric_name not in benchmark_scores:
        return None

    scores = benchmark_scores[metric_name]
    valid_mask = ~np.isnan(scores)

    if not valid_mask.any():
        return None

    fig, ax = plt.subplots(figsize=figsize)
    coords_2d = coords[:, dims]

    valid_coords = coords_2d[valid_mask]
    valid_scores = scores[valid_mask]

    if len(valid_scores) > 3:
        xi = np.linspace(coords_2d[:, 0].min(), coords_2d[:, 0].max(), 100)
        yi = np.linspace(coords_2d[:, 1].min(), coords_2d[:, 1].max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata(
            (valid_coords[:, 0], valid_coords[:, 1]), valid_scores, (xi, yi), method="linear", fill_value=np.nan
        )

        contour = ax.contourf(xi, yi, zi, levels=20, cmap="viridis", alpha=0.7)
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(f'{metric_name.replace("_", " ").title()} Score', rotation=270, labelpad=20)

    for dataset in dataset_names:
        start_idx = dataset_names.index(dataset) * len(checkpoint_numbers)
        end_idx = start_idx + len(checkpoint_numbers)
        trajectory = coords_2d[start_idx:end_idx]
        color = DATASET_COLORS.get(dataset, "gray")

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            alpha=0.8,
            linewidth=3,
            label=dataset.replace("full_", "").title(),
        )

        ax.scatter(trajectory[:, 0], trajectory[:, 1], color=color, s=50, alpha=0.8, edgecolors="white", linewidth=1)

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.set_xlabel(f"Dimension {dims[0]}", fontsize=12)
    ax.set_ylabel(f"Dimension {dims[1]}", fontsize=12)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig


def plot_benchmark_progression(
    benchmark_scores, dataset_names, checkpoint_numbers, smoothing_window=4, figsize=(16, 12)
):
    metrics = {
        "gsm8k_math": "GSM8K Math (Exact Match)",
        "hellaswag_reasoning": "HellaSwag Reasoning (Acc Norm)",
        "humaneval_coding": "HumanEval Coding (Pass@10)",
        "wmt16_translation": "WMT16 Translation (BLEU)",
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (metric_key, metric_title) in enumerate(metrics.items()):
        ax = axes[idx]

        if metric_key not in benchmark_scores:
            ax.text(
                0.5, 0.5, f"No data for {metric_title}", ha="center", va="center", transform=ax.transAxes, fontsize=14
            )
            sns.despine(trim=True, ax=ax)
            continue

        scores = benchmark_scores[metric_key]

        for dataset in dataset_names:
            start_idx = dataset_names.index(dataset) * len(checkpoint_numbers)
            end_idx = start_idx + len(checkpoint_numbers)
            dataset_scores = scores[start_idx:end_idx]
            color = DATASET_COLORS.get(dataset, "gray")
            label = dataset.replace("full_", "").title()

            valid_mask = ~np.isnan(dataset_scores)
            if valid_mask.any():
                valid_checkpoints = np.array(checkpoint_numbers)[valid_mask]
                valid_scores = dataset_scores[valid_mask]

                ax.plot(
                    valid_checkpoints, valid_scores, color=color, linewidth=1.5, marker="", linestyle="--", alpha=0.3
                )

                if len(valid_scores) >= smoothing_window:
                    smooth_scores = (
                        pd.Series(valid_scores).rolling(window=smoothing_window, center=True, min_periods=1).mean()
                    )
                    ax.plot(
                        valid_checkpoints,
                        smooth_scores,
                        color=color,
                        linewidth=3,
                        marker="o",
                        markersize=5,
                        label=label,
                    )
                else:
                    ax.plot(
                        valid_checkpoints, valid_scores, color=color, linewidth=3, marker="o", markersize=5, label=label
                    )

        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Score")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x > 0 else "0"))
        sns.despine(trim=True, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    data_dir = "/raid/s3/opengptx/mfrey/cp_analysis/dim_reduction"
    method = "pca"
    window = 100

    dataset_names = ["full_code", "full_general", "full_math", "full_mix"]
    checkpoint_numbers = [0] + list(range(5000, 105000, 5000))

    coords, benchmark_scores = load_data(data_dir, method, window)
    print(f"Loaded coordinates: {coords.shape}")
    print(f"Available metrics: {list(benchmark_scores.keys())}")

    output_dir = Path(data_dir) / "plots"
    output_dir.mkdir(exist_ok=True)

    n_dims = 5
    dim_pairs = [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]

    print("Generating benchmark progression plot...")
    fig = plot_benchmark_progression(benchmark_scores, dataset_names, checkpoint_numbers)
    plt.savefig(output_dir / "benchmark_progression_smoothed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Generating trajectory and contour plots...")
    for dims in dim_pairs:
        fig = plot_trajectories(coords, dataset_names, checkpoint_numbers, dims)
        plt.savefig(output_dir / f"trajectories_dims_{dims[0]}_{dims[1]}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        for metric_name in benchmark_scores.keys():
            fig = plot_contour_with_trajectories(
                coords, benchmark_scores, metric_name, dataset_names, checkpoint_numbers, dims
            )
            if fig:
                plt.savefig(
                    output_dir / f"contour_smoothed_{metric_name}_dims_{dims[0]}_{dims[1]}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

    print(f"Generated plots and saved to {output_dir}")


if __name__ == "__main__":
    main()
