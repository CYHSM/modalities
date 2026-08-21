#!/usr/bin/env python
"""Plots held-out LM loss against looped position for the loop-placement sweep.

Twelve arms, one per layer of the 12-layer base pattern ``MEM*EMEMEM*E``. Each arm builds that
one layer and loops it x6; every arm is otherwise identical (12 built / 17 executed layers). This
is an exhaustive sweep: every position in the stack was looped exactly once, so the 12 points
below tile the whole x-axis with no gaps.

Reads directly from docs/loopotron/position_sweep_stats.json and wave2_final_stats.json (for the
A0 baseline and its seed s.d.) -- no numbers are retyped here.

Usage:
    python scripts/plot_position_sweep.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "loopotron"

BASE_PATTERN = "MEM*EMEMEM*E"
FAMILY_OF_LETTER = {"M": "mamba", "E": "moe", "*": "attention"}
COLORS = {"mamba": "#2a78d6", "moe": "#eb6834", "attention": "#1baf7a"}
LABELS = {"mamba": "Mamba", "moe": "MoE", "attention": "Attention"}
ARM_PREFIX_TO_FAMILY = {"P": "mamba", "Q": "moe", "R": "attention"}
SEED_SD = 0.0021  # measured from 3 seeds of A1 (looping every Mamba layer); see loopotron.tex


def load_data() -> tuple[float, dict[str, list[tuple[int, str, float]]]]:
    sweep = json.load(open(DOCS_DIR / "position_sweep_stats.json"))["arms"]
    wave2 = json.load(open(DOCS_DIR / "wave2_final_stats.json"))
    baseline_loss = wave2["A0"]["LM"]["mean"]

    by_family: dict[str, list[tuple[int, str, float]]] = {"mamba": [], "moe": [], "attention": []}
    for arm_name, values in sweep.items():
        family = ARM_PREFIX_TO_FAMILY[arm_name[0]]
        built_index = int(arm_name.split("_at_")[1])
        by_family[family].append((built_index, arm_name.split("_loop_")[0], values["LM"]))
    for points in by_family.values():
        points.sort(key=lambda p: p[0])
    return baseline_loss, by_family


def main() -> None:
    baseline_loss, by_family = load_data()

    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=150)
    ax.set_ylim(2.488, 2.575)

    # Baseline: no loop, 12 built = 12 executed. Shaded band is +/- one seed s.d.
    ax.axhspan(baseline_loss - SEED_SD, baseline_loss + SEED_SD, color="0.85", zorder=0)
    ax.axhline(baseline_loss, color="0.4", linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        "A0 baseline (no loop)",
        xy=(11.2, baseline_loss),
        fontsize=9,
        color="0.35",
        va="bottom",
    )

    for family, points in by_family.items():
        xs = [p[0] for p in points]
        ys = [p[2] for p in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2,
            color=COLORS[family],
            label=LABELS[family],
            zorder=3,
        )

    # Callouts on the two headline points.
    best = min(by_family["mamba"], key=lambda p: p[2])
    worst = max(by_family["mamba"], key=lambda p: p[2])
    ax.annotate(
        f"best position\n{(best[2] - baseline_loss) / SEED_SD:+.1f} s.d. vs baseline",
        xy=(best[0], best[2]),
        xytext=(best[0] + 0.9, best[2] - 0.011),
        fontsize=8.5,
        color=COLORS["mamba"],
        arrowprops=dict(arrowstyle="-", color=COLORS["mamba"], linewidth=0.8),
    )
    ax.annotate(
        "worse than\nnot looping at all",
        xy=(worst[0], worst[2]),
        xytext=(worst[0] + 0.5, worst[2] + 0.006),
        fontsize=8.5,
        color=COLORS["mamba"],
        arrowprops=dict(arrowstyle="-", color=COLORS["mamba"], linewidth=0.8),
    )

    # X axis: built index 0..11, with the base-pattern letter for each position.
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{i}\n{BASE_PATTERN[i]}" for i in range(12)])
    ax.set_xlim(-0.6, 12.6)
    ax.set_xlabel("position in the stack (built layer index)", fontsize=10, labelpad=8)
    ax.set_ylabel("held-out LM loss  (lower is better)", fontsize=10)

    ax.set_title(
        "Looping one layer x6 -- where you put it changes the loss by up to 24 seed s.d.",
        fontsize=11,
        pad=14,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)

    fig.tight_layout()
    out_path = DOCS_DIR / "position_sweep.png"
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
