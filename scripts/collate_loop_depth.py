#!/usr/bin/env python
"""Collates the Phase 0 loop-depth results into the position table.

The question is whether a loop group's extra iterations are worth more early in the stack than late.
The decisive comparison is **within an arm**: every loop group in one arm loops the same operator, so
comparing groups against each other holds operator class fixed by construction -- which is exactly
what Wave 2's cross-arm ranking could not do.

Run from the repository root::

    python scripts/collate_loop_depth.py
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]

ARM_LABELS = {
    "A1": "Mamba",
    "A2": "MoE",
    "A3": "attention",
    "A4": "Mamba+MoE",
    "A5": "Mamba+attn",
    "A6": "attn+MoE",
}


def _family(run_name: str) -> str:
    return re.match(r"^([AN]\d+)", run_name).group(1)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def _spearman(first: list[float], second: list[float]) -> float:
    """Rank correlation, with ties averaged."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rank_first, rank_second = ranks(first), ranks(second)
    mean_first, mean_second = statistics.fmean(rank_first), statistics.fmean(rank_second)
    numerator = sum((a - mean_first) * (b - mean_second) for a, b in zip(rank_first, rank_second))
    denominator = (
        sum((a - mean_first) ** 2 for a in rank_first) * sum((b - mean_second) ** 2 for b in rank_second)
    ) ** 0.5
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_depth")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_depth.json")
    arguments = parser.parse_args()

    runs = defaultdict(list)
    for path in sorted(arguments.input_directory.glob("*.json")):
        runs[_family(path.stem)].append(json.loads(path.read_text()))

    print("\n=== Phase 0: what are a loop group's extra iterations worth, by position? ===\n")
    print("delta = paired loss increase (nats) when that group alone is reduced to K=1,")
    print("on identical tokens. Positive = those iterations were doing useful work.")
    print("Within one arm every group loops the SAME operator, so position is the only variable.\n")

    print(f"{'arm':4s} {'loops':12s} {'@exec':>6s} {'K':>3s} {'delta':>18s} {'s.e.':>8s}")
    print("-" * 60)
    collated = {}
    all_positions, all_deltas = [], []
    for family in sorted(runs):
        seeds = runs[family]
        entries = []
        for position in range(len(seeds[0]["per_group_ablation"])):
            group = seeds[0]["per_group_ablation"][position]
            deltas = [seed["per_group_ablation"][position]["delta"] for seed in seeds]
            mean, sd = _mean_sd(deltas)
            # Two independent uncertainties: spread across seeds, and the within-run sampling error
            # of the paired estimate. Report the larger, so the table never looks tighter than it is.
            sampling = statistics.fmean(
                seed["per_group_ablation"][position]["standard_error"] for seed in seeds
            )
            entries.append({**group, "delta_mean": mean, "delta_sd_across_seeds": sd, "sampling_error": sampling})
            spread = f"{mean:+.4f}±{sd:.4f}" if len(seeds) > 1 else f"{mean:+.4f}"
            print(
                f"{family:4s} {ARM_LABELS.get(family, '?'):12s} {group['executed_position']:6d} "
                f"{group['num_loops']:3d} {spread:>18s} {sampling:8.4f}"
            )
            all_positions.append(group["executed_position"])
            all_deltas.append(mean)

        within = (
            _spearman([entry["executed_position"] for entry in entries], [entry["delta_mean"] for entry in entries])
            if len(entries) > 1
            else float("nan")
        )
        collated[family] = {
            "n_seeds": len(seeds),
            "layer_pattern": seeds[0]["layer_pattern"],
            "baseline_loss": statistics.fmean(seed["baseline_loss"] for seed in seeds),
            "per_group": entries,
            "within_arm_spearman_position_vs_delta": within,
            "global_depth_sweep": [
                {
                    "num_loops": point["num_loops"],
                    "n_executed_layers": point["n_executed_layers"],
                    "delta_mean": statistics.fmean(
                        seed["global_depth_sweep"][index]["delta"] for seed in seeds
                    ),
                }
                for index, point in enumerate(seeds[0]["global_depth_sweep"])
            ],
        }
        if len(entries) > 1:
            print(f"{'':4s} {'':12s} within-arm Spearman(position, delta) = {within:+.3f}\n")

    print("--- pooled across all arms and groups ---")
    print(f"Spearman(executed position, delta) = {_spearman(all_positions, all_deltas):+.3f}  "
          f"(n={len(all_positions)} groups)")
    print("Negative = early groups' iterations are worth more, which is the position hypothesis.")

    print("\n--- global depth sweep: loss delta vs uniform K (trained K in parentheses) ---")
    counts = [point["num_loops"] for point in collated[sorted(collated)[0]]["global_depth_sweep"]]
    print(f"{'arm':4s} {'trained':>8s} " + " ".join(f"{'K=' + str(k):>9s}" for k in counts))
    for family in sorted(collated):
        trained = runs[family][0]["trained_num_loops"]
        cells = " ".join(f"{point['delta_mean']:+9.4f}" for point in collated[family]["global_depth_sweep"])
        print(f"{family:4s} {trained:8d} {cells}")

    arguments.output.write_text(json.dumps(collated, indent=1))
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
