#!/usr/bin/env python
"""Collates per-run loop-update JSONs into the comparison table.

Reads what `run_loop_updates.py` wrote for each Wave 2 run, aggregates the seeds of each arm, and
prints the table alongside the arm's published loss so the two can be read together.

Aggregation follows the shape of the data rather than flattening it:

* Within a run, each per-token quantity is summarized by its **median**, which is robust to the few
  very-high-norm tokens (attention sinks) that would otherwise drag a mean around.
* Across the iterations of one group, the medians are averaged -- a group with K iterations
  contributes K-1 consecutive cosines, and they are one trajectory.
* Across the groups of one arm, results are reported **per group as well as pooled**. Pooling alone
  would mislead: A1 has five loop groups spread through the stack while A3 has two, so an arm-level
  average weights early-stack behaviour differently in the two arms.
* Across seeds, mean and standard deviation.

Run from the repository root::

    python scripts/collate_loop_updates.py
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]

# Arm family -> what that arm loops, for the table's leftmost columns.
ARM_LABELS = {
    "A0": "none (baseline)",
    "A1": "Mamba",
    "A2": "MoE",
    "A3": "attention",
    "A4": "Mamba+MoE",
    "A5": "Mamba+attn",
    "A6": "attn+MoE",
    "N1": "none (+Mamba layers)",
    "N2": "none (+MoE layers)",
    "N3": "none (+attn layers)",
    "N4": "none (+attn/MoE layers)",
}


def _family(run_name: str) -> str:
    """
    Maps a run name onto its arm family, e.g. ``A4_loop_mamba_moe_seed2_redo`` -> ``A4``.

    Args:
        run_name (str): The run name.

    Returns:
        str: The family key.
    """
    return re.match(r"^([AN]\d+)", run_name).group(1)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """
    Mean and population standard deviation of a seed group.

    Args:
        values (list[float]): One value per seed.

    Returns:
        tuple[float, float]: Mean, and standard deviation (0.0 for a single seed).
    """
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def _format(mean: float, sd: float, n: int, digits: int = 3) -> str:
    """Renders a seed-aggregated value, showing spread only where there is more than one seed."""
    return f"{mean:.{digits}f}" if n < 2 else f"{mean:.{digits}f}±{sd:.{digits}f}"


def _run_summary(report: dict) -> dict:
    """
    Reduces one run's report to per-group and pooled scalars.

    Args:
        report (dict): The JSON written by `run_loop_updates.py`.

    Returns:
        dict: Per-group cosine/step-norm medians and their pooled averages.
    """
    groups = []
    for group in report["groups"]:
        cosines = [entry["median"] for entry in group["update_cosine"]]
        step_norms = [entry["median"] for entry in group["relative_step_norm"]]
        groups.append(
            {
                "first_executed_index": group["first_executed_index"],
                "composition": group["composition"],
                "num_loops": group["num_loops"],
                # Carried alongside the ratios because it is their denominator: the residual stream
                # grows with depth, so an early group's ratio is inflated relative to a late one's.
                "group_input_norm": group["group_input_norm"]["median"],
                "update_cosine": statistics.fmean(cosines),
                "update_cosine_per_pair": cosines,
                "relative_step_norm": statistics.fmean(step_norms),
                "relative_step_norm_per_iteration": step_norms,
                "first_to_last_step_ratio": (step_norms[-1] / step_norms[0]) if step_norms[0] else float("nan"),
                "between_member_cosine": (
                    statistics.fmean(
                        entry["median"]
                        for pair in group["members"]["between_member_cosine"]
                        for entry in pair["per_iteration"]
                    )
                    if group.get("members")
                    else None
                ),
                "member_types": (
                    [entry["layer_type"] for entry in group["members"]["per_member_relative_norm"]]
                    if group.get("members")
                    else None
                ),
                "member_step_norms": (
                    [
                        statistics.fmean(item["median"] for item in entry["per_iteration"])
                        for entry in group["members"]["per_member_relative_norm"]
                    ]
                    if group.get("members")
                    else None
                ),
            }
        )

    stack_cosines = [entry["median"] for entry in report["stack"]["update_cosine"]]
    return {
        "arm": report["arm"],
        "n_executed_layers": report["n_executed_layers"],
        "sanity_batch_loss": report["sanity_batch_loss"],
        "groups": groups,
        "pooled_update_cosine": statistics.fmean(g["update_cosine"] for g in groups) if groups else None,
        "pooled_relative_step_norm": statistics.fmean(g["relative_step_norm"] for g in groups) if groups else None,
        "stack_update_cosine": statistics.fmean(stack_cosines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_updates")
    parser.add_argument("--stats", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "wave2_final_stats.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_updates.json")
    arguments = parser.parse_args()

    losses = json.loads(arguments.stats.read_text())
    summaries = defaultdict(list)
    for path in sorted(arguments.input_directory.glob("*.json")):
        summaries[_family(path.stem)].append(_run_summary(json.loads(path.read_text())))

    print("\n=== Loop-update diagnostic: does each iteration do different work? ===\n")
    print("Update cosine: 1.0 = successive iterations push the same direction (redundant),")
    print("0.0 = orthogonal work, negative = later passes partly undo earlier ones.")
    print("Step norm: median ||delta_k|| / ||h_0||, averaged over a group's iterations.\n")
    header = (
        f"{'arm':4s} {'loops':22s} {'exec':>4s} {'LM loss':>16s} "
        f"{'update cosine':>16s} {'step norm':>16s} {'stack cosine':>16s}"
    )
    print(header)
    print("-" * len(header))

    collated = {}
    for family in sorted(summaries, key=lambda key: (key[0], int(key[1:]))):
        runs = summaries[family]
        n = len(runs)
        loss = losses.get(family, {}).get("LM", {})
        loss_text = "  —" if loss.get("mean") is None else _format(loss["mean"], loss.get("sd") or 0.0, n, digits=4)

        stack_mean, stack_sd = _mean_sd([run["stack_update_cosine"] for run in runs])
        if runs[0]["groups"]:
            cosine_mean, cosine_sd = _mean_sd([run["pooled_update_cosine"] for run in runs])
            norm_mean, norm_sd = _mean_sd([run["pooled_relative_step_norm"] for run in runs])
            cosine_text = _format(cosine_mean, cosine_sd, n)
            norm_text = _format(norm_mean, norm_sd, n)
        else:
            cosine_text = norm_text = "  —"

        print(
            f"{family:4s} {ARM_LABELS.get(family, '?'):22s} {runs[0]['n_executed_layers']:4d} "
            f"{loss_text:>16s} {cosine_text:>16s} {norm_text:>16s} "
            f"{_format(stack_mean, stack_sd, n):>16s}"
        )
        collated[family] = {"n_seeds": n, "runs": runs}

    # Per group, seeds aggregated. Groups sit at different depths in different arms, and the trend
    # with depth runs in opposite directions across arms, so the pooled column above hides the
    # structure that matters.
    print("\n--- per group, by depth (cos = mean consecutive-update cosine, seeds aggregated) ---")
    print(f"{'arm':4s} {'@exec [comp]^K':>16s} {'cos':>15s} {'step/||h0||':>15s} {'||h0||':>9s}")
    for family in sorted(summaries, key=lambda key: (key[0], int(key[1:]))):
        runs = summaries[family]
        for position in range(len(runs[0]["groups"])):
            group = runs[0]["groups"][position]
            cosine_mean, cosine_sd = _mean_sd([run["groups"][position]["update_cosine"] for run in runs])
            norm_mean, norm_sd = _mean_sd([run["groups"][position]["relative_step_norm"] for run in runs])
            input_norm, _ = _mean_sd([run["groups"][position]["group_input_norm"] for run in runs])
            label = f"@{group['first_executed_index']:>2d} [{group['composition']}]^{group['num_loops']}"
            print(
                f"{family:4s} {label:>16s} {_format(cosine_mean, cosine_sd, len(runs)):>15s} "
                f"{_format(norm_mean, norm_sd, len(runs)):>15s} {input_norm:9.1f}"
            )

    # Cross-arm comparison, made legitimate by bucketing on depth. Comparing arm-level averages
    # directly is invalid: A1 has five groups starting at executed index 0, A3 has two starting at 3,
    # so an arm average mixes "what this operator does when looped" with "where its groups happen to
    # sit". Buckets are on the group's first executed index, in thirds of a ~18-layer stack.
    print("\n--- matched-depth comparison: mean update cosine by stack position ---")
    buckets = {"early (0-5)": (0, 5), "mid (6-13)": (6, 13), "late (14+)": (14, 10**6)}
    print(f"{'arm':4s} " + " ".join(f"{name:>15s}" for name in buckets))
    for family in sorted(summaries, key=lambda key: (key[0], int(key[1:]))):
        runs = summaries[family]
        if not runs[0]["groups"]:
            continue
        cells = []
        for low, high in buckets.values():
            per_seed = [
                statistics.fmean(
                    group["update_cosine"] for group in run["groups"] if low <= group["first_executed_index"] <= high
                )
                for run in runs
                if any(low <= group["first_executed_index"] <= high for group in run["groups"])
            ]
            cells.append(_format(*_mean_sd(per_seed), len(per_seed)) if per_seed else "—")
        print(f"{family:4s} " + " ".join(f"{cell:>15s}" for cell in cells))

    member_lines = [
        (family, group)
        for family in sorted(summaries)
        for group in summaries[family][0]["groups"]
        if group["between_member_cosine"] is not None
    ]
    if member_lines:
        print("\n--- multi-operator groups: are the members complementary or cancelling? ---")
        print("(negative between-member cosine = the two layers partly undo each other)")
        for family, group in member_lines:
            types = "/".join(group["member_types"])
            norms = "/".join(f"{value:.3f}" for value in group["member_step_norms"])
            print(
                f"{family:4s} @{group['first_executed_index']:>2d}[{group['composition']}]^{group['num_loops']} "
                f"members={types:7s} step_norms={norms:15s} between_member_cos={group['between_member_cosine']:+.3f}"
            )

    arguments.output.write_text(json.dumps(collated, indent=1))
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
