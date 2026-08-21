#!/usr/bin/env python
"""Tests whether a baseline's per-layer update norms predict where looping pays off.

THE CLAIM UNDER TEST
--------------------
The position sweep (docs/loopotron/position_sweep.md) trained twelve arms to learn where extra depth
is worth spending: one layer looped at K=6, at each of the twelve positions the base pattern offers.
That cost twelve runs of ~11-13h each.

Three orderings coincided in that wave -- loop benefit, position sensitivity, and residual-stream
update magnitude -- suggesting a single underlying quantity. If that reading is right, the answer
should be readable from the UNLOOPED baseline alone: layer i's own-input-relative update norm,
measured on A0_baseline in one forward pass, should predict the gain from looping layer i.

If it holds, the twelve-run sweep is replaced by one forward pass over one checkpoint the study
already had. If it fails, the coincidence of orderings was descriptive and nothing more.

    predictor  x_i = A0's layer-i update norm, relative to that layer's own input (per-token median)
    outcome    y_i = A0 loss - (loss of the arm that loops layer i at K=6)

Both are indexed by BUILT layer index, 0..11, which is what makes them commensurable: A0 has exactly
one layer per index, and the sweep has exactly one arm per index.

WHAT IS PRE-SPECIFIED
---------------------
The hypothesis is directional -- larger update norm predicts larger gain -- so a positive rank
correlation supports it and a negative one refutes it. The primary statistic is the pooled Spearman
over all twelve positions. `relative_to_own_input` is the primary predictor because it is the
normalization the earlier `member_step_norms` finding used; `absolute_update_norm` and the
stack-relative ratio are reported as robustness, not as alternatives to pick from after the fact.

The practically important statistic is not a correlation at all but TOP-1 SELECTION: if you had only
the baseline and had to choose one layer to loop, would the predictor choose the one that actually
won? A heuristic can rank imperfectly and still be useful if it picks the right winner.

Run from the repository root (no GPU needed; reads the JSON the diagnostic wrote)::

    python scripts/analyze_update_norm_predictor.py
"""

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
FAMILIES = {"P": "Mamba", "Q": "MoE", "R": "attention"}
SEED_SD = 0.0021  # loopotron.tex, Wave 3: four runs of A1


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, no scipy dependency."""

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                ranks[order[position]] = average
            index = stop + 1
        return ranks

    rank_x, rank_y = rank(x), rank(y)
    n = len(x)
    mean_x, mean_y = sum(rank_x) / n, sum(rank_y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    var_x = sum((a - mean_x) ** 2 for a in rank_x) ** 0.5
    var_y = sum((b - mean_y) ** 2 for b in rank_y) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else float("nan")


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x) ** 0.5
    var_y = sum((b - mean_y) ** 2 for b in y) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=REPOSITORY_ROOT / "docs/loopotron/loop_updates/A0_baseline.json")
    parser.add_argument("--sweep", type=Path, default=REPOSITORY_ROOT / "docs/loopotron/position_sweep_stats.json")
    parser.add_argument("--wave2", type=Path, default=REPOSITORY_ROOT / "docs/loopotron/wave2_final_stats.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "docs/loopotron/update_norm_predictor.json")
    arguments = parser.parse_args()

    profile_report = json.loads(arguments.profile.read_text())
    if "layer_profile" not in profile_report:
        raise SystemExit(
            f"{arguments.profile} predates the layer_profile addition; re-run scripts/run_layer_profile.sh"
        )
    # A0 is unlooped, so execution order IS built order and step index IS built index.
    if profile_report["n_built_layers"] != profile_report["n_executed_layers"]:
        raise SystemExit("predictor must be measured on an UNLOOPED baseline, where step == built index")

    profile = {entry["step"]: entry for entry in profile_report["layer_profile"]}
    sweep = json.loads(arguments.sweep.read_text())["arms"]
    a0_loss = json.loads(arguments.wave2.read_text())["A0"]["LM"]["mean"]

    rows = []
    for arm_name, values in sweep.items():
        built_index = int(arm_name.rsplit("_at_", 1)[1])
        entry = profile[built_index]
        rows.append(
            {
                "arm": arm_name,
                "family": arm_name[0],
                "operator": FAMILIES[arm_name[0]],
                "built_index": built_index,
                "layer_type": entry["layer_type"],
                "predictor_relative": entry["relative_to_own_input"]["median"],
                "predictor_absolute": entry["absolute_update_norm"]["median"],
                "input_norm": entry["input_norm"]["median"],
                "loop_loss": values["LM"],
                "gain": a0_loss - values["LM"],
            }
        )
    rows.sort(key=lambda row: row["built_index"])

    print(f"A0 baseline loss {a0_loss:.4f}; predictor from {arguments.profile.name}\n")
    header = f"{'idx':>3} {'type':>4} {'arm':24s} {'upd/own':>8} {'|upd|':>8} {'loop LM':>8} {'gain':>8} {'gain sd':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['built_index']:3d} {row['layer_type']:>4} {row['arm']:24s} "
            f"{row['predictor_relative']:8.4f} {row['predictor_absolute']:8.3f} "
            f"{row['loop_loss']:8.4f} {row['gain']:+8.4f} {row['gain']/SEED_SD:+8.1f}"
        )

    predictor = [row["predictor_relative"] for row in rows]
    gain = [row["gain"] for row in rows]

    print("\n=== PRIMARY: pooled over all 12 positions ===")
    print(f"  Spearman(update norm, gain) = {spearman(predictor, gain):+.3f}")
    print(f"  Pearson (update norm, gain) = {pearson(predictor, gain):+.3f}")
    print(f"  robustness, absolute norm   = {spearman([r['predictor_absolute'] for r in rows], gain):+.3f} (Spearman)")

    print("\n=== within family (position effect, operator held fixed) ===")
    for family, operator in FAMILIES.items():
        subset = [row for row in rows if row["family"] == family]
        if len(subset) < 3:
            print(f"  {operator:10s} n={len(subset)}, too few for a rank correlation")
            continue
        print(
            f"  {operator:10s} n={len(subset)}  Spearman = "
            f"{spearman([r['predictor_relative'] for r in subset], [r['gain'] for r in subset]):+.3f}"
        )

    print("\n=== across operators (operator effect, averaged over position) ===")
    for family, operator in FAMILIES.items():
        subset = [row for row in rows if row["family"] == family]
        mean_predictor = sum(r["predictor_relative"] for r in subset) / len(subset)
        mean_gain = sum(r["gain"] for r in subset) / len(subset)
        print(f"  {operator:10s} mean update norm {mean_predictor:7.4f}   mean gain {mean_gain:+.4f}")

    print("\n=== TOP-1 SELECTION: pick one layer to loop, knowing only the baseline ===")
    chosen = max(rows, key=lambda row: row["predictor_relative"])
    actual = max(rows, key=lambda row: row["gain"])
    ranked = sorted(rows, key=lambda row: -row["gain"])
    rank_of_choice = ranked.index(chosen) + 1
    print(f"  predictor picks : {chosen['arm']} (idx {chosen['built_index']}, {chosen['layer_type']})")
    print(f"  actual best     : {actual['arm']} (idx {actual['built_index']}, {actual['layer_type']})")
    print(f"  -> the pick ranks {rank_of_choice} of {len(rows)} by true gain; "
          f"regret {actual['gain'] - chosen['gain']:+.4f} nats "
          f"({(actual['gain'] - chosen['gain'])/SEED_SD:.1f} s.d.)")

    result = {
        "a0_loss": a0_loss,
        "predictor_source": str(arguments.profile),
        "rows": rows,
        "spearman_pooled": spearman(predictor, gain),
        "pearson_pooled": pearson(predictor, gain),
        "top1_choice": chosen["arm"],
        "top1_actual": actual["arm"],
        "top1_rank_of_choice": rank_of_choice,
        "top1_regret_nats": actual["gain"] - chosen["gain"],
    }
    arguments.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
