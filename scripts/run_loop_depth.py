#!/usr/bin/env python
"""Measures what each loop group's extra iterations are worth, for one trained Wave 2 run.

Loads the run's checkpoint, evaluates it on a fixed token set, then re-evaluates it with loop
iterations removed -- once per looped group, and once per point of a global depth sweep -- and reports
every result as a **paired** difference against the unmodified model on the same tokens.

Nothing is trained. See `src/modalities/analysis/loop_depth.py` for what the numbers do and do not
support: removing iterations puts the model off its training distribution, so the magnitudes overstate
the causal value of those iterations and only the ordering across positions is worth reading.

Run under one GPU, from the repository root::

    python scripts/run_loop_depth.py --arm A1_loop_mamba \\
        --experiments-root /leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster

`scripts/run_loop_depth.sh` sweeps every run of the wave.
"""

import argparse
import json
from pathlib import Path

import torch

from modalities.analysis.checkpoints import arm_config_path, fixed_evaluation_batch, load_arm
from modalities.analysis.loop_depth import (
    group_executed_positions,
    looped_group_indices,
    override_loop_counts,
    paired_delta,
    per_token_losses,
)

REPOSITORY_ROOT = Path(__file__).parents[1]

# 128 sequences x 2048 tokens = 262,144 tokens. Chosen against measurement noise, not convenience: an
# 8-sequence batch moved by 0.14 nats across seeds in the update diagnostic, because 8 sequences is
# effectively 8 samples. The effects here are ~0.01 nats, and are read as paired differences.
NUM_SEQUENCES = 128
MICRO_BATCH_SIZE = 8

# Depth sweep either side of the trained value. Wave 2 arms train at K in {2, 3, 4}.
GLOBAL_LOOP_COUNTS = (1, 2, 3, 4, 6, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_depth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    config_path = arm_config_path(arguments.arm)
    checkpoint_info = arguments.experiments_root / arguments.arm / "checkpoints" / "last_checkpoint_info.json"
    checkpoint_directory = Path(json.loads(checkpoint_info.read_text())["checkpoint_folder_path"])

    model = load_arm(arguments.arm, checkpoint_directory, device)
    samples = fixed_evaluation_batch(config_path, model.sample_key, NUM_SEQUENCES).to(device)

    positions = group_executed_positions(model)
    looped = looped_group_indices(model)
    print(
        f"[{arguments.arm}] pattern={model.layer_pattern} executed={model.n_executed_layers} "
        f"looped groups={len(looped)} at executed positions {[positions[i] for i in looped]}",
        flush=True,
    )

    # The baseline goes through the same override machinery as every ablation (with num_loops=None),
    # so that any difference between the two paths would show up as a nonzero self-delta rather than
    # as a constant offset hidden in every reported number.
    with override_loop_counts(model, num_loops=None):
        baseline = per_token_losses(model, samples, MICRO_BATCH_SIZE)
    print(f"[{arguments.arm}] baseline loss {baseline.mean().item():.4f}", flush=True)

    per_group = []
    for schedule_index in looped:
        group = model._schedule[schedule_index]
        with override_loop_counts(model, num_loops=1, group_index=schedule_index):
            ablated = per_token_losses(model, samples, MICRO_BATCH_SIZE)
        result = paired_delta(baseline, ablated)
        per_group.append(
            {
                "schedule_index": schedule_index,
                "executed_position": positions[schedule_index],
                "num_loops": group.num_loops,
                "num_member_layers": len(group.layer_keys),
                **result,
            }
        )
        print(
            f"[{arguments.arm}]   group @exec {positions[schedule_index]:>2d} (K={group.num_loops}) "
            f"-> K=1: delta {result['delta']:+.4f} ± {result['standard_error']:.4f}",
            flush=True,
        )

    depth_sweep = []
    for num_loops in GLOBAL_LOOP_COUNTS:
        with override_loop_counts(model, num_loops=num_loops) as schedule:
            executed = sum(group.num_executed_layers for group in schedule)
            losses = per_token_losses(model, samples, MICRO_BATCH_SIZE)
        result = paired_delta(baseline, losses)
        depth_sweep.append({"num_loops": num_loops, "n_executed_layers": executed, **result})
        print(
            f"[{arguments.arm}]   all groups K={num_loops} ({executed} executed): "
            f"delta {result['delta']:+.4f} ± {result['standard_error']:.4f}",
            flush=True,
        )

    report = {
        "arm": arguments.arm,
        "config": config_path.name,
        "checkpoint": str(checkpoint_directory),
        "layer_pattern": model.layer_pattern,
        "n_executed_layers": model.n_executed_layers,
        "trained_num_loops": max((model._schedule[i].num_loops for i in looped), default=1),
        "evaluation": {"num_sequences": NUM_SEQUENCES, "sequence_length": int(model.sequence_length)},
        "baseline_loss": baseline.mean().item(),
        "per_group_ablation": per_group,
        "global_depth_sweep": depth_sweep,
    }
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_directory / f"{arguments.arm}.json"
    output_path.write_text(json.dumps(report, indent=1))
    print(f"[{arguments.arm}] -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
