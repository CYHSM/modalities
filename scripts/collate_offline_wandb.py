#!/usr/bin/env python
"""Collates final metrics for a wave straight out of the OFFLINE wandb run files.

Compute nodes have no internet, so every arm trains with ``WANDB_MODE=offline`` and leaves a
``run-<id>.wandb`` transaction log under ``<EXPERIMENTS_ROOT>/wandb/wandb/offline-run-*``. This reads
those logs directly rather than going through ``wandb sync`` and the public API, so collation needs
no network, cannot be affected by a partial sync, and gives byte-exact values.

The wandb 0.25 transaction log stores metrics as a stream of incremental ``summary`` records rather
than as one summary blob (``files/wandb-summary.json`` is not written for an offline run that was
not synced). Replaying every ``summary.update`` in order reconstructs the final summary state, whose
values are each metric's last logged value -- for the evaluation metrics that is the step-76,250
evaluation, which is exactly what the paper tables report.

Run it with the wandb venv, which is the one that has the SDK::

    /leonardo_work/EUHPC_D21_101/mfrey/wandb_env/bin/python scripts/collate_offline_wandb.py \
        --arms P0_loop_mamba_at_0 P1_loop_mamba_at_2 ... --output docs/loopotron/position_sweep_stats.json

Pass ``--validate-against docs/loopotron/wave2_final_stats.json`` to re-derive already-published arms
from the same code path and fail if any disagrees. Do that whenever the extraction changes: the
metric keys are not self-describing (``test WeightedSumLoss`` is the held-out LM loss; MATH and
TriviaQA are reported as ``answer_nll``, not as their WeightedSumLoss, because the configured
objective folds in the MoE auxiliary loss and is a mean of per-batch means), and picking the wrong
one produces plausible numbers that are quietly the wrong quantity.
"""

import argparse
import json
import statistics
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

DEFAULT_WANDB_ROOT = Path(
    "/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster/wandb/wandb"
)

# Metric name -> summary key. See the module docstring on why the two benchmarks use answer_nll.
METRICS = {
    "LM": "test WeightedSumLoss",
    "MATH": "minerva_math answer_nll",
    "TQA": "triviaqa answer_nll",
    "p_hop_1_acc": "p_hop_1 answer_accuracy",
    "p_hop_1_nll": "p_hop_1 answer_nll",
    "p_hop_2_nll": "p_hop_2 answer_nll",
    "p_hop_3_nll": "p_hop_3 answer_nll",
    "bind_3_nll": "bind_3 answer_nll",
    "consumed_tokens": "train consumed tokens",
    "samples_per_second": "train train samples/s",
    "grad_norm_avg": "train grad norm avg",
}


def read_offline_run(wandb_file: Path) -> tuple[str, dict]:
    """
    Replays one offline transaction log into its run name and final summary.

    Args:
        wandb_file (Path): Path to a ``run-<id>.wandb`` file.

    Returns:
        tuple[str, dict]: The run's display name and its reconstructed final summary.
    """
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    run_name = None
    summary: dict = {}
    while True:
        try:
            data = datastore.scan_data()
        except Exception:
            break  # A truncated tail is normal for a run killed mid-write; keep what was read.
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        record_type = record.WhichOneof("record_type")
        if record_type == "run" and run_name is None:
            run_name = record.run.display_name or record.run.run_id
        elif record_type == "summary":
            for update in record.summary.update:
                key = ".".join(update.nested_key) if update.nested_key else update.key
                try:
                    summary[key] = json.loads(update.value_json)
                except json.JSONDecodeError:
                    summary[key] = update.value_json
    return run_name, summary


def index_runs(wandb_root: Path) -> dict[str, dict]:
    """
    Builds a run-name -> {summary, path} index over every offline run under ``wandb_root``.

    A run name can appear more than once if an arm was relaunched. The most recently started
    directory wins, and the collision is reported, because silently averaging a discarded run with
    its replacement is exactly the failure Wave 2 spent a week undoing.

    Args:
        wandb_root (Path): Directory holding the ``offline-run-*`` directories.

    Returns:
        dict[str, dict]: Mapping from run name to its summary and source path.
    """
    index: dict[str, dict] = {}
    for run_directory in sorted(wandb_root.glob("offline-run-*")):
        wandb_files = list(run_directory.glob("run-*.wandb"))
        if not wandb_files:
            continue
        run_name, summary = read_offline_run(wandb_files[0])
        if run_name is None:
            continue
        if run_name in index:
            print(f"  ! duplicate run name {run_name}: keeping {run_directory.name}, ignoring earlier")
        index[run_name] = {"summary": summary, "path": str(run_directory)}
    return index


def collect(index: dict[str, dict], arms: list[str]) -> dict:
    """
    Extracts the metrics of interest for each requested arm.

    Args:
        index (dict[str, dict]): Output of :func:`index_runs`.
        arms (list[str]): Run names to extract.

    Returns:
        dict: Per-arm metric values, plus a list of arms not found.
    """
    results = {}
    missing = []
    for arm in arms:
        if arm not in index:
            missing.append(arm)
            continue
        summary = index[arm]["summary"]
        results[arm] = {
            name: summary.get(key) for name, key in METRICS.items()
        }
        results[arm]["_source"] = index[arm]["path"]
    return {"arms": results, "missing": missing}


def validate(index: dict[str, dict], reference_path: Path) -> int:
    """
    Re-derives published arms through this extractor and compares against the recorded figures.

    Args:
        index (dict[str, dict]): Output of :func:`index_runs`.
        reference_path (Path): A ``*_final_stats.json`` written by an earlier collation.

    Returns:
        int: Number of disagreements found.
    """
    reference = json.loads(reference_path.read_text())
    disagreements = 0
    print(f"\nValidating extraction against {reference_path.name}:")
    for arm, block in reference.items():
        for metric in ("LM", "MATH", "TQA"):
            if metric not in block:
                continue
            recorded = block[metric]["vals"]
            derived = []
            for run_name in block["runs"]:
                if run_name in index:
                    value = index[run_name]["summary"].get(METRICS[metric])
                    if value is not None:
                        derived.append(value)
            if len(derived) != len(recorded):
                print(f"  {arm:4s} {metric:5s} SKIP  ({len(derived)}/{len(recorded)} runs found offline)")
                continue
            worst = max(abs(a - b) for a, b in zip(sorted(derived), sorted(recorded)))
            status = "ok" if worst < 1e-6 else "MISMATCH"
            if worst >= 1e-6:
                disagreements += 1
            print(f"  {arm:4s} {metric:5s} {status:9s} max|Δ| = {worst:.2e}  (n={len(derived)})")
    return disagreements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, help="Run names (experiment ids) to collate.")
    parser.add_argument("--wandb-root", type=Path, default=DEFAULT_WANDB_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-against", type=Path, default=None)
    arguments = parser.parse_args()

    print(f"Scanning {arguments.wandb_root} ...")
    index = index_runs(arguments.wandb_root)
    print(f"Indexed {len(index)} offline runs.")

    if arguments.validate_against:
        disagreements = validate(index, arguments.validate_against)
        if disagreements:
            raise SystemExit(f"\n{disagreements} metric(s) disagree with the reference; extraction is wrong.")
        print("  -> extraction reproduces the reference exactly.")

    collected = collect(index, arguments.arms)
    if collected["missing"]:
        print(f"\n! not found offline: {', '.join(collected['missing'])}")

    print()
    header = f"{'arm':24s} {'LM':>9s} {'MATH':>8s} {'TQA':>8s} {'p_hop_1':>8s} {'tokens':>14s} {'samp/s':>7s}"
    print(header)
    print("-" * len(header))
    for arm, values in collected["arms"].items():
        print(
            f"{arm:24s} {values['LM']:9.4f} {values['MATH']:8.4f} {values['TQA']:8.4f} "
            f"{values['p_hop_1_acc']:8.4f} {int(values['consumed_tokens']):14d} {values['samples_per_second']:7.1f}"
        )

    lm_values = [v["LM"] for v in collected["arms"].values()]
    if len(lm_values) > 1:
        print(
            f"\nLM loss: min {min(lm_values):.4f}  max {max(lm_values):.4f}  "
            f"spread {max(lm_values) - min(lm_values):.4f}  sd {statistics.stdev(lm_values):.4f}"
        )

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(collected, indent=1) + "\n")
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
