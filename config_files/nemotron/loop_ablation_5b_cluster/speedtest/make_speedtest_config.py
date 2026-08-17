#!/usr/bin/env python
"""Builds a short, throughput-measuring variant of config_A2_loop_moe.yaml.

Used once, interactively, to pick the dp_degree/GPU count for the 5B-token wave -- see
docs/components/nemotron_loops_research_plan.md section 9. Not part of the training pipeline
itself: num_target_steps is small on purpose (just enough to get past warmup and CUDA graph /
kernel-cache warmup) and the run is timed end to end from wall-clock timestamps around the job,
not from an internal metric, so it needs no log-scraping to be trustworthy.

Global batch is held at 65,536 tokens/step (local_micro_batch_size x gradient_accumulation_steps x
dp_degree x 2048) for every variant, so this measures pure data-parallel scaling efficiency, not a
different optimization problem -- see the base config's step_profile comment.

Usage:
    python make_speedtest_config.py --dp 4 --strategy replicate --micro 8 --accum 1 --out /path/to/out.yaml
    python make_speedtest_config.py --dp 16 --strategy shard --micro 2 --accum 1 --out /path/to/out.yaml
"""
import argparse
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_CONFIG = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster/config_A2_loop_moe.yaml"

NUM_TEST_STEPS = 120


def sub_once(text: str, pattern: str, replacement: str, name: str) -> str:
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"expected exactly one match for {name}, found {n}")
    return new_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, required=True, help="dp_degree (total GPUs)")
    parser.add_argument("--strategy", choices=["replicate", "shard"], required=True)
    parser.add_argument("--micro", type=int, required=True, help="local_train_micro_batch_size")
    parser.add_argument("--accum", type=int, required=True, help="gradient_accumulation_steps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    global_batch_tokens = args.micro * args.accum * args.dp * 2048
    if global_batch_tokens != 65536:
        raise SystemExit(f"micro*accum*dp*2048 = {global_batch_tokens}, must be 65536")

    text = SOURCE_CONFIG.read_text()

    text = sub_once(text, r"^    gradient_accumulation_steps: \d+$", f"    gradient_accumulation_steps: {args.accum}", "accum")
    text = sub_once(text, r"^    local_train_micro_batch_size: \d+$", f"    local_train_micro_batch_size: {args.micro}", "micro")
    text = sub_once(text, r"^    num_target_tokens: \d+$", f"    num_target_tokens: {NUM_TEST_STEPS * 65536}", "num_target_tokens")
    text = sub_once(text, r"^    num_target_steps: \d+$", f"    num_target_steps: {NUM_TEST_STEPS}", "num_target_steps")
    text = sub_once(text, r"^    warmup_steps: \d+$", "    warmup_steps: 5", "warmup_steps")
    text = sub_once(text, r"^    checkpointing_interval_in_steps: \d+$", f"    checkpointing_interval_in_steps: {NUM_TEST_STEPS + 1}", "checkpointing_interval_in_steps")
    text = sub_once(text, r"^    evaluation_interval_in_steps: \d+$", f"    evaluation_interval_in_steps: {NUM_TEST_STEPS + 1}", "evaluation_interval_in_steps")
    text = sub_once(text, r"^    project: .*$", "    project: modalities_nemotron_speedtest", "wandb project")

    if args.strategy == "replicate":
        text = sub_once(text, r"^    data_parallel_replicate_degree: 1$", f"    data_parallel_replicate_degree: {args.dp}", "replicate_degree")
        text = sub_once(text, r"^    data_parallel_shard_degree: -1$", "    data_parallel_shard_degree: 1", "shard_degree")
    else:
        text = sub_once(text, r"^    data_parallel_replicate_degree: 1$", "    data_parallel_replicate_degree: 1", "replicate_degree (noop)")
        text = sub_once(text, r"^    data_parallel_shard_degree: -1$", f"    data_parallel_shard_degree: {args.dp}", "shard_degree")
        text = sub_once(
            text,
            r"^    layers_per_fsdp_unit: 1$",
            "    layers_per_fsdp_unit: 1\n    reshard_after_forward: false",
            "reshard_after_forward",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} (dp={args.dp} strategy={args.strategy} micro={args.micro} accum={args.accum})")


if __name__ == "__main__":
    main()
