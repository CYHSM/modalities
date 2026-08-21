#!/usr/bin/env python
"""Derives a `modalities warmstart`-compatible sibling of each position-sweep arm config.

Thin wrapper over loop_ablation_5b_cluster/generate_warmstart_configs.py, whose ``render_warmstart``
does the actual transformation; only the directory it scans differs. Reusing it means the warmstart
shape cannot drift between the two waves.

READ THIS BEFORE RESUMING ANYTHING. A run resumed with ``modalities warmstart`` is **not**
interchangeable with an uninterrupted one in this codebase: there is an unfixed checkpoint
round-trip defect on the tied embedding/output weight (`modalities run` writes a
``transformer.lm_head.weight`` that is not the trained shared tensor -- cosine 0.012 to it after ten
steps, and no optimizer state), and in Wave 2 five resumed runs came back 0.040-0.055 nats "better"
than their own first seeds, an order of magnitude past genuine seed noise, nearly overturning the
paper's headline result. See docs/loopotron/loopotron.tex section "Wave 2 seed replicates".

These configs therefore exist as damage control for a Slurm-initiated requeue, not as a normal path.
Every arm in this wave is expected to finish inside a single slot (17 executed layers, ~10-12h
against a 24h limit), so warmstart should never fire. **If any arm's log shows it resumed, discard
that run and relaunch it under a fresh experiment id rather than reporting its number.**

Run from the repository root, after generate_arm_configs.py::

    python config_files/nemotron/loop_ablation_position_sweep/generate_warmstart_configs.py
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster"))

from generate_warmstart_configs import render_warmstart  # noqa: E402

ARM_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_position_sweep"

BANNER_POINTER = (
    "config_files/nemotron/loop_ablation_5b_cluster/generate_warmstart_configs.py",
    "config_files/nemotron/loop_ablation_position_sweep/generate_warmstart_configs.py",
)


def main() -> None:
    training_configs = [p for p in sorted(ARM_DIRECTORY.glob("config_*.yaml")) if not p.stem.endswith("_warmstart")]
    if not training_configs:
        raise SystemExit(f"no arm configs found in {ARM_DIRECTORY}; run generate_arm_configs.py first")

    for training_config_path in training_configs:
        arm_name = training_config_path.stem.removeprefix("config_")
        output_path = ARM_DIRECTORY / f"config_{arm_name}_warmstart.yaml"
        text = render_warmstart(training_config_path.read_text(), arm_name).replace(*BANNER_POINTER)
        output_path.write_text(text)
        print(f"{arm_name:24s} -> {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
