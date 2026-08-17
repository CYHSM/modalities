#!/usr/bin/env python
"""Generates clean replacements for the 5 seed-replicate runs contaminated by cancel+warmstart.

Background: A4_seed2, A4_seed3, A5_seed2, A5_seed3 and A6_seed2 (see generate_seed_configs.py) were
cancelled mid-run on 2026-08-13 (to switch them off the account-capped boost_qos_lprod QOS) and
resumed via `modalities warmstart` from their last 5000-step checkpoint. That re-executed every
step between the checkpoint and the cancellation point (1800-4900 steps depending on the arm) --
real extra gradient updates beyond the matched 76,250-step budget every other arm gets. It shows up
as a 0.040-0.055 nat improvement over each arm's seed1 run, an order of magnitude past the ~0.002-
0.003 nat spread the six genuinely-uninterrupted seed replicates (A1 x3, A2 x3, A3 x3, A6 seed3)
show, and past the effect sizes (0.005-0.03 nats) this whole study is trying to resolve. Naively
averaging the contaminated numbers into a "3-seed" mean would have overturned the "A1 loop-Mamba is
best" headline finding for a reason that has nothing to do with architecture -- see the "Seeds"
limitation in docs/loopotron/loopotron.tex, updated 2026-08-14.

This script writes fresh single-shot configs for the same 5 (arm, seed-slot) pairs under new
experiment_ids (suffix "_redo") so they cannot collide with the contaminated checkpoints already on
disk, or be silently warmstarted from them. Derived directly from each arm's ORIGINAL (seed1)
config, exactly like generate_seed_configs.py -- same architecture, same data order (sampler
seed 42), same eval question sets (seed 1234), only weight init (unseeded) differs. These runs must
be left to complete in ONE slot: do not cancel/resubmit them, or the same contamination recurs.

Run from the repository root, after generate_arm_configs.py::

    python config_files/nemotron/loop_ablation_5b_cluster/generate_seed_redo_configs.py
"""

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARM_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster"

# (base arm name, new suffix) -- base arm name must match the pinned experiment_id in
# config_<base_arm_name>.yaml (i.e. the untouched seed1 config).
REDO_PAIRS = [
    ("A4_loop_mamba_moe", "seed2_redo"),
    ("A4_loop_mamba_moe", "seed3_redo"),
    ("A5_loop_mamba_attention", "seed2_redo"),
    ("A5_loop_mamba_attention", "seed3_redo"),
    ("A6_loop_attention_moe", "seed2_redo"),
]


def main() -> None:
    for arm_name, suffix in REDO_PAIRS:
        base_path = ARM_DIRECTORY / f"config_{arm_name}.yaml"
        base_text = base_path.read_text()

        new_arm_name = f"{arm_name}_{suffix}"
        output_path = ARM_DIRECTORY / f"config_{new_arm_name}.yaml"

        text, n = re.subn(
            rf"^  experiment_id: {re.escape(arm_name)}$",
            f"  experiment_id: {new_arm_name}",
            base_text,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            raise RuntimeError(
                f"expected exactly one pinned experiment_id line 'experiment_id: {arm_name}' "
                f"in {base_path}, found {n}"
            )
        output_path.write_text(text)
        print(f"{new_arm_name:44s} -> {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
