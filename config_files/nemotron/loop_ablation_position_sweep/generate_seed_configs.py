#!/usr/bin/env python
"""Adds extra-seed siblings of the position-sweep arms, for the "then extend" step.

This wave launches at n=1 per position deliberately: five runs read the trend across P0..P4 first,
and seeds are added only if that trend is marginal against the s.d. already measured for this
architecture elsewhere (0.0021 nats, four runs of A1 -- loopotron.tex "Wave 3"). Running this script
with, say, ``--suffixes seed2 seed3`` then produces the extra configs without regenerating anything
already on disk or on the queue.

Model weight initialization is unseeded in this codebase, so re-running an identical config under a
new experiment_id draws a fresh random init while the training data order (sampler seed 42) and the
synthetic-eval question sets (seed 1234) stay fixed -- the same mechanism every earlier replication
in this study used. The only edit needed is the pinned ``experiment_id`` line, which drives both the
checkpoint path and the wandb run name.

Run from the repository root, after generate_arm_configs.py::

    python config_files/nemotron/loop_ablation_position_sweep/generate_seed_configs.py --suffixes seed2
    python config_files/nemotron/loop_ablation_position_sweep/generate_warmstart_configs.py

then append the new names to arm_list.txt (or point the launcher at arm_list_seeds.txt) and resubmit
with a matching --array range.
"""

import argparse
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARM_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_position_sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=["seed2"],
        help="Seed suffixes to generate, e.g. --suffixes seed2 seed3 (default: seed2).",
    )
    arguments = parser.parse_args()

    base_paths = [
        p
        for p in sorted(ARM_DIRECTORY.glob("config_P*.yaml"))
        if not p.stem.endswith("_warmstart") and not re.search(r"_seed\d+$", p.stem)
    ]
    if not base_paths:
        raise SystemExit(f"no first-seed arm configs found in {ARM_DIRECTORY}; run generate_arm_configs.py first")

    generated = []
    for base_path in base_paths:
        arm_name = base_path.stem.removeprefix("config_")
        base_text = base_path.read_text()

        for suffix in arguments.suffixes:
            new_arm_name = f"{arm_name}_{suffix}"
            text, n = re.subn(
                rf"^  experiment_id: {re.escape(arm_name)}$",
                f"  experiment_id: {new_arm_name}",
                base_text,
                count=1,
                flags=re.MULTILINE,
            )
            if n != 1:
                raise RuntimeError(
                    f"expected exactly one pinned line 'experiment_id: {arm_name}' in {base_path}, found {n}"
                )
            (ARM_DIRECTORY / f"config_{new_arm_name}.yaml").write_text(text)
            generated.append(new_arm_name)
            print(f"{new_arm_name:32s} -> config_{new_arm_name}.yaml")

    (ARM_DIRECTORY / "arm_list_seeds.txt").write_text("".join(f"{name}\n" for name in generated))
    print(f"\narm_list_seeds.txt written with {len(generated)} arms; submit with --array=1-{len(generated)}.")


if __name__ == "__main__":
    main()
