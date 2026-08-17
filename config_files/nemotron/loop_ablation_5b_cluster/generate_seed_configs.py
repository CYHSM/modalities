#!/usr/bin/env python
"""Generates two extra-seed siblings of each A1-A6 config in this wave (seed2, seed3).

Every arm config already ran once (implicitly "seed1"). This adds two more runs of the *same*
config for A1-A6 only, so the paper table can report a mean +/- s.d. over 3 seeds for the six
looped variants instead of n=1 -- see the "Seeds" limitation in docs/loopotron/loopotron.tex.

Model weight initialization is unseeded in this codebase (no torch.manual_seed call anywhere in
the init path), so simply re-running the identical config under a new experiment_id draws a fresh
random initialization while holding everything else fixed: the training data order
(train_dataloader.config.sampler.seed: 42) and the synthetic-eval question sets
(seed: 1234 on every reasoning dataloader) are unchanged, exactly matching the methodology used for
the A6a x4 replication in Wave 1 (see loopotron.tex, "The seed-noise floor").

The only edit needed is the experiment_id line: it drives both the checkpoint path
(checkpoint_saving.config.checkpoint_path) and the wandb run name (evaluation_subscriber.config
.experiment_id), both via ${settings.experiment_id}, so patching it once gives each seed its own
checkpoint directory and wandb run with no other collision risk.

Run from the repository root, after generate_arm_configs.py::

    python config_files/nemotron/loop_ablation_5b_cluster/generate_seed_configs.py
"""

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARM_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster"

SEEDED_ARM_NAMES = [
    "A1_loop_mamba",
    "A2_loop_moe",
    "A3_loop_attention",
    "A4_loop_mamba_moe",
    "A5_loop_mamba_attention",
    "A6_loop_attention_moe",
]
EXTRA_SEED_SUFFIXES = ["seed2", "seed3"]


def main() -> None:
    for arm_name in SEEDED_ARM_NAMES:
        base_path = ARM_DIRECTORY / f"config_{arm_name}.yaml"
        base_text = base_path.read_text()

        for suffix in EXTRA_SEED_SUFFIXES:
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
                    f"in {base_path}, found {n} -- run generate_arm_configs.py first if this config "
                    "predates it, or the config's shape changed"
                )
            output_path.write_text(text)
            print(f"{new_arm_name:40s} -> {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
