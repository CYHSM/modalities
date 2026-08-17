#!/usr/bin/env python
"""Generates the 5B-token-wave arm configs from the shared base config.

Sibling of config_files/nemotron/loop_ablation/generate_arm_configs.py, which generates the
original 557M-token / Llama-3-tokenizer wave. This script writes to a SEPARATE directory
(loop_ablation_5b_cluster/) rather than overwriting the originals, because a run from an earlier
wave may still be reading its config file from disk when this runs -- see
docs/components/nemotron_loops_research_plan.md section 3.9.

Only the eleven arms docs/components/nemotron_loops_research_plan.md section 9.6 calls for in this
wave are generated: A0-A6 and their FLOP-matched anchors N1-N4. The dense baselines (D1/D2) and the
per-iteration-norm/input-injection 2x2 (A6a/A6b/A6c) are not part of this wave -- section 9.6 drops
the latter explicitly, since section 3.7 found no measurable effect from either refinement.

Run from the repository root::

    python config_files/nemotron/loop_ablation_5b_cluster/generate_arm_configs.py
"""

import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "config_files/nemotron/loop_ablation"))

from generate_arm_configs import ARMS as ALL_ARMS  # noqa: E402
from generate_arm_configs import _read_n_embd, _render_arm  # noqa: E402
from modalities.models.nemotron.layer_pattern import get_num_built_layers, get_num_layers  # noqa: E402

import generate_arm_configs as _original_module  # noqa: E402

THIS_WAVE_ARM_NAMES = [
    "A0_baseline",
    "A1_loop_mamba",
    "A2_loop_moe",
    "A3_loop_attention",
    "A4_loop_mamba_moe",
    "A5_loop_mamba_attention",
    "A6_loop_attention_moe",
    "N1_anchor_mamba",
    "N2_anchor_moe",
    "N3_anchor_attention",
    "N4_anchor_attention_moe",
]
ARMS = [arm for arm in ALL_ARMS if arm.name in THIS_WAVE_ARM_NAMES]
assert len(ARMS) == len(THIS_WAVE_ARM_NAMES), (
    f"expected {len(THIS_WAVE_ARM_NAMES)} arms, found {len(ARMS)} -- "
    f"the upstream ARMS list changed shape, update THIS_WAVE_ARM_NAMES"
)

# Patch the module-level paths the imported helpers close over, rather than duplicating them.
_original_module.BASE_CONFIG_PATH = REPOSITORY_ROOT / "config_files/nemotron/config_research_nemotron_loops_5b_cluster.yaml"
_original_module.OUTPUT_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster"
_original_module.WANDB_PROJECT = "modalities_nemotron_loops_5b_cluster"
BASE_CONFIG_PATH = _original_module.BASE_CONFIG_PATH
OUTPUT_DIRECTORY = _original_module.OUTPUT_DIRECTORY


def main() -> None:
    """Writes one config file per ablation arm in this wave."""
    base_config_text = BASE_CONFIG_PATH.read_text()
    n_embd = _read_n_embd(base_config_text)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        output_path = OUTPUT_DIRECTORY / f"config_{arm.name}.yaml"
        text = _render_arm(base_config_text, arm, n_embd)
        # Pinned, not ${modalities_env:experiment_id}: that resolver hashes the config path plus
        # the current timestamp, so it differs on every launch. `modalities warmstart` (unlike
        # `run`) exposes no --experiment_id flag to override it from the launcher, so the only way
        # to keep an arm's checkpoint/wandb directory stable across a requeue is to pin it here,
        # identically in the training config and its generate_warmstart_configs.py-derived sibling.
        text, n = re.subn(
            r"^  experiment_id: \$\{modalities_env:experiment_id\}$",
            f"  experiment_id: {arm.name}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            raise RuntimeError(f"expected exactly one experiment_id line for arm {arm.name}, found {n}")
        output_path.write_text(text)
        print(
            f"{arm.name:32s} {arm.layer_pattern:34s} "
            f"built={get_num_built_layers(arm.layer_pattern):3d} "
            f"executed={get_num_layers(arm.layer_pattern):3d} "
            f"-> {output_path.relative_to(REPOSITORY_ROOT)}"
        )


if __name__ == "__main__":
    main()
