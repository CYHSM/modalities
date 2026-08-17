#!/usr/bin/env python
"""Derives a `modalities warmstart`-compatible sibling of each arm config in this wave.

At 5B tokens no arm finishes inside one scheduler slot (see
docs/components/nemotron_loops_research_plan.md section 9.3-9.4), so `run_wave_5b.sh` resumes an
interrupted arm with `modalities warmstart --config_file_path config_<ARM>_warmstart.yaml
--last_checkpoint_info_file_path <path>` rather than `modalities run`. The warmstart config differs
from the training one in exactly the ways tutorials/warmstart/configs/warmstart_config.yaml differs
from its pre-training counterpart:

  * `training_progress` is derived FROM THE CHECKPOINT instead of zeroed, via four
    `number_conversion` components keyed off `settings.warmstart_checkpoint_paths`.
  * `settings.warmstart_checkpoint_paths: ${warmstart_env:checkpoint_paths}` is added -- populated
    by the `--last_checkpoint_info_file_path` CLI argument at launch, not written here.
  * `app_state` becomes the `dcp` variant wrapping the original (renamed `app_state_raw`) plus
    `checkpoint_dir_path`, so weights/optimizer/scheduler state load from the checkpoint instead of
    fresh-initializing.
  * A top-level `checkpoint_loading` component is added, matching the tutorial.

Every other component (dataset, model architecture, optimizer hyperparameters, evaluation) is
untouched -- it is the same run, resumed, so anything else differing would make the resumed
segment a different experiment from the one that was interrupted.

Run from the repository root, after generate_arm_configs.py::

    python config_files/nemotron/loop_ablation_5b_cluster/generate_warmstart_configs.py
"""

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARM_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster"

OLD_TRAINING_PROGRESS = """  training_progress:
    global_num_seen_tokens: 0
    num_seen_steps: 0
    num_seen_samples: 0
    last_step: -1"""

NEW_TRAINING_PROGRESS = """  training_progress:
    global_num_seen_tokens:
      component_key: number_conversion
      variant_key: global_num_seen_tokens_from_checkpoint_path
      config:
        checkpoint_path: ${settings.warmstart_checkpoint_paths.checkpoint_folder_path}
    num_seen_steps:
      component_key: number_conversion
      variant_key: num_seen_steps_from_checkpoint_path
      config:
        checkpoint_path: ${settings.warmstart_checkpoint_paths.checkpoint_folder_path}
    num_seen_samples:
      component_key: number_conversion
      variant_key: num_samples_from_num_tokens
      config:
        num_tokens: ${settings.training_progress.global_num_seen_tokens}
        sequence_length: ${settings.step_profile.sequence_length}
    last_step:
      component_key: number_conversion
      variant_key: last_step_from_checkpoint_path
      config:
        checkpoint_path: ${settings.warmstart_checkpoint_paths.checkpoint_folder_path}
  warmstart_checkpoint_paths: ${warmstart_env:checkpoint_paths}"""

OLD_APP_STATE = """app_state:
  component_key: app_state
  variant_key: raw
  config:
    model:
      instance_key: initialized_model
      pass_type: BY_REFERENCE
    optimizer:
      instance_key: optimizer
      pass_type: BY_REFERENCE
    lr_scheduler:
      instance_key: lr_scheduler
      pass_type: BY_REFERENCE"""

NEW_APP_STATE = """checkpoint_loading:
  component_key: checkpoint_loading
  variant_key: dcp
  config:
    global_rank: ${settings.cuda_env.global_rank}

app_state:
  component_key: app_state
  variant_key: dcp
  config:
    raw_app_state:
      instance_key: app_state_raw
      pass_type: BY_REFERENCE
    checkpoint_dir_path: ${settings.warmstart_checkpoint_paths.checkpoint_folder_path}

app_state_raw:
  component_key: app_state
  variant_key: raw
  config:
    model:
      instance_key: initialized_model
      pass_type: BY_REFERENCE
    optimizer:
      instance_key: optimizer
      pass_type: BY_REFERENCE
    lr_scheduler:
      instance_key: lr_scheduler
      pass_type: BY_REFERENCE"""

BANNER = """# GENERATED FILE -- do not edit. Edit config_{arm}.yaml and re-run
# config_files/nemotron/loop_ablation_5b_cluster/generate_warmstart_configs.py.
#
# Warmstart sibling of config_{arm}.yaml: same run, resumed from its most recent checkpoint.
# Launch with:
#   modalities warmstart --config_file_path config_{arm}_warmstart.yaml \\
#       --experiments_root_path <same experiments_root_path as the original run> \\
#       --last_checkpoint_info_file_path <path to that run's last_checkpoint_info.json>
"""


def render_warmstart(training_config_text: str, arm_name: str) -> str:
    text = training_config_text
    for old, new, name in (
        (OLD_TRAINING_PROGRESS, NEW_TRAINING_PROGRESS, "training_progress"),
        (OLD_APP_STATE, NEW_APP_STATE, "app_state"),
    ):
        if text.count(old) != 1:
            raise RuntimeError(f"expected exactly one occurrence of the {name} block in arm {arm_name}, "
                                f"found {text.count(old)} -- the arm config changed shape; update this generator")
        text = text.replace(old, new, 1)
    return BANNER.format(arm=arm_name) + "\n" + text


def main() -> None:
    training_configs = sorted(ARM_DIRECTORY.glob("config_*.yaml"))
    training_configs = [p for p in training_configs if not p.stem.endswith("_warmstart")]
    if not training_configs:
        raise SystemExit(f"no arm configs found in {ARM_DIRECTORY}; run generate_arm_configs.py first")

    for training_config_path in training_configs:
        arm_name = training_config_path.stem.removeprefix("config_")
        output_path = ARM_DIRECTORY / f"config_{arm_name}_warmstart.yaml"
        output_path.write_text(render_warmstart(training_config_path.read_text(), arm_name))
        print(f"{arm_name:32s} -> {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
