#!/usr/bin/env python3
"""Derive a warmstart config from a janus training config.

The two runs we compare (dual-path and the pure-wide control) must differ ONLY
in the model, so the warmstart transformation is applied mechanically from the
same source config rather than hand-edited per run.

Four changes, mirroring config_files/training/config_lorem_ipsum_long_fsdp2_warmstart.yaml:
  1. settings.training_progress.*      -- read back from the checkpoint instead of 0
  2. settings.warmstart_checkpoint_paths -- injected by `modalities warmstart`
  3. checkpoint_loading                -- new dcp component
  4. app_state                         -- renamed to app_state_raw, new dcp app_state
                                          that restores model+optimizer+scheduler

Usage: make_warmstart_config.py <in.yaml> <out.yaml>
"""

import re
import sys
from pathlib import Path

CKPT = "${settings.warmstart_checkpoint_paths.checkpoint_folder_path}"

TRAINING_PROGRESS = f"""  training_progress:
    global_num_seen_tokens:
      component_key: number_conversion
      variant_key: global_num_seen_tokens_from_checkpoint_path
      config:
        checkpoint_path: {CKPT}
    num_seen_steps:
      component_key: number_conversion
      variant_key: num_seen_steps_from_checkpoint_path
      config:
        checkpoint_path: {CKPT}
    num_seen_samples:
      component_key: number_conversion
      variant_key: num_samples_from_num_tokens
      config:
        num_tokens: ${{settings.training_progress.global_num_seen_tokens}}
        sequence_length: ${{settings.step_profile.sequence_length}}
    last_step:
      component_key: number_conversion
      variant_key: last_step_from_checkpoint_path
      config:
        checkpoint_path: {CKPT}
  warmstart_checkpoint_paths: ${{warmstart_env:checkpoint_paths}}
"""

CHECKPOINT_LOADING = """checkpoint_loading:
  component_key: checkpoint_loading
  variant_key: dcp
  config:
    global_rank: ${settings.cuda_env.global_rank}

"""

APP_STATE_DCP = f"""app_state:
  component_key: app_state
  variant_key: dcp
  config:
    raw_app_state:
      instance_key: app_state_raw
      pass_type: BY_REFERENCE
    checkpoint_dir_path: {CKPT}

"""


def transform(text: str) -> str:
    # 1 + 2: replace the flat training_progress block with checkpoint-derived values.
    progress_re = re.compile(
        r"^  training_progress:\n"
        r"    global_num_seen_tokens: 0\n"
        r"    num_seen_steps: 0\n"
        r"    num_seen_samples: 0\n"
        r"    last_step: -1\n",
        re.MULTILINE,
    )
    text, n = progress_re.subn(TRAINING_PROGRESS, text)
    if n != 1:
        raise SystemExit(f"expected exactly 1 training_progress block, found {n}")

    # 3 + 4: rename the raw app_state and prepend the dcp loader in front of it.
    app_state_re = re.compile(r"^app_state:\n(  component_key: app_state\n  variant_key: raw\n)", re.MULTILINE)
    if not app_state_re.search(text):
        raise SystemExit("could not find the `app_state` (variant raw) component")
    text = app_state_re.sub(CHECKPOINT_LOADING + APP_STATE_DCP + r"app_state_raw:\n\1", text, count=1)

    return text


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(transform(src.read_text()))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
