#!/usr/bin/env python
"""Builds a tiny checkpoint+warmstart round-trip test from config_A0_baseline.yaml.

Verifies the two things docs/components/nemotron_loops_research_plan.md section 9.4 flags as
failing silently: the LR schedule resuming at the right point, and the dataloader skipping the
right number of samples on resume.

warmstart_test_train.yaml stops at step 10 (== its checkpointing interval, so exactly one
checkpoint is written). warmstart_test_resume.yaml keeps the REAL target (20) and, given that
checkpoint via --last_checkpoint_info_file_path, should continue from step 11 to 20 with the LR
schedule picking up where it left off, not resetting to warmup.
"""
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TRAINING_SOURCE = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster/config_A0_baseline.yaml"
WARMSTART_SOURCE = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster/config_A0_baseline_warmstart.yaml"
OUT_DIR = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_5b_cluster/speedtest"

FULL_TARGET_STEPS = 20
STOP_AT_STEP = 10


def sub_once(text, pattern, replacement, name):
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"expected exactly one match for {name}, found {n}")
    return new_text


def patch(text, target_steps):
    text = sub_once(text, r"^    num_target_tokens: \d+$", f"    num_target_tokens: {target_steps * 65536}", "num_target_tokens")
    text = sub_once(text, r"^    num_target_steps: \d+$", f"    num_target_steps: {target_steps}", "num_target_steps")
    text = sub_once(text, r"^    checkpointing_interval_in_steps: \d+$", f"    checkpointing_interval_in_steps: {STOP_AT_STEP}", "checkpointing_interval_in_steps")
    text = sub_once(text, r"^    warmup_steps: \d+$", "    warmup_steps: 2", "warmup_steps")
    text = sub_once(text, r"^    evaluation_interval_in_steps: \d+$", f"    evaluation_interval_in_steps: {STOP_AT_STEP}", "evaluation_interval_in_steps")
    text = sub_once(text, r"^    project: .*$", "    project: modalities_nemotron_warmstart_test", "wandb project")
    return text


(OUT_DIR / "warmstart_test_train.yaml").write_text(patch(TRAINING_SOURCE.read_text(), STOP_AT_STEP))
(OUT_DIR / "warmstart_test_resume.yaml").write_text(patch(WARMSTART_SOURCE.read_text(), FULL_TARGET_STEPS))
(OUT_DIR / "warmstart_test_reference.yaml").write_text(patch(TRAINING_SOURCE.read_text(), FULL_TARGET_STEPS))

print(f"wrote {OUT_DIR / 'warmstart_test_train.yaml'} (stop at step {STOP_AT_STEP})")
print(f"wrote {OUT_DIR / 'warmstart_test_resume.yaml'} (resume to step {FULL_TARGET_STEPS})")
print(f"wrote {OUT_DIR / 'warmstart_test_reference.yaml'} (continuous run to step {FULL_TARGET_STEPS}, no interruption)")
