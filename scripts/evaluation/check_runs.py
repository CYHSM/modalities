#!/usr/bin/env python3
"""
Diagnostic script to check status, checkpoints, and potential errors across
all revision runs listed in all_configs.txt.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

SCRATCH_ROOT = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions")
ALL_CONFIGS_TXT = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/emnlp/sbatch_revisions/all_configs.txt")
LOGS_DIR = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/emnlp/sbatch_revisions/logs")


def get_active_jobs():
    """Run squeue to find active job IDs and array tasks."""
    active = {}
    try:
        res = subprocess.run(["squeue", "-u", "mfrey000", "-h", "-o", "%i %t"], capture_output=True, text=True, check=True)
        for line in res.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                job_id_str, status = parts[0], parts[1]
                # job_id_str can be like 49050942_7 or 49050942_[12-14]
                active[job_id_str] = status
    except Exception as e:
        print(f"Warning: could not run squeue: {e}")
    return active


def parse_slurm_logs():
    """Parse SLURM log files to map config files to jobs, array tasks, and check for errors."""
    logs_info = []
    if not LOGS_DIR.exists():
        return logs_info

    for f in LOGS_DIR.glob("*.err"):
        content_head = ""
        content_tail = ""
        try:
            with open(f, "r", errors="ignore") as file:
                lines = file.readlines()
                if lines:
                    content_head = "".join(lines[:30])
                    content_tail = "".join(lines[-100:])
        except Exception:
            continue

        # Extract array task index and config file from head
        array_task_match = re.search(r"Array Task (\d+) using config:\s*(\S+)", content_head)
        if not array_task_match:
            # Try parsing from CONFIG_FILE_PATH=
            config_match = re.search(r"CONFIG_FILE_PATH=(\S+)", content_head)
            task_match = re.search(r"SLURM_ARRAY_TASK_ID=(\d+)", content_head)
            if config_match and task_match:
                config_path = config_match.group(1)
                array_task = int(task_match.group(1))
            else:
                continue
        else:
            array_task = int(array_task_match.group(1))
            config_path = array_task_match.group(2)

        # Extract Job ID and Array ID from log filename or header
        # Filename is usually: modalities_run-JOBID_ARRAYID.err
        fn_match = re.search(r"modalities_run-(\d+)_(\d+)\.err", f.name)
        if fn_match:
            job_id = fn_match.group(1)
            array_id = fn_match.group(2)
        else:
            job_id = "unknown"
            array_id = str(array_task)

        # Check for training errors (ignoring the conversion/olmes failures)
        has_training_error = False
        error_msg = ""

        # Ignore conversion log failure output in slurm logs when assessing training crash
        clean_tail = content_tail
        # Remove conversion log failures block to check for actual training process errors
        clean_tail = re.sub(r"Model conversion failed.*", "", clean_tail, flags=re.DOTALL)
        clean_tail = re.sub(r"No hf_checkpoint found.*", "", clean_tail, flags=re.DOTALL)

        if "Traceback (most recent call last):" in clean_tail or "RuntimeError:" in clean_tail or "CUDA error:" in clean_tail or "OutOfMemoryError" in clean_tail:
            has_training_error = True
            # Extract last few lines for traceback description
            lines = [l.strip() for l in clean_tail.split("\n") if l.strip()]
            error_msg = " | ".join(lines[-4:])

        finished = "=== FINISHED ===" in content_tail or "END TIME:" in content_tail

        logs_info.append({
            "log_file": f.name,
            "job_id": job_id,
            "array_id": array_id,
            "array_task": array_task,
            "config_path": config_path,
            "finished": finished,
            "has_error": has_training_error,
            "error_msg": error_msg
        })
    return logs_info


def main():
    if not ALL_CONFIGS_TXT.exists():
        print(f"Error: {ALL_CONFIGS_TXT} not found.")
        sys.exit(1)

    with open(ALL_CONFIGS_TXT, "r") as f:
        config_lines = [line.strip() for line in f if line.strip()]

    active_jobs = get_active_jobs()
    logs_info = parse_slurm_logs()

    # List scratch run dirs
    scratch_subdirs = []
    if SCRATCH_ROOT.exists():
        scratch_subdirs = [SCRATCH_ROOT / d for d in os.listdir(SCRATCH_ROOT) if (SCRATCH_ROOT / d).is_dir()]

    print("=" * 110)
    print(f"REVISIONS RUN DIAGNOSTIC SUMMARY")
    print(f"Scratch dir: {SCRATCH_ROOT}")
    print("=" * 110)

    # Print header
    print(f"{'Idx':<4} {'Config File':<75} {'Runs Found':<10} {'Status':<15}")
    print("-" * 110)

    # Track how many times each config in 1-14 ran
    runs_1_14_counts = {}

    for idx_0, config_line in enumerate(config_lines):
        idx = idx_0 + 1
        config_name = os.path.basename(config_line)

        # 1. Find matching directories in scratch
        matching_dirs = []
        for d in scratch_subdirs:
            if (d / config_name).exists():
                # Get stats
                mtime = d.stat().st_mtime
                # Find step count from directories
                steps = []
                for item in d.iterdir():
                    if item.is_dir() and item.name.startswith("eid_"):
                        match = re.search(r"seen_steps_(\d+)", item.name)
                        if match:
                            steps.append(int(match.group(1)))
                max_step = max(steps) if steps else 0
                matching_dirs.append((d, mtime, max_step))

        # Sort matching dirs by mtime (oldest to newest)
        matching_dirs.sort(key=lambda x: x[1])

        # Match SLURM logs
        matching_logs = [l for l in logs_info if l["array_task"] == idx or os.path.basename(l["config_path"]) == config_name]

        # Determine overall status and seeds count
        run_count = len(matching_dirs)
        if idx <= 14:
            runs_1_14_counts[idx] = run_count

        status_str = "NO RUNS"
        if run_count > 0:
            status_details = []
            for d, _, max_step in matching_dirs:
                # Check if this directory belongs to an active job
                is_running = False
                is_pd = False
                for job_key, job_status in active_jobs.items():
                    # check if job_key matches job id in slurm logs
                    # Or try to infer if active job is running
                    # Let's check if the directory's last checkpoint is recent (within 1 hour)
                    pass

                # Let's inspect logs matching this directory
                dir_logs = []
                for l in matching_logs:
                    # SLURM logs can be mapped by checking if it shares the timestamp/id in folder name
                    # e.g. folder name is 2026-07-09__13-29-18_b8a9387bbd4a6472, look for it in slurm log
                    log_file_path = LOGS_DIR / l["log_file"]
                    try:
                        with open(log_file_path, "r", errors="ignore") as lf:
                            if d.name in lf.read():
                                dir_logs.append(l)
                    except Exception:
                        pass

                # If no direct match in slurm logs content, match by index
                if not dir_logs and len(matching_dirs) == len(matching_logs):
                    # Guess matching by order
                    dir_idx = matching_dirs.index((d, _, max_step))
                    dir_logs = [matching_logs[dir_idx]]

                dir_status = "Unknown"
                if dir_logs:
                    l = dir_logs[-1]
                    if l["has_error"]:
                        dir_status = "CRASHED"
                    elif l["finished"]:
                        dir_status = "FINISHED"
                    else:
                        dir_status = "RUNNING?"
                else:
                    # Fallback check
                    # Check if last_checkpoint_info.json has target steps completed
                    info_file = d / "last_checkpoint_info.json"
                    if info_file.exists():
                        import json
                        try:
                            with open(info_file, "r") as f_json:
                                info = json.load(f_json)
                                path_str = info.get("checkpoint_folder_path") or info.get("model_checkpoint_path", "")
                                if "target_steps" in path_str:
                                    # parse target_steps and seen_steps
                                    match_seen = re.search(r"seen_steps_(\d+)", path_str)
                                    match_target = re.search(r"target_steps_(\d+)", path_str)
                                    if match_seen and match_target and match_seen.group(1) == match_target.group(1):
                                        dir_status = "FINISHED"
                                    else:
                                        dir_status = "ACTIVE/INCOMPLETE"
                        except Exception:
                            pass

                status_details.append(f"{d.name[-8:]}({dir_status}:{max_step}s)")
            status_str = ", ".join(status_details)

        # Print row
        print(f"{idx:<4} {config_name:<75} {run_count:<10} {status_str:<15}")

        # If any of the SLURM logs indicates a crash, print the error detail below
        for l in matching_logs:
            if l["has_error"]:
                print(f"     ↳ [CRASHED] Job {l['job_id']}_{l['array_id']} ({l['log_file']}): {l['error_msg']}")

    print("=" * 110)
    print("RUN COUNTS FOR CONFIGS 1-14 (Target in experiments_emnlp_revisions: 2 runs per config):")
    print("-" * 110)
    for idx in range(1, 15):
        count = runs_1_14_counts.get(idx, 0)
        config_name = os.path.basename(config_lines[idx - 1])
        status = "OK" if count == 2 else ("LACKING (need to run more)" if count < 2 else "EXCESSIVE")
        print(f"Config {idx:<2}: {config_name:<70} | Runs count: {count} | Status: {status}")
    print("=" * 110)


if __name__ == "__main__":
    main()
