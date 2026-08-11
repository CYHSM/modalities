#!/usr/bin/env python3
import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("convert_and_evaluate")

SCRIPT_DIR = Path(__file__).resolve().parent
MODALITIES_ROOT = SCRIPT_DIR.parent.parent
OLMES_SCRIPT = SCRIPT_DIR / "run_olmes_sbatch.sh"
ALL_CONFIGS_TXT = MODALITIES_ROOT / "config_files/emnlp_revisions/sbatch/all_configs.txt"

DEFAULT_TASKS = [
    "minerva_math_algebra:bpb::olmes",
    "minerva_math_counting_and_probability:bpb::olmes",
    "minerva_math_geometry:bpb::olmes",
    "minerva_math_intermediate_algebra:bpb::olmes",
    "minerva_math_number_theory:bpb::olmes",
    "minerva_math_prealgebra:bpb::olmes",
    "minerva_math_precalculus:bpb::olmes",
    "arc_challenge:rc::olmes:full",
    "arc_easy:rc::olmes:full",
    "hellaswag:rc::olmes:full",
    "winogrande:rc::olmes:full",
    "socialiqa:rc::olmes:full",
    "piqa:rc::olmes:full",
    "qasper_yesno:rc::olmes",
    "lambada",
    "arc_challenge:rc:bpb::olmes:full",
    "arc_easy:rc:bpb::olmes:full",
    "hellaswag:rc:bpb::olmes:full",
    "winogrande:rc:bpb::olmes:full",
    "socialiqa:rc:bpb::olmes:full",
    "piqa:rc:bpb::olmes:full",
    "qasper_yesno:rc:bpb::olmes",
    "lambada:bpb",
    "gsm8k::olmes",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tasks_from_config(config_path: Path) -> list:
    try:
        cfg = load_yaml(config_path)
        evaluator = cfg.get("downstream_evaluator", {})
        config_block = evaluator.get("config", {})
        tasks = config_block.get("tasks")
        if tasks and isinstance(tasks, list):
            return tasks
    except Exception as e:
        logger.warning(f"Could not parse tasks from {config_path}: {e}. Using defaults.")
    return DEFAULT_TASKS


def find_run_directories(config_path: Path, scratch_subdirs: list[Path]) -> list[Path]:
    basename = config_path.name
    matching = []
    for d in scratch_subdirs:
        if (d / basename).exists():
            matching.append(d)
    matching.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matching


def main():
    parser = argparse.ArgumentParser(description="Convert and evaluate model checkpoints manually.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--config-index", type=int, help="1-based index of the configuration to process in all_configs.txt.")
    parser.add_argument("--last-only", action="store_true", help="Only evaluate the last checkpoint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not ALL_CONFIGS_TXT.exists():
        logger.error(f"Config list file {ALL_CONFIGS_TXT} does not exist.")
        sys.exit(1)

    with open(ALL_CONFIGS_TXT, "r") as f:
        config_lines = [line.strip() for line in f if line.strip()]

    if args.config_index is not None:
        if 1 <= args.config_index <= len(config_lines):
            config_lines = [config_lines[args.config_index - 1]]
            logger.info(f"Filtering to configuration at index {args.config_index}: {config_lines[0]}")
        else:
            logger.error(f"Config index {args.config_index} is out of bounds (1 to {len(config_lines)}).")
            sys.exit(1)

    config_paths = []
    for line in config_lines:
        path = MODALITIES_ROOT / line
        if path.exists():
            config_paths.append(path)
        else:
            path = Path(line)
            if path.exists():
                config_paths.append(path)
            else:
                logger.warning(f"Config path {line} not found.")

    if not config_paths:
        logger.error("No valid configurations found.")
        sys.exit(1)

    scratch_roots = [
        Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions"),
        Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"),
        Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_tests"),
    ]
    scratch_subdirs = []
    for s_root in scratch_roots:
        if s_root.exists():
            scratch_subdirs.extend([s_root / d for d in os.listdir(s_root) if (s_root / d).is_dir()])

    for config_path in config_paths:
        logger.info(f"Processing config: {config_path.name}")
        run_dirs = find_run_directories(config_path, scratch_subdirs)
        if not run_dirs:
            logger.info(f"  No run directory found in scratch for config {config_path.name}. Skipping.")
            continue

        for run_dir in run_dirs:
            logger.info(f"  Processing run directory: {run_dir.name}")

            checkpoint_dirs = []
            for item in run_dir.iterdir():
                if item.is_dir() and item.name.startswith("eid_"):
                    checkpoint_dirs.append(item)

            if not checkpoint_dirs:
                logger.info("  No checkpoint directories found yet.")
                continue

            def get_step(d: Path) -> int:
                match = re.search(r"seen_steps_(\d+)", d.name)
                return int(match.group(1)) if match else -1

            checkpoint_dirs.sort(key=get_step)

            if args.last_only and checkpoint_dirs:
                checkpoint_dirs = [checkpoint_dirs[-1]]

            for cp_dir in checkpoint_dirs:
                step = get_step(cp_dir)
                if step == -1:
                    logger.warning(f"  Could not extract step number from directory name: {cp_dir.name}")
                    continue

                logger.info(f"  Step {step}:")

                hf_checkpoint_dir = cp_dir / "hf_checkpoint"
                hf_config_file = hf_checkpoint_dir / "config.json"
                converted = False

                if hf_config_file.exists():
                    logger.info(f"    HF checkpoint already exists at {hf_checkpoint_dir.name}")
                    converted = True
                else:
                    conversion_cmd = [
                        sys.executable,
                        "-m",
                        "modalities.conversion",
                        str(config_path),
                        str(hf_checkpoint_dir),
                        "--checkpoint_path",
                        str(cp_dir),
                    ]
                    logger.info(f"    [CONVERT] {' '.join(conversion_cmd)}")
                    if not args.dry_run:
                        log_file = cp_dir / "conversion.log"
                        run_env = os.environ.copy()
                        src_dir = str(MODALITIES_ROOT / "src")
                        run_env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{run_env.get('PYTHONPATH', '')}"

                        try:
                            with open(log_file, "w") as out_log:
                                subprocess.run(conversion_cmd, stdout=out_log, stderr=subprocess.STDOUT, check=True, env=run_env)
                            logger.info("    ✓ Conversion succeeded.")
                            converted = True
                        except subprocess.CalledProcessError as e:
                            logger.error(f"    ✗ Conversion failed. Check log at {log_file}")

                if converted:
                    eval_metrics_file = hf_checkpoint_dir / f"olmes_eval_{step}" / "metrics-all.jsonl"
                    if eval_metrics_file.exists():
                        logger.info("    Downstream evaluation already completed.")
                    else:
                        tasks = get_tasks_from_config(config_path)
                        tasks_str = " ".join(tasks)
                        eval_cmd = [
                            "bash",
                            str(OLMES_SCRIPT),
                            str(hf_checkpoint_dir),
                            tasks_str,
                            str(step),
                            "128",
                            "1",
                        ]
                        logger.info(f"    [EVAL] {' '.join(eval_cmd)}")
                        if not args.dry_run:
                            try:
                                subprocess.run(eval_cmd, check=True)
                                logger.info("    ✓ Downstream evaluation job finished.")
                            except subprocess.CalledProcessError as e:
                                logger.error(f"    ✗ Downstream evaluation script failed: {e}")


if __name__ == "__main__":
    main()
