import os
import sys
import re
import json
import traceback
import time
import gc
from pathlib import Path
import multiprocessing as mp  # Use stdlib multiprocessing, NOT torch.multiprocessing
import queue  # Standard library queue

# NOTE: torch and olmes_evaluator are intentionally NOT imported here.
# They must be imported INSIDE the worker process AFTER setting
# CUDA_VISIBLE_DEVICES, otherwise CUDA initializes on GPU 0.

# --- CONFIGURATION MATCHING YOUR CLI ---
ENTITY = "cyhsm"
PROJECT = "loom"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/mfrey/loop/checkpoints"
BENCHMARK_ROOT = "/raid/s3/opengptx/mfrey/loop/benchmarks_full"

# WANDB TOGGLE
USE_WANDB = False  # Set to True to log to wandb, False to only save to disk

# 1. TASKS
TASKS_TO_RUN = [
    "modalities:base_easy:math_bpb",
    "modalities:base_easy:qa_rc",
    "modalities:base_easy:qa_bpb",
    # "gsm8k::olmes"
]

# 2. LIMIT
EVAL_LIMIT = None

# 3. BATCH SIZE
BATCH_SIZE = "auto"

# 4. RUN SELECTION
RUN_IDS_TO_EVALUATE = None

# 5. MAPPING (MANIFEST)
EXPERIMENT_MANIFEST = [
    # ==========================================
    # 1. BASELINES
    # ==========================================
    # {
    #     "name":   "baseline_buckets_mvd",
    #     "run_id": "dx7k84zu",
    #     "folder": "2026-01-13__16-31-27_68708f8c380c52c4",
    # },
    # {
    #     "name":   "baseline_isoflop_to_L1024G512memory_and_3loops",
    #     "run_id": "acw5j6ox",
    #     "folder": "2026-01-26__14-49-40_02da07efbc8445e6",
    # },
    # {
    #     "name":   "baseline_isoparam_to_L1024G512memory",
    #     "run_id": "zxwh48wu",
    #     "folder": "2026-01-26__11-06-54_e3e769aaa0033a7e",
    # },

    # # ==========================================
    # # 2. LOOP 1 VARIANTS (Memory Size Scaling)
    # # ==========================================
    # {
    #     "name":   "loop1_L512G1024",
    #     "run_id": "dy705ftt",
    #     "folder": "2026-01-26__13-52-26_a8c42b25b4d84f03",
    # },
    # {
    #     "name":   "loop1_L512G4096",
    #     "run_id": "1ket85g0",
    #     "folder": "2026-01-26__11-22-50_82a1729fd64c49f8",
    # },
    # {
    #     "name":   "loop1_L1024G512",
    #     "run_id": "2h4fgihj",
    #     "folder": "2026-01-26__11-16-53_96a467fd7a75f1db",
    # },
    # {
    #     "name":   "loop1_L4096G512",
    #     "run_id": "jhtbyhil",
    #     "folder": "2026-01-26__13-49-58_9ffb0931b75711c2",
    # },

    # ==========================================
    # 3. LOOP 3 VARIANTS (Core & MVD)
    # ==========================================
    {
        "name":   "loop3_buckets_mvd",
        "run_id": "6kibshcj",
        "folder": "2026-01-15__09-28-10_6293597212cdaed8",
    },
    {
        "name":   "loop3_iso_mvd",
        "run_id": "90ndflhs",
        "folder": "2026-01-13__16-17-05_cfb2b82b2d1f3129",
    },
    {
        "name":   "loop3_L1024G512_ponder001",
        "run_id": "7mzvjajg",
        "folder": "2026-01-27__15-24-04_043e421a1b54accb",
    },
    {
        "name":   "loop3_L1024G512_ponder-001",
        "run_id": "a8wysxso",
        "folder": "2026-01-27__15-23-57_938b4efba13f5ed5",
    },

    # ==========================================
    # 4. LOOP 3 VARIANTS (Individual Memory / Init)
    # ==========================================
    {
        "name":   "loop3_L1024G512_individualMemory_frozenmem",
        "run_id": "t5qfh6mu",
        "folder": "2026-01-25__01-09-00_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init0",
        "run_id": "fu2lz8ci",
        "folder": "2026-01-23__22-42-46_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init3",
        "run_id": "89w112fv",
        "folder": "2026-01-23__22-45-27_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init-3",
        "run_id": "zhvqr1sl",
        "folder": "2026-01-23__19-52-07_6756580fe4bb252b",
    },

    # ==========================================
    # 5. HIGH LOOPS (5, 7, 9)
    # ==========================================
    {
        "name":   "loop5_L1024G512",
        "run_id": "vvkqwmg6",
        "folder": "2026-01-27__21-51-37_ad8de814eeeade84",
    },
    {
        "name":   "loop5_buckets_mvd",
        "run_id": "8l32ixdg",
        "folder": "2026-01-13__16-15-53_ac0fe16b31e73093",
    },
    {
        "name":   "loop5_iso_mvd",
        "run_id": "zrgz6gwd",
        "folder": "2026-01-13__16-19-19_4fde1337716312ed",
    },
    {
        "name":   "loop7_buckets_mvd",
        "run_id": "6533grrb",
        "folder": "2026-01-14__08-46-21_d2cfb3a1e9e8f3b1",
    },
    {
        "name":   "loop9_buckets",
        "run_id": "lp4f3bc5",
        "folder": "2026-01-27__18-44-29_54e21865ce16b752",
    },
]


# =====================================================================
# HELPERS
# =====================================================================

def get_step_from_subfolder(subfolder_name):
    match = re.search(r"seen_steps_(\d+)", subfolder_name)
    return int(match.group(1)) if match else None


def parse_olmes_results(results_dict):
    """Parses OLMES results into a flat dictionary."""
    metrics = {}

    if isinstance(results_dict, list):
        all_metrics = results_dict
    else:
        all_metrics = results_dict.get("metrics", [])

    for item in all_metrics:
        task_name = item.get("task_config", {}).get("metadata", {}).get("alias")
        if not task_name:
            task_name = item.get("task_name", "unknown")

        score = item.get("metrics", {}).get("primary_score")
        if score is not None:
            metrics[f"eval/{task_name}"] = score

    return metrics


def save_metrics_to_disk(metrics, output_path):
    """Save metrics to a JSON file on disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"       Saved metrics to {output_path}")


def discover_checkpoints(exp_config):
    """
    For a single experiment, find its config yaml and all checkpoint steps.
    Returns a list of task dicts ready for the queue, or [] if nothing found.
    """
    folder_path = Path(CHECKPOINTS_ROOT) / exp_config["folder"]
    if not folder_path.exists():
        print(f"!!! WARNING: Folder for '{exp_config['name']}' not found at {folder_path}")
        return []

    # Find config yaml
    config_path = None
    yaml_files = list(folder_path.glob("*.yaml"))
    if yaml_files:
        config_path = str(yaml_files[0])

    # Find all checkpoint subdirectories
    checkpoints = []
    for item in folder_path.iterdir():
        if item.is_dir() and "seen_steps" in item.name:
            step = get_step_from_subfolder(item.name)
            if step is not None:
                checkpoints.append((step, item))
    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        print(f"    -> No checkpoints found for '{exp_config['name']}'. Skipping.")
        return []

    # Build one queue item per checkpoint
    tasks = []
    for step, ckpt_path in checkpoints:
        tasks.append({
            "name":        exp_config["name"],
            "run_id":      exp_config["run_id"],
            "folder":      exp_config["folder"],
            "step":        step,
            "ckpt_path":   str(ckpt_path),
            "config_path": config_path,
        })

    return tasks


# =====================================================================
# SINGLE-CHECKPOINT EVALUATION (runs inside a worker)
# =====================================================================

def evaluate_single_checkpoint(task):
    """
    Evaluate ONE checkpoint. Saves results to disk.
    Must be called from a process that already set CUDA_VISIBLE_DEVICES.
    """
    import torch
    from olmes_evaluator import evaluate_modalities_checkpoint

    name        = task["name"]
    run_id      = task["run_id"]
    step        = task["step"]
    ckpt_path   = task["ckpt_path"]
    config_path = task["config_path"]

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"    >> [{name}] Step {step} on GPU {gpu} — starting")

    # Check cache
    expected_json = Path(BENCHMARK_ROOT) / run_id / f"step_{step}" / "all_results.json"
    eval_output = None

    if expected_json.exists():
        print(f"    >> [{name}] Step {step} — CACHE HIT at {expected_json}")
        try:
            with open(expected_json, 'r') as f:
                eval_output = {"metrics": json.load(f)}
        except Exception as e:
            print(f"       Error reading cached JSON: {e}. Will re-run eval.")

    # Run eval if needed
    if eval_output is None:
        print(f"    >> [{name}] Step {step} — COMPUTING...")
        try:
            eval_output = evaluate_modalities_checkpoint(
                checkpoint_path=ckpt_path,
                config_path=config_path,
                tasks=TASKS_TO_RUN,
                limit=EVAL_LIMIT,
                batch_size=BATCH_SIZE,
                output_dir=f"{BENCHMARK_ROOT}/{run_id}/step_{step}",
            )
        except Exception as e:
            print(f"       ERROR on [{name}] step {step}: {e}")
            traceback.print_exc()
            return

    # Just confirm completion (all_results.json is written by evaluate_modalities_checkpoint)
    if eval_output:
        metrics = parse_olmes_results(eval_output)
        if metrics:
            print(f"    >> [{name}] Step {step} — {len(metrics)} metrics computed.")
        else:
            print(f"    >> [{name}] Step {step} — WARNING: no valid metrics found.")

    # Cleanup GPU memory after each checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# GPU WORKER
# =====================================================================

def persistent_gpu_worker(gpu_id, task_queue):
    """
    Sits on a specific GPU and evaluates individual checkpoints from the queue.
    CUDA_VISIBLE_DEVICES is set BEFORE importing torch.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    print(f"--- Worker started on GPU {gpu_id} ({device_name}) [PID {os.getpid()}] ---")

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            print(f"--- Worker on GPU {gpu_id} [PID {os.getpid()}] finished all tasks. ---")
            break

        try:
            evaluate_single_checkpoint(task)
        except Exception as e:
            print(f"!!! CRITICAL FAILURE [{task.get('name')} step {task.get('step')}]: {e}")
            traceback.print_exc()


# =====================================================================
# POST-PROCESSING: aggregate results & optional wandb logging
# =====================================================================

def aggregate_and_log(experiments):
    """
    After all workers finish, walk the results on disk, build summaries,
    and optionally log everything to wandb (sequentially, to avoid conflicts).
    """
    if USE_WANDB:
        import wandb

    for exp in experiments:
        run_id = exp["run_id"]
        name   = exp["name"]
        run_dir = Path(BENCHMARK_ROOT) / run_id

        # Collect all all_results.json files for this run and parse them
        all_step_metrics = []
        if run_dir.exists():
            for step_dir in sorted(run_dir.iterdir()):
                if not step_dir.is_dir():
                    continue
                results_file = step_dir / "all_results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            raw_results = json.load(f)
                        metrics = parse_olmes_results({"metrics": raw_results})
                        if metrics:
                            # Extract step from folder name
                            step = get_step_from_subfolder(step_dir.name)
                            if step is not None:
                                metrics["seen_steps"] = step
                                all_step_metrics.append(metrics)
                    except Exception as e:
                        print(f"    Warning: could not read {results_file}: {e}")

        # Sort by step
        all_step_metrics.sort(key=lambda m: m.get("seen_steps", 0))

        if not all_step_metrics:
            print(f"    [{name}] No metrics found, skipping aggregation.")
            continue

        # Save summary
        summary_path = run_dir / "eval_summary.json"
        save_metrics_to_disk({
            "experiment_name": name,
            "run_id": run_id,
            "metrics_by_step": all_step_metrics,
        }, summary_path)

        # Optional wandb logging (sequential — safe, no concurrent writes)
        if USE_WANDB:
            try:
                run = wandb.init(
                    id=run_id, project=PROJECT, entity=ENTITY,
                    resume="must", reinit=True,
                )
                print(f"    [{name}] Logging {len(all_step_metrics)} steps to WandB...")
                run.define_metric("seen_steps")
                run.define_metric("eval/*", step_metric="seen_steps")

                for step_metrics in all_step_metrics:
                    run.log(step_metrics)

                run.finish()
                print(f"    [{name}] WandB logging complete.")
            except Exception as e:
                print(f"    [{name}] WandB error: {e}")
                traceback.print_exc()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    GPU_IDS = [4, 5, 6, 7]       # <--- LIST YOUR GPUS HERE
    CONCURRENT_PER_GPU = 8        # <--- WORKERS PER GPU

    # ==========================================
    # 2. SETUP
    # ==========================================
    mp.set_start_method('spawn', force=True)
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)

    # Select experiments
    if RUN_IDS_TO_EVALUATE:
        experiments_to_run = [e for e in EXPERIMENT_MANIFEST if e["run_id"] in RUN_IDS_TO_EVALUATE]
    else:
        experiments_to_run = EXPERIMENT_MANIFEST

    # ==========================================
    # 3. DISCOVER ALL CHECKPOINTS → FLAT QUEUE
    # ==========================================
    all_tasks = []
    for exp in experiments_to_run:
        all_tasks.extend(discover_checkpoints(exp))

    total_workers = len(GPU_IDS) * CONCURRENT_PER_GPU

    print(f"--- Starting Parallel Evaluation Session ---")
    print(f"Experiments: {len(experiments_to_run)}")
    print(f"Total checkpoint tasks: {len(all_tasks)}")
    print(f"Workers: {total_workers} ({CONCURRENT_PER_GPU} per GPU x {len(GPU_IDS)} GPUs)")
    print()

    # ==========================================
    # 4. FILL THE TASK QUEUE
    # ==========================================
    task_queue = mp.Queue()
    for t in all_tasks:
        task_queue.put(t)

    # ==========================================
    # 5. START WORKERS
    # ==========================================
    processes = []
    for gpu_id in GPU_IDS:
        for _ in range(CONCURRENT_PER_GPU):
            p = mp.Process(target=persistent_gpu_worker, args=(gpu_id, task_queue))
            p.start()
            processes.append(p)

    # ==========================================
    # 6. WAIT FOR ALL EVALUATIONS TO FINISH
    # ==========================================
    for p in processes:
        p.join()

    print("\n=== All Checkpoint Evaluations Complete ===\n")

    # ==========================================
    # 7. AGGREGATE RESULTS & LOG TO WANDB
    # ==========================================
    print("--- Aggregating results and logging ---")
    aggregate_and_log(experiments_to_run)

    print("\n=== All Done ===")