import os
import sys
import re
import json
import traceback
import time
import gc
import torch 
from pathlib import Path

from olmes_evaluator import evaluate_modalities_checkpoint

# --- CONFIGURATION MATCHING YOUR CLI ---
ENTITY = "cyhsm"
PROJECT = "loom"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/mfrey/loop/checkpoints"
BENCHMARK_ROOT = "/raid/s3/opengptx/mfrey/loop/benchmarks_full" 

# WANDB TOGGLE
USE_WANDB = False  # Set to True to log to wandb, False to only save to disk

# Only import wandb if needed
if USE_WANDB:
    import wandb

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
    {
        "name":   "baseline_buckets_mvd",
        "run_id": "dx7k84zu",
        "folder": "2026-01-13__16-31-27_68708f8c380c52c4",
    },
    {
        "name":   "baseline_isoflop_to_L1024G512memory_and_3loops",
        "run_id": "acw5j6ox",
        "folder": "2026-01-26__14-49-40_02da07efbc8445e6",
    },
    {
        "name":   "baseline_isoparam_to_L1024G512memory",
        "run_id": "zxwh48wu",
        "folder": "2026-01-26__11-06-54_e3e769aaa0033a7e",
    },

    # ==========================================
    # 2. LOOP 1 VARIANTS (Memory Size Scaling)
    # ==========================================
    {
        "name":   "loop1_L512G1024",
        "run_id": "dy705ftt",
        "folder": "2026-01-26__13-52-26_a8c42b25b4d84f03",
    },
    {
        "name":   "loop1_L512G4096",
        "run_id": "1ket85g0",
        "folder": "2026-01-26__11-22-50_82a1729fd64c49f8",
    },
    {
        "name":   "loop1_L1024G512",
        "run_id": "2h4fgihj",
        "folder": "2026-01-26__11-16-53_96a467fd7a75f1db",
    },
    {
        "name":   "loop1_L4096G512",
        "run_id": "jhtbyhil",
        "folder": "2026-01-26__13-49-58_9ffb0931b75711c2",
    },

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
        "name":   "loop3_L1024G512_cyclical",
        "run_id": "b7l63bce",
        "folder": "2026-01-27__14-04-59_c8fdbd942cf02ca0",
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

def evaluate_model_folder(model_folder_path, run_id, exp_name="Unknown"):
    
    print(f"\n=== Processing Experiment: {exp_name} ===")
    print(f"    -> Folder: {model_folder_path.name}")
    print(f"    -> Run ID: {run_id}")
    print(f"    -> WandB Logging: {'Enabled' if USE_WANDB else 'Disabled (disk only)'}")

    # 1. Config & Checkpoints
    config_path = None
    yaml_files = list(model_folder_path.glob("*.yaml"))
    if yaml_files: config_path = str(yaml_files[0])

    checkpoints = []
    for item in model_folder_path.iterdir():
        if item.is_dir() and "seen_steps" in item.name:
            step = get_step_from_subfolder(item.name)
            if step is not None:
                checkpoints.append((step, item))
    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        print("    -> No checkpoints found. Skipping.")
        return

    # 2. INITIALIZE WANDB (only if enabled)
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(id=run_id, project=PROJECT, entity=ENTITY, resume="must", reinit=True)
            print(f"    -> CONNECTED: {run.url}") 
            
            run.define_metric("seen_steps")
            run.define_metric("eval/*", step_metric="seen_steps")
        except Exception as e:
            print(f"    !!! CRITICAL ERROR INITIALIZING WANDB: {e}")
            return
    
    # Collect all metrics for disk saving
    all_run_metrics = []

    # 3. Loop Steps
    for step, ckpt_path in checkpoints:
        
        expected_json = Path(BENCHMARK_ROOT) / run_id / f"step_{step}" / "all_results.json"
        eval_output = None
        
        if expected_json.exists():
            print(f"    >> [CACHE HIT] Found results at {expected_json}")
            try:
                with open(expected_json, 'r') as f:
                    eval_output = {"metrics": json.load(f)}
            except Exception as e:
                print(f"       Error reading JSON: {e}. Will re-run eval.")
        
        if eval_output is None:
            print(f"    >> [COMPUTING] Running Eval on Step {step}...")
            try:
                eval_output = evaluate_modalities_checkpoint(
                    checkpoint_path=str(ckpt_path),
                    config_path=config_path, 
                    tasks=TASKS_TO_RUN,          
                    limit=EVAL_LIMIT,            
                    batch_size=BATCH_SIZE,       
                    output_dir=f"{BENCHMARK_ROOT}/{run_id}/step_{step}",
                )
            except Exception as e:
                print(f"       ERROR executing eval on step {step}: {e}")
                traceback.print_exc()
                continue

        # --- LOGGING ---
        if eval_output:
            metrics_to_log = parse_olmes_results(eval_output)
            
            if metrics_to_log:
                metrics_to_log["seen_steps"] = step
                
                # Save to disk always
                all_run_metrics.append(metrics_to_log)
                
                # Log to wandb if enabled
                if USE_WANDB and run:
                    run.log(metrics_to_log)
                    print(f"       Logged {len(metrics_to_log)} metrics to WandB (seen_steps={step}).")
                else:
                    print(f"       Collected {len(metrics_to_log)} metrics (seen_steps={step}).")
            else:
                print("       Warning: No valid metrics found to log.")

    # Save aggregated metrics to disk
    if all_run_metrics:
        summary_path = Path(BENCHMARK_ROOT) / run_id / "eval_summary.json"
        save_metrics_to_disk({
            "experiment_name": exp_name,
            "run_id": run_id,
            "metrics_by_step": all_run_metrics
        }, summary_path)

    # 4. CLEANUP
    if USE_WANDB and run:
        print(f"    -> Finishing WandB run {run_id}...")
        run.finish()
    
    # 5. HARD RESET for Memory
    time.sleep(5) 
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"    -> Cleanup complete for {exp_name}.\n")

if __name__ == "__main__":
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)
    root = Path(CHECKPOINTS_ROOT)
    
    target_ids = RUN_IDS_TO_EVALUATE
    
    print(f"--- Starting Evaluation Session ---")
    print(f"WandB Logging: {'ENABLED' if USE_WANDB else 'DISABLED'}")
    if target_ids:
        print(f"Targeting {len(target_ids)} specific Run IDs.")
    else:
        print("Targeting ALL experiments in manifest.")
        
    for exp_config in EXPERIMENT_MANIFEST:
        name = exp_config["name"]
        run_id = exp_config["run_id"]
        folder_name = exp_config["folder"]
        
        if target_ids and run_id not in target_ids:
            continue
            
        full_folder_path = root / folder_name
        
        if not full_folder_path.exists():
            print(f"!!! WARNING: Folder for '{name}' ({run_id}) not found: {full_folder_path}")
            continue

        try:
            evaluate_model_folder(full_folder_path, run_id, exp_name=name)
        except Exception as e:
            print(f"!!! CRITICAL FAILURE processing experiment {name}: {e}")
            traceback.print_exc()
            continue