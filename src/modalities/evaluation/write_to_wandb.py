import wandb
import os
import sys
import re
import json
import traceback
from pathlib import Path

from olmes_evaluator import evaluate_modalities_checkpoint

# --- CONFIGURATION MATCHING YOUR CLI ---
ENTITY = "cyhsm"
PROJECT = "loop"
CHECKPOINTS_ROOT = "/raid/s3/opengptx/mfrey/loop/checkpoints"
BENCHMARK_ROOT = "./benchmarks_v1" 

# 1. TASKS
TASKS_TO_RUN = [
    "modalities:base_easy:math_bpb",
    "modalities:base_easy:qa_rc",
    "modalities:base_easy:qa_bpb",
    #"gsm8k::olmes"
]

# 2. LIMIT
EVAL_LIMIT = 64

# 3. BATCH SIZE
BATCH_SIZE = 1

# 4. MAPPING:
FOLDER_MAPPING = {
    # "folder_unique_identifier" : "wandb_run_id"
    "2026-01-07__09-43-23_17c44997f407444e": "lqd4mcij",
    "2026-01-07__09-44-35_ce0e1da419bf7641": "rm57x92v", 
    "2026-01-07__09-45-00_f87bce5b79839f9a": "hr3vk2b9",
    "2026-01-07__09-55-08_7873cf97080bff5f": "dz1libib",
}

def get_run_id_for_folder(folder_name):
    for key, run_id in FOLDER_MAPPING.items():
        if key in folder_name:
            return run_id
    return None

def get_step_from_subfolder(subfolder_name):
    match = re.search(r"seen_steps_(\d+)", subfolder_name)
    return int(match.group(1)) if match else None

def parse_olmes_results(results_dict):
    """Parses OLMES results into a flat WandB dictionary."""
    wandb_metrics = {}
    
    # Handle case where input is the raw list (from JSON file) vs wrapper dict (from function)
    if isinstance(results_dict, list):
        all_metrics = results_dict
    else:
        all_metrics = results_dict.get("metrics", [])
    
    for item in all_metrics:
        # Get Alias
        task_name = item.get("task_config", {}).get("metadata", {}).get("alias")
        if not task_name:
            task_name = item.get("task_name", "unknown")
            
        # Get Score
        score = item.get("metrics", {}).get("primary_score")
        if score is not None:
            wandb_metrics[f"eval/{task_name}"] = score
            
    return wandb_metrics

def evaluate_model_folder(model_folder_path):
    folder_name = model_folder_path.name
    run_id = get_run_id_for_folder(folder_name)
    
    if not run_id:
        return

    print(f"\n=== Processing: {folder_name} ===")
    print(f"    -> Mapped to WandB Run ID: {run_id}")

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

    if not checkpoints: return

    # 2. INITIALIZE WANDB (With Custom X-Axis Fix)
    try:
        wandb.init(id=run_id, project=PROJECT, entity=ENTITY, resume="must", reinit=True)
        print(f"    -> CONNECTED: {wandb.run.url}") 
        
        # Define X-Axis
        wandb.define_metric("seen_steps")
        wandb.define_metric("eval/*", step_metric="seen_steps")
    except Exception as e:
        print(f"    !!! CRITICAL ERROR: {e}")
        return

    # 3. Loop Steps
    for step, ckpt_path in checkpoints:
        
        # --- THE CACHE CHECK ---
        # Look for: ./benchmarks_v1/RUN_ID/step_STEP/all_results.json
        expected_json = Path(BENCHMARK_ROOT) / run_id / f"step_{step}" / "all_results.json"
        
        eval_output = None
        
        if expected_json.exists():
            print(f"    >> [CACHE HIT] Found results at {expected_json}")
            try:
                with open(expected_json, 'r') as f:
                    # Your file is a list [...], we load it directly
                    eval_output = {"metrics": json.load(f)}
            except Exception as e:
                print(f"       Error reading JSON: {e}. Will re-run eval.")
        
        # If no cache or error reading cache, run the eval
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
                # Add the custom X-axis step
                metrics_to_log["seen_steps"] = step
                
                # Log without 'step=' argument
                wandb.log(metrics_to_log)
                print(f"       Logged {len(metrics_to_log)} metrics (seen_steps={step}).")
            else:
                print("       Warning: No valid metrics found to log.")

    wandb.finish()

if __name__ == "__main__":
    # Ensure benchmark root exists
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)
    
    root = Path(CHECKPOINTS_ROOT)
    for item in root.iterdir():
        if item.is_dir():
            evaluate_model_folder(item)