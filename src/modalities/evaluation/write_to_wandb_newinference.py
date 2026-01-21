import wandb
import os
import sys
import re
import json
import traceback
import time
import gc
import torch # Assuming you are using PyTorch
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
EVAL_LIMIT = 128

# 3. BATCH SIZE
BATCH_SIZE = 1

# 4. MAPPING:
FOLDER_MAPPING = {
    # "folder_unique_identifier" : "wandb_run_id"
    #"2026-01-07__09-43-23_17c44997f407444e": "lqd4mcij",
    #"2026-01-07__09-44-35_ce0e1da419bf7641": "rm57x92v", 
    #"2026-01-07__09-45-00_f87bce5b79839f9a": "hr3vk2b9",
    #"2026-01-07__09-55-08_7873cf97080bff5f": "dz1libib",
    #"2026-01-09__14-39-48_6293597212cdaed8": "f24cucd6",
    #"2026-01-09__17-09-03_6293597212cdaed8": "hn9d90qx",
    #"2026-01-13__16-31-27_68708f8c380c52c4": "7lswnrp7",
    #"2026-01-13__16-19-19_4fde1337716312ed": "2qwnjk33",
    #"2026-01-13__16-17-05_cfb2b82b2d1f3129": "kb904d0r",
    "2026-01-13__16-15-53_ac0fe16b31e73093": "15dsfdv6", 
    "2026-01-14__08-46-21_d2cfb3a1e9e8f3b1": "pw702435",
    "2026-01-15__09-28-10_6293597212cdaed8": "2u49bm6x",
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
            wandb_metrics[f"eval/{task_name}"] = score
            
    return wandb_metrics

# --- NEW CONFIGURATION ---
INFERENCE_SUFFIX = "-HaltThreshold99_10loops"  # The tag to append to the name
NEW_ID_SUFFIX = "_halt99_10loops"                    # Suffix for the unique ID to ensure we can resume this eval run

def evaluate_model_folder(model_folder_path):
    folder_name = model_folder_path.name
    original_run_id = get_run_id_for_folder(folder_name)
    
    if not original_run_id:
        return

    print(f"\n=== Processing: {folder_name} ===")
    print(f"    -> Original Training Run ID: {original_run_id}")

    # 1. Config & Checkpoints setup (Same as before)
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

    # 2. FETCH ORIGINAL RUN DATA & INIT NEW RUN
    try:
        api = wandb.Api()
        # Fetch the original run to get its config and name
        # Note: We need the full path "entity/project/run_id"
        src_run = api.run(f"{ENTITY}/{PROJECT}/{original_run_id}")
        
        original_config = src_run.config
        original_name = src_run.name
        
        # Create new metadata
        new_run_name = f"{original_name}{INFERENCE_SUFFIX}"
        new_run_id = f"{original_run_id}{NEW_ID_SUFFIX}" 
        
        print(f"    -> FORKING to New Run: {new_run_name} (ID: {new_run_id})")

        # Initialize the NEW run
        # We merge the original config with any specific eval params you might want to track
        eval_config = original_config.copy()
        eval_config["inference_threshold"] = 0.99 # Example: log the specific param you changed
        
        run = wandb.init(
            id=new_run_id, 
            name=new_run_name,
            project=PROJECT, 
            entity=ENTITY, 
            config=eval_config,   # Pass the copied config here
            tags=["inference_eval", "forked"], # Helpful for filtering later
            resume="allow",       # Allow creation if new, resume if existing
            reinit=True
        )
        print(f"    -> CONNECTED: {run.url}") 
        
        # Define X-Axis
        run.define_metric("seen_steps")
        run.define_metric("eval/*", step_metric="seen_steps")
        
    except Exception as e:
        print(f"    !!! CRITICAL ERROR INITIALIZING WANDB: {e}")
        traceback.print_exc()
        return

    # 3. Loop Steps (Same logic, but logging to the new 'run' object)
    for step, ckpt_path in checkpoints:
        
        # NOTE: You might want to change the output directory to avoid overwriting 
        # previous standard benchmark results if you are running this locally.
        # I appended the suffix to the path here so they are stored separately.
        output_dir_path = Path(BENCHMARK_ROOT) / new_run_id / f"step_{step}"
        expected_json = output_dir_path / "all_results.json"
        
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
                    output_dir=str(output_dir_path), # Updated path
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
                run.log(metrics_to_log)
                print(f"       Logged {len(metrics_to_log)} metrics (seen_steps={step}).")
            else:
                print("       Warning: No valid metrics found to log.")

    # 4. CLEANUP
    print(f"    -> Finishing run {new_run_id}...")
    run.finish()
    
    time.sleep(2) 
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"    -> Cleanup complete for {folder_name}.\n")

if __name__ == "__main__":
    os.makedirs(BENCHMARK_ROOT, exist_ok=True)
    
    root = Path(CHECKPOINTS_ROOT)
    # Sort folders to ensure deterministic order (optional but helpful)
    all_folders = sorted([item for item in root.iterdir() if item.is_dir()])
    
    for item in all_folders:
        try:
            evaluate_model_folder(item)
        except Exception as e:
            print(f"!!! CRITICAL FAILURE processing folder {item.name}: {e}")
            traceback.print_exc()
            # If one folder crashes hard, try to continue to the next
            continue