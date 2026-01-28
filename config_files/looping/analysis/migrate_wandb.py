import wandb
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# --- CONFIGURATION ---
ENTITY = "cyhsm"
PROJECT = "loop"
DRY_RUN = False

DIRECT_MAP = {
    "train train/ce_loss_avg": "train loss/ce_avg",
    "train train/loss_avg": "train loss/avg",
    "train train/loss_last": "train loss/last",
    "train train/consumed_tokens": "train progress/consumed_tokens",
    "train train/grad_norm_avg": "train grads/norm_avg",
    "train train/grad_norm_last": "train grads/norm_last",
    "train train/ponder_loss_avg": "train ponder/loss_avg",
    "train train/ponder_cost_avg": "train adaptive/ponder_cost_avg",
    "train train/ponder_weight": "train adaptive/ponder_weight",
    "train train/normalized_steps_avg": "train adaptive/normalized_steps_avg",
    "train train/expected_steps_avg": "train adaptive/expected_steps_avg",
    "train train/local_mem_scale_mean": "train adaptive/avg_local_mem_scale",
    "train train/global_mem_scale_mean": "train adaptive/avg_global_mem_scale",
}

SIMPLE_AGGREGATES = {
    "ponder_cost": "train adaptive/avg_layer_cost",
    "cos_sim": "train adaptive/avg_layer_cos_sim",
    "local_mem_scale": "train adaptive/avg_local_mem_scale",
    "global_mem_scale": "train adaptive/avg_global_mem_scale"
}


def transform_row(row_dict):
    """Transform a single row dict, renaming metrics and computing aggregates."""
    new_row = {}
    simple_aggs = defaultdict(list)
    indexed_aggs = defaultdict(lambda: defaultdict(list))

    for key, value in row_dict.items():
        # Skip internal wandb keys
        if key.startswith("_"):
            continue
            
        # Skip NaN values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        # 1. Handle Direct Renames
        if key in DIRECT_MAP:
            new_row[DIRECT_MAP[key]] = value
            continue

        # 2. Handle Layer Transforms
        layer_match = re.search(r"train (?:train/)?layer_(\d+)/(.+)", key)

        if layer_match:
            layer_idx = layer_match.group(1)
            metric_name = layer_match.group(2)

            new_key = f"train layers/{layer_idx}/{metric_name}"
            new_row[new_key] = value

            # Aggregation for simple metrics
            if metric_name in SIMPLE_AGGREGATES:
                simple_aggs[metric_name].append(value)

            # Aggregation for indexed metrics (loop_scale_N, halt_prob_N)
            index_match = re.search(r"(.+)_(\d+)$", metric_name)
            if index_match:
                base_name = index_match.group(1)
                step_idx = int(index_match.group(2))
                if base_name in ["loop_scale", "halt_prob"]:
                    indexed_aggs[base_name][step_idx].append(value)
        else:
            # 3. Keep everything else AS-IS (eval metrics, system metrics, etc.)
            new_row[key] = value

    # Calculate simple averages
    for metric, target_name in SIMPLE_AGGREGATES.items():
        if simple_aggs[metric]:
            new_row[target_name] = np.mean(simple_aggs[metric])

    # Calculate indexed averages
    for base_name, step_dict in indexed_aggs.items():
        for step_idx, values in step_dict.items():
            avg_key = f"train adaptive/avg_{base_name}_step_{step_idx}"
            new_row[avg_key] = np.mean(values)

    return new_row


def migrate_run(entity, project, old_run_id, dry_run=True):
    api = wandb.Api(timeout=120)
    old_run = api.run(f"{entity}/{project}/{old_run_id}")
    print(f"Fetching history for: {old_run.name} ({old_run.id})")

    # === Use DataFrame instead of scan_history ===
    print("\n[1/3] Fetching FULL history using history() DataFrame...")
    print("  (This gets all steps, unlike scan_history which samples)")
    
    # Get maximum samples to avoid sampling
    df = old_run.history(samples=1000000, pandas=True)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Unique _step values: {df['_step'].nunique()}")
    print(f"  Step range: {df['_step'].min()} to {df['_step'].max()}")
    
    # Identify eval/ columns (benchmark metrics with custom x-axis)
    eval_cols = [c for c in df.columns if c.startswith('eval/')]
    print(f"  Found {len(eval_cols)} eval/ benchmark columns")
    
    # Check seen_steps
    if 'seen_steps' in df.columns:
        seen_steps_non_null = df['seen_steps'].notna().sum()
        print(f"  Rows with seen_steps: {seen_steps_non_null}")

    # === Analyze what we're migrating ===
    print("\n[2/3] Analyzing data structure...")
    
    # Get all columns except internal ones
    data_cols = [c for c in df.columns if not c.startswith('_')]
    train_cols = [c for c in data_cols if c.startswith('train')]
    other_cols = [c for c in data_cols if not c.startswith('train') and not c.startswith('eval/')]
    
    print(f"  Train columns: {len(train_cols)}")
    print(f"  Eval/ columns: {len(eval_cols)}")
    print(f"  Other columns: {len(other_cols)} ({other_cols[:5]}...)")

    if dry_run:
        print("\n=== DRY RUN - Summary ===")
        print(f"Total rows to migrate: {len(df)}")
        print(f"Unique steps: {df['_step'].nunique()}")
        
        # Sample transformation
        sample_idx = len(df) // 2
        sample_row = df.iloc[sample_idx].to_dict()
        print(f"\n--- Sample row transformation (step {sample_row.get('_step')}) ---")
        transformed = transform_row(sample_row)
        
        # Show some transformed keys
        new_train_keys = [k for k in transformed.keys() if k.startswith('train')][:8]
        print(f"  Transformed train keys: {new_train_keys}")
        
        # Check eval data
        eval_rows = df[df[eval_cols].notna().any(axis=1)] if eval_cols else pd.DataFrame()
        print(f"\n--- Eval benchmark data ---")
        print(f"  Rows with eval/ metrics: {len(eval_rows)}")
        if len(eval_rows) > 0 and 'seen_steps' in df.columns:
            seen_vals = sorted(eval_rows['seen_steps'].dropna().unique())
            print(f"  At seen_steps: {seen_vals}")
        
        print("\n=== Set DRY_RUN = False to execute migration ===")
        return

    # === ACTUAL MIGRATION ===
    print(f"\n[3/3] Starting migration...")
    
    with wandb.init(
        entity=entity,
        project="loom",
        name=f"{old_run.name}_mvd",
        config=old_run.config,
        tags=(old_run.tags or []) + ["mvd"]
    ) as new_run:
        
        # Define metrics for custom x-axis on eval/
        new_run.define_metric("seen_steps")
        new_run.define_metric("eval/*", step_metric="seen_steps")
        
        # Process each row in order
        print("  Logging data...")
        logged = 0
        
        # Sort by _step to ensure correct order
        df_sorted = df.sort_values('_step')
        
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            step = row.get('_step')
            if pd.isna(step):
                continue
            step = int(step)
            
            # Convert row to dict
            row_dict = row.to_dict()
            
            # Transform the row
            transformed = transform_row(row_dict)
            
            # Skip empty rows
            if not transformed:
                continue
            
            # Log
            new_run.log(transformed, step=step)
            logged += 1
            
            if idx % 1000 == 0:
                print(f"    Processed {idx}/{len(df_sorted)} rows (logged: {logged})", end='\r')
        
        print(f"\n  Logged {logged} rows across {df['_step'].nunique()} unique steps")
        print(f"\nMigration complete!")
        print(f"New run: {new_run.url}")


if __name__ == "__main__":
    migrate_run(ENTITY, PROJECT, "7lswnrp7", dry_run=DRY_RUN)
    migrate_run(ENTITY, PROJECT, "2qwnjk33", dry_run=DRY_RUN)
    migrate_run(ENTITY, PROJECT, "kb904d0r", dry_run=DRY_RUN)
    migrate_run(ENTITY, PROJECT, "15dsfdv6", dry_run=DRY_RUN)
    migrate_run(ENTITY, PROJECT, "f24cucd6", dry_run=DRY_RUN)
    migrate_run(ENTITY, PROJECT, "pw702435", dry_run=DRY_RUN)