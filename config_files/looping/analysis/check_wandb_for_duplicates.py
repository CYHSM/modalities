import wandb
import pandas as pd

api = wandb.Api()
# Pick one run ID that failed (e.g., your Baseline or Loop-3)
run = api.run("cyhsm/loop/runs/hxcaquix") 
metric_key = "train train/ce_loss_avg"

print(f"Fetching data for {run.name}...")
history = list(run.scan_history(keys=["_step", metric_key]))
df = pd.DataFrame(history).dropna(subset=[metric_key])

# Check for duplicates
duplicates = df[df.duplicated(subset=["_step"], keep=False)]

if not duplicates.empty:
    print(f"\nFound {len(duplicates)} duplicate entries!")
    print("Here are the first few duplicates:")
    print(duplicates.head(10))
else:
    print("\nNo duplicates found in this run. (Check the other runs!)")