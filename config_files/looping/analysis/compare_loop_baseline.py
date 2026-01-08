import wandb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Configuration ---
api = wandb.Api()
CACHE_FILE = "wandb_data_cache.csv"

# Replace with your actual Run IDs
# Format: "entity/project/run_id"
runs_config = {
    "Baseline": "cyhsm/loop/runs/hxcaquix", 
    "Loop-3":   "cyhsm/loop/runs/o1h0riku",
    "Loop-5":   "cyhsm/loop/runs/dkd86omq"
}

metric_key = "train train/ce_loss_avg"
BIN_SIZE = 10000  # Aggregates 500 steps into one point

def fetch_and_process_data():
    """Fetches data from WandB, cleans duplicates, and aligns runs."""
    dfs = []
    
    for name, run_id in runs_config.items():
        try:
            print(f"Fetching {name} from WandB...")
            run = api.run(run_id)
            
            # 1. Fetch dense history
            history = list(run.scan_history(keys=["_step", metric_key]))
            df = pd.DataFrame(history)
            
            # 2. Drop NaNs and rename
            df = df.dropna(subset=[metric_key])
            df = df.rename(columns={metric_key: f"loss_{name}"})
            
            # 3. FIX DUPLICATES: Group by step and average
            df = df.groupby("_step").mean()
            dfs.append(df)
            
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            return None

    if not dfs:
        return None

    print("Aligning steps and merging...")
    # Inner join to ensure we compare only common steps
    combined_df = pd.concat(dfs, axis=1, join="inner").reset_index()
    return combined_df

# --- Main Logic ---

# 1. Check for Cache
if os.path.exists(CACHE_FILE):
    print(f"Loading data from local cache: {CACHE_FILE}")
    combined_df = pd.read_csv(CACHE_FILE)
else:
    # 2. Fetch if no cache
    combined_df = fetch_and_process_data()
    if combined_df is not None:
        print(f"Saving data to cache: {CACHE_FILE}")
        combined_df.to_csv(CACHE_FILE, index=False)
    else:
        raise ValueError("Failed to fetch data.")

# 3. Calculate Difference: Loop - Baseline
# (Negative = Loop is better)
for name in runs_config.keys():
    if name == "Baseline": continue
    combined_df[f"diff_{name}"] = combined_df[f"loss_{name}"] - combined_df["loss_Baseline"]

# 4. Binning & Aggregation
print(f"Aggregating into {BIN_SIZE}-step bins...")
combined_df["step_bin"] = (combined_df["_step"] // BIN_SIZE) * BIN_SIZE

agg_df = combined_df.groupby("step_bin").agg(
    {f"diff_{name}": ["mean", "std"] for name in runs_config.keys() if name != "Baseline"}
)

agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df = agg_df.reset_index()

# 5. Plotting
fig = go.Figure()
colors = {"Loop-3": "#EF553B", "Loop-5": "#00CC96"} 

for name in runs_config.keys():
    if name == "Baseline": continue
    
    mean_col = f"diff_{name}_mean"
    std_col = f"diff_{name}_std"
    
    if mean_col not in agg_df.columns: continue

    fig.add_trace(go.Scatter(
        x=agg_df["step_bin"],
        y=agg_df[mean_col],
        name=f"{name} (vs Baseline)",
        mode='lines+markers',
        line=dict(color=colors.get(name, "blue"), width=2),
        error_y=dict(
            type='data',
            array=agg_df[std_col],
            visible=True,
            thickness=1.5,
            width=3
        ),
        hovertemplate=(
            f"<b>{name}</b><br>" +
            "Step: %{x}<br>" +
            "Diff: %{y:.5f}<br>" +
            "StdDev: %{error_y.array:.5f}<extra></extra>"
        )
    ))

# Zero Line (Baseline Parity)
fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")

fig.update_layout(
    title="<b>Improvement Over Baseline</b><br><sup>(Loop Loss - Baseline Loss). Lower (Negative) is Better.</sup>",
    xaxis_title="Training Steps",
    yaxis_title="Loss Difference",
    template="plotly_white",
    height=700, width=1000,
    hovermode="x unified"
)

fig.show()
fig.write_html("wandb_diff_analysis.html")