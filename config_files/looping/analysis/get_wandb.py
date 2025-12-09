import wandb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

api = wandb.Api()
runs_config = {
    5: "cyhsm/loop/runs/m477h4eu",
    10: "cyhsm/loop/runs/hln5m4w3",
    100: "cyhsm/loop/runs/qx6lkixt",
    200: "cyhsm/loop/runs/wekiuslf",
    1000: "cyhsm/loop/runs/cc4ycqfe"
}
max_step = 15000
window = 701
metric_key = "train train/ce_loss_avg"
ponder_key = "train train/ponder_weight"

def rolling_slope(y, window):
    x = np.arange(window)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()
    slopes = np.array([
        (y[i:i+window] @ x_centered) / denom 
        for i in range(len(y) - window + 1)
    ])
    return -slopes

def calculate_binned_percentiles(x_data, y_data, bins=15):
    """Calculates binned statistics for a single run."""
    df_bin = pd.DataFrame({'x': x_data, 'y': y_data})
    
    # Create bins across the specific range of this run
    # If a run has a very narrow range, this adapts automatically
    if len(df_bin) == 0: return None, None
    
    df_bin['bin'] = pd.cut(df_bin['x'], bins)
    
    # Calculate stats
    stats = df_bin.groupby('bin', observed=True)['y'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    bin_centers = [b.mid for b in stats.index]
    
    return bin_centers, stats

run_data = {}
ponder_data = {}
steps = None

print("Fetching data...")
for label, run_id in runs_config.items():
    try:
        run = api.run(run_id)
        history = list(run.scan_history(keys=["_step", metric_key, ponder_key], max_step=max_step))
        df = pd.DataFrame(history).sort_values("_step").dropna(subset=[metric_key])
        
        if len(df) > window:
            start_idx = window // 2
            end_idx = -(window // 2)
            if steps is None:
                steps = df["_step"].values[start_idx:end_idx]
            run_data[label] = rolling_slope(df[metric_key].values, window)
            ponder_data[label] = df[ponder_key].values[start_idx:end_idx]
            print(f"-> Processed Run {label} ({len(df)} points)")
    except Exception as e:
        print(f"Error fetching {run_id}: {e}")

baseline_curve = np.median(np.array(list(run_data.values())), axis=0)

fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=False, 
    vertical_spacing=0.08,
    subplot_titles=("Absolute Improvement Rate", "Relative Performance (vs Median)", "Correlation: Ponder Weight vs Perf Ratio")
)
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

for i, label in enumerate(sorted(run_data.keys())):
    grads = run_data[label]
    ponders = ponder_data[label]
    color = colors[i % len(colors)]
    
    relative_perf = grads / np.where(baseline_curve == 0, 1e-9, baseline_curve)
    
    hover_template_main = (
        "<b>Config %{text}</b><br>" +
        "Step: %{x}<br>" +
        "Val: %{y:.5f}<br>" +
        "Ponder: %{customdata:.4f}<extra></extra>"
    )

    # 1. Main Plot
    fig.add_trace(
        go.Scatter(
            x=steps, y=grads, name=f"Config {label}", legendgroup=f"g{label}",
            line=dict(color=color, width=2),
            customdata=ponders, text=[label]*len(steps),
            hovertemplate=hover_template_main
        ), row=1, col=1
    )
    
    # 2. Ratio Plot
    fig.add_trace(
        go.Scatter(
            x=steps, y=relative_perf, name=f"Config {label}", legendgroup=f"g{label}",
            line=dict(color=color, width=2), showlegend=False,
            customdata=ponders, text=[label]*len(steps),
            hovertemplate=hover_template_main
        ), row=2, col=1
    )

    # 3. Correlation Plot (Background Scatter)
    # Made distinctively transparent so lines pop out
    fig.add_trace(
        go.Scatter(
            x=ponders, y=relative_perf, name=f"Config {label}", legendgroup=f"g{label}",
            mode='markers', marker=dict(color=color, size=2, opacity=0.15),
            showlegend=False,
            hovertemplate="<b>Weight: %{x:.4f}</b><br>Ratio: %{y:.4f}<extra></extra>"
        ), row=3, col=1
    )

    # --- Add Per-Run Statistics to Plot 3 ---
    bin_x, stats = calculate_binned_percentiles(ponders, relative_perf, bins=20)
    
    if bin_x is not None:
        # Median Line (Thick, Solid)
        fig.add_trace(go.Scatter(
            x=bin_x, y=stats[0.5], mode='lines', 
            name=f"Config {label} Median", 
            legendgroup=f"g{label}", # <--- Same group, so it toggles with the rest
            line=dict(color=color, width=3),
            showlegend=False,
            hovertemplate=f"<b>Config {label} Median</b><br>%{{y:.3f}}<extra></extra>"
        ), row=3, col=1)

        # 25th/75th Percentiles (Thinner, Dashed) - Indicates standard variance
        for p in [0.25, 0.75]:
            fig.add_trace(go.Scatter(
                x=bin_x, y=stats[p], mode='lines', 
                legendgroup=f"g{label}",
                line=dict(color=color, width=1.5, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ), row=3, col=1)

fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=1)
fig.update_xaxes(matches='x', row=2, col=1)

fig.update_layout(
    title="Run Comparison & Correlation Analysis", 
    height=1200, width=1000, 
    hovermode="closest"
)

fig.update_yaxes(title_text="Improvement Rate", row=1, col=1)
fig.update_yaxes(title_text="Ratio vs Median", row=2, col=1)
fig.update_xaxes(title_text="Step", row=2, col=1)

fig.update_xaxes(title_text="Ponder Weight", row=3, col=1)
fig.update_yaxes(title_text="Ratio vs Median", row=3, col=1)

fig.show()
fig.write_html("wandb_run_comparison_correlation.html")