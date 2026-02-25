import wandb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Cross-Entropy Loss key (from training logs)
# Common options: "train/CrossEntropyLoss", "train/ce_loss", "train/loss"
CE_LOSS_KEY = "eval CLMCrossEntropyWithPonderLoss"

# Step column for evals (evals are logged at seen_steps, not _step)
SEEN_STEPS_KEY = "seen_steps"

# Commonsense Accuracy (Higher is better)
ACC_COMMONSENSE = {
    "ARC-C": "eval_full/arc_challenge:rc::olmes:full",
    "ARC-E": "eval_full/arc_easy:rc::olmes:full",
    "HellaSwag": "eval_full/hellaswag:rc::olmes:full",
    "Lambada": "eval_full/lambada",
    "PIQA": "eval_full/piqa:rc::olmes:full",
    "Qasper": "eval_full/qasper_yesno:rc::olmes",
    "SocialIQA": "eval_full/socialiqa:rc::olmes:full",
    "Winogrande": "eval_full/winogrande:rc::olmes:full",
}

# Commonsense BPB (Lower is better)
BPB_COMMONSENSE = {
    "ARC-C BPB": "eval_full/arc_challenge:rc:bpb::olmes:full",
    "ARC-E BPB": "eval_full/arc_easy:rc:bpb::olmes:full",
    "HellaSwag BPB": "eval_full/hellaswag:rc:bpb::olmes:full",
    "Lambada BPB": "eval_full/lambada:bpb",
    "PIQA BPB": "eval_full/piqa:rc:bpb::olmes:full",
    "Qasper BPB": "eval_full/qasper_yesno:rc:bpb::olmes",
}

# Math BPB (Lower is better)
BPB_MATH = {
    "Algebra": "eval_full/minerva_math_algebra:bpb::olmes",
    "CountProb": "eval_full/minerva_math_counting_and_probability:bpb::olmes",
    "Geometry": "eval_full/minerva_math_geometry:bpb::olmes",
    "IntAlg": "eval_full/minerva_math_intermediate_algebra:bpb::olmes",
    "NumTheory": "eval_full/minerva_math_number_theory:bpb::olmes",
    "PreAlg": "eval_full/minerva_math_prealgebra:bpb::olmes",
    "PreCalc": "eval_full/minerva_math_precalculus:bpb::olmes",
}

# Default checkpoint steps (based on seen_steps)
DEFAULT_CHECKPOINTS = {
    "early": 5000,
    "mid": 20000,
    "end": 35000,
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_wandb_api(timeout: int = 120):
    """Get WandB API instance with timeout."""
    return wandb.Api(timeout=timeout)


def fetch_run_history(
    entity: str,
    project: str,
    run_id: str,
    samples: int = 1000000,
) -> pd.DataFrame:
    """
    Fetch full run history as DataFrame.
    
    Args:
        entity: WandB entity/username
        project: WandB project name
        run_id: Run ID
        samples: Max samples to fetch (set high to get all)
    
    Returns:
        DataFrame with all logged metrics
    """
    api = get_wandb_api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    print(f"  Loading history for {run.name} ({run.id}), state: {run.state}")
    
    # Use history() with pandas=True for full data
    df = run.history(samples=samples, pandas=True)
    
    # Add step column
    if '_step' in df.columns:
        df['step'] = df['_step']
    else:
        df['step'] = range(len(df))
    
    df = df.sort_values('step').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def get_ce_loss_at_step(
    df: pd.DataFrame,
    target_step: int,
    ce_key: str = CE_LOSS_KEY,
    step_key: str = "step",
    avg_window: int = 100,
) -> Optional[float]:
    """
    Get CE loss at a specific step, averaged over a window.
    
    Uses 'step' column (based on _step) for CE loss, NOT seen_steps.
    
    Args:
        df: DataFrame with training history
        target_step: Target step number
        ce_key: Key for CE loss in history
        step_key: Key for step number (default: 'step')
        avg_window: Number of steps to average (before target)
    
    Returns:
        Averaged CE loss or None if not available
    """
    if ce_key not in df.columns:
        # Try to find similar column
        ce_cols = [c for c in df.columns if 'loss' in c.lower() or 'ce' in c.lower()]
        if ce_cols:
            print(f"  Warning: '{ce_key}' not found. Available loss columns: {ce_cols[:5]}")
        return None
    
    if step_key not in df.columns:
        return None
    
    # Get non-null CE values
    mask = df[ce_key].notna() & df[step_key].notna()
    subset = df.loc[mask, [step_key, ce_key]].copy()
    
    if subset.empty:
        return None
    
    # Get values in window [target_step - avg_window, target_step]
    window_mask = (subset[step_key] >= target_step - avg_window) & (subset[step_key] <= target_step)
    window_df = subset[window_mask]
    
    if window_df.empty:
        # If no values in window, get closest value
        subset["_dist"] = abs(subset[step_key] - target_step)
        closest_idx = subset["_dist"].idxmin()
        return subset.loc[closest_idx, ce_key]
    
    return window_df[ce_key].mean()


def get_metric_at_seen_step(
    df: pd.DataFrame,
    target_seen_step: int,
    metric_key: str,
    seen_step_key: str = SEEN_STEPS_KEY,
) -> Optional[float]:
    """
    Get a metric value at a specific seen_steps value.
    
    IMPORTANT: Evals are logged at seen_steps checkpoints, not at _step.
    
    Args:
        df: DataFrame with evaluation history
        target_seen_step: Target seen_steps value (e.g., 5000, 20000, 35000)
        metric_key: Key for the metric
        seen_step_key: Column name for seen_steps
    
    Returns:
        Metric value or None if not available
    """
    if metric_key not in df.columns:
        return None
    
    if seen_step_key not in df.columns:
        print(f"  Warning: '{seen_step_key}' column not found!")
        return None
    
    # Get rows where this metric is logged (non-null)
    mask = df[metric_key].notna() & df[seen_step_key].notna()
    subset = df.loc[mask, [seen_step_key, metric_key]].copy()
    
    if subset.empty:
        return None
    
    # Find exact match at target seen_step
    exact = subset[subset[seen_step_key] == target_seen_step]
    if not exact.empty:
        return exact[metric_key].iloc[0]
    
    # If no exact match, find closest
    subset["_dist"] = abs(subset[seen_step_key] - target_seen_step)
    closest_idx = subset["_dist"].idxmin()
    closest_step = subset.loc[closest_idx, seen_step_key]
    
    # Warn if we're far from target
    if abs(closest_step - target_seen_step) > 1000:
        print(f"  Warning: No eval at seen_steps={target_seen_step} for {metric_key}, "
              f"using closest at {closest_step}")
    
    return subset.loc[closest_idx, metric_key]


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def compute_avg(values: List[Optional[float]]) -> Optional[float]:
    """Compute average of non-None values."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return np.mean(valid) if valid else None


def get_metrics_at_checkpoint(
    df: pd.DataFrame,
    seen_step: int,
    ce_avg_window: int = 100,
) -> Dict[str, Optional[float]]:
    """
    Get all metrics at a specific checkpoint (seen_steps).
    
    IMPORTANT: Evals are logged at seen_steps, CE loss uses regular step.
    
    Args:
        df: DataFrame with full run history
        seen_step: Checkpoint based on seen_steps (e.g., 5000, 20000, 35000)
        ce_avg_window: Window for averaging CE loss
    
    Returns:
        Dictionary with CE, Math BPB (avg), CS BPB (avg), CS Acc (avg), and all individual metrics
    """
    result = {}
    
    # CE Loss (averaged) - uses 'step' column, not seen_steps
    # Assuming step and seen_step are roughly aligned for CE loss
    result["CE"] = get_ce_loss_at_step(df, seen_step, CE_LOSS_KEY, step_key="step", avg_window=ce_avg_window)
    
    # CS Accuracy - uses seen_steps
    acc_values = []
    for label, key in ACC_COMMONSENSE.items():
        val = get_metric_at_seen_step(df, seen_step, key)
        result[label] = val
        if val is not None:
            acc_values.append(val)
    result["CS Acc"] = compute_avg(acc_values)
    
    # CS BPB - uses seen_steps
    cs_bpb_values = []
    for label, key in BPB_COMMONSENSE.items():
        val = get_metric_at_seen_step(df, seen_step, key)
        result[label] = val
        if val is not None:
            cs_bpb_values.append(val)
    result["CS BPB"] = compute_avg(cs_bpb_values)
    
    # Math BPB - uses seen_steps
    math_bpb_values = []
    for label, key in BPB_MATH.items():
        val = get_metric_at_seen_step(df, seen_step, key)
        result[label] = val
        if val is not None:
            math_bpb_values.append(val)
    result["Math BPB"] = compute_avg(math_bpb_values)
    
    return result


def fetch_run_data(
    entity: str,
    project: str,
    run_id: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch full history for a run.
    
    Args:
        entity: WandB entity/username
        project: WandB project name
        run_id: Run ID
    
    Returns:
        DataFrame with full history, or None if error
    """
    try:
        return fetch_run_history(entity, project, run_id)
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None


# =============================================================================
# TABLE GENERATION
# =============================================================================

def build_metrics_table(
    entity: str,
    project: str,
    experiments: List[Dict],
    checkpoints: Dict[str, int] = None,
    ce_avg_window: int = 100,
) -> pd.DataFrame:
    """
    Build a table with metrics at multiple checkpoints for multiple experiments.
    
    Fetches full history once per run, then extracts metrics at each checkpoint.
    
    Args:
        entity: WandB entity/username
        project: WandB project name
        experiments: List of dicts with 'name', 'run_id', and 'display_name'
        checkpoints: Dict mapping checkpoint names to seen_steps values
        ce_avg_window: Window for averaging CE loss
    
    Returns:
        DataFrame with multi-level columns (checkpoint, metric)
    """
    if checkpoints is None:
        checkpoints = DEFAULT_CHECKPOINTS
    
    rows = []
    
    for exp in experiments:
        run_id = exp["run_id"]
        display_name = exp.get("display_name", exp["name"])
        
        print(f"Fetching: {display_name} ({run_id})...")
        
        row = {"Model": display_name}
        
        # Fetch full history once
        df = fetch_run_data(entity, project, run_id)
        
        if df is None or df.empty:
            print(f"  Warning: No data for {display_name}")
            rows.append(row)
            continue
        
        # Check if seen_steps column exists
        if SEEN_STEPS_KEY not in df.columns:
            print(f"  Warning: '{SEEN_STEPS_KEY}' not in columns. Available: {list(df.columns)[:10]}...")
            # Try to find alternative
            alt_keys = [c for c in df.columns if 'seen' in c.lower() or 'step' in c.lower()]
            print(f"  Step-related columns: {alt_keys}")
        
        # Extract metrics at each checkpoint
        for ckpt_name, ckpt_step in checkpoints.items():
            metrics = get_metrics_at_checkpoint(df, ckpt_step, ce_avg_window)
            
            # Add to row with checkpoint prefix
            for metric_name, value in metrics.items():
                col_name = (ckpt_name, metric_name)
                row[col_name] = value
        
        rows.append(row)
    
    # Create DataFrame
    df_result = pd.DataFrame(rows)
    
    return df_result


def format_main_table(
    df: pd.DataFrame,
    checkpoints: List[str] = None,
    metrics: List[str] = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Format the main results table with selected metrics.
    
    Args:
        df: Full metrics DataFrame
        checkpoints: List of checkpoint names to include
        metrics: List of metric names to include
        decimals: Number of decimal places
    
    Returns:
        Formatted DataFrame
    """
    if checkpoints is None:
        checkpoints = ["early", "mid", "end"]
    if metrics is None:
        metrics = ["CE", "Math BPB", "CS BPB", "CS Acc"]
    
    # Select columns
    cols = ["Model"]
    for ckpt in checkpoints:
        for metric in metrics:
            col = (ckpt, metric)
            if col in df.columns:
                cols.append(col)
    
    result = df[cols].copy()
    
    # Round numeric columns
    for col in result.columns:
        if col != "Model":
            result[col] = pd.to_numeric(result[col], errors="coerce").round(decimals)
    
    return result


def to_latex_multicolumn(
    df: pd.DataFrame,
    checkpoints: List[str] = None,
    metrics: List[str] = None,
    caption: str = "Experimental Results",
    label: str = "tab:results",
) -> str:
    """
    Generate LaTeX table with multicolumn headers.
    
    Args:
        df: Formatted DataFrame
        checkpoints: Checkpoint names for headers
        metrics: Metric names for subheaders
        caption: Table caption
        label: Table label
    
    Returns:
        LaTeX code as string
    """
    if checkpoints is None:
        checkpoints = ["early", "mid", "end"]
    if metrics is None:
        metrics = ["CE", "Math BPB", "CS BPB", "CS Acc"]
    
    n_metrics = len(metrics)
    n_ckpts = len(checkpoints)
    
    # Build column format
    col_format = "l" + "c" * (n_metrics * n_ckpts)
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\toprule")
    
    # Checkpoint header row (multicolumn)
    header1 = [""]
    for ckpt in checkpoints:
        ckpt_display = ckpt.capitalize()
        header1.append(r"\multicolumn{" + str(n_metrics) + r"}{c}{" + ckpt_display + "}")
    lines.append(" & ".join(header1) + r" \\")
    
    # Add cmidrules under each checkpoint
    cmidrules = []
    for i, ckpt in enumerate(checkpoints):
        start = 2 + i * n_metrics
        end = start + n_metrics - 1
        cmidrules.append(r"\cmidrule(lr){" + str(start) + "-" + str(end) + "}")
    lines.append(" ".join(cmidrules))
    
    # Metric header row
    header2 = ["Model"]
    for ckpt in checkpoints:
        for metric in metrics:
            # Add arrows for direction
            if metric in ["CE", "Math BPB", "CS BPB"]:
                header2.append(metric + r" $\downarrow$")
            else:
                header2.append(metric + r" $\uparrow$")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        row_data = [str(row["Model"])]
        for ckpt in checkpoints:
            for metric in metrics:
                val = row.get((ckpt, metric), None)
                if pd.isna(val) or val is None:
                    row_data.append("-")
                else:
                    row_data.append(f"{val:.4f}")
        lines.append(" & ".join(row_data) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def to_latex_transposed(
    df: pd.DataFrame,
    checkpoint: str,
    model_order: List[str],
    short_names: Dict[str, str],
    caption: str = "Results",
    label: str = "tab:results",
    bold_best: bool = True,
    decimals: int = 4,
) -> str:
    """
    Generate LaTeX table with benchmarks as rows and models as columns.
    Includes AVG row for each section.
    
    Args:
        df: Full metrics DataFrame from build_metrics_table
        checkpoint: Which checkpoint to use ("early", "mid", "end")
        model_order: List of display names in desired column order
        short_names: Dict mapping display names to short column headers
        caption: Table caption
        label: Table label
        bold_best: Whether to bold best values per row
        decimals: Number of decimal places
    
    Returns:
        LaTeX code as string
    """
    # Filter to models we want, in order
    df_filtered = df[df["Model"].isin(model_order)].copy()
    df_filtered["_order"] = df_filtered["Model"].apply(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    df_filtered = df_filtered.sort_values("_order")
    
    models = df_filtered["Model"].tolist()
    n_models = len(models)
    
    # Build header with short names
    header_names = [short_names.get(m, m) for m in models]
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    
    col_format = "l" + "c" * n_models
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\toprule")
    
    # Header row
    header = ["Bench"] + header_names
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    
    def get_values_for_bench(bench_label: str) -> List[Optional[float]]:
        """Get values for a benchmark across all models."""
        values = []
        for _, row in df_filtered.iterrows():
            val = row.get((checkpoint, bench_label), None)
            if pd.isna(val) or val is None:
                values.append(None)
            else:
                values.append(val)
        return values
    
    def format_row(bench_label: str, display_label: str, higher_better: bool, values: List[Optional[float]] = None):
        """Format a single benchmark row with optional bolding."""
        if values is None:
            values = get_values_for_bench(bench_label)
        
        # Find best value
        valid_values = [v for v in values if v is not None]
        if valid_values and bold_best:
            if higher_better:
                best_val = max(valid_values)
            else:
                best_val = min(valid_values)
        else:
            best_val = None
        
        # Format values
        formatted = [display_label]
        for val in values:
            if val is None:
                formatted.append("-")
            else:
                val_str = f"{val:.{decimals}f}"
                # Bold if this is the best
                if bold_best and best_val is not None and abs(val - best_val) < 1e-9:
                    val_str = r"\textbf{" + val_str + "}"
                formatted.append(val_str)
        
        return " & ".join(formatted) + r" \\"
    
    def compute_avg_row(bench_dict: Dict[str, str], higher_better: bool) -> str:
        """Compute AVG row for a section."""
        # For each model, compute average across benchmarks
        avg_values = []
        for _, row in df_filtered.iterrows():
            model_vals = []
            for bench_label in bench_dict.keys():
                val = row.get((checkpoint, bench_label), None)
                if val is not None and not pd.isna(val):
                    model_vals.append(val)
            if model_vals:
                avg_values.append(np.mean(model_vals))
            else:
                avg_values.append(None)
        
        return format_row("AVG", r"\textit{AVG}", higher_better, avg_values)
    
    # CS Accuracy section
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Commonsense Accuracy} $\uparrow$} \\")
    for bench in ACC_COMMONSENSE.keys():
        lines.append(format_row(bench, bench, higher_better=True))
    lines.append(compute_avg_row(ACC_COMMONSENSE, higher_better=True))
    
    lines.append(r"\midrule")
    
    # CS BPB section
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Commonsense BPB} $\downarrow$} \\")
    for bench, key in BPB_COMMONSENSE.items():
        display_name = bench.replace(" BPB", "")
        lines.append(format_row(bench, display_name, higher_better=False))
    lines.append(compute_avg_row(BPB_COMMONSENSE, higher_better=False))
    
    lines.append(r"\midrule")
    
    # Math BPB section
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Math BPB} $\downarrow$} \\")
    for bench in BPB_MATH.keys():
        lines.append(format_row(bench, bench, higher_better=False))
    lines.append(compute_avg_row(BPB_MATH, higher_better=False))
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def to_latex_other_runs(
    df: pd.DataFrame,
    checkpoint: str,
    exclude_models: List[str],
    short_names: Dict[str, str],
    caption: str = "Other Runs Results",
    label: str = "tab:other_runs",
    bold_best: bool = True,
    decimals: int = 4,
) -> str:
    """
    Generate LaTeX table for runs NOT in the main table.
    Same transposed format with abbreviated names.
    
    Args:
        df: Full metrics DataFrame
        checkpoint: Which checkpoint ("early", "mid", "end")
        exclude_models: List of model names to EXCLUDE (already in main table)
        short_names: Dict mapping full names to short column headers
        caption: Table caption
        label: Table label
        bold_best: Whether to bold best values
        decimals: Decimal places
    
    Returns:
        LaTeX code
    """
    # Filter to models NOT in exclude list
    df_filtered = df[~df["Model"].isin(exclude_models)].copy()
    
    if df_filtered.empty:
        return "% No additional runs to show"
    
    models = df_filtered["Model"].tolist()
    n_models = len(models)
    
    # Build header with short names
    header_names = [short_names.get(m, m) for m in models]
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    
    col_format = "l" + "c" * n_models
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\toprule")
    
    # Header row
    header = ["Bench"] + header_names
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    
    def get_values_for_bench(bench_label: str) -> List[Optional[float]]:
        """Get values for a benchmark across all models."""
        values = []
        for _, row in df_filtered.iterrows():
            val = row.get((checkpoint, bench_label), None)
            if pd.isna(val) or val is None:
                values.append(None)
            else:
                values.append(val)
        return values
    
    def format_row(bench_label: str, display_label: str, higher_better: bool, values: List[Optional[float]] = None):
        """Format a single benchmark row with optional bolding."""
        if values is None:
            values = get_values_for_bench(bench_label)
        
        valid_values = [v for v in values if v is not None]
        if valid_values and bold_best:
            best_val = max(valid_values) if higher_better else min(valid_values)
        else:
            best_val = None
        
        formatted = [display_label]
        for val in values:
            if val is None:
                formatted.append("-")
            else:
                val_str = f"{val:.{decimals}f}"
                if bold_best and best_val is not None and abs(val - best_val) < 1e-9:
                    val_str = r"\textbf{" + val_str + "}"
                formatted.append(val_str)
        
        return " & ".join(formatted) + r" \\"
    
    def compute_avg_row(bench_dict: Dict[str, str], higher_better: bool) -> str:
        """Compute AVG row for a section."""
        avg_values = []
        for _, row in df_filtered.iterrows():
            model_vals = []
            for bench_label in bench_dict.keys():
                val = row.get((checkpoint, bench_label), None)
                if val is not None and not pd.isna(val):
                    model_vals.append(val)
            if model_vals:
                avg_values.append(np.mean(model_vals))
            else:
                avg_values.append(None)
        
        return format_row("AVG", r"\textit{AVG}", higher_better, avg_values)
    
    # CS Accuracy
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{CS Acc} $\uparrow$} \\")
    for bench in ACC_COMMONSENSE.keys():
        lines.append(format_row(bench, bench, higher_better=True))
    lines.append(compute_avg_row(ACC_COMMONSENSE, higher_better=True))
    
    lines.append(r"\midrule")
    
    # CS BPB
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{CS BPB} $\downarrow$} \\")
    for bench in BPB_COMMONSENSE.keys():
        display = bench.replace(" BPB", "")
        lines.append(format_row(bench, display, higher_better=False))
    lines.append(compute_avg_row(BPB_COMMONSENSE, higher_better=False))
    
    lines.append(r"\midrule")
    
    # Math BPB
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Math BPB} $\downarrow$} \\")
    for bench in BPB_MATH.keys():
        lines.append(format_row(bench, bench, higher_better=False))
    lines.append(compute_avg_row(BPB_MATH, higher_better=False))
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# Keep old function for backwards compatibility but mark deprecated
def to_latex_all_runs(
    df: pd.DataFrame,
    checkpoint: str,
    caption: str = "All Runs Results",
    label: str = "tab:all_runs",
    bold_best: bool = True,
    decimals: int = 4,
) -> str:
    """
    Generate LaTeX table for ALL runs in the DataFrame.
    
    Same format as to_latex_transposed but includes all models.
    
    Args:
        df: Full metrics DataFrame
        checkpoint: Which checkpoint ("early", "mid", "end")
        caption: Table caption
        label: Table label
        bold_best: Whether to bold best values
        decimals: Decimal places
    
    Returns:
        LaTeX code
    """
    models = df["Model"].tolist()
    n_models = len(models)
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    
    col_format = "l" + "c" * n_models
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\toprule")
    
    # Header - use model names directly (might be long)
    header = ["Bench"] + models
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    
    def format_row(bench_label: str, higher_better: bool):
        values = []
        for _, row in df.iterrows():
            val = row.get((checkpoint, bench_label), None)
            if pd.isna(val) or val is None:
                values.append(None)
            else:
                values.append(val)
        
        valid_values = [v for v in values if v is not None]
        if valid_values and bold_best:
            best_val = max(valid_values) if higher_better else min(valid_values)
        else:
            best_val = None
        
        formatted = [bench_label]
        for val in values:
            if val is None:
                formatted.append("-")
            else:
                val_str = f"{val:.{decimals}f}"
                if bold_best and best_val is not None and abs(val - best_val) < 1e-9:
                    val_str = r"\textbf{" + val_str + "}"
                formatted.append(val_str)
        
        return " & ".join(formatted) + r" \\"
    
    # CS Accuracy
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{CS Acc} $\uparrow$} \\")
    for bench in ACC_COMMONSENSE.keys():
        lines.append(format_row(bench, higher_better=True))
    
    lines.append(r"\midrule")
    
    # CS BPB
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{CS BPB} $\downarrow$} \\")
    for bench in BPB_COMMONSENSE.keys():
        display = bench.replace(" BPB", "")
        row = format_row(bench, higher_better=False)
        lines.append(row.replace(bench + " &", display + " &"))
    
    lines.append(r"\midrule")
    
    # Math BPB
    lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Math BPB} $\downarrow$} \\")
    for bench in BPB_MATH.keys():
        lines.append(format_row(bench, higher_better=False))
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def to_latex_appendix(
    df: pd.DataFrame,
    checkpoints: List[str] = None,
    caption: str = "Full Benchmark Results",
    label: str = "tab:appendix",
) -> str:
    """
    Generate LaTeX appendix table with all individual benchmarks.
    Transposed: benchmarks as rows, models as columns.
    
    Args:
        df: Full metrics DataFrame
        checkpoints: Checkpoint names
        caption: Table caption
        label: Table label
    
    Returns:
        LaTeX code as string
    """
    if checkpoints is None:
        checkpoints = ["early", "mid", "end"]
    
    # Get all benchmark names
    all_benchmarks = (
        list(ACC_COMMONSENSE.keys()) + 
        list(BPB_COMMONSENSE.keys()) + 
        list(BPB_MATH.keys())
    )
    
    models = df["Model"].tolist()
    n_models = len(models)
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    
    # One table per checkpoint
    for ckpt in checkpoints:
        lines.append(r"\vspace{0.5em}")
        lines.append(r"\textbf{" + ckpt.capitalize() + r" (Step " + str(DEFAULT_CHECKPOINTS.get(ckpt, "?")) + r")}")
        lines.append(r"\vspace{0.3em}")
        lines.append("")
        
        col_format = "l" + "c" * n_models
        lines.append(r"\begin{tabular}{" + col_format + "}")
        lines.append(r"\toprule")
        
        # Header: Benchmark | Model1 | Model2 | ...
        header = ["Benchmark"] + models
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")
        
        # CS Accuracy section
        lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Commonsense Accuracy} $\uparrow$} \\")
        for bench in ACC_COMMONSENSE.keys():
            row_data = [bench]
            for _, row in df.iterrows():
                val = row.get((ckpt, bench), None)
                if pd.isna(val) or val is None:
                    row_data.append("-")
                else:
                    row_data.append(f"{val:.4f}")
            lines.append(" & ".join(row_data) + r" \\")
        
        lines.append(r"\midrule")
        
        # CS BPB section
        lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Commonsense BPB} $\downarrow$} \\")
        for bench in BPB_COMMONSENSE.keys():
            row_data = [bench]
            for _, row in df.iterrows():
                val = row.get((ckpt, bench), None)
                if pd.isna(val) or val is None:
                    row_data.append("-")
                else:
                    row_data.append(f"{val:.4f}")
            lines.append(" & ".join(row_data) + r" \\")
        
        lines.append(r"\midrule")
        
        # Math BPB section
        lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Math BPB} $\downarrow$} \\")
        for bench in BPB_MATH.keys():
            row_data = [bench]
            for _, row in df.iterrows():
                val = row.get((ckpt, bench), None)
                if pd.isna(val) or val is None:
                    row_data.append("-")
                else:
                    row_data.append(f"{val:.4f}")
            lines.append(" & ".join(row_data) + r" \\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")
    
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# =============================================================================
# DISPLAY NAME MAPPINGS
# =============================================================================

def get_display_name(name: str) -> str:
    """
    Convert experiment name to clean display name.
    """
    mappings = {
        # Baselines
        "baseline_buckets_mvd": "Baseline",
        "baseline_isoflop_to_L1024G512memory_and_3loops": "Baseline IsoFLOP→Memory",
        "baseline_isoparam_to_L1024G512memory": "Baseline IsoParam→Memory",
        
        # Loop variants
        "loop1_L512G1024": "Loop 1 (L512 G1024)",
        "loop1_L512G4096": "Loop 1 (L512 G4096)",
        "loop1_L1024G512": "Loop 1 (L1024 G512)",
        "loop1_L4096G512": "Loop 1 (L4096 G512)",
        
        # Loop 3
        "loop3_buckets_mvd": "Loop 3",
        "loop3_iso_mvd": "Loop 3 IsoFLOP",
        "loop3_L1024G512_cyclical": "Loop 3 Cyclical",
        "loop3_L1024G512_ponder001": "Loop 3 Ponder 0.01",
        "loop3_L1024G512_ponder-001": "Loop 3 Ponder -0.01",
        
        # Loop 3 Memory variants
        "loop3_L1024G512_individualMemory_frozenmem": "Loop 3 Memory (Frozen)",
        "loop3_L1024G512_individualMemory_init0": "Loop 3 Memory (Init 0)",
        "loop3_L1024G512_individualMemory_init3": "Loop 3 Memory (Init 3)",
        "loop3_L1024G512_individualMemory_init-3": "Loop 3 Memory (Init -3)",
        
        # High loops
        "loop5_L1024G512": "Loop 5",
        "loop5_buckets_mvd": "Loop 5",
        "loop5_iso_mvd": "Loop 5 IsoFLOP",
        "loop7_buckets_mvd": "Loop 7",
        "loop9_buckets": "Loop 9",
    }
    return mappings.get(name, name)


def add_display_names(experiments: List[Dict]) -> List[Dict]:
    """Add display_name field to experiment list."""
    result = []
    for exp in experiments:
        exp_copy = exp.copy()
        if "display_name" not in exp_copy:
            exp_copy["display_name"] = get_display_name(exp["name"])
        result.append(exp_copy)
    return result


# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================

def inspect_run(
    entity: str,
    project: str,
    run_id: str,
) -> Dict:
    """
    Inspect a run to understand its structure.
    
    Returns info about columns, seen_steps values, and available metrics.
    """
    df = fetch_run_history(entity, project, run_id)
    
    if df is None or df.empty:
        return {"error": "No data"}
    
    info = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }
    
    # Check for step columns
    step_cols = [c for c in df.columns if 'step' in c.lower()]
    info["step_columns"] = step_cols
    
    # Check seen_steps values
    if SEEN_STEPS_KEY in df.columns:
        seen_steps = df[SEEN_STEPS_KEY].dropna().unique()
        info["seen_steps_values"] = sorted(seen_steps)
        info["seen_steps_count"] = len(seen_steps)
    
    # Check eval columns
    eval_cols = [c for c in df.columns if c.startswith('eval_full/')]
    info["eval_columns_count"] = len(eval_cols)
    info["eval_columns_sample"] = eval_cols[:10]
    
    # Check loss columns
    loss_cols = [c for c in df.columns if 'loss' in c.lower() or 'ce' in c.lower()]
    info["loss_columns"] = loss_cols
    
    return info


def print_available_checkpoints(
    entity: str,
    project: str,
    run_id: str,
):
    """Print available seen_steps checkpoints for a run."""
    info = inspect_run(entity, project, run_id)
    
    if "error" in info:
        print(f"Error: {info['error']}")
        return
    
    print(f"Run {run_id}:")
    print(f"  Total rows: {info['total_rows']}")
    print(f"  Step columns: {info['step_columns']}")
    
    if 'seen_steps_values' in info:
        print(f"  Available seen_steps checkpoints: {info['seen_steps_values']}")
    else:
        print(f"  Warning: '{SEEN_STEPS_KEY}' column not found!")
    
    print(f"  Eval columns: {info['eval_columns_count']}")
    print(f"  Loss columns: {info['loss_columns']}")