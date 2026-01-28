import wandb
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
ENTITY = "cyhsm" 
PROJECT = "loop"

# Target step to compare all models at
TARGET_SEEN_STEPS = 10000

# Paste your specific Run IDs here
RUN_IDS = {
    "Baseline":             "7lswnrp7",
    "Loop 3":               "f24cucd6",
    "Loop 3 Isoflop":       "kb904d0r",
    "Loop 3 Memory 1024":   "775dn24r",
    "Loop 3 Memory 4096":   "f3jqn4le",
}

# --- 2. METRIC GROUPS ---

# Group A: Commonsense Accuracy (Higher is better)
acc_commonsense = {
    "ARC-C": "eval/arc_challenge:rc::olmes:full",
    "HellaSwag": "eval/hellaswag:rc::olmes:full",
    "ARC-E": "eval/arc_easy:rc::olmes:full",
    "Lambada": "eval/lambada",
    "PIQA": "eval/piqa:rc::olmes:full",
    "Qasper": "eval/qasper_yesno:rc::olmes",
    "SocialIQA": "eval/socialiqa:rc::olmes:full",
    "Winogrande": "eval/winogrande:rc::olmes:full"
}

# Group B: Commonsense BPB (Lower is better)
bpb_commonsense = {
    "Qasper BPB": "eval/qasper_yesno:rc:bpb::olmes",
    "PIQA BPB": "eval/piqa:rc:bpb::olmes:full",
    "Lambada BPB": "eval/lambada:bpb",
    "HellaSwag BPB": "eval/hellaswag:rc:bpb::olmes:full",
    "ARC-E BPB": "eval/arc_easy:rc:bpb::olmes:full",
    "ARC-C BPB": "eval/arc_challenge:rc:bpb::olmes:full"
}

# Group C: Math BPB (Lower is better)
bpb_math = {
    "PreCalc": "eval/minerva_math_precalculus:bpb::olmes",
    "PreAlg": "eval/minerva_math_prealgebra:bpb::olmes",
    "NumTheory": "eval/minerva_math_number_theory:bpb::olmes",
    "IntAlg": "eval/minerva_math_intermediate_algebra:bpb::olmes",
    "Geometry": "eval/minerva_math_geometry:bpb::olmes",
    "CountProb": "eval/minerva_math_counting_and_probability:bpb::olmes",
    "Algebra": "eval/minerva_math_algebra:bpb::olmes"
}

# Group D: Other (Lower is better)
other_metrics = {
    "CE Loss": "eval CLMCrossEntropyWithPonderLoss"
}

# --- 3. HELPER: Highlight Best Value ---
def highlight_best(df, minimize_cols, maximize_cols, format_type='latex'):
    df_out = df.copy()
    
    for col in minimize_cols:
        if col in df_out.columns:
            numeric_series = pd.to_numeric(df_out[col], errors='coerce')
            if numeric_series.notna().any():
                min_val = numeric_series.min()
                is_min = numeric_series == min_val
                if format_type == 'latex':
                    df_out.loc[is_min, col] = df_out.loc[is_min, col].apply(lambda x: f"\\textbf{{{x}}}")
                else:
                    df_out.loc[is_min, col] = df_out.loc[is_min, col].apply(lambda x: f"<b>{x}</b>")

    for col in maximize_cols:
        if col in df_out.columns:
            numeric_series = pd.to_numeric(df_out[col], errors='coerce')
            if numeric_series.notna().any():
                max_val = numeric_series.max()
                is_max = numeric_series == max_val
                if format_type == 'latex':
                    df_out.loc[is_max, col] = df_out.loc[is_max, col].apply(lambda x: f"\\textbf{{{x}}}")
                else:
                    df_out.loc[is_max, col] = df_out.loc[is_max, col].apply(lambda x: f"<b>{x}</b>")
                    
    return df_out

# --- 4. FETCHING LOGIC ---
def get_runs_data():
    api = wandb.Api()
    data_rows = []

    # All eval metric keys we care about
    all_eval_keys = (
        list(acc_commonsense.values()) + 
        list(bpb_commonsense.values()) + 
        list(bpb_math.values()) + 
        list(other_metrics.values())
    )

    print(f"Fetching data for {len(RUN_IDS)} runs at seen_steps={TARGET_SEEN_STEPS}...")

    for display_name, run_id in RUN_IDS.items():
        if "PLACEHOLDER" in run_id:
            print(f"⚠️  {display_name}: Skipping (placeholder ID)")
            data_rows.append({"Model": display_name})
            continue
            
        try:
            print(f"Processing {display_name} (Run ID: {run_id})")
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            
            # Fetch full history with large sample size
            history = run.history(samples=100000, pandas=True)
            
            if history.empty:
                print(f"⚠️  {display_name}: No history data found")
                data_rows.append({"Model": display_name})
                continue
            
            # Check if seen_steps column exists
            if 'seen_steps' not in history.columns:
                print(f"⚠️  {display_name}: No 'seen_steps' column found")
                data_rows.append({"Model": display_name})
                continue
            
            # Find eval columns that exist in this run
            existing_eval_cols = [c for c in all_eval_keys if c in history.columns]
            
            if not existing_eval_cols:
                print(f"⚠️  {display_name}: No eval columns found")
                data_rows.append({"Model": display_name})
                continue
            
            # Filter to rows with eval data (any eval metric non-null)
            eval_mask = history[existing_eval_cols].notna().any(axis=1)
            eval_data = history[eval_mask]
            
            # Filter to target seen_steps
            target_data = eval_data[eval_data['seen_steps'] == TARGET_SEEN_STEPS]
            
            if target_data.empty:
                available_steps = sorted(eval_data['seen_steps'].dropna().unique())
                print(f"⚠️  {display_name}: No data at seen_steps={TARGET_SEEN_STEPS}")
                print(f"   Available seen_steps: {available_steps}")
                data_rows.append({"Model": display_name})
                continue
            
            # Take the first matching row (they should all be the same for a given seen_steps)
            target_row = target_data.iloc[0]
            print(f"   ✓ Found data at seen_steps={TARGET_SEEN_STEPS} (_step={target_row['_step']})")

            row = {"Model": display_name}
            def get_val(key): 
                if key in target_row.index:
                    val = target_row[key]
                    return val if pd.notna(val) else None
                return None

            # 1. CE Loss - this is logged by _step, not seen_steps
            ce_loss_key = other_metrics["CE Loss"]
            if ce_loss_key in history.columns:
                ce_loss_data = history[history[ce_loss_key].notna()]
                ce_at_step = ce_loss_data[ce_loss_data['_step'] == TARGET_SEEN_STEPS]
                if not ce_at_step.empty:
                    row["CE Loss"] = ce_at_step.iloc[0][ce_loss_key]
                else:
                    # Fallback: get closest _step
                    closest_idx = (ce_loss_data['_step'] - TARGET_SEEN_STEPS).abs().idxmin()
                    row["CE Loss"] = ce_loss_data.loc[closest_idx, ce_loss_key]
                    print(f"   ⚠️  CE Loss: used _step={ce_loss_data.loc[closest_idx, '_step']} (closest to {TARGET_SEEN_STEPS})")
            else:
                row["CE Loss"] = None

            # 2. CS Acc
            vals_acc = []
            for label, key in acc_commonsense.items():
                val = get_val(key)
                row[label] = val
                if val is not None: vals_acc.append(val)
            row["AVG CS Acc"] = np.mean(vals_acc) if vals_acc else None

            # 3. CS BPB
            vals_cs_bpb = []
            for label, key in bpb_commonsense.items():
                val = get_val(key)
                row[label] = val
                if val is not None: vals_cs_bpb.append(val)
            row["AVG CS BPB"] = np.mean(vals_cs_bpb) if vals_cs_bpb else None

            # 4. Math BPB
            vals_math = []
            for label, key in bpb_math.items():
                val = get_val(key)
                row[label] = val
                if val is not None: vals_math.append(val)
            row["AVG Math BPB"] = np.mean(vals_math) if vals_math else None

            data_rows.append(row)

        except Exception as e:
            print(f"Error fetching {display_name}: {e}")
            import traceback
            traceback.print_exc()
            data_rows.append({"Model": display_name})

    return pd.DataFrame(data_rows)

if __name__ == "__main__":
    df = get_runs_data()
    df = df.round(4)

    # --- DEFINE COLUMNS TO MINIMIZE VS MAXIMIZE ---
    cols_min = ["CE Loss", "AVG CS BPB", "AVG Math BPB"] + \
               list(bpb_commonsense.keys()) + list(bpb_math.keys())
               
    cols_max = ["AVG CS Acc"] + list(acc_commonsense.keys())

    # --- DEFINE ORDER (Averages first) ---
    base_order = ["Model", "CE Loss", "AVG CS Acc", "AVG CS BPB", "AVG Math BPB"]
    detail_order = list(acc_commonsense.keys()) + list(bpb_commonsense.keys()) + list(bpb_math.keys())
    
    final_col_order = base_order + detail_order
    final_col_order = [c for c in final_col_order if c in df.columns]
    
    df = df[final_col_order]

    # ==========================================
    # 1. HTML EXPORT
    # ==========================================
    
    df_html_data = highlight_best(df, cols_min, cols_max, format_type='html')

    html_map = {
        "CE Loss": "CE Loss ↓",
        "AVG CS Acc": "AVG CS Acc ↑",
        "AVG CS BPB": "AVG CS BPB ↓",
        "AVG Math BPB": "AVG Math BPB ↓"
    }
    for k in acc_commonsense: html_map[k] = f"{k} ↑"
    for k in bpb_commonsense: html_map[k] = f"{k} ↓"
    for k in bpb_math: html_map[k] = f"{k} ↓"
    
    df_html_data = df_html_data.rename(columns=html_map)
    
    html_table = df_html_data.to_html(index=False, border=0, classes="styled-table", escape=False)
    
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            .styled-table {{
                border-collapse: collapse; margin: 25px 0; font-size: 0.9em;
                min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            }}
            .styled-table thead tr {{
                background-color: #009879; color: #ffffff; text-align: left;
                position: sticky; top: 0;
            }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; text-align: center; }}
            .styled-table td:first-child {{ text-align: left; font-weight: bold; }}
            .styled-table tbody tr {{ border-bottom: 1px solid #dddddd; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .styled-table tbody tr:hover {{ background-color: #f1f1f1; }}
        </style>
    </head>
    <body>
        <h2>WandB Metrics @ seen_steps={TARGET_SEEN_STEPS} (Best values bolded)</h2>
        {html_table}
    </body>
    </html>
    """

    with open("results_step10k.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("\n✅ Saved 'results_step10k.html'")

    # ==========================================
    # 2. LATEX EXPORT
    # ==========================================
    
    df_latex_data = highlight_best(df, cols_min, cols_max, format_type='latex')

    latex_map = {
        "CE Loss": r"CE Loss $\downarrow$",
        "AVG CS Acc": r"\textbf{AVG CS Acc} $\uparrow$",
        "AVG CS BPB": r"\textbf{AVG CS BPB} $\downarrow$",
        "AVG Math BPB": r"\textbf{AVG Math BPB} $\downarrow$"
    }
    for k in acc_commonsense: latex_map[k] = f"{k} $\\uparrow$"
    for k in bpb_commonsense: latex_map[k] = f"{k} $\\downarrow$"
    for k in bpb_math: latex_map[k] = f"{k} $\\downarrow$"
    
    df_latex_data = df_latex_data.rename(columns=latex_map)

    print("\n--- LATEX CODE ---")
    print(df_latex_data.to_latex(
        index=False, 
        escape=False,
        na_rep="-",
        column_format="l" + "c"*(len(df_latex_data.columns)-1)
    ))