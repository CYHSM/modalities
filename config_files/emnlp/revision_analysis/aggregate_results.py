#!/usr/bin/env python3
import json
from pathlib import Path

def format_cell(val, baseline_val):
    if val is None or baseline_val is None:
        return "N/A"
    delta = val - baseline_val
    sign = "+" if delta >= 0 else "-"
    return f"{val:.3f} \\ssub{{{sign}{abs(delta):.2f}}}"

def generate_latex_table(results_dir="config_files/emnlp/revision_analysis/results"):
    base_path = Path(results_dir)
    if not base_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Look for model directories
    model_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print(f"No model directories found in {results_dir}")
        return

    print("======================================================")
    print(" LaTeX Output ")
    print("======================================================")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Intervention} & \\textbf{GSM8K} & \\textbf{TriviaQA} & \\textbf{WikiText-103} \\\\")
    
    # We want these exact columns
    source_keys = ["gsm8k", "triviaqa", "wikitext"]

    for model_dir in model_dirs:
        model_name = model_dir.name
        
        baseline_path = model_dir / "learned.json"
        if not baseline_path.exists():
            continue
            
        with open(baseline_path) as f:
            baseline_data = json.load(f)
            
        def get_source_loss(data, src):
            # Try exact match, or fallback to first matching prefix (e.g. wikitext-103)
            if src in data.get("sources", {}):
                return data["sources"][src]["mean_loss"]
            for k, v in data.get("sources", {}).items():
                if src in k.lower():
                    return v["mean_loss"]
            return None

        baseline_losses = {src: get_source_loss(baseline_data, src) for src in source_keys}
        
        print(f"\\midrule")
        print(f"\\multicolumn{{4}}{{l}}{{\\emph{{Token-class gate overrides ({model_name})}}}} \\\\")
        
        # Define the row definitions
        rows = [
            ("Arithmetic forced wide ($g_d{=}0, g_w{=}1$)", "gd0p0_gw1p0_mask_NUM+SYM.json"),
            ("Random control forced wide", "gd0p0_gw1p0_mask_RANDOM.json"),
            ("Function words forced deep ($g_d{=}1, g_w{=}0$)", "gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+VERB.json"),
            ("Random control forced deep", "gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOM+VERB.json")
        ]
        
        for row_label, filename in rows:
            abl_path = model_dir / filename
            if not abl_path.exists():
                print(f"\\quad {row_label:<45} & N/A & N/A & N/A \\\\")
                continue
                
            with open(abl_path) as f:
                abl_data = json.load(f)
                
            abl_losses = {src: get_source_loss(abl_data, src) for src in source_keys}
            
            cells = []
            for src in source_keys:
                cells.append(format_cell(abl_losses[src], baseline_losses[src]))
                
            print(f"\\quad {row_label:<45} & {cells[0]:<20} & {cells[1]:<20} & {cells[2]:<20} \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Token-class gate overrides analysis.}")
    print("\\label{tab:gate_overrides}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_latex_table()
