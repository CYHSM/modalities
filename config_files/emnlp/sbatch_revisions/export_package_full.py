import os
import re
import json
import shutil
import yaml
import zipfile
from pathlib import Path

ALL_CONFIGS_TXT = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/emnlp/sbatch_revisions/all_configs.txt")
MODALITIES_ROOT = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities")
SCRATCH_REVISIONS = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions")
SCRATCH_MAIN = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp")

EXPORT_DIR = Path("/leonardo_scratch/large/userexternal/mfrey000/emnlp_eval_export_full")
CONFIGS_OUT_DIR = EXPORT_DIR / "configs"
METRICS_OUT_DIR = EXPORT_DIR / "metrics"
ZIP_OUT_FILE = Path("/leonardo_scratch/large/userexternal/mfrey000/emnlp_eval_export_full")

OLD_ZIP = Path("/leonardo_scratch/large/userexternal/mfrey000/emnlp_eval_export.zip")
TEMP_EXTRACT_DIR = Path("/leonardo_scratch/large/userexternal/mfrey000/emnlp_eval_export_temp")

def extract_config_params(config_path):
    try:
        with open(config_path, "r") as f:
            content = yaml.safe_load(f)
            
        settings = content.get("settings", {})
        model_raw = content.get("model_raw", {})
        model_config = model_raw.get("config", {})
        adaptive_config = model_config.get("adaptive_config", {})
        
        target_tokens = settings.get("training_target", {}).get("num_target_tokens", "N/A")
        
        params = {
            "n_layer": model_config.get("n_layer", "N/A"),
            "n_embd": model_config.get("n_embd", "N/A"),
            "max_loops": adaptive_config.get("max_loops", "N/A"),
            "wide_ffn_hidden": adaptive_config.get("wide_ffn_hidden", "N/A"),
            "ffn_hidden": model_config.get("ffn_hidden", "N/A"),
            "use_cross": adaptive_config.get("use_cross", "N/A"),
            "layer_types": adaptive_config.get("layer_types", []),
            "target_tokens": f"{target_tokens / 1e8:.1f}e7" if isinstance(target_tokens, (int, float)) else target_tokens
        }
        return params
    except Exception:
        return {}

def parse_metrics_jsonl(metrics_file):
    scores = {}
    if not metrics_file.exists():
        return scores
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_name = data.get("task_name")
                metrics = data.get("metrics", {})
                if task_name:
                    scores[task_name] = metrics
    except Exception as e:
        print(f"Error parsing metrics {metrics_file}: {e}")
        
    # Robust fallback: if metrics-all.jsonl has fewer than 24 tasks, load individual task-*-metrics.json files
    if len(scores) < 24:
        eval_dir = metrics_file.parent
        if eval_dir.exists():
            for item in os.listdir(eval_dir):
                if item.startswith("task-") and item.endswith("-metrics.json"):
                    try:
                        with open(eval_dir / item, "r") as f_task:
                            data = json.load(f_task)
                            task_name = data.get("task_name")
                            metrics = data.get("metrics", {})
                            if task_name:
                                scores[task_name] = metrics
                    except Exception as e:
                        pass
    return scores

def main():
    # 1. Unzip the old archive to temp dir
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)
    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)
    
    if OLD_ZIP.exists():
        print(f"Unzipping {OLD_ZIP} to {TEMP_EXTRACT_DIR}")
        with zipfile.ZipFile(OLD_ZIP, 'r') as zip_ref:
            zip_ref.extractall(TEMP_EXTRACT_DIR)
    else:
        print(f"Warning: {OLD_ZIP} not found. Cannot merge old evaluations.")
        
    # 2. Setup the export folders
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(CONFIGS_OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)

    if not ALL_CONFIGS_TXT.exists():
        print(f"Error: {ALL_CONFIGS_TXT} not found.")
        return

    with open(ALL_CONFIGS_TXT, "r") as f:
        config_lines = [line.strip() for line in f if line.strip()]

    # Collect run directories from scratch
    revisions_dirs = [SCRATCH_REVISIONS / d for d in os.listdir(SCRATCH_REVISIONS) if (SCRATCH_REVISIONS / d).is_dir() and d != "wandb"] if SCRATCH_REVISIONS.exists() else []
    main_dirs = [SCRATCH_MAIN / d for d in os.listdir(SCRATCH_MAIN) if (SCRATCH_MAIN / d).is_dir() and d != "wandb"] if SCRATCH_MAIN.exists() else []
    all_scratch_dirs = revisions_dirs + main_dirs

    markdown_lines = []
    markdown_lines.append("# EMNLP Revisions: Evaluation Results and Runs Catalog\n")
    markdown_lines.append("This directory contains the configs, raw evaluation metrics, and a compiled summary of the EMNLP revisions experiments. This package is structured to be read by standard data loading code or by another Gemini instance to analyze downstream performance and parameter correlation.\n")
    
    markdown_lines.append("## 📁 Package Structure")
    markdown_lines.append("```")
    markdown_lines.append("emnlp_eval_export_full/")
    markdown_lines.append("├── summary.md                  # This document")
    markdown_lines.append("├── configs/                     # Model architecture YAML files (numbered 01 to 21)")
    markdown_lines.append("│   ├── config_01.yaml")
    markdown_lines.append("│   └── ...")
    markdown_lines.append("└── metrics/                     # Raw down-stream evaluation results (metrics-all.jsonl format)")
    markdown_lines.append("    ├── run_01_2026-07-09_metrics.jsonl")
    markdown_lines.append("    └── ...")
    markdown_lines.append("```\n")

    markdown_lines.append("## 📊 Downstream Performance Summary Table\n")
    markdown_lines.append("| ID | Config File | Loops | FFN Hidden/Wide | Run Directory | GSM8k (Acc) | HellaSwag (Acc) | Arc-Challenge (Acc) | Paloma C4 (PPL) | Paloma Wiki (PPL) | Minerva Math Avg (bpb) |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    # We process all configs in the txt file
    for idx, config_rel_path in enumerate(config_lines, 1):
        config_path = MODALITIES_ROOT / config_rel_path
        config_name = config_path.name
        
        # Copy config file to package
        config_out_name = f"config_{idx:02d}.yaml"
        shutil.copy(config_path, CONFIGS_OUT_DIR / config_out_name)
        
        params = extract_config_params(config_path)
        
        # Find matching run directories on scratch
        matching_scratch_runs = []
        for d in all_scratch_dirs:
            if (d / config_name).exists():
                matching_scratch_runs.append(d)
        matching_scratch_runs.sort(key=lambda x: x.name)
        
        # Find matching run directories in the old zip
        old_zip_run_names = []
        old_metrics_dir = TEMP_EXTRACT_DIR / "metrics"
        if old_metrics_dir.exists():
            # Old files are named run_{idx:02d}_{run_name}_metrics.jsonl
            pattern = re.compile(rf"^run_{idx:02d}_(.*)_metrics\.jsonl$")
            for filename in os.listdir(old_metrics_dir):
                match = pattern.match(filename)
                if match:
                    old_zip_run_names.append(match.group(1))
                    
        # Union of run names, keeping order
        all_runs_dict = {}
        for r_dir in matching_scratch_runs:
            all_runs_dict[r_dir.name] = {
                "name": r_dir.name,
                "scratch_dir": r_dir,
                "old_zip_metrics": None
            }
        for r_name in old_zip_run_names:
            old_m_file = old_metrics_dir / f"run_{idx:02d}_{r_name}_metrics.jsonl"
            if r_name in all_runs_dict:
                all_runs_dict[r_name]["old_zip_metrics"] = old_m_file
            else:
                all_runs_dict[r_name] = {
                    "name": r_name,
                    "scratch_dir": None,
                    "old_zip_metrics": old_m_file
                }
                
        sorted_run_names = sorted(all_runs_dict.keys())
        
        if not sorted_run_names:
            ffn_str = f"D: {params.get('ffn_hidden')} / W: {params.get('wide_ffn_hidden')}"
            markdown_lines.append(f"| {idx} | `{config_name}` | {params.get('max_loops')} | {ffn_str} | *No runs completed* | - | - | - | - | - | - |")
            continue
            
        for r_name in sorted_run_names:
            run_info = all_runs_dict[r_name]
            run_dir = run_info["scratch_dir"]
            old_metrics_file = run_info["old_zip_metrics"]
            
            # Determine path to read metrics
            metrics_file_to_read = None
            copied_filename = f"run_{idx:02d}_{r_name}_metrics.jsonl"
            
            # Try to get it from scratch first
            if run_dir:
                checkpoint_dirs = []
                for item in run_dir.iterdir():
                    if item.is_dir() and item.name.startswith("eid_"):
                        checkpoint_dirs.append(item)
                
                def get_step(d: Path) -> int:
                    match = re.search(r"seen_steps_(\d+)", d.name)
                    return int(match.group(1)) if match else -1
                checkpoint_dirs.sort(key=get_step)
                
                if checkpoint_dirs:
                    last_cp_dir = checkpoint_dirs[-1]
                    step = get_step(last_cp_dir)
                    
                    candidate1 = last_cp_dir / "hf_checkpoint" / f"olmes_eval_{step}" / "metrics-all.jsonl"
                    candidate2 = last_cp_dir / "hf_checkpoint" / "olmes_eval" / "metrics-all.jsonl"
                    if candidate1.exists():
                        metrics_file_to_read = candidate1
                    elif candidate2.exists():
                        metrics_file_to_read = candidate2
                        
            # If scratch has metrics, copy and use it.
            if metrics_file_to_read:
                print(f"Found new/updated metrics on scratch for Config {idx} run {r_name}")
                shutil.copy(metrics_file_to_read, METRICS_OUT_DIR / copied_filename)
                scores = parse_metrics_jsonl(metrics_file_to_read)
            elif old_metrics_file and old_metrics_file.exists():
                print(f"Using metrics from old zip for Config {idx} run {r_name}")
                shutil.copy(old_metrics_file, METRICS_OUT_DIR / copied_filename)
                scores = parse_metrics_jsonl(old_metrics_file)
            else:
                scores = {}
                
            # Extract specific scores
            def get_score_val(task_name, metric_key):
                task_data = scores.get(task_name)
                if not task_data:
                    return "-"
                val = task_data.get(metric_key)
                return val if val is not None else "-"

            gsm8k = get_score_val("gsm8k", "primary_score")
            if isinstance(gsm8k, float):
                gsm8k = f"{gsm8k*100:.2f}%" if gsm8k <= 1.0 else f"{gsm8k:.2f}"
                
            hellaswag = get_score_val("hellaswag", "primary_score")
            if isinstance(hellaswag, float):
                hellaswag = f"{hellaswag*100:.2f}%" if hellaswag <= 1.0 else f"{hellaswag:.2f}"
                
            arc_challenge = get_score_val("arc_challenge", "primary_score")
            if isinstance(arc_challenge, float):
                arc_challenge = f"{arc_challenge*100:.2f}%" if arc_challenge <= 1.0 else f"{arc_challenge:.2f}"
            
            paloma_c4 = get_score_val("paloma_c4_en", "ppl_word")
            if isinstance(paloma_c4, float):
                paloma_c4 = f"{paloma_c4:.2f}"
                
            paloma_wiki = get_score_val("paloma_wikitext_103", "ppl_word")
            if isinstance(paloma_wiki, float):
                paloma_wiki = f"{paloma_wiki:.2f}"
            
            # Average bpb across Minerva Math subtasks
            math_scores = []
            for k, m_dict in scores.items():
                if "minerva_math" in k:
                    val = m_dict.get("primary_score")
                    if isinstance(val, (int, float)):
                        math_scores.append(val)
            minerva_math_avg = f"{sum(math_scores)/len(math_scores):.4f}" if math_scores else "-"
            
            # Determine run label / type
            if run_dir:
                run_lbl = f"`revisions/{r_name[:10]}`" if run_dir.parent == SCRATCH_REVISIONS else f"`main/{r_name[:10]}`"
            else:
                run_lbl = f"`revisions/{r_name[:10]} (Preserved)`"
                
            ffn_str = f"D: {params.get('ffn_hidden')} / W: {params.get('wide_ffn_hidden')}"
            
            markdown_lines.append(
                f"| {idx} | `{config_name}` | {params.get('max_loops')} | {ffn_str} | {run_lbl} | {gsm8k} | {hellaswag} | {arc_challenge} | {paloma_c4} | {paloma_wiki} | {minerva_math_avg} |"
            )

    markdown_lines.append("\n## 📝 Configurations & Runs Directory Details\n")
    
    for idx, config_rel_path in enumerate(config_lines, 1):
        config_path = MODALITIES_ROOT / config_rel_path
        config_name = config_path.name
        params = extract_config_params(config_path)
        
        markdown_lines.append(f"### ID {idx}: `{config_name}`")
        markdown_lines.append(f"* **Original Config path:** `{config_rel_path}`")
        markdown_lines.append(f"* **Key Parameters:**")
        markdown_lines.append(f"  * Layers: `{params.get('n_layer')}`")
        markdown_lines.append(f"  * Embed Dimension: `{params.get('n_embd')}`")
        markdown_lines.append(f"  * Loops (Max Loops): `{params.get('max_loops')}`")
        markdown_lines.append(f"  * Direct path FFN hidden size (D): `{params.get('ffn_hidden')}`")
        markdown_lines.append(f"  * Wide/loop path FFN hidden size (W): `{params.get('wide_ffn_hidden')}`")
        markdown_lines.append(f"  * Cross-path connections: `{params.get('use_cross')}`")
        markdown_lines.append(f"  * Target Tokens count: `{params.get('target_tokens')}`")
        
        # Check matching run folders
        matching_scratch_runs = []
        for d in all_scratch_dirs:
            if (d / config_name).exists():
                matching_scratch_runs.append(d)
        matching_scratch_runs.sort(key=lambda x: x.name)
        
        old_zip_run_names = []
        old_metrics_dir = TEMP_EXTRACT_DIR / "metrics"
        if old_metrics_dir.exists():
            pattern = re.compile(rf"^run_{idx:02d}_(.*)_metrics\.jsonl$")
            for filename in os.listdir(old_metrics_dir):
                match = pattern.match(filename)
                if match:
                    old_zip_run_names.append(match.group(1))
                    
        all_runs_dict = {}
        for r_dir in matching_scratch_runs:
            all_runs_dict[r_dir.name] = {
                "name": r_dir.name,
                "scratch_dir": r_dir,
                "old_zip_metrics": None
            }
        for r_name in old_zip_run_names:
            old_m_file = old_metrics_dir / f"run_{idx:02d}_{r_name}_metrics.jsonl"
            if r_name in all_runs_dict:
                all_runs_dict[r_name]["old_zip_metrics"] = old_m_file
            else:
                all_runs_dict[r_name] = {
                    "name": r_name,
                    "scratch_dir": None,
                    "old_zip_metrics": old_m_file
                }
                
        sorted_run_names = sorted(all_runs_dict.keys())
        
        if not sorted_run_names:
            markdown_lines.append("  * **Active Runs:** None found completed on scratch.\n")
            continue
            
        markdown_lines.append("  * **Active Runs:**")
        for run_idx, r_name in enumerate(sorted_run_names, 1):
            run_info = all_runs_dict[r_name]
            run_dir = run_info["scratch_dir"]
            old_metrics_file = run_info["old_zip_metrics"]
            
            last_cp = "N/A"
            eval_str = "No evaluation results yet"
            
            metrics_file_to_read = None
            if run_dir:
                checkpoint_dirs = []
                for item in run_dir.iterdir():
                    if item.is_dir() and item.name.startswith("eid_"):
                        checkpoint_dirs.append(item)
                
                def get_step(d: Path) -> int:
                    match = re.search(r"seen_steps_(\d+)", d.name)
                    return int(match.group(1)) if match else -1
                checkpoint_dirs.sort(key=get_step)
                
                if checkpoint_dirs:
                    last_cp = checkpoint_dirs[-1].name
                    step = get_step(checkpoint_dirs[-1])
                    candidate1 = checkpoint_dirs[-1] / "hf_checkpoint" / f"olmes_eval_{step}" / "metrics-all.jsonl"
                    candidate2 = checkpoint_dirs[-1] / "hf_checkpoint" / "olmes_eval" / "metrics-all.jsonl"
                    if candidate1.exists() or candidate2.exists():
                        eval_str = "Evaluated successfully"
            elif old_metrics_file and old_metrics_file.exists():
                last_cp = "Preserved from previous zip"
                eval_str = "Evaluated successfully (Preserved)"
                
            scratch_path_type = "experiments_emnlp"
            if run_dir:
                scratch_path_type = "experiments_emnlp_revisions" if run_dir.parent == SCRATCH_REVISIONS else "experiments_emnlp"
            else:
                scratch_path_type = "experiments_emnlp_revisions (Preserved)"
                
            markdown_lines.append(f"    {run_idx}. Run `{r_name}` (in `{scratch_path_type}`)")
            markdown_lines.append(f"       * Last Checkpoint saved: `{last_cp}`")
            markdown_lines.append(f"       * Downstream Evaluation: `{eval_str}`")
        markdown_lines.append("")

    with open(EXPORT_DIR / "summary.md", "w", encoding="utf-8") as out:
        out.write("\n".join(markdown_lines))

    # Compress the output folder to a single zip file
    shutil.make_archive(ZIP_OUT_FILE, 'zip', EXPORT_DIR)
    print(f"Successfully packaged EMNLP revisions. Export file written to {ZIP_OUT_FILE}.zip")
    
    # Clean up temp folder
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)

if __name__ == "__main__":
    main()
