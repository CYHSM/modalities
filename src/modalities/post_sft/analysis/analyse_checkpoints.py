"""Simple checkpoint analysis script."""
import json
import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def run_evaluation(checkpoint_path, output_file, device="cuda:7", batch_size=8, limit=0.01):
    """Run lm_eval on a checkpoint."""
    tasks = "hellaswag,gsm8k_cot,humaneval_instruct,wmt16-de-en"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},trust_remote_code=True",
        "--tasks", tasks,
        "--device", device,
        "--batch_size", str(batch_size),
        "--limit", str(limit),
        "--output_path", str(output_file),
        "--confirm_run_unsafe_code"
    ]
    
    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    
    print(f"Running: {checkpoint_path} Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0

def collect_results(results_dir):
    """Load all JSON results into DataFrame."""
    data = []
    
    for json_file in Path(results_dir).glob("*.json"):
        with open(json_file) as f:
            result = json.load(f)
        
        # Parse filename: subset_step_XXXXX.json
        name = json_file.stem
        parts = name.split('_')
        subset = f"{parts[0]}_{parts[1]}"  # full_math, full_code, etc
        step = int(parts[3])
        
        row = {"subset": subset, "step": step}
        
        # Extract all metrics
        for task, metrics in result["results"].items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f"{task}_{metric}"] = value
        
        data.append(row)
    
    return pd.DataFrame(data).sort_values(["subset", "step"])

def plot_results(df):
    """Create plots comparing subsets."""
    # Print available columns for debugging
    print("Available columns:")
    metric_cols = [col for col in df.columns if col not in ["subset", "step"]]
    for col in metric_cols:
        print(f"  {col}")
    
    # Key metrics to plot - try to find the right column names
    possible_metrics = {
        "hellaswag": ["hellaswag_acc_norm", "hellaswag_acc_norm,none"],
        "gsm8k_cot": ["gsm8k_cot_exact_match", "gsm8k_cot_exact_match,strict-match", "gsm8k_cot_exact_match,flexible-extract"], 
        "humaneval_instruct": ["humaneval_instruct_pass@10", "humaneval_instruct_pass@10,create_test"],
        "wmt16-de-en": ["wmt16-de-en_bleu", "wmt16-de-en_bleu,none"]
    }
    
    # Find which metrics actually exist
    key_metrics = {}
    for task, candidates in possible_metrics.items():
        for candidate in candidates:
            if candidate in df.columns:
                key_metrics[task] = candidate
                break
        if task not in key_metrics:
            print(f"Warning: No metric found for {task}")
    
    print(f"Using metrics: {key_metrics}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (task, metric) in enumerate(key_metrics.items()):
        if i >= len(axes):
            break
            
        for subset in df['subset'].unique():
            data = df[df['subset'] == subset]
            if not data.empty and metric in data.columns:
                axes[i].plot(data['step'], data[metric], 'o-', 
                           label=subset.replace('full_', ''), linewidth=2)
        
        axes[i].set_title(task.upper())
        axes[i].set_xlabel('Training Step')
        axes[i].set_ylabel(metric.split('_')[-1].split(',')[0])
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(key_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze model checkpoints")
    parser.add_argument("--base-dir", default="/raid/s3/opengptx/mfrey/instruct/checkpoints_sft", 
                       help="Base directory with checkpoints")
    parser.add_argument("--results-dir", default="/raid/s3/opengptx/mfrey/cp_analysis/benchmark_cps",
                       help="Directory to save results")
    parser.add_argument("--device", default="cuda:7", help="GPU device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=float, default=0.01, help="Fraction of data to evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just plot")
    parser.add_argument("--subsets", nargs="+", default=["full_general", "full_math", "full_code", "full_mix"],
                       help="Training subsets to evaluate")
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(5000, 100000, 5000)),
                       help="Checkpoint steps to evaluate")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print(f"Subsets: {args.subsets}")
    if not args.skip_eval:
        # Run evaluations
        for subset in args.subsets:
            for step in args.steps:
                checkpoint_path = base_dir / subset / f"checkpoint-{step}"
                output_file = results_dir / f"{subset}_step_{step}.json"
                
                pattern = f"{subset}_step_{step}_*.json"
                existing_files = list(results_dir.glob(pattern))
                if existing_files:
                    print(f"Skipping existing: {existing_files[0]}")
                    continue
                    
                if not checkpoint_path.exists():
                    print(f"Missing checkpoint: {checkpoint_path}")
                    continue
                
                run_evaluation(checkpoint_path, output_file, args.device, args.batch_size, args.limit)
    
    # Analyze results
    df = collect_results(results_dir)
    if not df.empty:
        print(f"Loaded {len(df)} results")
        df.to_csv(results_dir / "results.csv", index=False)
        plot_results(df)
    else:
        print("No results found")

if __name__ == "__main__":
    main()