import os
import re
import json
from pathlib import Path

ALL_CONFIGS_TXT = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/emnlp/sbatch_revisions/all_configs.txt")
MODALITIES_ROOT = Path("/leonardo_work/EUHPC_D21_101/mfrey/modalities")
SCRATCH_REVISIONS = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions")
SCRATCH_MAIN = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp")

def main():
    if not ALL_CONFIGS_TXT.exists():
        print(f"Error: {ALL_CONFIGS_TXT} not found.")
        return

    with open(ALL_CONFIGS_TXT, "r") as f:
        config_lines = [line.strip() for line in f if line.strip()]

    revisions_dirs = [SCRATCH_REVISIONS / d for d in os.listdir(SCRATCH_REVISIONS) if (SCRATCH_REVISIONS / d).is_dir() and d != "wandb"] if SCRATCH_REVISIONS.exists() else []
    main_dirs = [SCRATCH_MAIN / d for d in os.listdir(SCRATCH_MAIN) if (SCRATCH_MAIN / d).is_dir() and d != "wandb"] if SCRATCH_MAIN.exists() else []
    all_scratch_dirs = revisions_dirs + main_dirs

    missing = []
    found_count = 0
    
    for idx, config_rel_path in enumerate(config_lines, 1):
        config_path = MODALITIES_ROOT / config_rel_path
        config_name = config_path.name
        
        # Find matching run directories
        matching_runs = []
        for d in all_scratch_dirs:
            if (d / config_name).exists():
                matching_runs.append(d)
        matching_runs.sort(key=lambda x: x.name)
        
        for run_dir in matching_runs:
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
                
                # Check for metrics
                candidate1 = last_cp_dir / "hf_checkpoint" / f"olmes_eval_{step}" / "metrics-all.jsonl"
                candidate2 = last_cp_dir / "hf_checkpoint" / "olmes_eval" / "metrics-all.jsonl"
                
                metrics_file = None
                if candidate1.exists():
                    metrics_file = candidate1
                elif candidate2.exists():
                    metrics_file = candidate2
                
                has_paloma = False
                if metrics_file:
                    try:
                        with open(metrics_file, "r") as f_met:
                            content = f_met.read()
                            if "paloma_c4_en" in content:
                                has_paloma = True
                    except Exception as e:
                        pass
                
                if metrics_file and not has_paloma:
                    missing.append({
                        "config_idx": idx,
                        "config_name": config_name,
                        "run_dir": str(run_dir),
                        "checkpoint_dir": str(last_cp_dir),
                        "step": step,
                        "metrics_file": str(metrics_file)
                    })
                elif metrics_file and has_paloma:
                    found_count += 1

    print(f"Total active scratch runs with evaluations: {found_count + len(missing)}")
    print(f"Runs with Paloma C4/Wiki: {found_count}")
    print(f"Runs MISSING Paloma C4/Wiki: {len(missing)}")
    for m in missing:
        print(f"  - Config {m['config_idx']} ({m['config_name']}) run {Path(m['run_dir']).name} step {m['step']}")

if __name__ == "__main__":
    main()
