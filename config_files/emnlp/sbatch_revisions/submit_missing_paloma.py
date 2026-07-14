import os
import re
import json
import subprocess
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
                original_out_dir = None
                if candidate1.exists():
                    metrics_file = candidate1
                    original_out_dir = last_cp_dir / "hf_checkpoint" / f"olmes_eval_{step}"
                elif candidate2.exists():
                    metrics_file = candidate2
                    original_out_dir = last_cp_dir / "hf_checkpoint" / "olmes_eval"
                
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
                        "run_dir": run_dir,
                        "checkpoint_dir": last_cp_dir,
                        "step": step,
                        "metrics_file": metrics_file,
                        "original_out_dir": original_out_dir
                    })

    print(f"Submitting {len(missing)} parallel Slurm jobs for missing Paloma evaluations...")

    for m in missing:
        idx = m["config_idx"]
        step = m["step"]
        run_name = m["run_dir"].name
        hf_checkpoint_dir = m["checkpoint_dir"] / "hf_checkpoint"
        original_out_dir = m["original_out_dir"]
        paloma_out_dir = hf_checkpoint_dir / f"olmes_eval_{step}_paloma"
        
        # Create custom sbatch script content
        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=paloma-eval-{idx}-{step}
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output={original_out_dir}_paloma_slurm.log
#SBATCH --error={original_out_dir}_paloma_slurm.err
set -e
ulimit -c 0

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_tests"
OLMES_VENV="${{MY_ROOT}}/venvs/olmes_uv"

export TIKTOKEN_CACHE_DIR="${{MY_ROOT}}/tiktoken_cache"
export HF_HOME="${{EXPERIMENTS_DIR}}/../hf_cache"
export HF_DATASETS_CACHE="${{HF_HOME}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export WANDB_MODE=offline
export NLTK_DATA="${{MY_ROOT}}/nltk_data"
export TRITON_HOME=/dev/shm/
export TRITON_CACHE_DIR=/dev/shm/.triton_cache_${{SLURM_JOB_ID}}
mkdir -p "${{TRITON_CACHE_DIR}}"

unset PYTHONPATH
unset VIRTUAL_ENV

source ${{OLMES_VENV}}/bin/activate
export PYTHONPATH="${{MY_ROOT}}/olmes:${{PYTHONPATH:-}}"
set -u

echo "Starting evaluation of paloma C4 and Wiki..."
python ${{OLMES_VENV}}/bin/olmes \\
    --model '{hf_checkpoint_dir}' \\
    --model-type hf \\
    --model-args '{{"trust_remote_code": true, "max_length": 4096}}' \\
    --task 'paloma_c4_en paloma_wikitext_103' \\
    --limit 128 \\
    --batch-size 1 \\
    --output-dir '{paloma_out_dir}'

# Copy other task files (excluding metrics-all.jsonl and metrics.json)
rsync -av --exclude='metrics-all.jsonl' --exclude='metrics.json' '{paloma_out_dir}/' '{original_out_dir}/'

# Post-processing: merge the results back into the main metrics-all.jsonl
echo "Merging Paloma results into main metrics-all.jsonl..."
cat '{paloma_out_dir}/metrics-all.jsonl' >> '{original_out_dir}/metrics-all.jsonl'

# Cleanup temporary paloma directory
rm -rf '{paloma_out_dir}'
echo "Paloma evaluation merged successfully!"
"""
        # Write sbatch script to a temporary file
        sbatch_file = MODALITIES_ROOT / f"config_files/emnlp/sbatch_revisions/logs/temp_paloma_{idx}_{run_name}.sh"
        with open(sbatch_file, "w") as f_sb:
            f_sb.write(sbatch_content)
            
        # Submit the job
        try:
            res = subprocess.run(["sbatch", str(sbatch_file)], capture_output=True, text=True, check=True)
            job_id = res.stdout.strip().split()[-1]
            print(f"Submitted Job ID {job_id} for Config {idx} run {run_name}")
        except Exception as e:
            print(f"Error submitting Job for Config {idx} run {run_name}: {e}")
            
        # Delete temporary script file
        if sbatch_file.exists():
            os.remove(sbatch_file)

if __name__ == "__main__":
    main()
