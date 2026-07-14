#!/bin/bash
#SBATCH --job-name=eval_array
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/eval_array-%A_%a.out
#SBATCH --error=logs/eval_array-%A_%a.err

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Run with --array"
    exit 1
fi

# Ensure we run from the repository root
cd /leonardo_work/EUHPC_D21_101/mfrey/modalities

# Run evaluation script for the specific config index
/leonardo_work/EUHPC_D21_101/mfrey/modalities/.venv/bin/python scripts/evaluation/convert_and_evaluate_revisions.py --config-index "$SLURM_ARRAY_TASK_ID" --last-only
