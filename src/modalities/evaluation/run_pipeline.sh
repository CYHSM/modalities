#!/bin/bash
set -u

# Ensure we're in the correct directory (the directory of this script)
cd "$(dirname "$0")"

MY_ROOT="/users/markusfrey"
MODALITIES_DIR="${MY_ROOT}/Github/modalities"
EXPERIMENTS_DIR="/capstor/scratch/cscs/markusfrey/experiments"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
export LIST_FILE="$(pwd)/checkpoints_to_eval.txt"

echo "=== 0. Checking for checkpoints needing conversion ==="
needs_conversion=0
for run_dir in "${EXPERIMENTS_DIR}"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    [ "$run_name" = "wandb" ] && continue
    for ckpt_dir in "${run_dir}"eid_*/; do
        [ -d "$ckpt_dir" ] || continue
        hf_out="${ckpt_dir}hf_checkpoint"
        if [ ! -f "${hf_out}/config.json" ]; then
            needs_conversion=$((needs_conversion+1))
        fi
    done
done
echo ""

if [ "$needs_conversion" -gt 0 ]; then
    echo "Found $needs_conversion checkpoints to convert."
    echo "=== 1. Submitting conversion job ==="
    CONV_JOB_ID=$(sbatch --parsable slurm_jobs/conversion.sh)
    echo "Submitted conversion job: $CONV_JOB_ID"
    echo "Waiting for conversion job to complete (polling every 60s)..."
    # Poll until squeue no longer lists this job. Discard standard error from squeue.
    while squeue -u "$USER" -j "$CONV_JOB_ID" 2>/dev/null | grep -q "$CONV_JOB_ID"; do
        sleep 60
    done
    echo "Conversion job completed."
    echo ""
else
    echo "All checkpoints are already converted. Skipping conversion step."
    echo ""
fi

echo "=== 2. Finding checkpoints to evaluate ==="
rm -f "$LIST_FILE"
found=0

for run_dir in "${EXPERIMENTS_DIR}"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    [ "$run_name" = "wandb" ] && continue
    for ckpt_dir in "${run_dir}"eid_*/; do
        hf_dir="${ckpt_dir}hf_checkpoint"
        if [ -d "$hf_dir" ] && [ -f "${hf_dir}/config.json" ]; then
            # If the eval metric file does not exist, queue it for eval
            if [ ! -f "${hf_dir}/olmes_eval/metrics-all.jsonl" ]; then
                # Append to list file (removing trailing slash from path)
                echo "${hf_dir%/}" >> "$LIST_FILE"
                found=$((found+1))
            fi
        fi
    done
done

if [ "$found" -eq 0 ]; then
    echo "No new checkpoints need evaluation."
    echo "Touching empty $LIST_FILE to avoid errors."
    touch "$LIST_FILE"
else
    echo "Found $found checkpoints to evaluate."
    echo ""
    echo "=== 3. Submitting evaluation array job ==="
    EVAL_JOB_ID=$(sbatch --parsable --array=1-${found} slurm_jobs/eval_array.sbatch)
    echo "Submitted eval array job: $EVAL_JOB_ID"
    echo "Waiting for evaluation jobs to complete (polling every 60s)..."
    while squeue -u "$USER" -j "$EVAL_JOB_ID" 2>/dev/null | grep -q "$EVAL_JOB_ID"; do
        sleep 60
    done
    echo "Evaluation array jobs completed."
fi
echo ""

echo "=== 4. Syncing to Weights & Biases ==="
# We use uenv run here so the venv's Python symlinks resolve correctly on the login node
uenv run pytorch/v2.9.1 --view=default -- bash -c "
    set -e -u
    source ${OLMES_VENV}/bin/activate
    python ${MODALITIES_DIR}/src/modalities/evaluation/python_scripts/sync_evals_to_wandb.py
"

echo "=== Pipeline Finished ==="