#!/bin/bash
set -u

# Ensure we're in the correct directory (the directory of this script)
cd "$(dirname "$0")"

# --- Safety flag to control deletion of old checkpoints/evals ---
FORCE_RERUN=${FORCE_RERUN:-0}

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
MODALITIES_DIR="${MY_ROOT}/modalities"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
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

        # --- FORCE_RERUN: only deletes hf_checkpoint. Old olmes_eval is preserved by rename step below. ---
        if [ "$FORCE_RERUN" = "1" ]; then
            if [ -d "$hf_out" ]; then
                echo "[FORCE_RERUN=1] Deleting existing hf_checkpoint: $hf_out"
                rm -rf "$hf_out"
            fi
            # NOTE: Removed deletion of olmes_eval here — handled by rename step 0.5.
        fi

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
    while squeue -u "$USER" -j "$CONV_JOB_ID" 2>/dev/null | grep -q "$CONV_JOB_ID"; do
        sleep 60
    done
    echo "Conversion job completed."
    echo ""
else
    echo "All checkpoints are already converted. Skipping conversion step."
    echo ""
fi

# --- NEW: Preserve old 128-sample evals before running the full benchmark ---
# echo "=== 1.5. Renaming existing olmes_eval -> olmes_eval_128 (preserving 128-sample runs) ==="
# renamed=0
# skipped_exists=0
# skipped_incomplete=0
# for run_dir in "${EXPERIMENTS_DIR}"/*/; do
#     [ -d "$run_dir" ] || continue
#     run_name=$(basename "$run_dir")
#     [ "$run_name" = "wandb" ] && continue
#     for ckpt_dir in "${run_dir}"eid_*/; do
#         [ -d "$ckpt_dir" ] || continue
#         hf_out="${ckpt_dir}hf_checkpoint"
#         eval_out="${hf_out}/olmes_eval"
#         eval_128="${hf_out}/olmes_eval_128"

#         # Only rename if olmes_eval exists AND olmes_eval_128 does not already exist
#         if [ -d "$eval_out" ] && [ ! -d "$eval_128" ]; then
#             # Sanity check: only rename if it looks like a completed eval
#             if [ -f "${eval_out}/metrics-all.jsonl" ]; then
#                 echo "Renaming: $eval_out -> $eval_128"
#                 mv "$eval_out" "$eval_128"
#                 renamed=$((renamed+1))
#             else
#                 echo "Skipping (no metrics-all.jsonl, likely incomplete): $eval_out"
#                 skipped_incomplete=$((skipped_incomplete+1))
#             fi
#         elif [ -d "$eval_out" ] && [ -d "$eval_128" ]; then
#             echo "Skipping (olmes_eval_128 already exists): $eval_out"
#             skipped_exists=$((skipped_exists+1))
#         fi
#     done
# done
# echo "Renamed: $renamed | Skipped (backup already exists): $skipped_exists | Skipped (incomplete): $skipped_incomplete"
# echo ""

# 2. Clean up the failed eval dirs (the ones without metrics-all.jsonl).
# #    Run this from the same dir as the pipeline script:
# for run_dir in /leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp/*/; do
#     [ -d "$run_dir" ] || continue
#     [ "$(basename "$run_dir")" = "wandb" ] && continue
#     for ckpt_dir in "${run_dir}"eid_*/; do
#         eval_out="${ckpt_dir}hf_checkpoint/olmes_eval"
#         if [ -d "$eval_out" ] && [ ! -f "${eval_out}/metrics-all.jsonl" ]; then
#             echo "Removing incomplete eval: $eval_out"
#             rm -rf "$eval_out"
#         fi
#     done
# done

# echo "=== 2. Finding checkpoints to evaluate ==="
# rm -f "$LIST_FILE"
# found=0
# for run_dir in "${EXPERIMENTS_DIR}"/*/; do
#     [ -d "$run_dir" ] || continue
#     run_name=$(basename "$run_dir")
#     [ "$run_name" = "wandb" ] && continue
#     for ckpt_dir in "${run_dir}"eid_*/; do
#         hf_dir="${ckpt_dir}hf_checkpoint"
#         if [ -d "$hf_dir" ] && [ -f "${hf_dir}/config.json" ]; then
#             # If the eval metric file does not exist, queue it for eval
#             if [ ! -f "${hf_dir}/olmes_eval/metrics-all.jsonl" ]; then
#                 # Append to list file (removing trailing slash from path)
#                 echo "${hf_dir%/}" >> "$LIST_FILE"
#                 found=$((found+1))
#             fi
#         fi
#     done
# done

# if [ "$found" -eq 0 ]; then
#     echo "No new checkpoints need evaluation."
#     echo "Touching empty $LIST_FILE to avoid errors."
#     touch "$LIST_FILE"
# else
#     echo "Found $found checkpoints to evaluate."
#     echo ""
#     echo "=== 3. Submitting evaluation array job ==="
#     EVAL_JOB_ID=$(sbatch --parsable --array=1-${found} slurm_jobs/eval_array.sbatch)
#     echo "Submitted eval array job: $EVAL_JOB_ID"
#     echo "Waiting for evaluation jobs to complete (polling every 60s)..."
#     while squeue -u "$USER" -j "$EVAL_JOB_ID" 2>/dev/null | grep -q "$EVAL_JOB_ID"; do
#         sleep 60
#     done
#     echo "Evaluation array jobs completed."
# fi

# echo ""
# echo "=== 4. Syncing to Weights & Biases ==="
# singularity exec --bind "${MY_ROOT}:${MY_ROOT}" --bind "/leonardo_scratch:/leonardo_scratch" "$CONTAINER" bash -c "
#     set -e -u
#     source ${OLMES_VENV}/bin/activate
#     export PYTHONPATH=${MODALITIES_DIR}/src:\$PYTHONPATH
#     python ${MODALITIES_DIR}/src/modalities/evaluation/python_scripts/sync_evals_to_wandb.py
# "

# echo "=== Pipeline Finished ==="