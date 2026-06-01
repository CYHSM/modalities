#!/bin/bash
set -u

# Ensure we're in the correct directory (the directory of this script)
cd "$(dirname "$0")"

# --- Safety flag to control deletion of old checkpoints/evals ---
FORCE_RERUN=${FORCE_RERUN:-0}

# --- Max concurrent conversion tasks (tune to your QoS limits) ---
MAX_CONCURRENT_CONVERSIONS=${MAX_CONCURRENT_CONVERSIONS:-64}

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
MODALITIES_DIR="${MY_ROOT}/modalities"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
export LIST_FILE="$(pwd)/checkpoints_to_eval.txt"
export CONV_LIST="$(pwd)/checkpoints_to_convert.txt"

# echo "=== 0. Building list of checkpoints to convert ==="
# rm -f "$CONV_LIST"
# needs_conversion=0
# for run_dir in "${EXPERIMENTS_DIR}"/*/; do
#     [ -d "$run_dir" ] || continue
#     run_name=$(basename "$run_dir")
#     [ "$run_name" = "wandb" ] && continue

#     config=$(find "$run_dir" -maxdepth 1 -name '*.yaml' ! -name '*.resolved' | head -1)
#     if [ -z "$config" ]; then
#         echo "⚠️  no yaml config in $run_name, skipping"
#         continue
#     fi

#     for ckpt_dir in "${run_dir}"eid_*/; do
#         [ -d "$ckpt_dir" ] || continue
#         hf_out="${ckpt_dir}hf_checkpoint"

#         # --- FORCE_RERUN: only deletes hf_checkpoint. Old olmes_eval is preserved by rename step below. ---
#         if [ "$FORCE_RERUN" = "1" ]; then
#             if [ -d "$hf_out" ]; then
#                 echo "[FORCE_RERUN=1] Deleting existing hf_checkpoint: $hf_out"
#                 rm -rf "$hf_out"
#             fi
#         fi

#         if [ ! -f "${hf_out}/config.json" ]; then
#             # tab-separated: ckpt_dir <TAB> hf_out <TAB> config
#             printf "%s\t%s\t%s\n" "${ckpt_dir%/}" "$hf_out" "$config" >> "$CONV_LIST"
#             needs_conversion=$((needs_conversion+1))
#         fi
#     done
# done

# echo ""
# if [ "$needs_conversion" -gt 0 ]; then
#     echo "Found $needs_conversion checkpoints to convert."
#     echo "=== 1. Submitting conversion array job (max ${MAX_CONCURRENT_CONVERSIONS} concurrent) ==="
#     CONV_JOB_ID=$(sbatch --parsable \
#         --array=1-${needs_conversion}%${MAX_CONCURRENT_CONVERSIONS} \
#         slurm_jobs/conversion_array.sbatch)
#     echo "Submitted conversion array: $CONV_JOB_ID"
#     echo "Waiting for conversion job to complete (polling every 60s)..."
#     while squeue -u "$USER" -j "$CONV_JOB_ID" 2>/dev/null | grep -q "$CONV_JOB_ID"; do
#         sleep 60
#     done
#     echo "Conversion job completed."
#     echo ""
# else
#     echo "All checkpoints are already converted. Skipping conversion step."
#     echo ""
# fi

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

echo ""
echo "=== 4. Syncing to Weights & Biases ==="
singularity exec --bind "${MY_ROOT}:${MY_ROOT}" --bind "/leonardo_scratch:/leonardo_scratch" "$CONTAINER" bash -c "
    set -e -u
    source ${OLMES_VENV}/bin/activate
    export PYTHONPATH=${MODALITIES_DIR}/src:\$PYTHONPATH
    python ${MODALITIES_DIR}/src/modalities/evaluation/python_scripts/sync_evals_to_wandb.py
"

echo "=== Pipeline Finished ==="