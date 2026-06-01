#!/bin/bash
#SBATCH --job-name=ablation-eval
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/abl-%A_%a.out
#SBATCH --error=logs/abl-%A_%a.err
#
# Run the paper-table ablation sweep against one checkpoint.
#
# Submit with:
#   sbatch --array=0-9 --export=HF_CKPT=/path/to/hf_checkpoint slurm_jobs/ablation.sbatch
#
# The array index picks which ablation to run. Add/remove from ABLATIONS below.
# Output: ${HF_CKPT}/paloma_diagnostics/ablations/<name>.json per array task.

set -e -u
ulimit -c 0

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
REPO_DIR=${REPO_DIR:-"${MY_ROOT}/modalities/src/modalities/evaluation"}
SCRIPT="${REPO_DIR}/python_scripts/ablation_eval.py"

[ -f "${SCRIPT}" ] || { echo "Error: ${SCRIPT} not found"; exit 1; }
[ -z "${HF_CKPT:-}" ] && { echo "Error: HF_CKPT not set"; exit 1; }
[ -f "${HF_CKPT}/config.json" ] || { echo "Error: ${HF_CKPT}/config.json missing"; exit 1; }

DIAG_DIR="${HF_CKPT}/paloma_diagnostics"
[ -d "${DIAG_DIR}" ] || { echo "Error: ${DIAG_DIR} missing — run diagnostics first"; exit 1; }

# The ablation matrix. Order matters: index = SLURM_ARRAY_TASK_ID.
# Keep 'learned' at index 0 so it's the first reference number you see.
ABLATIONS=(
    "learned"                       # 0: baseline
    "g_deep=0.5,g_wide=0.5"         # 1: uniform router
    "shuffle"                       # 2: shuffled router (the killer ablation)
    "g_deep=1,g_wide=0"             # 3: loop-only at inference
    "g_deep=0,g_wide=1"             # 4: wide-only at inference
    "g_deep=1,g_wide=1"             # 5: both fully open
    "force_loops=1"                 # 6: K=1
    "force_loops=2"                 # 7: K=2
    "force_loops=3"                 # 8: K=3
    "force_loops=4"                 # 9: K=4 — should match learned exactly
    "force_loops=5"                 # 10: K=5 (OOD: 1 step past trained)
    "force_loops=6"                 # 11: K=6
    "force_loops=7"                 # 12: K=7
    "force_loops=8"                 # 13: K=8
    "force_loops=9"                 # 14: K=9
    "force_loops=10"                # 15: K=10 (OOD: 6 steps past trained)
    # --- adaptive halting: 10-point linear sweep, T in [0.1, 0.99] ---
    # Lower thresholds (<0.1) are flat — router halts ~all tokens within 2
    # steps on its own (trained K=4 with avg E[steps]≈1.94 at T=0.05). The
    # threshold only bites past ~0.4; we still include 0.1-0.3 to show the
    # flat region for the curve. T=0.99 instead of 1.0 since 1.0 is exactly
    # K=1 (already at index 6).
    "halt_threshold=0.1"            # 16
    "halt_threshold=0.2"            # 17
    "halt_threshold=0.3"            # 18
    "halt_threshold=0.4"            # 19
    "halt_threshold=0.5"            # 20
    "halt_threshold=0.6"            # 21
    "halt_threshold=0.7"            # 22
    "halt_threshold=0.8"            # 23
    "halt_threshold=0.9"            # 24
    "halt_threshold=0.99"           # 25
)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: this job must be submitted with --array"
    echo "Available ablations:"
    for i in "${!ABLATIONS[@]}"; do
        echo "  $i: ${ABLATIONS[$i]}"
    done
    exit 1
fi

ABLATION="${ABLATIONS[$SLURM_ARRAY_TASK_ID]}"
[ -z "${ABLATION:-}" ] && { echo "No ablation at index $SLURM_ARRAY_TASK_ID"; exit 1; }

echo "=========================================="
echo "Array index: ${SLURM_ARRAY_TASK_ID}"
echo "Ablation:    ${ABLATION}"
echo "Checkpoint:  ${HF_CKPT}"
echo "Diag dir:    ${DIAG_DIR}"
echo "START:       $(date)"
echo "=========================================="

singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "${CONTAINER}" bash -c "
        set -e -u
        source ${OLMES_VENV}/bin/activate
        export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
        export HF_HOME=${EXPERIMENTS_DIR}/../hf_cache
        export HF_DATASETS_CACHE=\${HF_HOME}
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export TRITON_HOME=/dev/shm/
        export TRITON_CACHE_DIR=/dev/shm/.triton_cache_abl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
        mkdir -p \${TRITON_CACHE_DIR}
        export CUDA_VISIBLE_DEVICES=0

        python '${SCRIPT}' \\
            --ckpt '${HF_CKPT}' \\
            --paloma-dir '${DIAG_DIR}' \\
            --ablation '${ABLATION}'
    "

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
echo "Result: ${DIAG_DIR}/ablations/$(echo ${ABLATION} | tr ',=' '__' | tr -d '.' | sed 's/learned/learned/').json"
