#!/bin/bash
#SBATCH --job-name=modalities_run
#SBATCH --account=a0164
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -x -e

# ============================================================================
# 0. ARRAY SETUP
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array"
    exit 1
fi

CONFIG_LIST="all_configs.txt"
CONFIG_FILE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")
if [ -z "$CONFIG_FILE_PATH" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Array Task $SLURM_ARRAY_TASK_ID using config: $CONFIG_FILE_PATH"

# ============================================================================
# 1. PATH CONFIGURATION
# ============================================================================
MY_ROOT="/capstor/store/cscs/swissai/a0164/markusfrey"
MODALITIES_DIR="${MY_ROOT}/modalities"
EXPERIMENT_ROOT="/capstor/scratch/cscs/markusfrey/experiments"

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
# Activate uv venv (no modules, no containers)
source "${MODALITIES_DIR}/.venv/bin/activate"
export PYTHONPATH="${MODALITIES_DIR}/src:${PYTHONPATH}"

# Torch / NCCL — Slingshot, not InfiniBand. Do NOT set NCCL_IB_* or NCCL_SOCKET_IFNAME=ib0.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN                       # flip to INFO to debug comm issues
# If you see NCCL errors, try: export NCCL_NET="Socket"  (slow fallback, diagnostic only)

# W&B offline on compute nodes (no internet)
export WANDB_MODE=offline

# Rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID % 10000) + SLURM_ARRAY_TASK_ID ))

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array ID:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Nodes:       ${SLURM_JOB_NODELIST}"
echo "Num nodes:   ${SLURM_JOB_NUM_NODES}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:      ${CONFIG_FILE_PATH}"
echo "Experiment:  ${EXPERIMENT_ROOT}"
echo "START TIME:  $(date)"
echo "=========================================="

# ============================================================================
# 3. RUN TRAINING
# ============================================================================
cd "${MODALITIES_DIR}"

srun bash -c "
    torchrun \
        --node_rank=\$SLURM_PROCID \
        --nnodes=\$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        src/modalities/__main__.py run \
        --config_file_path '${CONFIG_FILE_PATH}' \
        --experiments_root_path '${EXPERIMENT_ROOT}'
"

echo "END TIME: $(date)"
echo "=== FINISHED ==="