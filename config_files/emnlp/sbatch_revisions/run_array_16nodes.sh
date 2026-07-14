#!/bin/bash
#SBATCH --job-name=modalities_run
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=16
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=logs/%x-%A_%a.err

set -x -e

# ============================================================================
# 0. ARRAY SETUP
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please submit with --array"
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
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
CONTAINER_IMAGE="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

HOST_CODE_DIR="${MY_ROOT}/modalities"
CONTAINER_CODE_DIR="/opt/repos/modalities"

HOST_DATA_DIR="/leonardo_work/EUHPC_E05_119/mfrey/tokenized"
CONTAINER_DATA_DIR="/data"

EXPERIMENT_ROOT="${MY_ROOT}/experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXPERIMENT_ROOT}"

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
module purge

export CXX=g++
export CC=gcc

# NCCL / interconnect settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22     # MAX safe value. Equals roughly 17 seconds.
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14   # Bumped up from 10 to give it more attempts

export WANDB_MODE=offline

# Rendezvous network parameters
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
export CONTAINER_SRC_DIR="${CONTAINER_CODE_DIR}/src"

srun singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "${HOST_CODE_DIR}:${CONTAINER_CODE_DIR}" \
    --bind "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
    --bind "${MY_ROOT}/experiments:${MY_ROOT}/experiments" \
    --bind "${MY_ROOT}/tokenizer:${MY_ROOT}/tokenizer" \
    "${CONTAINER_IMAGE}" bash -c "
        export PYTHONPATH='${CONTAINER_SRC_DIR}':\$PYTHONPATH
        export TOKENIZERS_PARALLELISM=false
        cd ${CONTAINER_CODE_DIR}
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
