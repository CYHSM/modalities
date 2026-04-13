#!/bin/bash
#SBATCH --job-name=modalities_run
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out  
#SBATCH --error=logs/%x-%A_%a.err

set -e 

# ============================================================================
# 0. ARRAY SETUP
# ============================================================================
# Check if SLURM_ARRAY_TASK_ID is set (failsafe if run without --array)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please submit with --array"
    exit 1
fi

# Read the Nth line from the config list based on the array task ID
CONFIG_LIST="configs_list.txt"
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

# Make experiment root unique per array task so they don't overwrite each other
EXPERIMENT_ROOT="${MY_ROOT}/experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${EXPERIMENT_ROOT}"

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
module purge
export WANDB_MODE=offline
export MASTER_ADDR=localhost 
# Use the array task ID to ensure distinct ports if multiple jobs land on the same node
export MASTER_PORT=$(( 10000 + (SLURM_ARRAY_JOB_ID % 40000) + SLURM_ARRAY_TASK_ID ))
export PYTHONPATH="${CONTAINER_CODE_DIR}:${PYTHONPATH}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID} | Array ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:   ${SLURM_JOB_NODELIST}"
echo "Config: ${CONFIG_FILE_PATH}"
echo "=========================================="

# ============================================================================
# 3. RUN TRAINING
# ============================================================================
export CONTAINER_SRC_DIR="${CONTAINER_CODE_DIR}/src"

srun singularity exec --nv \
--bind "${HOST_CODE_DIR}:${CONTAINER_CODE_DIR}" \
--bind "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
--bind "${MY_ROOT}/experiments:${MY_ROOT}/experiments" \
--bind "${MY_ROOT}/tokenizer:${MY_ROOT}/tokenizer" \
"${CONTAINER_IMAGE}" bash -c "
    export PYTHONPATH='${CONTAINER_SRC_DIR}':\$PYTHONPATH
    cd ${CONTAINER_CODE_DIR}
    
    torchrun \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
        --nnodes 1 \
        --nproc_per_node 1 \
        --rdzv_backend c10d \
        src/modalities/__main__.py run \
        --config_file_path '${CONFIG_FILE_PATH}' \
        --experiments_root_path '${EXPERIMENT_ROOT}'
"