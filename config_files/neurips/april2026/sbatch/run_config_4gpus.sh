#!/bin/bash
#SBATCH --job-name=modalities_run
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e # Exit immediately if a command fails

# ============================================================================
# 1. PATH CONFIGURATION (User defined)
# ============================================================================

# Your Project Root in D21_101
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
MY_SCRATCH="/leonardo_scratch/fast/EUHPC_D21_101/mfrey"

# path to container
CONTAINER_IMAGE="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

# path to code
HOST_CODE_DIR="${MY_ROOT}/modalities"
CONTAINER_CODE_DIR="/opt/repos/modalities" 

# path to data
HOST_DATA_DIR="/leonardo_work/EUHPC_E05_119/mfrey/tokenized"
CONTAINER_DATA_DIR="/data"

# config file
CONFIG_FILE_PATH="config_files/loom/SepGates/loop5_512deep_11776wide_nomix_loop_enrich_gate_inputnorm.yaml"

# Output directory for logs/checkpoints
EXPERIMENT_ROOT="${MY_SCRATCH}/experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${EXPERIMENT_ROOT}"

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================

module purge

export WANDB_MODE=offline
export MASTER_ADDR=localhost # Safest for single-node multi-GPU
export MASTER_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) + (RANDOM % 1000) ))
export PYTHONPATH="${CONTAINER_CODE_DIR}:${PYTHONPATH}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_JOB_NODELIST}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Data (Host): ${HOST_DATA_DIR}"
echo "Data (Container): ${CONTAINER_DATA_DIR}"
echo "=========================================="

# ============================================================================
# 3. RUN TRAINING (Single Node, 4 GPUs)
# ============================================================================
# Check if "src" exists in your code path, otherwise remove "/src" from the line below
export CONTAINER_SRC_DIR="${CONTAINER_CODE_DIR}/src"

srun singularity exec --nv \
--bind "${HOST_CODE_DIR}:${CONTAINER_CODE_DIR}" \
--bind "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
--bind "${MY_SCRATCH}/experiments:${MY_SCRATCH}/experiments" \
--bind "${MY_ROOT}/tokenizer:${MY_ROOT}/tokenizer" \
"${CONTAINER_IMAGE}" bash -c "
    # Force Python to look in your local code first
    export PYTHONPATH='${CONTAINER_SRC_DIR}':\$PYTHONPATH
    
    cd ${CONTAINER_CODE_DIR}
    
    echo 'Starting Torchrun...'
    torchrun \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
        --nnodes 1 \
        --nproc_per_node 4 \
        --rdzv_backend c10d \
        src/modalities/__main__.py run \
        --config_file_path '${CONFIG_FILE_PATH}' \
        --experiments_root_path '${EXPERIMENT_ROOT}'
"