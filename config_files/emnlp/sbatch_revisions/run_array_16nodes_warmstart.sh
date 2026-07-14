#!/bin/bash
#SBATCH --job-name=modalities_run_warmstart
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
# 0. CONFIGURATION FOR JOB 18 WARMSTART (RESUMING IN-PLACE)
# ============================================================================
CONFIG_FILE_PATH="config_files/emnlp/final_revisions_addedbenchmark/16e7_loops4__dm768_L64_loop1_F160M_ffnD4864_ffnW4864_dual_a50_nocross_expanded_warmstart.yaml"
EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
EXPERIMENT_ID="2026-07-13__09-51-14_6107ee4d75d206a3"
LAST_CHECKPOINT_INFO_PATH="${EXPERIMENT_ROOT}/${EXPERIMENT_ID}/last_checkpoint_info.json"

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
CONTAINER_IMAGE="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

HOST_CODE_DIR="${MY_ROOT}/modalities"
CONTAINER_CODE_DIR="/opt/repos/modalities"

HOST_DATA_DIR="/leonardo_work/EUHPC_E05_119/mfrey/tokenized"
CONTAINER_DATA_DIR="/data"

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
module purge

export CXX=g++
export CC=gcc

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14

export WANDB_MODE=offline

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Nodes:         ${SLURM_JOB_NODELIST}"
echo "Num nodes:     ${SLURM_JOB_NUM_NODES}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:        ${CONFIG_FILE_PATH}"
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "START TIME:    $(date)"
echo "=========================================="

# ============================================================================
# 3. RUN TRAINING (WARMSTART IN-PLACE)
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
            src/modalities/__main__.py warmstart \
            --config_file_path '${CONFIG_FILE_PATH}' \
            --experiments_root_path '${EXPERIMENT_ROOT}' \
            --experiment_id '${EXPERIMENT_ID}' \
            --last_checkpoint_info_file_path '${LAST_CHECKPOINT_INFO_PATH}'
    "

echo "END TIME: $(date)"
echo "=== FINISHED ==="
