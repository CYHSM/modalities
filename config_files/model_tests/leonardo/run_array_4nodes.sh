#!/bin/bash
#SBATCH --job-name=modalities_test
#SBATCH --account=AIFAC_S07_154
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --error=logs/%x-%A_%a.err

set -x -e

# ============================================================================
# 0. ARRAY SETUP
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Please submit with --array"
    exit 1
fi

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG_LIST="${REPO_DIR}/config_files/model_tests/leonardo/all_configs.txt"

CONFIG_FILE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")

if [ -z "$CONFIG_FILE_PATH" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [[ "$CONFIG_FILE_PATH" != /* ]]; then
    CONFIG_FILE_PATH="${REPO_DIR}/${CONFIG_FILE_PATH}"
fi

echo "Array Task $SLURM_ARRAY_TASK_ID using config: $CONFIG_FILE_PATH"

EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_tests/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p "${EXPERIMENT_ROOT}" logs

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
module purge

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14
export NCCL_IB_HCA=mlx5
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID % 10000) + SLURM_ARRAY_TASK_ID ))

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array ID:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Nodes:       ${SLURM_JOB_NODELIST}"
echo "Num nodes:   ${SLURM_JOB_NUM_NODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:      ${CONFIG_FILE_PATH}"
echo "Experiment:  ${EXPERIMENT_ROOT}"
echo "Venv:        ${VENV_DIR}"
echo "START TIME:  $(date)"
echo "=========================================="

export REPO_DIR VENV_DIR CONFIG_FILE_PATH EXPERIMENT_ROOT MASTER_ADDR MASTER_PORT GPUS_PER_NODE

# ============================================================================
# 3. RUN TRAINING
# ============================================================================
srun bash -c '
  source "${VENV_DIR}/bin/activate"
  cd "${REPO_DIR}"
  export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

  torchrun \
      --node_rank="${SLURM_PROCID}" \
      --nnodes="${SLURM_JOB_NUM_NODES}" \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      "$(which modalities)" run \
      --config_file_path "${CONFIG_FILE_PATH}" \
      --experiments_root_path "${EXPERIMENT_ROOT}"
'

echo "END TIME: $(date)"
echo "=== FINISHED ==="
