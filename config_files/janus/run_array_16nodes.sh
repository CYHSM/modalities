#!/bin/bash
#SBATCH --job-name=janus
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
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --error=logs/%x-%A_%a.err

# Native .venv launcher (no Singularity). The configs in this folder use absolute
# data paths, so no /data bind mount is needed.
#
#   sbatch --array=1-17 config_files/janus/run_array_16nodes.sh

set -x -e

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array=1-17"
    exit 1
fi

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG_LIST="${CONFIG_LIST:-${REPO_DIR}/config_files/janus/all_configs.txt}"

CONFIG_FILE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")
if [ -z "$CONFIG_FILE_PATH" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
[[ "$CONFIG_FILE_PATH" != /* ]] && CONFIG_FILE_PATH="${REPO_DIR}/${CONFIG_FILE_PATH}"

# Must match settings.paths.experiments_root_path inside the configs, otherwise the
# model converter and downstream evaluator look for the checkpoint in the wrong place.
EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest"
mkdir -p "${EXPERIMENT_ROOT}" "${REPO_DIR}/logs"

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
# Pass through to run_convert_and_eval_sbatch.sh, which by default evaluates only
# the final checkpoint. Submit with JANUS_EVAL_ALL_STEPS=1 to evaluate every one.
export JANUS_EVAL_ALL_STEPS="${JANUS_EVAL_ALL_STEPS:-0}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID % 10000) + SLURM_ARRAY_TASK_ID ))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "=========================================="
echo "Array:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Nodes:      ${SLURM_JOB_NUM_NODES} x ${GPUS_PER_NODE} GPU"
echo "Config:     ${CONFIG_FILE_PATH}"
echo "Experiment: ${EXPERIMENT_ROOT}"
echo "START:      $(date)"
echo "=========================================="

export REPO_DIR VENV_DIR CONFIG_FILE_PATH EXPERIMENT_ROOT MASTER_ADDR MASTER_PORT GPUS_PER_NODE JANUS_EVAL_ALL_STEPS

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

echo "END: $(date)"
