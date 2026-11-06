#!/bin/bash
#SBATCH --job-name=janus-livecheck
#SBATCH --account=AIFAC_S07_154
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

# End-to-end live-check of the full pipeline: 8 training steps -> 2 checkpoints
# (step 4, step 8) -> HF conversion -> inline OLMES downstream eval (LIMIT=8,
# batch=1) -> logged into the SAME live wandb run via downstream_evaluator,
# so it never hits the append-only-history / offline-resume step problems a
# post-hoc retrofit does. qos=normal + 1h cap (not debug-QOS's 30 min) because
# the training job blocks on two nested `sbatch --wait` OLMES jobs at the end.
#
#   sbatch config_files/janus/run_livecheck.sh

set -x -e

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG_FILE_PATH="${REPO_DIR}/config_files/janus/8e7_a50_nocross_livecheck.yaml"

EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_janus_livecheck"
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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "=========================================="
echo "Job:        ${SLURM_JOB_ID}"
echo "Nodes:      ${SLURM_JOB_NUM_NODES} x ${GPUS_PER_NODE} GPU"
echo "Config:     ${CONFIG_FILE_PATH}"
echo "Experiment: ${EXPERIMENT_ROOT}"
echo "START:      $(date)"
echo "=========================================="

export REPO_DIR VENV_DIR CONFIG_FILE_PATH EXPERIMENT_ROOT MASTER_ADDR MASTER_PORT GPUS_PER_NODE

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
