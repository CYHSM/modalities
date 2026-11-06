#!/bin/bash
#SBATCH --job-name=janusprobe
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

# Throughput probe on the debug QoS: 8 nodes / 32 GPUs, 40 steps per variant.
#   CONFIG_LIST=<path/to/groupA.txt> sbatch --export=ALL,CONFIG_LIST=... run_probe_debug.sh
set -x -e

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG_LIST="${CONFIG_LIST:?set CONFIG_LIST to the probe list.txt}"
[ -r "$CONFIG_LIST" ] || { echo "FATAL: CONFIG_LIST not readable from compute node: $CONFIG_LIST"; exit 1; }


EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_probe"
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
export MASTER_PORT=$(( 30000 + (SLURM_JOB_ID % 10000) ))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"


export REPO_DIR VENV_DIR EXPERIMENT_ROOT MASTER_ADDR MASTER_PORT GPUS_PER_NODE

set +e
while read -r CONFIG_FILE_PATH <&3; do
  [ -z "$CONFIG_FILE_PATH" ] && continue
  export CONFIG_FILE_PATH
  echo "######## PROBE VARIANT: $(basename ${CONFIG_FILE_PATH}) START $(date +%T) ########"
  srun </dev/null bash -c '
    source "${VENV_DIR}/bin/activate"
    cd "${REPO_DIR}"
    export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
    torchrun --node_rank="${SLURM_PROCID}" --nnodes="${SLURM_JOB_NUM_NODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "$(which modalities)" run --config_file_path "${CONFIG_FILE_PATH}" \
        --experiments_root_path "${EXPERIMENT_ROOT}"
  '
  rc=$?
  echo "######## PROBE VARIANT: $(basename ${CONFIG_FILE_PATH}) END rc=${rc} $(date +%T) ########"
  [ $rc -ne 0 ] && echo "!!!! VARIANT FAILED: $(basename ${CONFIG_FILE_PATH}) rc=${rc}"
  MASTER_PORT=$(( MASTER_PORT + 1 )); export MASTER_PORT
done 3< "$CONFIG_LIST"

echo "END: $(date)"
