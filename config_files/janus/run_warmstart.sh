#!/bin/bash
#SBATCH --job-name=janus-ws
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
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Resume a janus run from its last DCP checkpoint and train to num_target_steps.
#
# A full 18360-step run does not fit in the 24h partition cap, so each arm is
# run as (24h job) + (this, queued with --dependency=afterany). Both arms use
# the identical mechanism so the comparison stays matched.
#
#   sbatch --dependency=afterany:<trainjob> config_files/janus/run_warmstart.sh \
#          <WARMSTART_CONFIG> <BASE_CONFIG_NAME>
#
# BASE_CONFIG_NAME is the basename of the *original* (non-warmstart) config; it
# selects which experiment directory to resume, since the experiment id is only
# known after the training job has started. The newest matching one wins.
set -x -e

WARMSTART_CONFIG="$1"
BASE_CONFIG_NAME="$2"

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest"

[ -f "${WARMSTART_CONFIG}" ] || { echo "no such config: ${WARMSTART_CONFIG}"; exit 1; }

# Resolve the experiment to resume: newest dir holding a copy of the config AND a
# last_checkpoint_info.json written by a completed checkpoint.
#
# A warmstart writes its own experiment dir containing only the *_WARMSTART.yaml
# copy, not the base config. Matching the base name alone therefore made every
# link after the first rediscover the ORIGINAL dir and redo the segment the
# previous link had already done -- a chain that never advances. Accept either
# name; `ls -1dt` puts the newest (i.e. the furthest-along) link first.
WS_CONFIG_NAME="$(basename "${WARMSTART_CONFIG}")"
EXP_DIR=""
for d in $(ls -1dt "${EXPERIMENT_ROOT}"/*/); do
    if [ -f "${d}/last_checkpoint_info.json" ] && \
       { [ -f "${d}/${BASE_CONFIG_NAME}" ] || [ -f "${d}/${WS_CONFIG_NAME}" ]; }; then
        EXP_DIR="${d%/}"
        break
    fi
done
[ -n "${EXP_DIR}" ] || { echo "no resumable experiment found for ${BASE_CONFIG_NAME}"; exit 1; }

CKPT_INFO="${EXP_DIR}/last_checkpoint_info.json"
echo "Resuming experiment: ${EXP_DIR}"
cat "${CKPT_INFO}"

# These runs need several chained warmstarts to reach the target, and the chain
# is queued up front. Once the target is reached the remaining links have
# nothing to do, so exit cleanly rather than starting a 0-step run (which the
# config validators reject).
CKPT_DIR=$(grep -oE '"checkpoint_folder_path": *"[^"]+"' "${CKPT_INFO}" | sed 's/.*: *"//; s/"$//')
SEEN=$(echo "${CKPT_DIR}" | grep -oE 'seen_steps_[0-9]+' | grep -oE '[0-9]+')
TARGET=$(echo "${CKPT_DIR}" | grep -oE 'target_steps_[0-9]+' | grep -oE '[0-9]+')
echo "checkpoint at step ${SEEN} of ${TARGET}"
if [ -n "${SEEN}" ] && [ -n "${TARGET}" ] && [ "${SEEN}" -ge "${TARGET}" ]; then
    echo "already at target -- nothing to do, exiting cleanly."
    exit 0
fi

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
export JANUS_EVAL_ALL_STEPS="${JANUS_EVAL_ALL_STEPS:-0}"
# The end-of-training eval inherits this; a 64-layer model needs >10h for the
# 25-task suite, so the eval script's 6h default would time out silently.
export JANUS_EVAL_TIME="${JANUS_EVAL_TIME:-20:00:00}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 50000 + (SLURM_JOB_ID % 10000) ))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

export REPO_DIR VENV_DIR WARMSTART_CONFIG CKPT_INFO EXPERIMENT_ROOT JANUS_EVAL_ALL_STEPS JANUS_EVAL_TIME
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE

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
      "$(which modalities)" warmstart \
      --config_file_path "${WARMSTART_CONFIG}" \
      --last_checkpoint_info_file_path "${CKPT_INFO}" \
      --experiments_root_path "${EXPERIMENT_ROOT}"
' </dev/null
echo "END: $(date)"
