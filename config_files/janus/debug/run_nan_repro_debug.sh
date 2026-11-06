#!/bin/bash
#SBATCH --job-name=janus-nan-repro
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=16
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

# Instrumented repro of the dual a=50 L32 F80M NaN (task 23): reproduced twice
# now at ~step 1324-1325, immediately after the un-clamped scale fix also
# failed at the same step -- pointing to a deterministic backward-pass
# trigger (grad_norm goes nan one step before loss does), not a slow
# scale-drift divergence. Runs the exact same real config with:
#   - MODALITIES_DETECT_ANOMALY=1 -> torch.autograd.set_detect_anomaly(True),
#     so autograd raises at the exact op whose backward first produces a
#     non-finite gradient, with a traceback into the forward call site.
#   - TORCHDYNAMO_DISABLE=1 -> disables torch.compile, so the AdaptiveRecursiveBlock
#     runs eagerly and anomaly detection can attribute cleanly instead of
#     pointing into an opaque compiled/fused kernel.
# Both add real overhead (roughly 2-4x slower); 8h budget is generous for
# reaching step ~1325 and catching the raise.
#   sbatch config_files/janus/debug/run_nan_repro_debug.sh
set -x -e

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG_FILE_PATH="${REPO_DIR}/config_files/janus/stability/8e7_loops4__dm768_L32_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross.yaml"

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
export MODALITIES_DETECT_ANOMALY=1
export TORCHDYNAMO_DISABLE=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 50000 + (SLURM_JOB_ID % 10000) ))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "REPRO config: ${CONFIG_FILE_PATH}"
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
