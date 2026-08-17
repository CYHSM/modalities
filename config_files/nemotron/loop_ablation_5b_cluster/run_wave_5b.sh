#!/bin/bash
#SBATCH --job-name=nemo5b
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_lprod
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --array=1-11
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/nemo5b-%x-%A_%a.out
#SBATCH --error=logs/nemo5b-%x-%A_%a.err
#
# NODE COUNT: 1 node (4 GPUs) per arm, decided empirically on the debug queue 2026-08-11 -- see
# speedtest/run_speedtest.sh and docs/components/nemotron_loops_research_plan.md section 9.3.
# dp_degree 4 (1 node, pure FSDP2 replicate) measured ~59-63 samples/s on the slowest arm
# (A2_loop_moe), a clean ~4x over the single-GPU baseline; dp_degree 16 (4 nodes) measured WORSE,
# ~38 samples/s -- cross-node all-reduce overhead and thinner per-GPU MoE routing batches outweigh
# the extra GPUs at this model size. At ~59 samples/s the slowest arm finishes 76,250 steps in
# ~11.5h, comfortably inside this script's 20h slot with no chaining needed; --requeue below is a
# safety net, not a requirement. Every arm MUST use the same dp_degree: arms are compared on their
# answers to the same resumable_distributed_sampler(seed=42) stream, which depends on world_size,
# so changing node count per arm would make them read data in a different order and break the
# comparison -- do not change --nodes for a subset of arms.
#
# CPUS-PER-TASK: 8, not the node's full core count. This job is GPU-bound (dataloading is a packed
# mmap read, not CPU-heavy tokenization), and --exclusive + --mem=0 already reserve the whole node
# for this task regardless of the declared CPU count, so a lower number costs nothing in practice.
# It does matter for AIFAC_S07_154's account-wide boost_qos_lprod quota (MaxTRESPerAccount:
# cpu=256, gpu=32, node=8, shared with every other user on the account, not just this study):
# discovered 2026-08-12 when 9 of 12 seed-wave array tasks queued on MaxCpuPerAccount while another
# user's concurrent jobs (also 1 node/4 GPU/32 CPU each) were consuming the other half of the same
# 256-CPU pool. The node/GPU caps bind at the same 8 concurrent jobs regardless of this setting, but
# a smaller declared CPU request can still let a task slip in first once anyone's slot frees up, and
# it is simply a more honest ask on a shared account.
#
# 11 arms (A0-A6, N1-N4; see arm_list.txt), one SLURM array task each, each task exclusively
# owning 1 node x 4 GPUs. Bare-metal (no singularity): the container image's mamba_ssm was
# ABI-incompatible with its own torch build (undefined symbol at import, independent of backend or
# node count -- discovered when the first speed test failed identically on every config), so this
# uses /leonardo_work/EUHPC_D21_101/mfrey/modalities/.venv directly, with mamba-ssm/causal-conv1d
# rebuilt into it against that venv's torch (2.10.0+cu126) -- see the mamba_ssm rebuild note in
# docs/components/nemotron_loops_research_plan.md section 9 (added 2026-08-11).
#
# CHECKPOINT / RESUME: each arm's experiment_id is pinned (not timestamped) in its config, so
# checkpoints always land at the same path across attempts. On launch, this script checks for that
# arm's last_checkpoint_info.json: if present, it resumes with `modalities warmstart`; if absent,
# it starts fresh with `modalities run`. Combined with --requeue and the SIGUSR1 trap below (SLURM
# sends USR1 180s before the time limit; the trap asks SLURM to requeue this exact array task
# rather than letting it die), an arm that does not finish inside one slot continues automatically
# across as many slots as it needs -- no manual resubmission required. Checkpointing interval is
# 5000 steps (~12 checkpoints over 76,250 steps); checkpoint_saving.k: 1 in the base config keeps
# only the most recent, bounding disk use per arm regardless of how many slots it takes.
#
# WANDB: OFFLINE (compute nodes have no internet). Sync from the login node with
# config_files/nemotron/loop_ablation_5b_cluster/sync_wandb.sh (add --watch for live sync while
# the wave runs).
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/run_wave_5b.sh

set -x

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array (already set in the header above)." >&2
    exit 1
fi

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
ARM_LIST="${REPO}/config_files/nemotron/loop_ablation_5b_cluster/arm_list.txt"
ARM_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ARM_LIST}")
if [ -z "${ARM_NAME}" ]; then
    echo "Error: no arm at line ${SLURM_ARRAY_TASK_ID} of ${ARM_LIST}" >&2
    exit 1
fi

EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster"
mkdir -p "${EXPERIMENTS_ROOT}" "${REPO}/logs"

module purge
module load cuda/12.6 gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID % 10000) + SLURM_ARRAY_TASK_ID ))

echo "=========================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID}  arm: ${ARM_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}  Nodes: ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES})"
echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "START TIME: $(date)"
echo "=========================================="

# SLURM sends USR1 180s before the wall-clock limit. Requeue this exact array task rather than
# letting torchrun be killed mid-step; the checkpoint/warmstart detection below picks up from the
# most recent checkpoint on the next attempt. --requeue in the header makes this an allowed action.
trap 'echo "[${ARM_NAME}] caught USR1, requeuing job ${SLURM_JOB_ID}"; scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

LAST_CKPT="${EXPERIMENTS_ROOT}/${ARM_NAME}/checkpoints/last_checkpoint_info.json"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_ablation_5b_cluster"

if [[ -f "${LAST_CKPT}" ]]; then
    MODE="warmstart"
    CONFIG_FILE_PATH="${CONFIG_DIR}/config_${ARM_NAME}_warmstart.yaml"
    MODALITIES_ARGS=(warmstart
        --config_file_path "${CONFIG_FILE_PATH}"
        --experiments_root_path "${EXPERIMENTS_ROOT}"
        --last_checkpoint_info_file_path "${LAST_CKPT}")
    echo "[${ARM_NAME}] found ${LAST_CKPT} -> resuming with modalities warmstart"
else
    MODE="run"
    CONFIG_FILE_PATH="${CONFIG_DIR}/config_${ARM_NAME}.yaml"
    MODALITIES_ARGS=(run
        --config_file_path "${CONFIG_FILE_PATH}"
        --experiments_root_path "${EXPERIMENTS_ROOT}")
    echo "[${ARM_NAME}] no checkpoint found -> starting fresh with modalities run"
fi

cd "${REPO}"
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    bash -c "
        ${REPO}/.venv/bin/torchrun \
            --node_rank=\${SLURM_PROCID} \
            --nnodes=${SLURM_JOB_NUM_NODES} \
            --nproc_per_node=4 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            ${REPO}/.venv/bin/modalities ${MODALITIES_ARGS[*]}
    "

echo "END TIME: $(date)"
echo "=== ${ARM_NAME} (array task ${SLURM_ARRAY_TASK_ID}) FINISHED THIS SLOT ==="
