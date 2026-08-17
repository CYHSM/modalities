#!/bin/bash
#SBATCH --job-name=nemo5bseed
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --array=1-12
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/nemo5bseed-%x-%A_%a.out
#SBATCH --error=logs/nemo5bseed-%x-%A_%a.err
#
# Seed-noise-floor companion to run_wave_5b.sh: two extra seeds (seed2, seed3) of each A1-A6 looped
# variant, so Wave 2's headline ranking (A1 loop-mamba best, A3 loop-attention worst, the A4/A5 swap
# vs. Wave 1) can be checked against measured run-to-run noise instead of resting on n=1 per arm --
# see the "Seeds" limitation in docs/loopotron/loopotron.tex. A0 and N1-N4 are NOT re-seeded here:
# the open question is specifically about the ranking *among loop types*, which only concerns
# A1-A6.
#
# Everything else is identical to run_wave_5b.sh and its rationale applies unchanged here: same 1
# node x 4 GPUs per arm (dp_degree 4, empirically the fastest single-node setting), same bare-metal
# .venv, same checkpoint/warmstart/--requeue/USR1-trap machinery, same offline wandb + sync_wandb.sh,
# and the same --cpus-per-task=8 (this job is GPU-bound; see the CPUS-PER-TASK note in
# run_wave_5b.sh for why that number matters on AIFAC_S07_154's shared account-wide CPU quota).
#
# QOS: normal, not boost_qos_lprod (which run_wave_5b.sh used and which the first 6 seed-wave tasks
# ran under -- see run_wave_5b.sh). Discovered 2026-08-13, from the cluster's published QOS table:
# boost_qos_lprod caps the WHOLE ACCOUNT at 8 nodes / 32 GPUs concurrently (shared across every user
# on AIFAC_S07_154, not just this study) -- MaxCpuPerAccount/MaxNodePerAccount queuing on tasks 7-12
# was this cap, not a CPU-declaration problem (--exclusive grants the full node's real CPU count
# regardless of --cpus-per-task once a job actually starts, so the earlier cpu=8 change only ever
# helped at admission time, not while running). `normal` on the same boost_usr_prod partition has no
# such account-wide TRES cap (`sacctmgr show qos normal` -- MaxTRESPA/MaxTRESPU/GrpTRES all empty),
# just a per-job cap of 64 nodes and a 24h walltime, comfortably above this script's 20h. Tasks 7-12
# were cancelled and resubmitted under `normal`; tasks 1-6 (three already finished, three still
# running) stay on boost_qos_lprod undisturbed -- --requeue reuses each job's original submission
# QOS, so switching this file does not change already-submitted tasks.
#
# The only structural difference is which arm list and config directory contents get read.
#
# SEEDING MECHANISM: model weight initialization is unseeded in this codebase, so re-running an
# identical config under a new experiment_id draws a fresh random init while the training data
# order (sampler seed 42) and the synthetic-eval question sets (seed 1234) stay fixed -- the same
# mechanism Wave 1 used for the A6a x4 replication. See generate_seed_configs.py, which produced
# config_A{1..6}_..._seed{2,3}.yaml (and generate_warmstart_configs.py, re-run to give them
# warmstart siblings) from the unmodified seed1 configs already used by run_wave_5b.sh.
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/run_wave_5b_seeds.sh

set -x

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array (already set in the header above)." >&2
    exit 1
fi

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
ARM_LIST="${REPO}/config_files/nemotron/loop_ablation_5b_cluster/arm_list_seeds.txt"
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
