#!/bin/bash
#SBATCH --job-name=nemo5bredo
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
#SBATCH --array=1-5
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/nemo5bredo-%x-%A_%a.out
#SBATCH --error=logs/nemo5bredo-%x-%A_%a.err
#
# Clean replacement for the 5 seed-replicate runs contaminated by a cancel+warmstart on 2026-08-13
# (A4_seed2, A4_seed3, A5_seed2, A5_seed3, A6_seed2 -- see run_wave_5b_seeds.sh and the "Seeds"
# limitation in docs/loopotron/loopotron.tex). Cancelling those mid-run and resuming via
# `modalities warmstart` showed up as a 0.040-0.055 nat improvement over each arm's seed1 run,
# dwarfing the ~0.001-0.003 nat spread the genuinely-uninterrupted seed replicates (A1 x3, A2 x3,
# A3 x3, A6 seed3) show. NOTE: an earlier revision of this comment blamed re-executed gradient
# steps past the checkpoint. That is WRONG and was disproved -- the resumed runs consumed exactly
# 4,997,120,000 tokens like every other arm, and their logged LR matches the single 76,250-step
# cosine to 6 s.f. What a debug-queue repro did find is a checkpoint round-trip defect on the tied
# embedding/output weight: `modalities run` writes a transformer.lm_head.weight that is not the
# trained shared tensor (cosine 0.012 to it after 10 steps, no optimizer state), while the live
# model is correctly tied at every construction stage in both paths. Loading that writes both names
# into the one tied tensor, so the resumed model depends on which is applied last. That is enough to
# know a resumed run is not interchangeable with an uninterrupted one; it is not yet a full account
# of the 0.04-0.055 nat shift. See speedtest/{inspect_weight_tying.py,run_tying_test.sh}. Also see
# generate_seed_redo_configs.py, which produced config_A{4,5,6}_..._seed{2,3}_redo.yaml under fresh
# experiment_ids so they cannot collide with (or accidentally warmstart from) the contaminated
# checkpoints already on disk.
#
# DO NOT CANCEL AND RESUBMIT THESE TASKS. That is exactly what caused the contamination being fixed
# here. If a task genuinely needs more than one 20h slot, let --requeue and the USR1 trap below
# handle it automatically at the real wall-clock boundary (SLURM-triggered, unavoidable, and the
# reason checkpointing exists) rather than cancelling it by hand mid-run -- a SLURM-triggered
# requeue also re-executes back to the last checkpoint, but there every arm needing one takes the
# same hit, so it stays a fair, matched-budget comparison; a hand-picked subset does not.
#
# Otherwise identical to run_wave_5b_seeds.sh: same 1 node x 4 GPUs, dp_degree 4, bare-metal .venv,
# --cpus-per-task=8 (see the CPUS-PER-TASK note in run_wave_5b.sh), qos=normal (see the QOS note in
# run_wave_5b_seeds.sh -- boost_qos_lprod's account-wide 8-node/32-GPU cap is shared with every user
# on AIFAC_S07_154), offline wandb + sync_wandb.sh. Only the arm list and config directory contents
# read differ.
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/run_wave_5b_seeds_redo.sh

set -x

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array (already set in the header above)." >&2
    exit 1
fi

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
ARM_LIST="${REPO}/config_files/nemotron/loop_ablation_5b_cluster/arm_list_seeds_redo.txt"
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
# This is the ONLY sanctioned path back into warmstart for these tasks -- see the header note above.
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
