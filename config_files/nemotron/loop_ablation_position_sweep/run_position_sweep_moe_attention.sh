#!/bin/bash
#SBATCH --job-name=nemopos2
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
#SBATCH --array=1-7
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/nemopos2-%x-%A_%a.out
#SBATCH --error=logs/nemopos2-%x-%A_%a.err
#
# THE LOOP-POSITION SWEEP, FAMILIES 2 AND 3: MoE and attention.
#
# Companion to run_position_sweep.sh, which ran the Mamba family (P0..P4) and found a 0.0498-nat
# spread across positions -- 23.7x the 0.0021 seed s.d., LARGER than Wave 2's entire 0.0361-nat
# operator spread -- with the optimum at built index 2 and index 9 actually WORSE than not looping at
# all. Position is therefore established as a large causal variable for Mamba. These two families ask
# whether that curve belongs to the operator or to the stack.
#
#   Q0..Q4  one MoE layer at K=6, built indices 1, 4, 6, 8, 11   (array tasks 1-5)
#   R0..R1  one attention layer at K=6, built indices 3, 10      (array tasks 6-7)
#
# Every arm in both families is 12 built / 17 executed, so WITHIN a family active parameters and
# FLOPs are identical by construction and position is the only variable -- which is all the position
# claim rests on. ACROSS families the cost differs (an extra MoE execution costs ~2x an extra Mamba
# one, attention ~0.59x), so the cross-family comparison is of curve SHAPE -- where the optimum sits,
# whether loss degrades with depth -- and NOT of magnitude. K, not FLOPs, is held fixed on purpose:
# it keeps "same loop, different operator, different place" as the contrast, and it maximises signal
# in the two weaker operators, so that a null is a real answer rather than an underpowered one.
#
# R is a single contrast, not a curve: the base pattern MEM*EMEMEM*E holds only two attention layers.
# That is a property of the stack, not a choice -- widening it needs a different base pattern, which
# would break comparability with A0 and every Wave 2 arm. It runs at n=1 like the rest; if the R0/R1
# gap lands under ~5 s.d. it needs seeds before it can be read, via generate_seed_configs.py.
#
# Reading it: if all three families peak in the same region, position is a stack property and Wave 2's
# operator ranking is largely a positional artifact. If the optima differ by operator, position and
# operator interact, and neither Wave 2's ranking nor a single position curve tells the whole story.
#
# Everything operational is inherited from run_wave_5b_seeds_redo.sh and its rationale applies
# unchanged: same 1 node x 4 GPUs per arm (dp_degree 4, empirically the fastest single-node setting
# -- 4 nodes measured WORSE), same bare-metal .venv, same offline wandb + sync_wandb.sh, same
# --cpus-per-task=8 for admission against the account-wide CPU quota, and qos=normal rather than
# boost_qos_lprod (which caps the WHOLE ACCOUNT at 8 nodes / 32 GPUs concurrently).
#
# Expected ~11h per arm: the Mamba family measured 11h02m-11h14m at the same 17 executed layers.
# The MoE family is the heavier of the two here (10 MoE executions against the P family's 5), so
# allow for the upper end. The 20h request keeps headroom, which matters here -- see below.
#
# DO NOT CANCEL AND RESUBMIT THESE TASKS, and do not manually warmstart them. There is an unfixed
# checkpoint round-trip defect on the tied embedding/output weight, and in Wave 2 it made five
# resumed runs land 0.040-0.055 nats "better" than their own first seeds -- an order of magnitude
# past genuine seed noise, and enough to have overturned the paper's headline finding had it been
# averaged in. The warmstart configs and the USR1 trap below exist only so that a SLURM-initiated
# requeue at a real wall-clock boundary does not lose the run outright. IF AN ARM'S LOG SHOWS IT
# RESUMED, DISCARD THAT RUN AND RELAUNCH IT UNDER A FRESH EXPERIMENT ID rather than reporting its
# number. See docs/loopotron/loopotron.tex, "Wave 2 seed replicates".
#
# Submit from the repository root:
#   sbatch config_files/nemotron/loop_ablation_position_sweep/run_position_sweep_moe_attention.sh

set -x

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array (already set in the header above)." >&2
    exit 1
fi

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_ablation_position_sweep"
ARM_LIST="${CONFIG_DIR}/arm_list_moe_attention.txt"
ARM_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ARM_LIST}")
if [ -z "${ARM_NAME}" ]; then
    echo "Error: no arm at line ${SLURM_ARRAY_TASK_ID} of ${ARM_LIST}" >&2
    exit 1
fi

# Same experiments root as Wave 2: these arms are read directly against A0 and A1, and the
# collation scripts in scripts/ already point here.
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
# letting torchrun be killed mid-step. --requeue in the header makes this an allowed action.
trap 'echo "[${ARM_NAME}] caught USR1, requeuing job ${SLURM_JOB_ID}"; scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

LAST_CKPT="${EXPERIMENTS_ROOT}/${ARM_NAME}/checkpoints/last_checkpoint_info.json"

if [[ -f "${LAST_CKPT}" ]]; then
    MODE="warmstart"
    CONFIG_FILE_PATH="${CONFIG_DIR}/config_${ARM_NAME}_warmstart.yaml"
    MODALITIES_ARGS=(warmstart
        --config_file_path "${CONFIG_FILE_PATH}"
        --experiments_root_path "${EXPERIMENTS_ROOT}"
        --last_checkpoint_info_file_path "${LAST_CKPT}")
    echo "[${ARM_NAME}] !!! RESUMING FROM CHECKPOINT -- this run is CONTAMINATED for reporting"
    echo "[${ARM_NAME}] !!! (tied embedding/output weight checkpoint defect; see this script's header)"
    echo "[${ARM_NAME}] !!! discard its final number and relaunch under a fresh experiment id"
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
