#!/bin/bash
#SBATCH --job-name=nemorefine
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/nemorefine-%x-%A_%a.out
#SBATCH --error=logs/nemorefine-%x-%A_%a.err
#
# Launches the loop-refinement wave: A1's Mamba loop at K in {3, 6}, with and without the per-group
# refinements (stabilized recurrence, injection norm, FiLM iteration conditioning), all at Wave 2's
# exact 5B-token budget so that R_k3_simple is a straight replication of A1_loop_mamba.
#
# Submit from the repository root:
#
#   sbatch --array=1-7 config_files/nemotron/loop_refinements/run_refinement_wave.sh \
#       config_files/nemotron/loop_refinements/arm_list_1node.txt
#
# Every arm runs on ONE node, which is what keeps the whole wave on Wave 2's data stream: the
# resumable sampler's order depends on world size, so a different node count would silently make an
# arm incomparable to both Wave 2 and the rest of the wave. The --nodes value must therefore match
# the node count generate_refinement_configs.py baked into the configs (NODES_PER_LOOP_COUNT), since
# dp_degree and the per-GPU micro batch were set from it. A mismatch does not crash: the run trains
# happily at the wrong global batch and produces a number that looks fine and means nothing. The
# guard below checks it rather than trusting the submitter.
#
# Expected wall clock from the measured throughputs: K=3 ~12.4h, K=6 ~18.0h, both inside one 24h
# slot with headroom. That is the point -- a requeue resumes through `modalities warmstart`, the path
# with the open checkpoint defect (loopotron.tex section 3.6). The USR1 trap below is a safety net
# for a genuine wall-clock overrun, not a routine path.

set -x

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array (see the header above)." >&2
    exit 1
fi

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
ARM_LIST="${1:?Error: pass the arm list as the first argument (see the header above)}"
ARM_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ARM_LIST}")
if [ -z "${ARM_NAME}" ]; then
    echo "Error: no arm at line ${SLURM_ARRAY_TASK_ID} of ${ARM_LIST}" >&2
    exit 1
fi

EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_refinements"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_refinements"
CONFIG_FILE_PATH="${CONFIG_DIR}/config_${ARM_NAME}.yaml"
mkdir -p "${EXPERIMENTS_ROOT}" "${REPO}/logs"

# The allocation must match what the config was generated for, or the global batch silently changes.
CONFIG_DP_DEGREE=$(sed -n 's/^    data_parallel_replicate_degree: \([0-9]*\)$/\1/p' "${CONFIG_FILE_PATH}")
ALLOCATED_DP_DEGREE=$((SLURM_JOB_NUM_NODES * 4))
if [ "${CONFIG_DP_DEGREE}" != "${ALLOCATED_DP_DEGREE}" ]; then
    echo "Error: ${ARM_NAME} was generated for dp_degree ${CONFIG_DP_DEGREE}, but this allocation of" \
         "${SLURM_JOB_NUM_NODES} node(s) x 4 GPUs gives ${ALLOCATED_DP_DEGREE}. Submit this arm list" \
         "with --nodes=$((CONFIG_DP_DEGREE / 4))." >&2
    exit 1
fi

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
echo "dp_degree:  ${ALLOCATED_DP_DEGREE}  (config expects ${CONFIG_DP_DEGREE})"
echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "START TIME: $(date)"
echo "=========================================="

trap 'echo "[${ARM_NAME}] caught USR1, requeuing job ${SLURM_JOB_ID}"; scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

LAST_CKPT="${EXPERIMENTS_ROOT}/${ARM_NAME}/checkpoints/last_checkpoint_info.json"
if [[ -f "${LAST_CKPT}" ]]; then
    CONFIG_FILE_PATH="${CONFIG_DIR}/config_${ARM_NAME}_warmstart.yaml"
    MODALITIES_ARGS=(warmstart
        --config_file_path "${CONFIG_FILE_PATH}"
        --experiments_root_path "${EXPERIMENTS_ROOT}"
        --last_checkpoint_info_file_path "${LAST_CKPT}")
    echo "[${ARM_NAME}] found ${LAST_CKPT} -> resuming with modalities warmstart"
    echo "[${ARM_NAME}] WARNING: warmstart carries the open checkpoint defect (loopotron.tex 3.6)." \
         "This arm is no longer cleanly comparable; note it before using the result."
else
    MODALITIES_ARGS=(run
        --config_file_path "${CONFIG_FILE_PATH}"
        --experiments_root_path "${EXPERIMENTS_ROOT}")
    echo "[${ARM_NAME}] no checkpoint found -> starting fresh with modalities run"
fi

cd "${REPO}"
srun --input none --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    bash -c "
        ${REPO}/.venv/bin/torchrun \
            --node_rank=\${SLURM_PROCID} \
            --nnodes=${SLURM_JOB_NUM_NODES} \
            --nproc_per_node=4 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            ${REPO}/.venv/bin/modalities ${MODALITIES_ARGS[*]}
    "
SRUN_STATUS=$?

echo "END TIME: $(date)"
# srun's exit code is NOT propagated by the launcher otherwise, which is how a NCCL crash in the 5B
# seed wave came back as COMPLETED with a corrupt checkpoint on disk. Report it explicitly.
echo "=== ${ARM_NAME} (array task ${SLURM_ARRAY_TASK_ID}) FINISHED THIS SLOT, srun exit ${SRUN_STATUS} ==="
exit ${SRUN_STATUS}
