#!/bin/bash
#SBATCH --job-name=nemo_warmstart_test
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=logs/warmstart_test-%j.out
#SBATCH --error=logs/warmstart_test-%j.err

# Verifies the checkpoint/warmstart round trip before launching the real wave -- see
# docs/components/nemotron_loops_research_plan.md section 9.4, which calls this mandatory:
# both the LR schedule resume point and the dataloader sample-skip fail silently if wrong.
#
#   1. warmstart_test_train.yaml    -- A0_baseline, stopped at step 10 (one checkpoint written)
#   2. warmstart_test_resume.yaml   -- warmstart from that checkpoint, continues to step 20
#   3. warmstart_test_reference.yaml -- A0_baseline run straight through to step 20, uninterrupted
#
# Compare (2)'s and (3)'s logged LR at each step 11-20: they must match exactly (the scheduler is
# deterministic in the step count). Loss will not match bit-for-bit (MoE routing uses CUDA atomics,
# see section 4 of the research plan), but should track closely given identical data order.
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/speedtest/run_warmstart_test.sh

set -x -e

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
RESULTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/nemo_warmstart_test"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_ablation_5b_cluster/speedtest"
mkdir -p "${RESULTS_ROOT}" "${REPO}/logs"

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

run_torchrun () {
    local PORT=$1
    shift
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c "
        ${REPO}/.venv/bin/torchrun \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
            --nnodes=1 \
            --nproc_per_node=4 \
            ${REPO}/.venv/bin/modalities $*
    "
}

cd "${REPO}"

echo "=== [1/3] train to step 10 ==="
run_torchrun 29701 run \
    --config_file_path "${CONFIG_DIR}/warmstart_test_train.yaml" \
    --experiments_root_path "${RESULTS_ROOT}"

LAST_CKPT="${RESULTS_ROOT}/A0_baseline/checkpoints/last_checkpoint_info.json"
if [[ ! -f "${LAST_CKPT}" ]]; then
    echo "FAIL: no checkpoint written at ${LAST_CKPT}" >&2
    exit 1
fi
echo "checkpoint found: ${LAST_CKPT}"
cat "${LAST_CKPT}"

echo "=== [2/3] warmstart resume to step 20 ==="
run_torchrun 29702 warmstart \
    --config_file_path "${CONFIG_DIR}/warmstart_test_resume.yaml" \
    --experiments_root_path "${RESULTS_ROOT}" \
    --last_checkpoint_info_file_path "${LAST_CKPT}"

echo "=== [3/3] continuous reference run to step 20 ==="
run_torchrun 29703 run \
    --config_file_path "${CONFIG_DIR}/warmstart_test_reference.yaml" \
    --experiments_root_path "${RESULTS_ROOT}"

echo "=== WARMSTART TEST FINISHED -- compare LR columns of [2] and [3] by hand ==="
