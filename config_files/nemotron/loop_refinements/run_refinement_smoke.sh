#!/bin/bash
#SBATCH --job-name=refine_smoke
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/refine_smoke-%j.out
#SBATCH --error=logs/refine_smoke-%j.err
#
# Smoke-tests every arm of the loop-refinement wave before any of them is launched for real, and
# measures the throughput the wave's token budget was sized against.
#
# Two things need a GPU and therefore cannot be checked on the login node:
#   1. That each config BUILDS. The schema test (tests/models/nemotron/test_shipped_config_schemas.py)
#      only validates the config against the pydantic models; it never instantiates the model, and
#      Main's experiment-id sync calls .cuda() unconditionally.
#   2. Throughput per arm. generate_refinement_configs.py picked 23,000 steps from an ESTIMATE that
#      K=12 runs ~3x slower than K=3 (67 vs 22 executed layers). If that estimate is wrong the K=12
#      arms overrun the 20h slot, get requeued, and resume through `modalities warmstart` -- the
#      path with the open checkpoint defect. Better to find out in 30 minutes on the debug queue.
#
# Each arm runs for a handful of steps against the real data and the real 4-GPU layout, so this also
# catches a refinement that only fails at scale: an FSDP2 wrapping problem with the new root-level
# `transformer.loop_mods` parameters, or a dtype mismatch between the fp32 recurrence vectors and
# bf16 hidden states.
#
# boost_qos_dbg caps a job at 30 minutes, so the arm list is filtered by an optional regex argument
# (and the per-arm window by an optional second argument). Submit from the repository root.
#
# What actually needs measuring is throughput at each loop count, since that is what the wave's
# token budget was sized against -- and the heaviest arm at each K is the conservative one to
# measure. Three arms at 7 minutes fits one debug slot:
#
#   sbatch config_files/nemotron/loop_refinements/run_refinement_smoke.sh '_all$' 420
#
# The remaining six arms differ from these only in loop_config flags, and both the build and the
# parameter accounting of every flag combination are already pinned by
# tests/models/nemotron/test_layer_loops.py.
#
# With no argument every arm runs, which overruns the debug QOS -- use that form only on a longer
# QOS.

set -x

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_refinements"
ARM_LIST="${CONFIG_DIR}/arm_list_refinements.txt"
EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_refine_smoke"
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
PORT=29850

cd "${REPO}"

ARM_FILTER="${1:-.}"
# Seconds per arm. Must cover model build + FSDP2 wrap + at least training_log_interval_in_steps (5)
# steps, or the run is killed before it reports any throughput at all -- which is what 240 did to the
# K=12 arms, whose build alone takes ~2 minutes. 420 leaves ~4 minutes of actual training there.
WINDOW="${2:-420}"

# Shortest-first, so a build error in the cheap K=3 arms surfaces within a couple of minutes rather
# than after the K=12 ones have burned the slot. `timeout` sits INSIDE srun so the signal reaches
# torchrun and its four workers directly; killing srun from here leaves the workers holding the GPUs.
while read -r ARM_NAME; do
    [ -z "${ARM_NAME}" ] && continue
    echo "=============================================================="
    echo "SMOKE: ${ARM_NAME}   $(date)"
    echo "=============================================================="
    PORT=$((PORT + 1))
    # --input none is load-bearing: srun otherwise reads this shell's stdin and forwards it to the
    # task, swallowing the process substitution that feeds `while read`. The loop then runs exactly
    # one arm and exits looking like a success. (Cost one debug-queue slot on 2026-08-14.)
    srun --input none --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c "
        timeout --signal=INT --kill-after=30 ${WINDOW} \
        ${REPO}/.venv/bin/torchrun \
            --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
            --nnodes=1 --nproc_per_node=4 \
            ${REPO}/.venv/bin/modalities run \
                --config_file_path ${CONFIG_DIR}/config_${ARM_NAME}.yaml \
                --experiments_root_path ${EXPERIMENTS_ROOT}
    "
    echo "--- ${ARM_NAME} stopped after its ${WINDOW}s smoke window (timeout exit 124 is expected) ---"
done < <(sort -t k -k2 -n "${ARM_LIST}" | grep -E "${ARM_FILTER}")

echo "=============================================================="
echo "SMOKE RESULTS: throughput and loss per arm"
echo "=============================================================="
# `samples/s` is what the wave's wall-clock estimate was built on; a step is 32 samples
# (8 micro-batch x 4 dp).
grep -hoE "\[[^]]*\] (samples/s|train loss avg): [0-9.]+" "${REPO}/logs/refine_smoke-${SLURM_JOB_ID}.out" | tail -60 || true
echo "=== SMOKE TEST FINISHED ==="
