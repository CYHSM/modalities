#!/usr/bin/env bash
# Launches one arm of the Nemotron layer-loop ablation on a single GPU.
#
#   ./config_files/nemotron/loop_ablation/run_arm.sh <ARM_NAME> <GPU_ID> [RDZV_PORT]
#
# Example, running six arms concurrently on GPUs 0-5:
#
#   for i in "A0_baseline 0" "A1_loop_mamba 1" "A2_loop_moe 2" \
#            "A3_loop_attention 3" "A4_loop_mamba_moe 4" "A5_loop_mamba_attention 5"; do
#     ./config_files/nemotron/loop_ablation/run_arm.sh $i &
#   done; wait
#
# The RDZV port defaults to 29500 + GPU_ID so concurrent arms do not collide. The experiment id is
# the arm name plus a timestamp, which becomes the wandb run name and the results directory, so
# re-running an arm never overwrites an earlier run.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <ARM_NAME> <GPU_ID> [RDZV_PORT]" >&2
    echo "arms:" >&2
    ls "$(dirname "$0")"/config_*.yaml | sed 's|.*/config_||; s|\.yaml$||; s|^|  |' >&2
    exit 1
fi

ARM_NAME=$1
GPU_ID=$2
RDZV_PORT=${3:-$((29500 + GPU_ID))}

REPOSITORY_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CONFIG_PATH="${REPOSITORY_ROOT}/config_files/nemotron/loop_ablation/config_${ARM_NAME}.yaml"

# Work whether or not the calling shell has the venv activated: a detached or non-interactive
# launch otherwise dies immediately with "torchrun: command not found".
if [[ -x "${REPOSITORY_ROOT}/.venv/bin/torchrun" ]]; then
    export PATH="${REPOSITORY_ROOT}/.venv/bin:${PATH}"
fi
if ! command -v torchrun > /dev/null; then
    echo "torchrun not found; activate the environment or create ${REPOSITORY_ROOT}/.venv" >&2
    exit 1
fi

if [[ ! -f ${CONFIG_PATH} ]]; then
    echo "No config for arm '${ARM_NAME}' at ${CONFIG_PATH}" >&2
    exit 1
fi

EXPERIMENT_ID="${ARM_NAME}__$(date +%Y%m%d-%H%M%S)"

echo "arm=${ARM_NAME} gpu=${GPU_ID} port=${RDZV_PORT} experiment_id=${EXPERIMENT_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" torchrun \
    --rdzv-endpoint "localhost:${RDZV_PORT}" \
    --nnodes 1 \
    --nproc_per_node 1 \
    "$(which modalities)" run \
    --experiments_root_path "${REPOSITORY_ROOT}/results" \
    --experiment_id "${EXPERIMENT_ID}" \
    --config_file_path "${CONFIG_PATH}"
