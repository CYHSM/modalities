#!/bin/bash
set -e

# Helper script to run ablation test configs (GPT-2 & DualPath)
# This tests the full cycle: training (10 steps) -> DCP checkpoint saving -> automatic HF conversion -> OLMES downstream evaluation -> W&B logging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODALITIES_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_EXEC="${PYTHON_EXEC:-/leonardo_work/EUHPC_D21_101/mfrey/venvs/olmes_uv/bin/python}"
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG="${1:-${SCRIPT_DIR}/dualpath_test.yaml}"

echo "=========================================="
echo "Running ablation test with config:"
echo "Config: ${CONFIG}"
echo "GPUs:   ${NUM_GPUS}"
echo "Python: ${PYTHON_EXEC}"
echo "=========================================="

export PYTHONPATH="${MODALITIES_ROOT}/src:${PYTHONPATH:-}"

if [ "${NUM_GPUS}" -eq 1 ]; then
    ${PYTHON_EXEC} -m modalities.main run "${CONFIG}"
else
    ${PYTHON_EXEC} -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" -m modalities.main run "${CONFIG}"
fi

echo "=========================================="
echo "Test run completed!"
echo "=========================================="
