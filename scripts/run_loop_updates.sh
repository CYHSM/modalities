#!/bin/bash
#SBATCH --job-name=loopupd
#SBATCH --account=AIFAC_S07_154
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/loopupd-%j.out
#SBATCH --error=logs/loopupd-%j.err
#
# Runs the loop-update diagnostic over every Wave 2 run that the paper actually reports.
#
# The run list is read from docs/loopotron/wave2_final_stats.json rather than from the contents of
# the experiments directory. Those differ: several discarded warmstart runs (e.g.
# A4_loop_mamba_moe_seed2, superseded by ..._seed2_redo) still have checkpoints on disk, and
# analyzing them would describe models the paper excludes.
#
# Each arm runs as its own process so that one arm's captured activations -- roughly 1.5 GB in
# float32 for a 22-layer arm at 8x2048 tokens -- are released before the next one starts. This is a
# single forward pass per arm; nothing is trained and no optimizer state is read.
#
# Submit from the repository root:
#
#   sbatch scripts/run_loop_updates.sh

set -x

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster"
STATS="${REPO}/docs/loopotron/wave2_final_stats.json"
mkdir -p "${REPO}/logs"

module purge
module load cuda/12.6 gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

cd "${REPO}"

ARMS=$("${REPO}/.venv/bin/python" -c "
import json
runs = json.load(open('${STATS}'))
print(' '.join(run for arm in runs.values() for run in arm['runs']))
")

echo "=== ${#ARMS} arms to process ==="
FAILED=()
for ARM in ${ARMS}; do
    echo "---------------------------------------------------------------"
    echo "ARM ${ARM}  $(date)"
    echo "---------------------------------------------------------------"
    "${REPO}/.venv/bin/python" "${REPO}/scripts/run_loop_updates.py" \
        --arm "${ARM}" \
        --experiments-root "${EXPERIMENTS_ROOT}" || FAILED+=("${ARM}")
done

echo "=== FINISHED $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED ARMS: ${FAILED[*]}" >&2
    exit 1
fi
echo "All arms succeeded."
