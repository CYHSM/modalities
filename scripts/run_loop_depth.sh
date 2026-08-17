#!/bin/bash
#SBATCH --job-name=loopdepth
#SBATCH --account=AIFAC_S07_154
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/loopdepth-%j.out
#SBATCH --error=logs/loopdepth-%j.err
#
# Phase 0: what are each loop group's extra iterations worth?
#
# Only the LOOPED arms are processed. A0 and the N anchors have no loop groups, so every setting would
# be identical to the baseline and the run would contribute nothing but wall clock. The run list is
# read from docs/loopotron/wave2_final_stats.json rather than from the experiments directory, which
# still holds discarded warmstart runs (e.g. A4_loop_mamba_moe_seed2, superseded by ..._seed2_redo)
# that the paper excludes.
#
# Each run evaluates the same 262k tokens once per setting: one baseline, one per looped group, and
# six points of the global depth sweep. Nothing is trained.
#
# Submit from the repository root:
#
#   sbatch scripts/run_loop_depth.sh

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

# A1-A6 are the looped arms; A0 and N* have no loop groups.
ARMS=$("${REPO}/.venv/bin/python" -c "
import json
runs = json.load(open('${STATS}'))
print(' '.join(
    run
    for arm, entry in runs.items()
    if arm.startswith('A') and arm != 'A0'
    for run in entry['runs']
))
")

echo "=== arms: ${ARMS} ==="
FAILED=()
for ARM in ${ARMS}; do
    echo "---------------------------------------------------------------"
    echo "ARM ${ARM}  $(date)"
    echo "---------------------------------------------------------------"
    "${REPO}/.venv/bin/python" "${REPO}/scripts/run_loop_depth.py" \
        --arm "${ARM}" \
        --experiments-root "${EXPERIMENTS_ROOT}" || FAILED+=("${ARM}")
done

echo "=== FINISHED $(date) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED ARMS: ${FAILED[*]}" >&2
    exit 1
fi
echo "All arms succeeded."
