#!/bin/bash
#SBATCH --job-name=layerprof
#SBATCH --account=AIFAC_S07_154
#SBATCH --qos=normal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/layerprof-%j.out
#SBATCH --error=logs/layerprof-%j.err
#
# Measures the per-layer update profile used by the update-norm predictor
# (docs/loopotron/update_norm_predictor.md).
#
# The predictor's claim is that you can tell WHERE extra depth will pay off from a single UNLOOPED
# baseline checkpoint, before training any loop arm. So the arm that matters here is A0_baseline: its
# twelve layers cover every position the position sweep trains an arm for, and its own-input-relative
# update norms are the candidate predictor of those twelve arms' outcomes.
#
# One forward pass over the same fixed 8-sequence test batch every other diagnostic uses. Nothing is
# trained and no optimizer state is read. Adding arms as arguments profiles them too, which is only
# useful as a consistency check -- the point of the experiment is that A0 alone suffices.
#
#   sbatch scripts/run_layer_profile.sh                  # A0_baseline
#   sbatch scripts/run_layer_profile.sh A0_baseline A1_loop_mamba

set -x

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster"
ARMS=("${@:-A0_baseline}")
mkdir -p "${REPO}/logs"

module purge
module load cuda/12.6 gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

cd "${REPO}"
FAILED=()
for ARM in "${ARMS[@]}"; do
    echo "--- ${ARM}  $(date) ---"
    "${REPO}/.venv/bin/python" "${REPO}/scripts/run_loop_updates.py" \
        --arm "${ARM}" \
        --experiments-root "${EXPERIMENTS_ROOT}" || FAILED+=("${ARM}")
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo "=== done $(date) ==="
