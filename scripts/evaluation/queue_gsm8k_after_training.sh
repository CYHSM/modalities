#!/bin/bash
#SBATCH --job-name=gsm8k-after
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Run the gsm8k eval on a run's FINAL checkpoint once its training chain is done.
#
# The newer janus configs dropped `gsm8k::olmes` from their task list, so the
# automatic end-of-training eval covers 25 tasks but not Math Acc. This job is
# queued with --dependency=afterany on the last warmstart link, discovers the
# experiment directory the same way run_warmstart.sh does, and fills the gap
# without anyone having to remember to launch it.
#
#   sbatch --dependency=afterany:<last_ws_jobid> \
#          scripts/evaluation/queue_gsm8k_after_training.sh <BASE_CONFIG_NAME>
#
# This job itself uses no GPU: run_convert_and_eval_sbatch.sh submits the real
# work as its own 1-GPU allocation and blocks on `sbatch --wait`.
set -x

BASE_CONFIG_NAME="$1"
REPO_DIR="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest"

cd "${REPO_DIR}" || exit 1

# Newest experiment dir holding this config AND a completed final checkpoint.
EXP_DIR=""
for d in $(ls -1dt "${EXPERIMENT_ROOT}"/*/); do
    # A warmstart writes its own dir holding only the *_WARMSTART.yaml copy, so
    # the final checkpoint of a chained run lives under that name, not the base.
    if ls -d "${d}"/eid_*seen_steps_18360-* >/dev/null 2>&1 && \
       { [ -f "${d}/${BASE_CONFIG_NAME}" ] || [ -f "${d}/${BASE_CONFIG_NAME%.yaml}_WARMSTART.yaml" ]; }; then
        EXP_DIR="${d%/}"
        break
    fi
done
if [ -z "${EXP_DIR}" ]; then
    echo "no finished (18360-step) experiment found for ${BASE_CONFIG_NAME}; nothing to do."
    exit 0
fi

CK=$(ls -d "${EXP_DIR}"/eid_*seen_steps_18360-* | head -1)
CFG=$(ls "${EXP_DIR}"/*.yaml.resolved | head -1)
if [ -f "${CK}/hf_checkpoint/olmes_eval_18360_gsm8k/metrics-all.jsonl" ]; then
    echo "gsm8k already evaluated for ${EXP_DIR}; nothing to do."
    exit 0
fi

# Final checkpoint only, and generous walltime: gsm8k is generative and a deep
# model does not finish 1024 examples in the default 6h.
export JANUS_EVAL_ALL_STEPS=1
export JANUS_EVAL_TIME=20:00:00

bash scripts/evaluation/run_convert_and_eval_sbatch.sh \
    "${CK}" "${CFG}" "${CK}/hf_checkpoint" "gsm8k::olmes" "18360_gsm8k" 1024 1
echo "END: $(date)"
