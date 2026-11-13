#!/bin/bash
#SBATCH --job-name=eval-after
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
# Convert and evaluate the FINAL checkpoint of a run once its training chain is done.
#
# The review-round configs disable the inline downstream eval: training would otherwise
# keep all 16 nodes idle while it waits on the OLMES job. Queue this behind the last
# training link instead:
#
#   sbatch --dependency=afterany:<last_train_jobid> \
#          scripts/evaluation/queue_eval_after_training.sh <BASE_CONFIG_NAME>
#
# It finds the run the same way run_warmstart.sh does and runs the full paper suite
# (25 tasks + gsm8k) at limit=1024 on the final checkpoint, as one 1-GPU job
# (run_convert_and_eval_sbatch.sh). This job itself uses no GPU; it waits on that one.
# It exits cleanly if the run has not reached its final step or was already evaluated.
set -x

BASE_CONFIG_NAME="$1"
REPO_DIR="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest}"
TASKS="minerva_math_algebra:bpb::olmes minerva_math_counting_and_probability:bpb::olmes \
minerva_math_geometry:bpb::olmes minerva_math_intermediate_algebra:bpb::olmes \
minerva_math_number_theory:bpb::olmes minerva_math_prealgebra:bpb::olmes \
minerva_math_precalculus:bpb::olmes arc_challenge:rc::olmes:full arc_easy:rc::olmes:full \
hellaswag:rc::olmes:full winogrande:rc::olmes:full socialiqa:rc::olmes:full piqa:rc::olmes:full \
qasper_yesno:rc::olmes lambada arc_challenge:rc:bpb::olmes:full arc_easy:rc:bpb::olmes:full \
hellaswag:rc:bpb::olmes:full winogrande:rc:bpb::olmes:full socialiqa:rc:bpb::olmes:full \
piqa:rc:bpb::olmes:full qasper_yesno:rc:bpb::olmes lambada:bpb paloma_c4_en::paloma \
paloma_wikitext_103::paloma gsm8k::olmes"

cd "${REPO_DIR}" || exit 1

# Newest experiment dir holding this config (or its _WARMSTART twin, which is where a
# chained run's final checkpoint lives) AND a checkpoint at its target step.
EXP_DIR=""; CK=""
for d in $(ls -1dt "${EXPERIMENT_ROOT}"/*/); do
    { [ -f "${d}/${BASE_CONFIG_NAME}" ] || [ -f "${d}/${BASE_CONFIG_NAME%.yaml}_WARMSTART.yaml" ]; } || continue
    for c in "${d}"/eid_*; do
        seen=$(echo "$c" | grep -oE 'seen_steps_[0-9]+' | grep -oE '[0-9]+')
        target=$(echo "$c" | grep -oE 'target_steps_[0-9]+' | grep -oE '[0-9]+')
        if [ -n "$seen" ] && [ "$seen" = "$target" ]; then EXP_DIR="${d%/}"; CK="$c"; STEP="$seen"; break 2; fi
    done
done
if [ -z "${CK}" ]; then
    echo "no finished experiment found for ${BASE_CONFIG_NAME}; nothing to do."
    exit 0
fi
CFG=$(ls "${EXP_DIR}"/*.yaml.resolved | head -1)
if [ -f "${CK}/hf_checkpoint/olmes_eval_${STEP}/metrics-all.jsonl" ]; then
    echo "already evaluated: ${CK}"; exit 0
fi

# Two eval jobs can target the same checkpoint (duplicate runs, eval-after-train plus
# eval-after-warmstart). Take a lock so only one converts and evaluates it; a lock whose
# job is no longer in the queue is stale and taken over.
LOCK="${CK}/.eval_lock_${STEP}"
if [ -f "${LOCK}" ] && squeue -h -j "$(cat "${LOCK}")" 2>/dev/null | grep -q .; then
    echo "being evaluated by job $(cat "${LOCK}"): ${CK}"; exit 0
fi
echo "${SLURM_JOB_ID:-$$}" > "${LOCK}"
# Final step only; 26 tasks incl. generative gsm8k take ~7h for an L16 model.
export JANUS_EVAL_ALL_STEPS=1
export JANUS_EVAL_TIME="${JANUS_EVAL_TIME:-14:00:00}"
bash scripts/evaluation/run_convert_and_eval_sbatch.sh \
    "${CK}" "${CFG}" "${CK}/hf_checkpoint" "${TASKS}" "${STEP}" 1024 1
echo "END: $(date)"
