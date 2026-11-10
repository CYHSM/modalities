#!/bin/bash
# Launch a list of review configs: training array -> one warmstart link per run -> final eval.
#
#   bash config_files/janus/review/launch.sh config_files/janus/review/attnmatched.txt
#
# The warmstart link (afterany) exits cleanly when training already reached its target,
# but it still needs 16 nodes to get through the queue first, which took days in Nov 2026.
# So each run gets two eval jobs: one right after training (the usual case) and one after
# the warmstart link (for runs that hit the 24h limit, e.g. the widened F160M PureWide,
# ~24.4h). Both evaluate the final checkpoint only and exit cleanly if the run is not
# finished yet or already evaluated.
set -euo pipefail

LIST="$1"
REPO_DIR="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
cd "${REPO_DIR}"
N=$(grep -c . "${LIST}")

ARRAY_ID=$(CONFIG_LIST="${REPO_DIR}/${LIST}" sbatch --parsable --array=1-"${N}" \
    config_files/janus/run_array_16nodes.sh)
echo "training array: ${ARRAY_ID} (${N} runs)"

i=0
while read -r CFG; do
    [ -z "${CFG}" ] && continue
    i=$((i + 1))
    BASE=$(basename "${CFG}")
    WS="${CFG%.yaml}_WARMSTART.yaml"
    WS_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID}_${i}" \
        config_files/janus/run_warmstart.sh "${REPO_DIR}/${WS}" "${BASE}")
    EV1_ID=$(sbatch --parsable --dependency=afterany:"${ARRAY_ID}_${i}" \
        scripts/evaluation/queue_eval_after_training.sh "${BASE}")
    EV2_ID=$(sbatch --parsable --dependency=afterany:"${WS_ID}" \
        scripts/evaluation/queue_eval_after_training.sh "${BASE}")
    echo "${ARRAY_ID}_${i}  ws=${WS_ID}  eval=${EV1_ID},${EV2_ID}  ${BASE}"
done < "${LIST}"
