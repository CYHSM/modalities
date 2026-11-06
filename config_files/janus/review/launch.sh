#!/bin/bash
# Launch a list of review configs: training array -> one warmstart link per run -> final eval.
#
#   bash config_files/janus/review/launch.sh config_files/janus/review/attnmatched.txt
#
# The warmstart link (afterany) exits cleanly when training already reached its target,
# so it only costs time for runs that hit the 24h limit (the widened F160M PureWide
# needs ~24.4h). The eval job then converts and evaluates the final checkpoint only.
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
    EV_ID=$(sbatch --parsable --dependency=afterany:"${WS_ID}" \
        scripts/evaluation/queue_eval_after_training.sh "${BASE}")
    echo "${ARRAY_ID}_${i}  ws=${WS_ID}  eval=${EV_ID}  ${BASE}"
done < "${LIST}"
