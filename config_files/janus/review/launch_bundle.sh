#!/bin/bash
# Launch a bundle list (see run_bundle_bprod.sh) as ONE boost_qos_bprod job, plus per run:
# an eval right after the bundle, and a warmstart link (normal QOS, for a run that hits the
# 24h limit) with a second eval after it. Both evals exit cleanly if the run is not finished
# or already evaluated; the warmstart link exits cleanly if the run is finished.
#
#   bash config_files/janus/review/launch_bundle.sh config_files/janus/review/bundle_bprod.txt
set -euo pipefail
LIST="$1"
REPO_DIR="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
cd "${REPO_DIR}"
N=$(grep -c . "${LIST}")
NODES=$((16 * N))
[ "${NODES}" -lt 65 ] && NODES=65   # boost_qos_bprod minimum; extra nodes stay idle
TIME="${TIME:-24:00:00}"           # shorter bundles fit into scheduling gaps sooner
BUNDLE_ID=$(CONFIG_LIST="${REPO_DIR}/${LIST}" sbatch --parsable --nodes="${NODES}" --time="${TIME}" \
    config_files/janus/run_bundle_bprod.sh)
echo "bundle: ${BUNDLE_ID} (${N} runs, ${NODES} nodes, ${TIME})"
while read -r CFG _; do
    [ -z "${CFG}" ] && continue
    BASE=$(basename "${CFG}")
    WS="${CFG%.yaml}_WARMSTART.yaml"
    EV1=$(sbatch --parsable --dependency=afterany:"${BUNDLE_ID}" \
        scripts/evaluation/queue_eval_after_training.sh "${BASE}")
    WS_ID=$(sbatch --parsable --dependency=afterany:"${BUNDLE_ID}" \
        config_files/janus/run_warmstart.sh "${REPO_DIR}/${WS}" "${BASE}")
    EV2=$(sbatch --parsable --dependency=afterany:"${WS_ID}" \
        scripts/evaluation/queue_eval_after_training.sh "${BASE}")
    echo "  ${BASE}: eval=${EV1} ws=${WS_ID} eval2=${EV2}"
done < "${LIST}"
