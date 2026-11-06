#!/bin/bash
#SBATCH --job-name=s3-backup
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Upload the FINAL HF-converted checkpoint (+ its OLMES evals) of every run
# behind the main.tex tables to S3. DCP training shards are NOT uploaded --
# laid out by paper table so the mapping is obvious:
#
#   <DEST>/README.md                      table/column -> folder mapping
#   <DEST>/paper_checkpoints_layout.tsv   same, machine readable
#   <DEST>/<group>/<label>/               one folder per run
#        SOURCE.txt                       original scratch path + step
#        *.yaml, *.yaml.resolved          training config
#        hf_checkpoint/                   HF weights, tokenizer, adapter code,
#                                         olmes_eval*/ (the paper's metrics)
#
# `rclone copy` skips objects already present, so this is safe to re-run and
# is chained until everything is up (the serial partition caps jobs at 4h):
#
#   sbatch scripts/backup/upload_paper_checkpoints.sh fraunhofer:<bucket>/<prefix>
set -uo pipefail

DEST="${1:?usage: upload_paper_checkpoints.sh <remote:bucket/prefix>}"
REPO="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
SCRATCH="/leonardo_scratch/large/userexternal/mfrey000"
LAYOUT="${REPO}/scripts/backup/paper_checkpoints_layout.tsv"
RCLONE="${HOME}/bin/rclone"
OPTS=(--transfers 8 --checkers 16 --s3-chunk-size 64M --s3-upload-concurrency 4
      --retries 5 --low-level-retries 20 --stats 5m --stats-one-line)

cd "${REPO}" || exit 1
"${RCLONE}" copyto "${LAYOUT}" "${DEST}/paper_checkpoints_layout.tsv" "${OPTS[@]}"
"${RCLONE}" copyto "${REPO}/scripts/backup/README_s3.md" "${DEST}/README.md" "${OPTS[@]}"

fail=0
while IFS=$'\t' read -r group label src note; do
    [[ -z "${group}" || "${group}" == \#* ]] && continue
    d="${SCRATCH}/${src}"
    # final = highest seen_steps checkpoint (18360, or the latest one for unfinished runs)
    step=$(ls "${d}" | grep -oE 'seen_steps_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
    ck=$(ls -d "${d}"/eid_*seen_steps_${step}-* 2>/dev/null | head -1)
    if [ -z "${ck}" ] || [ ! -f "${ck}/hf_checkpoint/config.json" ]; then
        echo "MISSING hf_checkpoint ${group}/${label}: ${d}"; fail=1; continue
    fi
    out="${DEST}/${group}/${label}"
    echo "=== ${group}/${label}  (step ${step})"
    printf "source: %s\ncheckpoint: %s\nstep: %s\npaper: %s\n" "${d}" "$(basename "${ck}")" "${step}" "${note}" \
        | "${RCLONE}" rcat "${out}/SOURCE.txt"
    "${RCLONE}" copy "${d}" "${out}" --max-depth 1 --include "*.yaml" --include "*.yaml.resolved" \
        --include "last_checkpoint_info.json" "${OPTS[@]}" || fail=1
    "${RCLONE}" copy "${ck}/hf_checkpoint" "${out}/hf_checkpoint" "${OPTS[@]}" || fail=1
done < "${LAYOUT}"

echo "END $(date)  fail=${fail}"
exit ${fail}
