#!/bin/bash
#SBATCH --job-name=copy-tokenized
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Copy the tokenized training data out of EUHPC_E05_119 before that project's
# work area is locked. The project ended 2026-07-25, and EUHPC_D21_101 became
# unreadable about six months after it ended, which silently killed the L64
# runs. All four score buckets are copied in full, not just the files the
# janus configs reference, since nothing else will be readable afterwards.
#
# A second rsync pass with --checksum compares the contents of every file on
# both sides. The job fails if anything differs.
#
#   sbatch scripts/data/copy_tokenized_to_scratch.sh
set -uo pipefail

SRC="/leonardo_work/EUHPC_E05_119/mfrey/tokenized"
DST="/leonardo_scratch/large/userexternal/mfrey000/tokenized"
DIRS=(score3.0-3.5 score3.5-4.0 score4.0-4.5 score4.5+ .cache)

mkdir -p "${DST}"
echo "START copy: $(date)"
pids=()
for d in "${DIRS[@]}"; do
    rsync -a "${SRC}/${d}" "${DST}/" &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "END copy: $(date)  fail=${fail}"
[ "${fail}" -eq 0 ] || exit 1

echo "START verify: $(date)"
diffs=$(mktemp)
for d in "${DIRS[@]}"; do
    rsync -a --checksum --dry-run --itemize-changes "${SRC}/${d}" "${DST}/" >> "${diffs}" &
done
wait
echo "END verify: $(date)"
if [ -s "${diffs}" ]; then
    echo "MISMATCH:"; cat "${diffs}"; exit 1
fi
for d in "${DIRS[@]}"; do
    echo "${d}: src=$(find "${SRC}/${d}" -type f | wc -l) dst=$(find "${DST}/${d}" -type f | wc -l) files"
done
echo "VERIFIED OK"
