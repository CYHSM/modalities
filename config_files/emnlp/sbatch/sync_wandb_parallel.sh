#!/bin/bash
# sync_wandb_parallel.sh — run on a Leonardo LOGIN node (only place with internet)
set -u

SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
WANDB_DIR="${SCRATCH_ROOT}/wandb/wandb"
TARGET_PROJECT="dualpathflops-final"
PARALLEL="${PARALLEL:-8}"

# --- Credentials ---
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "ERROR: No WANDB_API_KEY and no ~/.netrc entry."
  exit 1
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB__SERVICE_WAIT=300
export WANDB_HTTP_TIMEOUT=60

source "$HOME/wandb_env_home/bin/activate"

LOGDIR="${SCRATCH_ROOT}/wandb_sync_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"
echo "Parallel: $PARALLEL"

sync_one() {
  d="$1"
  name=$(basename "$d")
  log="$LOGDIR/$name.log"
  if wandb sync "$d" --project "$TARGET_PROJECT" >"$log" 2>&1; then
    echo "OK   $name"
  else
    rc=$?
    echo "FAIL $name (rc=$rc)"
    return 1
  fi
}
export -f sync_one
export TARGET_PROJECT LOGDIR

# If a "todo" file is passed, sync only those dirs; otherwise sync all
if [ "${1:-}" = "--todo" ] && [ -f "${2:-}" ]; then
  INPUT_SRC=$(cat "$2")
  echo "Mode: retry from $2"
else
  INPUT_SRC=$(find "$WANDB_DIR" -maxdepth 1 -type d -name 'offline-run-*')
  echo "Mode: all runs"
fi

echo "$INPUT_SRC" \
  | xargs -d '\n' -n1 -P "$PARALLEL" bash -c 'sync_one "$0"'

# Build a retry list of anything that didn't fully upload
RETRY_FILE="$LOGDIR/retry.txt"
> "$RETRY_FILE"
while IFS= read -r d; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  log="$LOGDIR/$name.log"
  if [ ! -f "$log" ] || ! grep -q "uploaded\|Syncing.*done\|.wandb.*finished" "$log"; then
    echo "$d" >> "$RETRY_FILE"
  fi
done <<< "$INPUT_SRC"

TOTAL=$(echo "$INPUT_SRC" | wc -l)
FAILED=$(wc -l < "$RETRY_FILE")
echo
echo "Total: $TOTAL  Incomplete: $FAILED"
[ "$FAILED" -gt 0 ] && echo "Retry with: $0 --todo $RETRY_FILE"