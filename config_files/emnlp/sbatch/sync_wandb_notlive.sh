#!/bin/bash
set -u
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions"
WANDB_DIR="${SCRATCH_ROOT}/wandb/wandb"
TARGET_PROJECT="dualfull"

# --- Credentials ---
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "❌ No WANDB_API_KEY and no ~/.netrc entry."
  exit 1
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Give wandb more patience on a proxied login node
export WANDB__SERVICE_WAIT=300
export WANDB_HTTP_TIMEOUT=60

echo "⏳ Starting sync of offline WandB runs..."

# Activate your up-to-date host environment
#source /leonardo_work/EUHPC_D21_101/mfrey/wandb_env/bin/activate
source $HOME/wandb_env_home/bin/activate

# Sync each run individually so one failure doesn't abort everything
FAILED=0
for d in "$WANDB_DIR"/offline-run-*; do
  [ -d "$d" ] || continue
  echo "--- syncing $d"
  if ! wandb sync "$d" --project "$TARGET_PROJECT"; then
    echo "⚠️  FAILED: $d"
    FAILED=1
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo "✅ Sync complete."
else
  echo "⚠️ Sync encountered an error (one or more runs failed)."
  exit 1
fi