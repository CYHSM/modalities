#!/bin/bash
set -u

SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
# SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
WANDB_DIR="${SCRATCH_ROOT}/wandb/wandb"
TARGET_PROJECT="dualpathflops"

# --- Credentials ---
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "❌ No WANDB_API_KEY and no ~/.netrc entry."
    exit 1
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

echo "⏳ Starting live sync of offline WandB runs..."

# Activate your new, up-to-date host environment
source /leonardo_work/EUHPC_D21_101/mfrey/wandb_env/bin/activate
# source $HOME/wandb_env_home/bin/activate

# Run the sync directly on the host!
if wandb beta sync "$WANDB_DIR" --live -p "$TARGET_PROJECT" -n 200; then
    echo "✅ Sync complete."
else
    echo "⚠️ Sync encountered an error."
    exit 1
fi