#!/usr/bin/env bash
# Syncs offline wandb runs from the 5B-token cluster wave to the wandb backend.
#
# Compute nodes have no internet, so every arm trains with WANDB_MODE=offline (see run_wave_5b.sh)
# and writes its run under <EXPERIMENTS_ROOT>/<ARM_NAME>/wandb/offline-run-*. This syncs from
# wherever it runs, so run it on the LOGIN node (verified to have internet), not from inside an
# sbatch job.
#
#   ./sync_wandb.sh                                 # one-shot: sync everything found now
#   ./sync_wandb.sh --watch                          # loop every 600s until Ctrl-C ("live" sync)
#   ./sync_wandb.sh --watch 120                       # loop every 120s instead
#   ./sync_wandb.sh --root /other/experiments/root    # override the default experiments root
#
# Uses the dedicated wandb_env venv (already has valid api-key creds in ~/.netrc -- confirmed
# working: `curl -sI https://api.wandb.ai` succeeds from the login node). Safe to re-run: `wandb
# sync` is idempotent per run directory and skips ones already fully synced.
set -euo pipefail

EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster"
WANDB_BIN="/leonardo_work/EUHPC_D21_101/mfrey/wandb_env/bin/wandb"
WATCH=false
INTERVAL=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)
            WATCH=true
            if [[ ${2:-} =~ ^[0-9]+$ ]]; then
                INTERVAL=$2
                shift
            fi
            shift
            ;;
        --root)
            EXPERIMENTS_ROOT=$2
            shift 2
            ;;
        *)
            echo "usage: $0 [--watch [INTERVAL_SECONDS]] [--root PATH]" >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${WANDB_BIN}" ]]; then
    echo "wandb binary not found at ${WANDB_BIN} -- is wandb_env still there?" >&2
    exit 1
fi

sync_once () {
    local found=0
    while IFS= read -r -d '' run_dir; do
        found=$((found + 1))
        echo "[$(date +%H:%M:%S)] syncing ${run_dir#"${EXPERIMENTS_ROOT}/"}"
        "${WANDB_BIN}" sync "${run_dir}" || echo "  -> sync failed for ${run_dir}, will retry next pass" >&2
    done < <(find "${EXPERIMENTS_ROOT}" -maxdepth 4 -type d -name "offline-run-*" -print0 2> /dev/null)
    echo "[$(date +%H:%M:%S)] pass complete, ${found} offline run(s) found under ${EXPERIMENTS_ROOT}"
}

if [[ "${WATCH}" == true ]]; then
    echo "watching ${EXPERIMENTS_ROOT} every ${INTERVAL}s, Ctrl-C to stop"
    while true; do
        sync_once
        sleep "${INTERVAL}"
    done
else
    sync_once
fi
