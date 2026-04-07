#!/bin/bash
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
WANDB_DIR="${MY_ROOT}/modalities/wandb_storage/wandb"

export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"

for run in ${WANDB_DIR}/offline-run-*; do
    if [ -d "$run" ]; then
        echo "⏳ Syncing: $run"
        singularity exec \
            --bind "${MY_ROOT}:${MY_ROOT}" \
            --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
            "$CONTAINER" \
            wandb sync "$run"
        
        if [ $? -ne 0 ]; then
            echo "⚠️ Failed: $run — skipping"
        else
            echo "✅ Done: $run"
        fi
        
        # Give the system a moment to reclaim memory
        sleep 5
    fi
done