#!/bin/bash

# Define paths based on your exact environment
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
WANDB_DIR="${MY_ROOT}/modalities/wandb_storage/wandb"

echo "=========================================="
echo "          W&B Offline Sync Tool           "
echo "=========================================="

# 1. Sync all offline runs
echo "Syncing runs to the cloud..."

# We bind both the host root AND recreate the training code directory 
# so wandb can find the artifacts exactly where it left them.
singularity exec \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
    "$CONTAINER" \
    bash -c "wandb sync ${WANDB_DIR}/offline-run-*"

echo "=========================================="
echo "                 Done!                    "
echo "=========================================="