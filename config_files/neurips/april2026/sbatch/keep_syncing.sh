#!/bin/bash

SOURCE="/leonardo_scratch/fast/EUHPC_D21_101/mfrey/experiments"
DEST="/leonardo_scratch/large/userexternal/mfrey000/"

while true; do
    echo "--- Starting rsync attempt at $(date) ---"
    
    # Run rsync
    rsync -avh --progress --partial "$SOURCE" "$DEST"
    
    # Capture the exit code of the rsync command specifically
    exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Transfer complete!"
        exit 0
    else
        echo "Rsync failed (Exit code: $exit_status). Retrying in 10 seconds..."
        sleep 10
    fi
done