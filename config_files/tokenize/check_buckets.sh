#!/bin/bash

# Define the buckets based on your dataset configuration.
# The /data/ prefixes have been replaced with the specified Leonardo path.
BUCKETS=(
    "/leonardo_work/EUHPC_E05_119/mfrey/tokenized/score3.0-3.5"
    "/leonardo_work/EUHPC_E05_119/mfrey/tokenized/score3.5-4.0"
    "/leonardo_work/EUHPC_E05_119/mfrey/tokenized/score4.0-4.5"
    "/leonardo_work/EUHPC_E05_119/mfrey/tokenized/score4.5+"
    "/leonardo_scratch/large/userexternal/mfrey000/tokenized/nemotron_math_gpt2"
)

for BUCKET in "${BUCKETS[@]}"; do
    echo "========================================================================"
    echo "📦 BUCKET: $BUCKET"
    echo "========================================================================"

    # Check if the directory exists to avoid errors
    if [ -d "$BUCKET" ]; then
        
        # 1. Get the total size of the directory
        echo -e "📊 Total Directory Size:"
        du -sh "$BUCKET"
        
        # 2. Count the exact number of .pbin files (using find avoids 'Argument list too long' errors)
        FILE_COUNT=$(find "$BUCKET" -maxdepth 1 -name "*.pbin" | wc -l)
        echo -e "\n📁 Contains: $FILE_COUNT .pbin files."
        
        # 3. Show a detailed preview of the contents
        echo -e "\n🔍 Content Preview (First 5 files):"
        # ls -lh lists files with human-readable sizes, grep filters for .pbin, head limits to 5
        ls -lh "$BUCKET" | grep "\.pbin" | head -n 5
        
        if [ "$FILE_COUNT" -gt 5 ]; then
            echo "   ... (and $((FILE_COUNT - 5)) more files)"
        fi

    else
        echo "⚠️  Directory not found or not accessible on this node!"
    fi
    echo -e "\n"
done