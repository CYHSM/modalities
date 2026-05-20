#!/bin/bash
#SBATCH --job-name=hf-convert
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

set -u

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
MODALITIES_DIR="${MY_ROOT}/modalities"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
CONTAINER_IMAGE="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

CONVERT_SCRIPT="src/modalities/evaluation/python_scripts/convert_adaptive_gpt.py"
VERIFY_SCRIPT="src/modalities/evaluation/python_scripts/verify_logits.py"  # adjust if elsewhere

DO_VERIFY=1

run_py() {
    local script="$1"; shift
    singularity exec --nv \
        --bind "${MY_ROOT}:${MY_ROOT}" \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
        "$CONTAINER_IMAGE" bash -c "
            export PYTHONPATH=/opt/repos/modalities/src:\$PYTHONPATH
            cd /opt/repos/modalities
            python '$script' $(printf "'%s' " "$@")
        "
}

total=0; converted=0; skipped=0; failed=0; verified=0; verify_failed=0

for run_dir in "${EXPERIMENTS_DIR}"/*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")
    [ "$run_name" = "wandb" ] && continue

    config=$(find "$run_dir" -maxdepth 1 -name '*.yaml' ! -name '*.resolved' | head -1)
    if [ -z "$config" ]; then
        echo "⚠️  no yaml config in $run_name, skipping"
        continue
    fi

    for ckpt_dir in "${run_dir}"eid_*/; do
        [ -d "$ckpt_dir" ] || continue
        total=$((total+1))
        ckpt_name=$(basename "$ckpt_dir")
        hf_out="${ckpt_dir}hf_checkpoint"

        echo ""
        echo "=== [$total] $run_name / $ckpt_name ==="

        if [ -f "${hf_out}/config.json" ]; then
            echo "⏭  already converted, skipping"
            skipped=$((skipped+1))
            continue
        else
            echo "⏳ converting..."
            if run_py "$CONVERT_SCRIPT" "$ckpt_dir" "$hf_out" --modalities_config "$config"; then
                echo "✅ converted"
                converted=$((converted+1))
            else
                echo "❌ conversion failed"
                failed=$((failed+1))
                rm -rf "$hf_out"
                continue
            fi
        fi

        if [ "$DO_VERIFY" -eq 1 ]; then
            echo "🔍 verifying..."
            if run_py "$VERIFY_SCRIPT" "$config" "$ckpt_dir" "$hf_out"; then
                echo "✅ verified"
                verified=$((verified+1))
            else
                echo "⚠️  verification failed"
                verify_failed=$((verify_failed+1))
            fi
        fi
    done
done

echo ""
echo "===================="
echo "Total checkpoints: $total"
echo "Converted:         $converted"
echo "Skipped (done):    $skipped"
echo "Convert failed:    $failed"
if [ "$DO_VERIFY" -eq 1 ]; then
    echo "Verified:          $verified"
    echo "Verify failed:     $verify_failed"
fi

[ "$failed" -eq 0 ]