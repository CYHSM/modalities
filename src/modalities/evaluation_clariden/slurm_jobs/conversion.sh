#!/bin/bash
#SBATCH --job-name=hf-convert
#SBATCH --account=a0164
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --uenv=pytorch/v2.9.1:/user-environment
#SBATCH --view=default
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

set -u

MY_ROOT="/users/markusfrey"
MODALITIES_DIR="${MY_ROOT}/Github/modalities"
EXPERIMENTS_DIR="/capstor/scratch/cscs/markusfrey/experiments"

CONVERT_SCRIPT="src/modalities/evaluation/python_scripts/convert_adaptive_gpt.py"
VERIFY_SCRIPT="src/modalities/evaluation/python_scripts/verify_logits.py"  # adjust if elsewhere

DO_VERIFY=1

run_py() {
    local script="$1"; shift
    source ${MODALITIES_DIR}/.venv/bin/activate
    export PYTHONPATH=${MODALITIES_DIR}/src:$PYTHONPATH
    cd ${MODALITIES_DIR}
    python "$script" "$@"
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