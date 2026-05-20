#!/bin/bash
#SBATCH --job-name=hf-convert
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.out

set -euo pipefail

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
MODALITIES_DIR="${MY_ROOT}/modalities"
CONTAINER_IMAGE="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
CONVERT_SCRIPT="src/modalities/evaluation/python_scripts/convert_adaptive_gpt.py"
VERIFY_SCRIPT="src/modalities/evaluation/python_scripts/verify_logits.py"
DO_VERIFY=1

: "${CONV_LIST:?CONV_LIST must be exported by the submitter}"

# Pick the line for this array index
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONV_LIST")
if [ -z "$line" ]; then
    echo "No line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $CONV_LIST"
    exit 1
fi

ckpt_dir=$(echo "$line" | cut -f1)
hf_out=$(echo   "$line" | cut -f2)
config=$(echo   "$line" | cut -f3)

echo "=== Task ${SLURM_ARRAY_TASK_ID} ==="
echo "ckpt_dir: $ckpt_dir"
echo "hf_out:   $hf_out"
echo "config:   $config"
echo ""

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

# Skip if already converted (idempotent)
if [ -f "${hf_out}/config.json" ]; then
    echo "⏭  already converted, exiting"
    exit 0
fi

echo "⏳ converting..."
if ! run_py "$CONVERT_SCRIPT" "$ckpt_dir" "$hf_out" --modalities_config "$config"; then
    echo "❌ conversion failed"
    rm -rf "$hf_out"
    exit 1
fi
echo "✅ converted"

if [ "$DO_VERIFY" -eq 1 ]; then
    echo "🔍 verifying..."
    if run_py "$VERIFY_SCRIPT" "$config" "$ckpt_dir" "$hf_out"; then
        echo "✅ verified"
    else
        echo "⚠️  verification failed (non-fatal)"
    fi
fi
