#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=euhpc_e05_119
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-19
#SBATCH --output=logs/tokenize-%A_%a.out
#SBATCH --error=logs/tokenize-%A_%a.out

set -e
mkdir -p logs

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

RAW_DIR="/leonardo/pub/userexternal/bshomali/data/data/nvidia___nemotron-cc-math-v1_JSONL"
OUT_DIR="${SCRATCH_ROOT}/tokenized/nemotron_math_gpt2"
TEMPLATE="/leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/tokenize/tokenizer_config.yaml"

mkdir -p "$OUT_DIR"

i=${SLURM_ARRAY_TASK_ID}
SRC="${RAW_DIR}/Nemotron-CC-Math-v1_train_4plus_part${i}.jsonl"
IDX="${OUT_DIR}/part${i}.idx"
DST="${OUT_DIR}/part${i}.pbin"

if [ ! -f "$SRC" ]; then
    echo "SKIP missing: $SRC"
    exit 0
fi
if [ -f "$DST" ]; then
    echo "SKIP done: $DST"
    exit 0
fi

echo "=== Part $i ==="
echo "SRC: $SRC"
echo "DST: $DST"
date

singularity exec \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
    --bind "/leonardo/pub/userexternal/bshomali:/leonardo/pub/userexternal/bshomali" \
    "$CONTAINER" bash -c "
        export PYTHONPATH=/opt/repos/modalities/src:\$PYTHONPATH
        export HF_HOME=${SCRATCH_ROOT}/hf_cache
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        echo '--- indexing ---'
        python /opt/repos/modalities/src/modalities/__main__.py data create_raw_index \
            --index_path '${IDX}' '${SRC}'

        echo '--- tokenizing ---'
        CFG=\$(mktemp)
        sed -e 's|__SRC__|${SRC}|' \
            -e 's|__DST__|${DST}|' \
            -e 's|__IDX__|${IDX}|' \
            '${TEMPLATE}' > \"\$CFG\"

        python /opt/repos/modalities/src/modalities/__main__.py data pack_encoded_data \"\$CFG\"
        rm -f \"\$CFG\"
    "

echo "DONE: $(date)"