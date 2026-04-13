#!/bin/bash
#SBATCH --job-name=olmes-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_e05_119
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.out

set -u
mkdir -p logs

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
LIST_FILE="/leonardo_work/EUHPC_D21_101/mfrey/modalities/src/modalities/evaluation/from_hf/checkpoints_to_eval.txt"

# --- Task selection ---
# Set INCLUDE_GENERATIVE=1 to also run gsm8k + minerva_math (generative, slower)
INCLUDE_GENERATIVE=1
LIMIT=128
BATCH_SIZE=1

# math_bpb
TASKS="minerva_math_algebra:bpb::olmes \
minerva_math_counting_and_probability:bpb::olmes \
minerva_math_geometry:bpb::olmes \
minerva_math_intermediate_algebra:bpb::olmes \
minerva_math_number_theory:bpb::olmes \
minerva_math_prealgebra:bpb::olmes \
minerva_math_precalculus:bpb::olmes \
arc_challenge:rc::olmes:full \
arc_easy:rc::olmes:full \
hellaswag:rc::olmes:full \
winogrande:rc::olmes:full \
socialiqa:rc::olmes:full \
piqa:rc::olmes:full \
qasper_yesno:rc::olmes \
lambada \
arc_challenge:rc:bpb::olmes:full \
arc_easy:rc:bpb::olmes:full \
hellaswag:rc:bpb::olmes:full \
winogrande:rc:bpb::olmes:full \
socialiqa:rc:bpb::olmes:full \
piqa:rc:bpb::olmes:full \
qasper_yesno:rc:bpb::olmes \
lambada:bpb"

if [ "$INCLUDE_GENERATIVE" = "1" ]; then
    # TASKS="${TASKS} gsm8k::olmes minerva_math::olmes"
    TASKS="${TASKS} gsm8k::olmes"
fi

# --- Pick checkpoint from list ---
HF_CKPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
[ -z "$HF_CKPT" ] && { echo "no checkpoint for task $SLURM_ARRAY_TASK_ID"; exit 0; }
OUT_DIR="${HF_CKPT}/olmes_eval"

echo "Evaluating: $HF_CKPT"
echo "Generative tasks: $INCLUDE_GENERATIVE"

singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    "$CONTAINER" bash -c "
        source ${OLMES_VENV}/bin/activate
        export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
        export TIKTOKEN_CACHE_DIR=${MY_ROOT}/tiktoken_cache
        export HF_DATASETS_CACHE=${SCRATCH_ROOT}/hf_cache
        export HF_HOME=${SCRATCH_ROOT}/hf_cache
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        olmes \
            --model '${HF_CKPT}' \
            --model-type hf \
            --model-args '{\"trust_remote_code\": true}' \
            --task ${TASKS} \
            --limit ${LIMIT} \
            --batch-size ${BATCH_SIZE} \
            --output-dir '${OUT_DIR}'
    "