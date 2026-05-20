#!/bin/bash
#SBATCH --job-name=olmes-eval
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.err
set -e -u
ulimit -c 0
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
LIST_FILE=${LIST_FILE:-"$(pwd)/checkpoints_to_eval.txt"}
INCLUDE_GENERATIVE=1
LIMIT=1024
BATCH_SIZE=1

# --- Two Paloma subsets for capacity-vs-compute contrast ---
#   paloma_wikitext_103 -> knowledge/capacity-leaning (entity- and fact-dense Wikipedia)
#   paloma_c4_en        -> pattern/compute-leaning   (generic web text)
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
lambada:bpb \
paloma_c4_en::paloma \
paloma_wikitext_103::paloma"

if [ "$INCLUDE_GENERATIVE" = "1" ]; then
    TASKS="${TASKS} gsm8k::olmes"
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array"
    exit 1
fi

HF_CKPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
if [ -z "$HF_CKPT" ]; then
    echo "Error: No checkpoint found for array index $SLURM_ARRAY_TASK_ID"
    exit 0
fi
OUT_DIR="${HF_CKPT}/olmes_eval"

echo "=========================================="
echo "Job:         ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:        ${SLURM_JOB_NODELIST}"
echo "Checkpoint:  ${HF_CKPT}"
echo "START:       $(date)"
echo "=========================================="

# Use Singularity to execute the python evaluation inside the container environment
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$CONTAINER" bash -c "
    source ${OLMES_VENV}/bin/activate
    export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
    export TIKTOKEN_CACHE_DIR=${MY_ROOT}/tiktoken_cache
    export HF_HOME=${EXPERIMENTS_DIR}/../hf_cache
    export HF_DATASETS_CACHE=\${HF_HOME}
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export TRITON_HOME=/dev/shm/
    export TRITON_CACHE_DIR=/dev/shm/.triton_cache_\${SLURM_ARRAY_TASK_ID}
    mkdir -p \${TRITON_CACHE_DIR}
    export CUDA_VISIBLE_DEVICES=0
    python ${OLMES_VENV}/bin/olmes \
        --model '${HF_CKPT}' \
        --model-type hf \
        --model-args '{\"trust_remote_code\": true, \"max_length\": 4096}' \
        --task ${TASKS} \
        --limit ${LIMIT} \
        --batch-size ${BATCH_SIZE} \
        --output-dir '${OUT_DIR}'
"

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
