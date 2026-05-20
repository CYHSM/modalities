#!/bin/bash
#SBATCH --job-name=olmes-eval
#SBATCH --account=a0164
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.err

set -e -u
ulimit -c 0

MY_ROOT="/users/markusfrey"
EXPERIMENTS_DIR="/capstor/scratch/cscs/markusfrey/experiments"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
LIST_FILE=${LIST_FILE:-"$(pwd)/checkpoints_to_eval.txt"}

INCLUDE_GENERATIVE=1
LIMIT=128
BATCH_SIZE=1

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

# Use CSCS's native uenv tool to guarantee the mount and view, completely bypassing srun.
uenv run pytorch/v2.9.1 --view=default -- bash -c "
    set -e -u
    source ${OLMES_VENV}/bin/activate
    
    export TIKTOKEN_CACHE_DIR=${MY_ROOT}/tiktoken_cache
    export HF_HOME=${EXPERIMENTS_DIR}/../hf_cache
    export HF_DATASETS_CACHE=\${HF_HOME}
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    
    export TRITON_HOME=/dev/shm/
    # We use \$SLURM_ARRAY_TASK_ID since we are bypassing srun's PROCID
    export TRITON_CACHE_DIR=/dev/shm/.triton_cache_\${SLURM_ARRAY_TASK_ID}
    mkdir -p \${TRITON_CACHE_DIR}

    echo '--- env check ---'
    which python
    python -c 'import torch, torchvision; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"tv\", torchvision.__version__, \"cuda_ok\", torch.cuda.is_available())'
    echo '-----------------'

    export CUDA_VISIBLE_DEVICES=0

    # Force Python to run the script directly to avoid any lingering shebang issues
    python ${OLMES_VENV}/bin/olmes \
        --model '${HF_CKPT}' \
        --model-type hf \
        --model-args '{\"trust_remote_code\": true}' \
        --task ${TASKS} \
        --limit ${LIMIT} \
        --batch-size ${BATCH_SIZE} \
        --output-dir '${OUT_DIR}'
"

echo "=========================================="
echo "END: $(date)"
echo "=========================================="