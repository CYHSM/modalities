#!/bin/bash
HF_MODEL_DIR="$1"
TASKS="$2"
STEP="$3"
LIMIT="${4:-128}"
BATCH_SIZE="${5:-1}"
OUT_DIR="${HF_MODEL_DIR}/olmes_eval_${STEP}"
mkdir -p "${HF_MODEL_DIR}"

sbatch --wait <<EOF
#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=olmes-eval-${STEP}
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=${OUT_DIR}_slurm.log
#SBATCH --error=${OUT_DIR}_slurm.err
set -e
ulimit -c 0

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_tests"
OLMES_VENV="\${MY_ROOT}/venvs/olmes_uv"

echo "=========================================="
echo "Job:        \${SLURM_JOB_ID}"
echo "Node:       \${SLURM_JOB_NODELIST}"
echo "Checkpoint: ${HF_MODEL_DIR}"
echo "START:      \$(date)"
echo "=========================================="

# --- Offline HF setup (compute nodes have no internet; datasets must be precached here) ---
export TIKTOKEN_CACHE_DIR="\${MY_ROOT}/tiktoken_cache"
export HF_HOME="\${EXPERIMENTS_DIR}/../hf_cache"
export HF_DATASETS_CACHE="\${HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export WANDB_MODE=offline
export NLTK_DATA="\${MY_ROOT}/nltk_data"
export TRITON_HOME=/dev/shm/
export TRITON_CACHE_DIR=/dev/shm/.triton_cache_\${SLURM_JOB_ID}
mkdir -p "\${TRITON_CACHE_DIR}"

# Clear inherited python environment from the modalities training job
unset PYTHONPATH
unset VIRTUAL_ENV

source \${OLMES_VENV}/bin/activate
# Add OLMES source dir to PYTHONPATH just in case it is not installed in the venv
export PYTHONPATH="\${MY_ROOT}/olmes:\${PYTHONPATH:-}"
set -u

python \${OLMES_VENV}/bin/olmes \\
    --model '${HF_MODEL_DIR}' \\
    --model-type hf \\
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \\
    --task ${TASKS} \\
    --limit ${LIMIT} \\
    --batch-size ${BATCH_SIZE} \\
    --output-dir '${OUT_DIR}'

echo "=========================================="
echo "END: \$(date)"
echo "=========================================="
EOF