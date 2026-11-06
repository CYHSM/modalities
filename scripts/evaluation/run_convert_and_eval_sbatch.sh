#!/bin/bash
# Combined HF-conversion + OLMES-eval launcher, run as its own dedicated sbatch
# job (1 GPU) so it never shares compute with, or blocks, the training job.
#
# Historically the DCP->HF conversion ran synchronously on rank 0 inside the
# training process (ModelConverter.convert(), subprocess.run(...)). For
# checkpoints that take >10 min to convert, that stalled rank 0 past NCCL's
# collective timeout while the other ranks moved on to the next training
# step, killing the whole job (SIGABRT). This script decouples conversion
# from training the same way OLMES eval already was decoupled: submitted
# here via a non-blocking Popen in evaluator.py, running on its own
# allocation, with only this script's *own* shell (not the training
# process) blocking on `sbatch --wait`.
CHECKPOINT_PATH="$1"
MODALITIES_CONFIG="$2"
HF_MODEL_DIR="$3"
TASKS="$4"
STEP="$5"
LIMIT="${6:-1024}"
BATCH_SIZE="${7:-1}"
OUT_DIR="${HF_MODEL_DIR}/olmes_eval_${STEP}"
# Slurm creates the --output/--error files (and their parent dirs) *before* the
# job script runs. Pointing them inside ${HF_MODEL_DIR} therefore created
# hf_checkpoint/ itself, so the `already converted?` guard below always fired
# and conversion was silently skipped -- OLMES then ran against an empty dir and
# died with "Unrecognized model ... should have a `model_type` key". Keep the
# slurm logs one level up, in the DCP checkpoint dir, which always pre-exists.
LOG_PREFIX="${CHECKPOINT_PATH}/olmes_eval_${STEP}"

# Only the final checkpoint is evaluated. Intermediate evals cost a GPU-hours-
# heavy job each and are not currently used, so skip them here rather than in
# the config: this script is re-read on every invocation, so the guard also
# applies to training jobs that are already running with a smaller
# eval_interval baked into the config they started from.
# Set JANUS_EVAL_ALL_STEPS=1 to evaluate every checkpoint again.
# Set JANUS_EVAL_TIME to raise the 6h walltime: generative tasks (gsm8k runs
# generate_until at ~15 s/it on a 32-layer model) do not finish 1024 examples
# in 6h at depth, and a timeout leaves no metrics-all.jsonl at all.
TARGET_STEP=$(echo "${CHECKPOINT_PATH}" | grep -oE 'target_steps_[0-9]+' | grep -oE '[0-9]+')
if [ "${JANUS_EVAL_ALL_STEPS:-0}" != "1" ] && [ -n "${TARGET_STEP}" ] && [ "${STEP}" != "${TARGET_STEP}" ]; then
    echo "step ${STEP} is not the final checkpoint (${TARGET_STEP}); skipping conversion+eval."
    exit 0
fi

sbatch --wait <<EOF
#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=conv-eval-${STEP}
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=${JANUS_EVAL_TIME:-06:00:00}
#SBATCH --mem=64G
#SBATCH --output=${LOG_PREFIX}_slurm.log
#SBATCH --error=${LOG_PREFIX}_slurm.err
set -e
ulimit -c 0

MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="\${MY_ROOT}/modalities"

echo "=========================================="
echo "Job:        \${SLURM_JOB_ID}"
echo "Node:       \${SLURM_JOB_NODELIST}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "START:      \$(date)"
echo "=========================================="

# --- Stage 1: convert DCP checkpoint to HF format (skip if already done) ---
# Test for the actual converted artifact, not just the directory -- a bare dir
# can be left behind by a crashed conversion (or created by slurm) and would
# otherwise mask a missing model.
if [ -f "${HF_MODEL_DIR}/config.json" ]; then
    echo "hf_checkpoint already converted at ${HF_MODEL_DIR}, skipping conversion."
else
    source "\${REPO_DIR}/.venv/bin/activate"
    export PYTHONPATH="\${REPO_DIR}/src:\${PYTHONPATH:-}"
    python -m modalities.conversion "${MODALITIES_CONFIG}" "${HF_MODEL_DIR}" \\
        --checkpoint_path "${CHECKPOINT_PATH}" \\
        --tokenizer_dir /leonardo_work/AIFAC_S07_154/mfrey/tokenizer/local_gpt2_tokenizer/ \\
        > "${CHECKPOINT_PATH}/conversion.log" 2>&1
    deactivate
    unset PYTHONPATH
    unset VIRTUAL_ENV
    echo "Conversion done: \$(date)"
fi

# --- Stage 2: OLMES downstream eval (skip if already done) ---
if [ -f "${OUT_DIR}/metrics-all.jsonl" ]; then
    echo "metrics-all.jsonl already exists at ${OUT_DIR}, skipping eval."
else
    mkdir -p "${OUT_DIR}"

    EVAL_MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
    EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_tests"
    OLMES_VENV="\${EVAL_MY_ROOT}/venvs/olmes_uv"

    # --- Offline HF setup (compute nodes have no internet; datasets must be precached here) ---
    export TIKTOKEN_CACHE_DIR="\${EVAL_MY_ROOT}/tiktoken_cache"
    export HF_HOME="\${EXPERIMENTS_DIR}/../hf_cache"
    export HF_DATASETS_CACHE="\${HF_HOME}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export LITELLM_LOCAL_MODEL_COST_MAP=True
    export WANDB_MODE=offline
    export NLTK_DATA="\${EVAL_MY_ROOT}/nltk_data"
    export TRITON_HOME=/dev/shm/
    export TRITON_CACHE_DIR=/dev/shm/.triton_cache_\${SLURM_JOB_ID}
    mkdir -p "\${TRITON_CACHE_DIR}"

    source \${OLMES_VENV}/bin/activate
    # Belt-and-braces: the olmes venv's editable modalities install used to point
    # at the stale repo copy under EUHPC_D21_101, which rejected newer config
    # fields ('gate_sum_cap: Extra inputs are not permitted') and silently failed
    # every eval while conversion still looked fine. The .pth now points at the
    # primary repo, but pin it here too so a venv rebuild cannot reintroduce it.
    export PYTHONPATH="\${REPO_DIR}/src:\${EVAL_MY_ROOT}/olmes:\${PYTHONPATH:-}"
    set -u

    LIMIT_ARG=()
    if [ "${LIMIT}" != "None" ] && [ -n "${LIMIT}" ]; then
        LIMIT_ARG=(--limit "${LIMIT}")
    fi

    python \${OLMES_VENV}/bin/olmes \\
        --model '${HF_MODEL_DIR}' \\
        --model-type hf \\
        --model-args '{"trust_remote_code": true, "max_length": 4096}' \\
        --task ${TASKS} \\
        "\${LIMIT_ARG[@]}" \\
        --batch-size ${BATCH_SIZE} \\
        --output-dir '${OUT_DIR}'
fi

echo "=========================================="
echo "END: \$(date)"
echo "=========================================="
EOF
