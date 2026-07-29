#!/bin/bash
#SBATCH --job-name=paloma-pipeline
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/pipeline-%j.out
#SBATCH --error=logs/pipeline-%j.err
#
# Two-stage pipeline using the merged paloma_diagnostics.py:
#   1) `run`     model pass -> paloma_<source>.parquet + static.json
#   2) `tag-pos` POS tagging -> paloma_<source>_pos.parquet
# Figures are produced elsewhere via paloma_figures.py.
#
# Usage:
#   sbatch --export=HF_CKPT=/path/to/hf_checkpoint slurm_jobs/pipeline.sbatch
#
# Optional env vars:
#   SOURCES="wikitext_103 gsm8k triviaqa"      # default: all three
#   MAX_DOCS=1000                              # docs per source (stage 1)
#   POS_MAX_CHUNKS=1000                        # chunks per source (stage 2)
#   ALLOW_ONLINE=1                             # let HF reach out
#   SKIP_STAGE_DIAG=1   / SKIP_STAGE_POS=1     # skip individual stages

set -e -u
ulimit -c 0

# ---- Paths --------------------------------------------------------------
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_viewer"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
REPO_DIR=${REPO_DIR:-"${MY_ROOT}/modalities/src/modalities/evaluation"}
SCRIPTS_DIR="${REPO_DIR}/python_scripts"

DIAG_SCRIPT="${SCRIPTS_DIR}/paloma_diagnostics.py"

[ -f "${DIAG_SCRIPT}" ] || { echo "Error: ${DIAG_SCRIPT} not found."; exit 1; }

if [ -z "${HF_CKPT:-}" ]; then
    echo "Error: HF_CKPT not set."
    echo "  sbatch --export=HF_CKPT=/path/to/hf_checkpoint slurm_jobs/pipeline.sbatch"
    exit 1
fi
[ -f "${HF_CKPT}/config.json" ] || { echo "Error: ${HF_CKPT}/config.json not found"; exit 1; }

# ---- Defaults -----------------------------------------------------------
MAX_DOCS=${MAX_DOCS:-500}
POS_MAX_CHUNKS=${POS_MAX_CHUNKS:-500}
ALLOW_ONLINE=${ALLOW_ONLINE:-0}

OUT_DIR="${HF_CKPT}/paloma_diagnostics"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Job:               ${SLURM_JOB_ID}"
echo "Node:              ${SLURM_JOB_NODELIST}"
echo "Checkpoint:        ${HF_CKPT}"
echo "Out dir:           ${OUT_DIR}"
echo "Sources:           ${SOURCES:-(default: wikitext_103 gsm8k triviaqa)}"
echo "MAX_DOCS:          ${MAX_DOCS}"
echo "POS_MAX_CHUNKS:    ${POS_MAX_CHUNKS}"
echo "ALLOW_ONLINE:      ${ALLOW_ONLINE}"
echo "START:             $(date)"
echo "=========================================="

# ---- Run ----------------------------------------------------------------
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$CONTAINER" bash -c "
        set -e -u
        source ${OLMES_VENV}/bin/activate
        export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
        export TIKTOKEN_CACHE_DIR=${MY_ROOT}/tiktoken_cache
        export HF_HOME=${EXPERIMENTS_DIR}/../hf_cache
        export HF_DATASETS_CACHE=\${HF_HOME}

        if [ '${ALLOW_ONLINE}' = '1' ]; then
            echo 'ALLOW_ONLINE=1 — letting HF reach out for missing datasets.'
            unset HF_HUB_OFFLINE 2>/dev/null || true
            unset TRANSFORMERS_OFFLINE 2>/dev/null || true
        else
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
        fi

        export TRITON_HOME=/dev/shm/
        export TRITON_CACHE_DIR=/dev/shm/.triton_cache_pipe_${SLURM_JOB_ID}
        mkdir -p \${TRITON_CACHE_DIR}
        export CUDA_VISIBLE_DEVICES=0

        # ------ Stage 1: diagnostics ------
        if [ -z \"\${SKIP_STAGE_DIAG:-}\" ]; then
            echo ''
            echo '=== Stage 1/2: paloma_diagnostics.py run ==='
            python '${DIAG_SCRIPT}' run \\
                --ckpt '${HF_CKPT}' \\
                --out-dir '${OUT_DIR}' \\
                --max-docs-per-source ${MAX_DOCS} \\
                ${SOURCES:+--sources ${SOURCES}}
        else
            echo 'SKIP_STAGE_DIAG set — skipping diagnostics.'
        fi

        # ------ Stage 2: POS tagging ------
        if [ -z \"\${SKIP_STAGE_POS:-}\" ]; then
            echo ''
            echo '=== Stage 2/2: paloma_diagnostics.py tag-pos ==='
            if ! python -c 'import spacy; spacy.load(\"en_core_web_sm\")' 2>/dev/null; then
                echo 'WARN: spaCy or en_core_web_sm not installed; skipping POS stage.'
                echo '      Install on login node: pip install spacy && python -m spacy download en_core_web_sm'
            else
                python '${DIAG_SCRIPT}' tag-pos \\
                    --in-dir '${OUT_DIR}' \\
                    --ckpt   '${HF_CKPT}' \\
                    --max-chunks-per-source ${POS_MAX_CHUNKS} \\
                    ${SOURCES:+--sources ${SOURCES}}
            fi
        else
            echo 'SKIP_STAGE_POS set — skipping POS tagging.'
        fi
    "

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
echo ""
echo "Download data locally for figure generation with:"
echo "  scp -r leonardo:${OUT_DIR} ./"
echo ""
echo "Then on your machine:"
echo "  python paloma_figures.py --in-dir <local_dir> --out-dir figs/ --all --ckpt <ckpt>"
