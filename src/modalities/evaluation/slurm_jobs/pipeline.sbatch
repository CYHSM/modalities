#!/bin/bash
#SBATCH --job-name=paloma-pipeline
#SBATCH --account=euhpc_e05_119
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
# Unified pipeline: diagnostics -> POS tag -> viewer.
# Replaces the old separate diagnostics.sbatch and pos_tag.sbatch.
#
# Usage:
#   sbatch --export=HF_CKPT=/path/to/hf_checkpoint slurm_jobs/pipeline.sbatch
#
# Optional env vars:
#   SOURCES="wikitext_103 gsm8k triviaqa"      # which sources (default: all 3)
#   MAX_DOCS=200                               # docs per source for diagnostics
#   POS_MAX_CHUNKS=200                         # chunks per source for POS
#   N_EXPLORER_CHUNKS=1                        # chunks shown in viewer explorer
#   ALLOW_ONLINE=1                             # let HF reach out (default: offline,
#                                              # matches your pre-cached setup)
#   SKIP_STAGE_DIAG=1   / SKIP_STAGE_POS=1 / SKIP_STAGE_VIEW=1
#                                              # skip individual stages
#
# Output:
#   ${HF_CKPT}/paloma_diagnostics/paloma_<source>.parquet
#   ${HF_CKPT}/paloma_diagnostics/paloma_<source>_pos.parquet
#   ${HF_CKPT}/paloma_diagnostics/static.json
#   ${HF_CKPT}/paloma_diagnostics/viewer.html

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
POS_SCRIPT="${SCRIPTS_DIR}/paloma_pos_tag.py"
VIEWER_SCRIPT="${SCRIPTS_DIR}/paloma_viewer.py"

for f in "${DIAG_SCRIPT}" "${POS_SCRIPT}" "${VIEWER_SCRIPT}"; do
    [ -f "${f}" ] || { echo "Error: ${f} not found."; exit 1; }
done

if [ -z "${HF_CKPT:-}" ]; then
    echo "Error: HF_CKPT not set."
    echo "  sbatch --export=HF_CKPT=/path/to/hf_checkpoint slurm_jobs/pipeline.sbatch"
    exit 1
fi
[ -f "${HF_CKPT}/config.json" ] || { echo "Error: ${HF_CKPT}/config.json not found"; exit 1; }

# ---- Defaults -----------------------------------------------------------
MAX_DOCS=${MAX_DOCS:-200}
POS_MAX_CHUNKS=${POS_MAX_CHUNKS:-200}
N_EXPLORER_CHUNKS=${N_EXPLORER_CHUNKS:-5}
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
echo "N_EXPLORER_CHUNKS: ${N_EXPLORER_CHUNKS}"
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

        # Compute nodes are offline; datasets must be in the cache. The
        # diagnostics script now prints full tracebacks on load failure,
        # so if a source can't be found you'll see exactly why.
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
            echo '=== Stage 1/3: paloma_diagnostics.py ==='
            python '${DIAG_SCRIPT}' \\
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
            echo '=== Stage 2/3: paloma_pos_tag.py ==='
            if ! python -c 'import spacy; spacy.load(\"en_core_web_sm\")' 2>/dev/null; then
                echo 'WARN: spaCy or en_core_web_sm not installed; skipping POS stage.'
                echo '      Install on login node: pip install spacy && python -m spacy download en_core_web_sm'
            else
                python '${POS_SCRIPT}' \\
                    --in-dir '${OUT_DIR}' \\
                    --ckpt   '${HF_CKPT}' \\
                    --max-chunks-per-source ${POS_MAX_CHUNKS} \\
                    ${SOURCES:+--sources ${SOURCES}}
            fi
        else
            echo 'SKIP_STAGE_POS set — skipping POS tagging.'
        fi

        # ------ Stage 3: viewer regeneration ------
        if [ -z \"\${SKIP_STAGE_VIEW:-}\" ]; then
            echo ''
            echo '=== Stage 3/3: paloma_viewer.py ==='
            python '${VIEWER_SCRIPT}' \\
                --in-dir '${OUT_DIR}' \\
                --out    '${OUT_DIR}/viewer.html' \\
                --ckpt   '${HF_CKPT}' \\
                --n-explorer-chunks ${N_EXPLORER_CHUNKS}
            echo \"Wrote viewer to: ${OUT_DIR}/viewer.html\"
        else
            echo 'SKIP_STAGE_VIEW set — skipping viewer regen.'
        fi
    "

echo "=========================================="
echo "END: $(date)"
echo "=========================================="
echo ""
echo "Download the viewer locally with:"
echo "  scp leonardo:${OUT_DIR}/viewer.html ./"