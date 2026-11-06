#!/bin/bash
#SBATCH --job-name=olmes-setup
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Build the OLMES evaluation environment used by run_convert_and_eval_sbatch.sh.
#
# The original env lived under /leonardo_work/EUHPC_D21_101/mfrey, which became
# unreadable once that project ended. This rebuilds the same layout under
# EVAL_ROOT:
#   olmes/           allenai/olmes at OLMES_COMMIT (put on PYTHONPATH by the eval script)
#   venvs/olmes_uv/  python 3.11 venv: olmes (no vLLM, we use --model-type hf) + modalities
#   tiktoken_cache/  pre-fetched encodings; litellm loads one at import, compute nodes are offline
#   nltk_data/       punkt etc., same reason
#
# Runs on the serial partition: login nodes have internet, but kill processes
# after 10 min of CPU time, which a full install exceeds.
#
#   sbatch scripts/evaluation/setup_olmes_env.sh
set -euo pipefail

EVAL_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${EVAL_ROOT}/modalities"
OLMES_DIR="${EVAL_ROOT}/olmes"
VENV="${EVAL_ROOT}/venvs/olmes_uv"
# Upstream HEAD since 2026-03-24, so every eval behind the paper (Apr-Sep 2026) ran on it.
OLMES_COMMIT="5a51f50"
# Keep the uv cache off /leonardo home: it filled the 50G quota once.
export UV_CACHE_DIR="/leonardo_scratch/large/userexternal/mfrey000/.uv_cache"
UV="${HOME}/.local/bin/uv"

echo "START: $(date)"
[ -d "${OLMES_DIR}/.git" ] || git clone https://github.com/allenai/olmes.git "${OLMES_DIR}"
git -C "${OLMES_DIR}" checkout --quiet "${OLMES_COMMIT}"
# pytrec_eval (HELMET only, unused here) has no wheels and fails to compile; the
# terrier fork is a drop-in with prebuilt wheels.
sed -i 's/"pytrec_eval", # For HELMET/"pytrec-eval-terrier", # For HELMET/' "${OLMES_DIR}/pyproject.toml"

"${UV}" venv --python 3.11 "${VENV}"
"${UV}" pip install --python "${VENV}/bin/python" -e "${OLMES_DIR}"
# Base deps only (no torch extra): modalities pins torch>=2.10, olmes torch<2.9.
"${UV}" pip install --python "${VENV}/bin/python" -e "${REPO_DIR}"
# olmes pins numpy<2, but the resolver picks a pyarrow (>=20) that needs numpy 2, and
# `import datasets` then fails inside every eval. 19.x still supports numpy 1.26.
"${UV}" pip install --python "${VENV}/bin/python" "pyarrow<20"
# uv venvs ship without setuptools, but olmes still imports pkg_resources (removed in setuptools 81).
"${UV}" pip install --python "${VENV}/bin/python" "setuptools<81"

export TIKTOKEN_CACHE_DIR="${EVAL_ROOT}/tiktoken_cache"
export NLTK_DATA="${EVAL_ROOT}/nltk_data"
mkdir -p "${TIKTOKEN_CACHE_DIR}" "${NLTK_DATA}"
"${VENV}/bin/python" - <<'EOF'
import os, nltk, tiktoken
for enc in ["cl100k_base", "o200k_base", "p50k_base", "r50k_base", "gpt2"]:
    tiktoken.get_encoding(enc)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, download_dir=os.environ["NLTK_DATA"], quiet=True)
print("caches ok")
EOF

# Import check under the same conditions as a compute node.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
PYTHONPATH="${REPO_DIR}/src:${OLMES_DIR}" "${VENV}/bin/python" - <<'EOF'
import torch, transformers, litellm, datasets
import oe_eval.run_eval  # the module olmes actually runs; importing oe_eval alone missed the pyarrow break
from modalities.conversion import generic_adapter
print("torch", torch.__version__, "transformers", transformers.__version__, "imports ok")
EOF
echo "END: $(date)"
