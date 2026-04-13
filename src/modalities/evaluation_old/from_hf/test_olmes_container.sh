MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"

HF_CKPT="${SCRATCH_ROOT}/experiments/2026-04-08__08-05-08_fe9b502424bcad8e/eid_2026-04-08__08-05-08_fe9b502424bcad8e-seen_steps_11016-seen_tokens_23102226432-target_steps_18360-target_tokens_38503710720/hf_checkpoint"
OUT_DIR="${HF_CKPT}/olmes_smoke_test"

export TIKTOKEN_CACHE_DIR="${MY_ROOT}/tiktoken_cache"
mkdir -p "$TIKTOKEN_CACHE_DIR"

singularity exec --bind "${MY_ROOT}:${MY_ROOT}" "$CONTAINER" bash -c "
    source ${OLMES_VENV}/bin/activate
    export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
    export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR}
    python -c 'import tiktoken; tiktoken.get_encoding(\"cl100k_base\"); print(\"cached\")'
    ls -la ${TIKTOKEN_CACHE_DIR}
"