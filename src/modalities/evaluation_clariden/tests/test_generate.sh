#!/bin/bash
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"

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
        python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ckpt = '/leonardo_scratch/large/userexternal/mfrey000/experiments/2026-04-08__08-05-08_fe9b502424bcad8e/eid_2026-04-08__08-05-08_fe9b502424bcad8e-seen_steps_12852-seen_tokens_26952597504-target_steps_18360-target_tokens_38503710720/hf_checkpoint'
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True).cuda()
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
inp = tok('The answer is', return_tensors='pt').input_ids.cuda()
out = model.generate(inp, max_new_tokens=20, do_sample=False)
print(tok.decode(out[0]))
EOF
    "