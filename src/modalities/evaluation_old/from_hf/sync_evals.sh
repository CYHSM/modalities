#!/bin/bash
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"

singularity exec --bind "${MY_ROOT}:${MY_ROOT}" --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" "$CONTAINER" bash -c "
    source ${OLMES_VENV}/bin/activate
    python ${MY_ROOT}/modalities/src/modalities/evaluation/from_hf/sync_evals_to_wandb.py
"