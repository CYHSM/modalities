#!/bin/bash
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

CKPT="/leonardo_scratch/large/userexternal/mfrey000/experiments/2026-04-08__08-05-08_fe9b502424bcad8e/eid_2026-04-08__08-05-08_fe9b502424bcad8e-seen_steps_1836-seen_tokens_3850371072-target_steps_18360-target_tokens_38503710720"
HF_OUT="/leonardo_scratch/large/userexternal/mfrey000/experiments/2026-04-08__08-05-08_fe9b502424bcad8e/hf_test"
CONFIG="/leonardo_scratch/large/userexternal/mfrey000/experiments/2026-04-08__08-05-08_fe9b502424bcad8e/d1280_dual_r30w70d_6272deep_8576wide_12L_loop3.yaml"

echo "=== CONVERT ==="
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
    "$CONTAINER" bash -c "
        export PYTHONPATH=/opt/repos/modalities/src:\$PYTHONPATH
        cd /opt/repos/modalities
        python src/modalities/conversion/loop/convert_adaptive_gpt.py \
            '$CKPT' '$HF_OUT' --modalities_config '$CONFIG'
    " || { echo '❌ convert failed'; exit 1; }

echo ""
echo "=== VERIFY ==="
singularity exec --nv \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --bind "${MY_ROOT}/modalities:/opt/repos/modalities" \
    "$CONTAINER" bash -c "
        export PYTHONPATH=/opt/repos/modalities/src:\$PYTHONPATH
        cd /opt/repos/modalities
        python src/modalities/conversion/loop/verify_logits.py \
            '$CONFIG' '$CKPT' '$HF_OUT'
    " || { echo '❌ verify failed'; exit 1; }

echo ""
echo "✅ all good"