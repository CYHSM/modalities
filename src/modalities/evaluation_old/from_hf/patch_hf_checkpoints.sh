#!/bin/bash
set -u

MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
EXPERIMENTS_DIR="${SCRATCH_ROOT}/experiments"
MODELING_SRC="${MY_ROOT}/modalities/src/modalities/conversion/loop/modeling_adaptive_gpt.py"

[ -f "$MODELING_SRC" ] || { echo "modeling file not found: $MODELING_SRC"; exit 1; }

patched=0
for run_dir in "${EXPERIMENTS_DIR}"/*/; do
    run_name=$(basename "$run_dir")
    [ "$run_name" = "wandb" ] && continue
    for ckpt_dir in "${run_dir}"eid_*/; do
        hf="${ckpt_dir}hf_checkpoint"
        [ -f "${hf}/config.json" ] || continue

        # Copy the modeling file in
        cp "$MODELING_SRC" "${hf}/modeling_adaptive_gpt.py"

        # Add auto_map to config.json if not already present
        python3 - <<EOF
import json
p = "${hf}/config.json"
with open(p) as f:
    cfg = json.load(f)
cfg["auto_map"] = {
    "AutoConfig": "modeling_adaptive_gpt.AdaptiveGPTConfig",
    "AutoModelForCausalLM": "modeling_adaptive_gpt.AdaptiveGPTForCausalLM",
}
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
EOF
        patched=$((patched+1))
        echo "✅ patched $hf"
    done
done
echo "Done. Patched $patched checkpoints."