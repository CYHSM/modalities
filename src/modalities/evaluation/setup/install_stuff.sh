MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
LIST_FILE="/leonardo_work/EUHPC_D21_101/mfrey/modalities/src/modalities/evaluation/from_hf/checkpoints_to_eval.txt"

> "$LIST_FILE"
for run_dir in "${SCRATCH_ROOT}"/experiments/*/; do
    run_name=$(basename "$run_dir")
    [ "$run_name" = "wandb" ] && continue
    for ckpt_dir in "${run_dir}"eid_*/; do
        hf="${ckpt_dir}hf_checkpoint"
        [ -f "${hf}/config.json" ] || continue
        [ -f "${hf}/olmes_eval/metrics-all.jsonl" ] && continue  # skip done
        echo "$hf" >> "$LIST_FILE"
    done
done
wc -l "$LIST_FILE"