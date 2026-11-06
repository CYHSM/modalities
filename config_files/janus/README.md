# Janus — dual-path ablation suite (seed 43)

17 configs recreated from `hpc-leonardo-emnlp-final:config_files/emnlp/final_budapest_seed43`,
corrected to run on this branch. Training hyperparameters, token budgets, intervals, data
mixture and seed (43) are **byte-identical** to the originals — only the items below changed.

## What was changed, and why

1. **Data paths.** The originals used `/data/...`, which only exists inside the Singularity
   image (`--bind /leonardo_work/EUHPC_E05_119/mfrey/tokenized:/data`). All 436 `raw_data_path`
   entries per config now use the absolute host path, so these run under the native `.venv`.
2. **`model_raw.variant_key: gpt2` → `dualpath`.** On the EMNLP branch the adaptive model *was*
   the gpt2 variant. On this branch it is a separate `dualpath` variant and `gpt2` no longer
   accepts `adaptive_config`. Without this the config fails validation.
3. **Removed `tokenizer` from `evaluation_subscriber`.** `WandBEvaluationResultSubscriberConfig`
   does not define that field here and the component factory rejects unknown keys.
4. **Added `model_converter` + `downstream_evaluator`** (the originals had neither), wiring the
   26-task OLMES suite with `LIMIT=1024`, `BATCH_SIZE=1`, `INCLUDE_GENERATIVE=1`.

## Running

```bash
sbatch --array=1-17 config_files/janus/run_array_16nodes.sh
```

16 nodes x 4 GPUs, native `.venv`, no container. `EXPERIMENT_ROOT` in the launcher must stay in
sync with `settings.paths.experiments_root_path` in the configs
(`/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest`) — the converter and
downstream evaluator resolve the checkpoint through the config value.

## Benchmark cache — read before running

Compute nodes have no internet (`HF_HUB_OFFLINE=1`), so every task must be precached from a
login node into `/leonardo_scratch/large/userexternal/mfrey000/hf_cache`.

**All 26 tasks are cached and offline-verified** (verified with `HF_HUB_OFFLINE=1` and no
token in the environment, i.e. exactly the conditions on a compute node).

`allenai/paloma` is a **gated** dataset, so `paloma_c4_en::paloma` and
`paloma_wikitext_103::paloma` needed an authenticated download once. That is already done — the
data now sits in the cache and compute nodes read it offline without any credential. You only
need a token again if the cache is wiped or another gated dataset is added.

Note that `HF_HOME` also determines where the HF **token** is read from. Pointing it at the
scratch cache moves token lookup away from `~/.cache/huggingface/`, so export `HF_TOKEN`
explicitly when fetching a gated dataset.

## Eval walltime

`scripts/evaluation/run_olmes_sbatch.sh` requests `--time=06:00:00`. 26 tasks at
`limit=1024`, `batch_size=1`, including generative `gsm8k::olmes`, may exceed that on one GPU.
Raise the walltime or the batch size if evals start hitting TIMEOUT.
