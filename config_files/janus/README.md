# Janus: dual-path training configs

Start with the repo-root `CLAUDE.md`. This file covers only this folder.

| Folder | What | Paper |
|---|---|---|
| `l16_sweep/` | L16 PureWide / PureLoop K=2,3,4 / Dual a25,a50,a75 (K=4) at F80M and F160M | main table; the F160M seeds 2 and 3 were trained from these |
| `softcap/` | L16 Dual a50 K=4 with `gate_sum_cap: 1.0`, F80M and F160M | soft-cap table. **Baseline for new ablations** |
| `depth/` | F80M: L8/L32/L48/L64 PureWide and Dual a50 soft cap. F160M: L32/L48 Dual (uncapped) and PureWide | depth table (L8/16/32), L64 (stalled at step 12852) |
| `gate_variants/` | L16 a50 with convex, fixed and softmax gates | not in the paper |
| `stability/` | Attempts to fix the L32 NaN divergence before the soft cap: lr, warmup, grad clip, fp32 reduce, init, convex gate, compile | not in the paper |
| `debug/` | Smoke test, livecheck and NaN-repro configs and their launchers | |
| `tools/` | `generate_configs.py` (FLOP-matched widths), `make_depth_variant.py`, `make_warmstart_config.py` | |

`*_WARMSTART.yaml` resumes its base config from the last checkpoint (made by
`tools/make_warmstart_config.py`). `paper_checkpoints_manifest.tsv` maps each paper table cell
to its experiment directory. `all_configs.txt` is the default array list for
`run_array_16nodes.sh`. Run directories store their config by basename, so moving a config
between folders does not break `run_warmstart.sh` discovery.

## Running

```bash
# full run(s): 16 nodes x 4 GPUs, 24h limit
CONFIG_LIST=my_list.txt sbatch --array=1-N config_files/janus/run_array_16nodes.sh
# continue past 24h (queue several links up front)
sbatch --dependency=afterany:<job> config_files/janus/run_warmstart.sh <WS_CONFIG> <BASE_CONFIG_NAME>
# 200-step smoke test on one node, debug QOS
sbatch config_files/janus/debug/run_smoketest_debug.sh [CONFIG]
```

`EXPERIMENT_ROOT` in the launchers must match `settings.paths.experiments_root_path` in the
config (`.../experiments_budapest`): the converter and evaluator find the checkpoint through the
config value.

## How these differ from the EMNLP-branch originals

The `l16_sweep/` configs are recreated from `hpc-leonardo-emnlp-final:config_files/emnlp`.
Hyperparameters, token budget, intervals, data mixture and seed 43 are identical. Changes:
absolute host data paths (no container), `variant_key: dualpath` (on that branch the model was
the `gpt2` variant), no `tokenizer` on the W&B subscriber, and an added `downstream_evaluator`.
Data paths have since moved to scratch and the tokenizer to `AIFAC_S07_154`, see `CLAUDE.md`.

## Eval notes

Compute nodes are offline. All OLMES tasks are cached in
`/leonardo_scratch/large/userexternal/mfrey000/hf_cache`, including the gated Paloma, which
needs `HF_TOKEN` only if the cache is ever rebuilt. Generative gsm8k is slow at depth (about
15 s/it at L32); the deep runs set `JANUS_EVAL_TIME=20:00:00`.
