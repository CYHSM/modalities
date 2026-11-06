# Dual-path paper: review/rebuttal working branch (`iclr-reviews`)

This branch of the Modalities fork holds everything behind the dual-path paper (under double-blind
review): the model, every training config, launchers, the eval pipeline and the analysis code.
Read this file first. The upstream Modalities framework docs (README.md, docs/) are general
background and not needed for this work.

**This repo is public.** Do not commit the paper text, reviews or anything else that ties the
submission to its authors (`main.tex` etc. live in the git-ignored `paper/` folder).

## The idea in one paragraph

Each layer has two parallel paths on the same input. The *deep* path is a sublayer re-applied K
times with shared weights (compute). The *wide* path has a larger FFN and is applied once
(capacity). Two independent per-token sigmoid gates mix them: `y = g_d * h_deep + g_w * h_wide`.
Baselines at matched FFN FLOPs: PureLoop (all deep) and PureWide (all wide). `alpha` = share of
FFN FLOPs on the deep path (25/50/75). Budgets: F80M / F160M = per-layer FFN FLOPs per token.

## Where things are

| What | Where |
|---|---|
| Model | `src/modalities/models/looped/dualpath.py`, `variant_key: dualpath` |
| Training configs | `config_files/janus/` (layout in its README) |
| Which run is behind which table | `config_files/janus/paper_checkpoints_manifest.tsv` |
| Launchers | `config_files/janus/run_array_16nodes.sh`, `run_warmstart.sh` |
| Config tools | `config_files/janus/tools/` (FLOP-matched widths, depth variants, warmstart configs) |
| Convert + OLMES eval | `scripts/evaluation/run_convert_and_eval_sbatch.sh` (fired by the config's `downstream_evaluator`) |
| Paper tables from eval output | `scripts/evaluation/render_paper_tables.py` |
| Routing analyses (deep share, POS, gate/loop overrides) | `scripts/analysis/` (see its README; not yet ported) |
| S3 backup of final checkpoints | `scripts/backup/` |

Model config knobs (`model_raw.config.adaptive_config`): `layer_types` (`dual`/`loop`/`wide` per
layer; PureLoop/PureWide are the same model with all `loop`/all `wide`), `max_loops` (K),
`ffn_hidden` (deep FFN), `wide_ffn_hidden`, `gate_mode: two_gates`, `use_cross: false` (the
"nocross" in config names), `gate_sum_cap` (the soft cap c; `1.0` in all newer runs).

Config names: `<8e7|16e7>_loops<N>__dm768_L<layers>_loop<K>_F<80|160>M_<widths>_<arm>[_suffix].yaml`.
`8e7`/`16e7` = F80M/F160M. Widths: `ffnD`/`ffnW` = deep/wide FFN for dual, `ffnL` for PureLoop,
`ffnW` for PureWide. A `_WARMSTART` twin resumes the base config from its last checkpoint.

## Paths on Leonardo

| What | Path |
|---|---|
| Training data (`.pbin`) | `/leonardo_scratch/large/userexternal/mfrey000/tokenized/{score3.0-3.5,score3.5-4.0,score4.0-4.5,score4.5+,nemotron_math_gpt2}` |
| Tokenizer (GPT-2) | `/leonardo_work/AIFAC_S07_154/mfrey/tokenizer/local_gpt2_tokenizer/` |
| Training venv | `<repo>/.venv` (python 3.11, torch 2.10) |
| OLMES eval env | `/leonardo_work/AIFAC_S07_154/mfrey/{venvs/olmes_uv,olmes,tiktoken_cache,nltk_data}`, built by `scripts/evaluation/setup_olmes_env.sh` (olmes `5a51f50`, torch 2.8) |
| OLMES dataset cache (offline) | `/leonardo_scratch/large/userexternal/mfrey000/hf_cache` |
| Experiments: May L16 sweep | `/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp` |
| Experiments: seeds, soft cap, depth, new runs | `/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest` |
| Smoke tests | `/leonardo_scratch/large/userexternal/mfrey000/experiments_janus_test` |

Run layout: `<root>/<exp_id>/eid_<exp_id>-seen_steps_<N>-...-target_steps_18360-.../` (DCP shards),
with `hf_checkpoint/` inside it and the OLMES output in `hf_checkpoint/olmes_eval_<step>/metrics-all.jsonl`.
`<root>/<exp_id>/last_checkpoint_info.json` points at the newest checkpoint.

## How to

- **Train**: list config paths in a text file, then
  `CONFIG_LIST=<file> sbatch --array=1-N config_files/janus/run_array_16nodes.sh`.
  16 nodes x 4 A100, 38.5B tokens = 18360 steps, a checkpoint every 1836 steps, seed 43.
  L16 takes 13-21h, so it fits the 24h limit.
- **Longer runs** (L>=32): make a warmstart config with `tools/make_warmstart_config.py`, then
  chain `sbatch --dependency=afterany:<job> config_files/janus/run_warmstart.sh <WS_CONFIG> <BASE_CONFIG_NAME>`.
  Queue several links up front; a link that finds the run finished exits cleanly.
- **Smoke test** (1 node, debug QOS, 200 steps): `sbatch config_files/janus/debug/run_smoketest_debug.sh [CONFIG]`.
- **Eval**: happens automatically at the final step (one 1-GPU job: DCP->HF conversion, then 25
  OLMES tasks at limit=1024). Newer configs leave out gsm8k: queue
  `scripts/evaluation/queue_gsm8k_after_training.sh` after the last training link. To re-run an
  eval by hand: `JANUS_EVAL_ALL_STEPS=1 JANUS_EVAL_TIME=12:00:00 bash scripts/evaluation/run_convert_and_eval_sbatch.sh <ckpt_dir> <resolved_cfg> <ckpt_dir>/hf_checkpoint '<tasks>' <label> 1024 1`.
  It writes to `olmes_eval_<label>` and skips anything already there.
- **Paper numbers**: `python scripts/evaluation/render_paper_tables.py`. Conventions: use the
  limit=1024 dir (not `_128`) and each task's `primary_score`. Commonsense = mean over ARC-c, ARC-e,
  HellaSwag, PIQA, SIQA, WinoGrande. Math BPB = `olmo3:base_easy:math_bpb`. C4/Wiki = Paloma BPB.
  Only rows with `limit == 1024` count, so smoke-test dirs next to a real eval are ignored.
- **Review-round tables** (compute-matched, gate controls, capped sweep; mean +- std over seeds):
  `python scripts/evaluation/render_review_tables.py [--apply <main.tex>]`. Configs and launch
  lists are in `config_files/janus/review/`.
- **New FLOP-matched arm**: get widths from `tools/generate_configs.py`. Its output follows the
  older EMNLP layout, so copy the closest existing janus config and change only the model block
  rather than using the output directly.

## Cluster gotchas (all of these have bitten before)

- **Allocation `AIFAC_S07_154` ends 2026-12-01.** No compute after that. The repo and work area
  are on the same project.
- **Expired projects get locked about 6 months after they end.** `EUHPC_D21_101` became
  unreadable in Sep 2026 and silently killed runs (tokenizer path). `EUHPC_E05_119` ended
  2026-07-25 and holds the original data copy; don't reference it. Grep new configs and scripts
  for paths under expired projects.
- **Scratch is purged.** F80M seeds 2 and 3 of the main table were lost to it. Final HF
  checkpoints are backed up on S3 (`scripts/backup/README_s3.md`); DCP shards are not.
- Login nodes kill processes after 10 min of CPU time. Use `--partition=lrd_all_serial`, which
  has internet, for installs and copies. Compute nodes are offline, so precache HF datasets from
  a login node. Paloma is gated but already cached.
- Keep `UV_CACHE_DIR` on scratch: `~/.cache/uv` once filled the 50G home quota.
- W&B runs are offline (project `dualcross`). A walltime kill while training waits for the
  final eval skips the downstream W&B sync; the metrics are still on disk.

## Comparability caveats for new ablations

- The May L16 sweep (main table, `experiments_emnlp`) was trained from branch
  `hpc-leonardo-emnlp-final`, i.e. older model code. The soft-cap and depth runs (Sep,
  `experiments_budapest`) use this branch's code. Compare new ablations against the Sep runs:
  the baseline config is `softcap/8e7_loops4__dm768_L16_loop4_F80M_dual_a50_nocross_softcap1.yaml`.
- Since Sep, the code clamps the loop/wide residual scales at 2.0.
- Without the soft cap, dual-path diverges to NaN at L>=32 (`stability/` holds the attempts).
- Every paper run uses `model_initializer.model_type: gpt2`. On the dual-path model those name
  filters match no parameters, so weights keep the default `nn.Linear` init. Keep it that way for
  comparability; the `*_initfix` configs test the fix.
- Every paper run sets `weight_decay_groups_excluded: [embedding, pytorch_rms_norm]`, but the
  model's norm group is called `layernorm`. So only the embedding is excluded, and the RMSNorm
  weights, `loop_scales`/`wide_scale` and the gate biases all get weight decay 0.3 (the optimizer
  logs a warning). Paper Appendix A says norms are excluded, which is wrong. Keep the setting
  as-is for new ablations so they stay comparable.
- The analysis scripts expect the standalone `AdaptiveGPTForCausalLM` HF export that the May
  checkpoints have. Sep checkpoints are exported as `ModalitiesHFAdapter`, so the analyses need
  porting before they can run on them.

## Status (2026-11-06)

- Done: L16 sweeps at F80M and F160M (3 seeds each; the F80M seed-2/3 checkpoints survive only
  as metrics), soft cap (F80M/F160M), depth L8/16/32. The S3 backup of every final checkpoint is
  complete.
- 2026-11-06: data moved to scratch and verified. The soft-cap smoke test passed (200 steps,
  checkpoints written). The rebuilt OLMES env reproduces the published soft-cap F80M numbers
  exactly (C4 1.0361, Wiki 0.9332, math BPB 0.5116, GSM8K 0.1348; output in
  `hf_checkpoint_recheck/olmes_eval_18360_recheck` of run `2026-09-14__01-44-11_4767e14501931d3a`).
- Stalled: L64 dual (soft cap) and PureWide, both at step 12852/18360 (the D21_101 lockout).
  Resume with `run_warmstart.sh`; each needs about 25h more.
- Missing from git: the Table 4 token-class override analysis (Wilcoxon over 18 checkpoints).
- Other branches: `auto-downstream-eval` (parent of this one), `hpc-leonardo-emnlp-final` (May
  sweep configs, original analysis), `loopotron` (a separate Nemotron looping project, unrelated).
