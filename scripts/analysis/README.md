# Routing analyses (copied from `hpc-leonardo-emnlp-final`, not yet ported)

These are copied unchanged from `src/modalities/evaluation/python_scripts` and
`src/modalities/evaluation/slurm_jobs` on branch `hpc-leonardo-emnlp-final`. They produced the
paper's routing figures (deep share by layer, task and POS; gate densities) and the
inference-time ablations (forced K, gate overrides, shuffled gates).

| Script | Purpose |
|---|---|
| `paloma_diagnostics.py` | `run`: per-token, per-layer gates, update norms and deep share on Paloma sources, written to parquet. `tag-pos`: spaCy UPOS sidecars. Also the analysis API the figures use. |
| `paloma_viewer.py` | Figures and interactive viewer over the diagnostics parquets. |
| `ablation_eval.py`, `aggregate_ablations.py`, `ablation_plot_halt_curve.py` | Inference-time ablations: gate overrides, shuffled gates, forced loop count. |
| `convert_adaptive_gpt.py`, `modeling_adaptive_gpt.py`, `verify_logits.py` | Export to the standalone `AdaptiveGPTForCausalLM` HF model the scripts above load. |
| `sync_evals_to_wandb.py` | Push OLMES results to W&B. |
| `slurm/` | The original job scripts, kept for their arguments only. |

## Before using them

- **The `slurm/` scripts do not run as-is.** They use a Singularity image and a python 3.12 venv
  under `/leonardo_work/EUHPC_D21_101`, which is no longer readable. Port them to the repo `.venv`
  or to `venvs/olmes_uv`; the paths are in the root `CLAUDE.md`.
- **They expect the `AdaptiveGPTForCausalLM` export.** The May L16 sweep checkpoints have it. The
  Sep soft-cap and depth checkpoints are exported as the generic `ModalitiesHFAdapter`. Either
  export those with `convert_adaptive_gpt.py`, which predates `gate_sum_cap` and the scale clamp
  and must learn both first, or teach the scripts to read gates from the adapter. Validate with
  `verify_logits.py`.
- The Table 4 analysis (token-class gate overrides vs a size-matched random control, Wilcoxon
  over 18 checkpoints) is not in this folder or anywhere in git history.
