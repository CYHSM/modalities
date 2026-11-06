# DualPath (janus) paper checkpoints

Final checkpoint of every training run behind the tables in `main.tex`, one folder per run, grouped by table.

Each run folder holds `SOURCE.txt` (original Leonardo scratch path, step, which table), the training config
(`*.yaml`, `*.yaml.resolved`), and `hf_checkpoint/`: HF weights (`pytorch_model.bin`), tokenizer,
`modalities_adapter.py` (load with `trust_remote_code=True`) and the OLMES eval outputs (`olmes_eval*/`).
DCP training shards are not included, so runs cannot be resumed from this backup.

Metrics in the paper are read from `hf_checkpoint/olmes_eval*/metrics-all.jsonl` (limit=1024 dirs, not `_128`),
using each task's `primary_score`; `scripts/evaluation/render_paper_tables.py` in the modalities repo reproduces them.

**Not included:** F80M seeds 2 and 3 of `tab:main`. Those checkpoints were deleted by the Leonardo scratch
purge before this backup (Sep 12 2026); only their published metrics remain.

## `L16_F80M_sweep/`

Full F80M sweep, L=16 (tab:appendix-f80m). Also seed 1 of every F80M row in tab:main.

| folder | paper | source run |
|---|---|---|
| `PureWide` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__02-00-20_7ad77be5d81edda7` |
| `PureLoop_loop2` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-15__04-29-48_f2cc5cc7dee9c20b` |
| `PureLoop_loop3` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__04-56-22_7772cb76cea79b1e` |
| `PureLoop_loop4` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__05-28-26_cba0ba4a0338106f` |
| `Dual_a25_loop2` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__13-49-25_46661c3a0490b08d` |
| `Dual_a50_loop2` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__02-05-13_fa51525556e2cd3b` |
| `Dual_a75_loop2` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__19-28-55_0d6659323c9236d8` |
| `Dual_a25_loop3` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__12-12-23_207cde6cd1273411` |
| `Dual_a50_loop3` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__02-21-25_03070170b2aa73fa` |
| `Dual_a75_loop3` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__19-28-58_635662eb51947f62` |
| `Dual_a25_loop4` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__23-27-56_3af89f56dea89107` |
| `Dual_a50_loop4` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-13__05-25-03_e3f7d3d2ad461681` |
| `Dual_a75_loop4` | seed 1 of tab:main; column of tab:appendix-f80m | `experiments_emnlp/2026-05-18__19-28-56_159b57f1f3f20442` |

## `L16_F160M_sweep/`

Full F160M sweep, L=16 (tab:appendix-f160m). Also seed 1 of every F160M row in tab:main.

| folder | paper | source run |
|---|---|---|
| `PureWide` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-12__06-08-00_e2024d4668888064` |
| `PureLoop_loop2` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-12__06-22-22_aa12cd089a6c295f` |
| `PureLoop_loop3` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-14__08-53-02_4216c33dec516135` |
| `PureLoop_loop4` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-12__08-25-36_59f1db86439d8912` |
| `Dual_a25_loop2` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__12-12-23_59f7fd9274a1eb2c` |
| `Dual_a50_loop2` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-12__06-09-41_7254f5ea8908ea68` |
| `Dual_a75_loop2` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__19-28-56_7d6f149ca64eea51` |
| `Dual_a25_loop3` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__12-12-23_2b5348bcf89b949f` |
| `Dual_a50_loop3` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-14__08-07-59_50122419e575e16d` |
| `Dual_a75_loop3` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__19-28-59_b5263d4c7baf76af` |
| `Dual_a25_loop4` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__12-12-23_355c95b231e4474c` |
| `Dual_a50_loop4` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-12__06-22-54_3719e6b84f11c9c1` |
| `Dual_a75_loop4` | seed 1 of tab:main; column of tab:appendix-f160m | `experiments_emnlp/2026-05-18__19-28-56_55916ccdd76d9adc` |

## `L16_F160M_seeds/`

Seeds 2 and 3 of the F160M rows in tab:main (mean +- std over seed 1 from L16_F160M_sweep plus these).

| folder | paper | source run |
|---|---|---|
| `PureWide_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__18-39-17_8bbbdb520711f375` |
| `PureWide_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-43-59_9bf743665565e1b5` |
| `PureLoop_loop2_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__18-41-25_36b265498cedb5f7` |
| `PureLoop_loop2_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-44-16_3cdff85aa15370b9` |
| `PureLoop_loop3_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__18-43-42_ffd20f609cbe702b` |
| `PureLoop_loop3_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-49-08_c8dd392cdda6300c` |
| `PureLoop_loop4_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__18-51-54_a8ce64b4a2b88bd3` |
| `PureLoop_loop4_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-55-59_a63b386ae5a3df8f` |
| `Dual_a25_loop4_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__19-52-44_48105428c1fd2d82` |
| `Dual_a25_loop4_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-56-05_d035cae6de83dd00` |
| `Dual_a50_loop4_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-29-47_8868c7e041195def` |
| `Dual_a50_loop4_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__21-33-30_86448890015102b6` |
| `Dual_a75_loop4_seed2` | seed 2 of tab:main (F160M) | `experiments_budapest/2026-07-24__20-35-33_54b4363cb17b7875` |
| `Dual_a75_loop4_seed3` | seed 3 of tab:main (F160M) | `experiments_budapest/2026-07-24__22-07-53_e2cd3461d055b1fe` |

## `unrolled_baseline/`

PureWide-Exp: L*K distinct layers at 1/K per-layer budget (tab:unrolled).

| folder | paper | source run |
|---|---|---|
| `PureWideExp_F80M_K2_L32` | tab:unrolled | `experiments_emnlp/2026-05-18__19-29-00_09776dacfbd02318` |
| `PureWideExp_F80M_K3_L48` | tab:unrolled | `experiments_emnlp/2026-05-18__19-28-58_b0f6ff56da44e1d2` |
| `PureWideExp_F80M_K4_L64` | tab:unrolled | `experiments_emnlp/2026-05-18__19-28-56_35921c8a7d8e2634` |
| `PureWideExp_F160M_K2_L32` | tab:unrolled | `experiments_emnlp/2026-05-18__19-28-59_78bcbfab173bc55b` |
| `PureWideExp_F160M_K3_L48` | tab:unrolled | `experiments_emnlp/2026-05-18__19-28-58_b701cc1f92e023b7` |
| `PureWideExp_F160M_K4_L64` | tab:unrolled | `experiments_emnlp/2026-05-18__22-41-23_d8e95812e2a31843` |

## `softcap/`

DualPath a50 loop4 with gate_sum_cap=1.0 at L=16 (tab:softcap).

| folder | paper | source run |
|---|---|---|
| `L16_F80M_Dual_a50_loop4_cap1` | tab:softcap; L16 row of tab:depth-sweep | `experiments_budapest/2026-09-14__01-44-11_4767e14501931d3a` |
| `L16_F160M_Dual_a50_loop4_cap1` | tab:softcap | `experiments_budapest/2026-09-13__03-57-19_7baca097a6a8a0d2` |

## `depth_sweep_F80M/`

F80M depth sweep, PureWide vs DualPath+cap (tab:depth-sweep). L16 rows: see L16_F80M_sweep/PureWide and softcap/L16_F80M_*. L64 runs are not included: training stopped at step 12852/18360 and was never converted to HF.

| folder | paper | source run |
|---|---|---|
| `L8_PureWide` | tab:depth-sweep | `experiments_budapest/2026-09-14__01-40-27_bfe620344dd90688` |
| `L8_Dual_a50_loop4_cap1` | tab:depth-sweep | `experiments_budapest/2026-09-14__01-39-55_a60bd94f927862fe` |
| `L32_PureWide` | tab:depth-sweep | `experiments_budapest/2026-09-10__05-47-34_4c2694a69d16531d` |
| `L32_Dual_a50_loop4_cap1` | tab:depth-sweep | `experiments_budapest/2026-09-13__03-54-08_e2578e41d2128d01` |
