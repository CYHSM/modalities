# Nemotron looped-model research: state and plan

**Purpose.** This document is a self-contained handoff. It assumes no knowledge of earlier sessions.
Read sections 1-3 to understand what exists, section 4 for facts that are expensive to rediscover,
sections 5-8 for background and queued work, and **section 9 for what to run next on the cluster**.

**Status as of 2026-08-11.** The cluster is available again; **section 9 is the handoff for the
5B-token wave on a new dataset**, and its first two items (regenerate the tokenizer-dependent
evaluation assets, verify warmstart) are prerequisites that fail silently if skipped.

**Earlier status, 2026-08-10.** Work items A (reasoning evaluations) and B (per-iteration norms and
input injection) are implemented and have been run; dense-transformer baselines exist for the first
time. The headline results are in section 3.6: the hybrid beats an iso-FLOP dense transformer by
0.20 nats, and per-iteration norms are the first loop refinement to show a reasoning gain. Neither
is trustworthy until the seed-noise floor is measured, which is the first item in section 9.

**Companion documents.** [nemotron_loops.md](nemotron_loops.md) is the user-facing reference for the
loop mechanism (notation, semantics, arm table, verified runs). [nemotron.md](nemotron.md) covers the
underlying hybrid Mamba-Transformer model. This document is the *research plan*; those two are the
*implementation reference*. Keep them separate.

---

## 1. What the research question is

Nemotron-style hybrid models are a stack of single-operator residual layers described by a pattern
string: `M` = Mamba-2 mixer, `E` = sparse MoE feed-forward, `*` = grouped-query attention,
`-` = dense feed-forward. Because each layer type is a self-contained residual block, a **loop**
(executing a run of layers several times while reusing one set of weights) can be placed around any
subset of layer types.

The question: **where is extra depth worth spending — on the Mamba mixers, the MoE feed-forwards,
the attention layers, or on mixed blocks?** Looping trades parameters for effective depth, so each
arm holds parameters fixed while raising FLOPs.

---

## 2. What is implemented

### 2.1 Loop grammar

A loop group is written `[<symbols>]^<K>` inside the layer pattern, e.g. `M[ME]^3*E`. Brackets are
mandatory. The repeat operator is `^`, **not** `*`, because `*` is the attention symbol and
`M[ME]*3*E` reads as though the 3 were multiplied by an attention layer. Nesting is unsupported.
Bracket-free patterns parse exactly as before, so all pre-existing configs are unaffected.

`n_layer` counts **built** layers (sets of weights), not executed ones. `M[ME]^3E` has `n_layer: 4`.

### 2.2 Files

| File | What it holds |
|---|---|
| [layer_pattern.py](../../src/modalities/models/nemotron/layer_pattern.py) | `LoopGroup`, `parse_layer_schedule`, `get_num_built_layers`; `count_layers_by_type` counts *executions* |
| [nemotron_model.py](../../src/modalities/models/nemotron/nemotron_model.py) | Schedule-driven forward, `LoopConfig`, `n_executed_layers`, `get_execution_counts` |
| [nemotron_model_factory.py](../../src/modalities/models/nemotron/nemotron_model_factory.py) | Passes `loop_config` through |
| [moe.py](../../src/modalities/models/components/moe/moe.py) | Aux loss accumulates per visit; `reset_aux_loss()` |
| [nemotron_mfu.py](../../src/modalities/utils/nemotron_mfu.py) | Active params execution-weighted |
| [number_conversion.py](../../src/modalities/utils/number_conversion.py) | `num_executed_layers_from_layer_pattern` |
| [nemotron_layers.py](../../src/modalities/models/nemotron/nemotron_layers.py) | `PerIterationNorm`; layers take an iteration index (section 7) |
| [test_layer_loops.py](../../tests/models/nemotron/test_layer_loops.py) | 49 tests for the above |

### 2.3 The critical design decision

`transformer.h` stays a **flat** `nn.ModuleDict` with one entry per set of weights. The loop lives in
a separate execution schedule:

```python
for group in self._schedule:            # list[LoopGroup]
    h = self._run_loop_group(group, h)
```

This is why activation checkpointing (`layers_fqn: transformer.h`), FSDP2 block wrapping, the
weight-decay group regexes and the initialization parameter filters all work on looped models with
**zero** changes — parameter names are still `transformer.h.<idx>...`. Do not nest loop groups into
their own module; it breaks all four.

`NemotronLLM._run_loop_group` is the single extension point for loop semantics, selected by
`loop_config.variant` (currently only `"simple"`, which chains iterations). `LoopConfig` permits
extra keys so a new variant can carry hyperparameters without a schema change.

### 2.4 Things that must follow executions, not parameters

Getting any of these wrong biases an ablation silently rather than raising an error.

| Quantity | Handling |
|---|---|
| **Effective depth** (init scaling) | `n_executed_layers`. Residual variance grows with layer applications. The config derives `num_layers` for the initializer from the pattern via a `number_conversion`, so it cannot go stale. **This is the single most important detail.** |
| **MoE auxiliary loss** | Accumulated across visits, reset per forward. A looped MoE layer routes independently each visit. Accumulation is out-of-place so an activation-checkpointing replay cannot corrupt the tensor already handed to the loss. |
| **MFU active parameters** | Execution-weighted via `get_execution_counts()`. |

`router.tokens_per_expert` already used `+=`, so the auxiliary-loss-free expert bias needed no change.

---

## 3. Experimental state

### 3.1 The research model

[config_research_nemotron_loops_1gpu.yaml](../../config_files/nemotron/config_research_nemotron_loops_1gpu.yaml)
is the base/template. It is a geometric scale-down of Nemotron-3 Nano 30B-A3B that preserves the
reference's **cost structure**:

| Property | Reference 30B-A3B | Research config |
|---|---|---|
| Model dim / built layers | 2688 / 52 | 1024 / 12 |
| Parameters | 31.58B total, 3.58B active | 1.105B total, 225M active |
| Non-embedding active/total | 9.3% | 9.7% |
| Per-layer active cost `M : * : E` | 1 : 0.60 : 2.07 | 1 : 0.59 : 2.01 |
| Expert sparsity | top-6 of 128 | top-6 of 128 |

Why 1.1B and not 400M: the embedding table and the Mamba layers are dense and always active, so a
400M model with a 128k vocab cannot be ~10% active; forcing the size down distorts the `M : * : E`
cost ratio, which is exactly what the ablation is about. Training FLOPs scale with *active*
parameters, so this trains at roughly the speed of a 225M dense model; only memory holds 1.1B.

The headline 20.4% active (vs the reference's 11.3%) is entirely the vocabulary tax — tied Llama-3
embeddings are 12% of this model but 2.2% of the 30B. It is constant across arms and cancels.

### 3.2 Arms

Generated from the base config by
[generate_arm_configs.py](../../config_files/nemotron/loop_ablation/generate_arm_configs.py) — edit
the base and regenerate, so arms cannot drift. Only `layer_pattern` and `n_layer` differ.

```bash
python config_files/nemotron/loop_ablation/generate_arm_configs.py
./config_files/nemotron/loop_ablation/run_arm.sh A1_loop_mamba 4      # arm, GPU
```

| Arm | Pattern | Built/Exec | Active | rel. FLOPs |
|---|---|---|---|---|
| `A0_baseline` | `MEM*EMEMEM*E` | 12/12 | 225.5M | 1.000 |
| `A1_loop_mamba` | `[M]^3E[M]^3*E[M]^3E[M]^3E[M]^3*E` | 12/22 | 283.6M | 1.244 |
| `A2_loop_moe` | `M[E]^2M*[E]^2M[E]^2M[E]^2M*[E]^2` | 12/17 | 283.9M | 1.245 |
| `A3_loop_attention` | `MEM[*]^4EMEMEM[*]^4E` | 12/18 | 246.0M | 1.244 |
| `A4_loop_mamba_moe` | `[ME]^2M*E[ME]^2[ME]^2M*E` | 12/18 | 277.9M | 1.220 |
| `A5_loop_mamba_attention` | `ME[M*]^3EMEME[M*]^3E` | 12/20 | 262.4M | 1.261 |
| `A6_loop_attention_moe` | `MEM[*E]^2MEMEM[*E]^2` | 12/16 | 255.7M | 1.179 |
| `N1_anchor_mamba` | `MMMEMMM*EMMMEMMMEMMM*E` | 22/22 | 283.6M | 1.244 |
| `N2_anchor_moe` | `MEEM*EEMEEMEEM*EE` | 17/17 | 283.9M | 1.245 |
| `N3_anchor_attention` | `MEM****EMEMEM****E` | 18/18 | 246.0M | 1.244 |
| `N4_anchor_attention_moe` | `MEM*E*EMEMEM*E*E` | 16/16 | 255.7M | 1.179 |
| `A6a_..._per_iteration_norm` | `A6`'s, + own pre-norm per iteration | 12/16 | 255.7M | 1.179 |
| `A6b_..._input_injection` | `A6`'s, + input re-injected per iteration | 12/16 | 255.7M | 1.179 |
| `A6c_..._norm_and_injection` | `A6`'s, + both | 12/16 | 255.7M | 1.179 |

All `A*` arms are 1.105B parameters, except that `A6a` and `A6c` carry 4,096 extra per-iteration
norm parameters (section 7.3). Anchors are 1.163B / 2.043B / 1.125B / 1.487B respectively.

The last three arms differ from `A6` only in `loop_config`, not in `layer_pattern`; they are the
2x2 of section 7.4.

**Arms are matched on FLOPs, not loop count.** An extra MoE execution costs ~2x an extra Mamba one
and attention ~1.65x, so equal `K` across types would compare unequal budgets. Hence `[M]^3`,
`[E]^2`, `[*]^4`. The loop is also spread over *every* layer of the type, so "which layer got looped"
is not a confound.

Each anchor matches its loop arm **exactly** — same executed count of every layer type, same active
parameters, same FLOPs — differing only in shared vs fresh weights. Two comparisons, two questions:
*arm vs `A0`* (iso-parameter) asks whether looping buys usable compute; *arm vs its anchor*
(iso-FLOP) asks whether sharing is as good as fresh weights.

`A6` now has its anchor, `N4_anchor_attention_moe` (16 built, 1.487B, 1.179x, exact FLOP match).
`A4` and `A5` still have none, so for them only the iso-parameter comparison against `A0` is
available. `A4`'s anchor would be `MEMEM*EMEMEMEMEM*E` (18 built, 8 Mamba / 8 MoE / 2 attention
executions -- verified an exact match to `A4`); `A5`'s has not been worked out.

### 3.3 Results so far

Two 2-hour waves on 1x A100-80GB each, wandb project `nemo`. Wave 1: `A0`, `A1`, `A2`, `N2`.
Wave 2: `A3`, `N3`, `N1`, `A4`, `A5`. Wave 3 (below) was the first to run `A6`.

Train loss at matched steps (all arms use identical tokens/step and LR schedule, so matched step =
matched tokens = matched LR):

All eight arms that ran, at step 3000 (~197M tokens), which is inside every arm's range:

| Loop arm (1.105B) | loss | | Anchor | loss | params |
|---|---|---|---|---|---|
| `A0_baseline` | 3.72 | | `N1_anchor_mamba` | 3.68 | 1.163B |
| `A1_loop_mamba` | 3.72 | | `N2_anchor_moe` | 3.69 | 2.043B |
| `A2_loop_moe` | 3.72 | | `N3_anchor_attention` | 3.70 | 1.125B |
| `A3_loop_attention` | 3.71 | | | | |
| `A4_loop_mamba_moe` | 3.72 | | | | |
| `A5_loop_mamba_attention` | 3.71 | | | | |

**No signal between loop arms.** All six are within 0.01 of each other — below the seed-noise floor
(which has not been measured; see section 8.5). At ~200M tokens a 1.24x compute difference is far too
small to separate.

The one consistent pattern: **every anchor beats every loop arm** by 0.02-0.04, ordered by parameter
count (`N1` 3.68 at 1.163B, `N2` 3.69 at 2.043B, `N3` 3.70 at 1.125B). Fresh weights beat shared
weights on perplexity at matched FLOPs — exactly the published result, and exactly why reasoning
evals are needed before concluding anything.

On wall-clock, `A0` wins outright (3.55 at step 4980 vs 3.68-3.73 at ~3100) simply by doing more
steps per hour.

`A6_loop_attention_moe` had never been run before wave 3.

This is the expected outcome at this budget, and it is exactly what the literature predicts you
should see when measuring perplexity alone (see section 5).

### 3.4 Wave 3 -- KILLED at step ~935, no usable results

Launched 2026-08-07 16:16 on GPUs 1-4, `timeout 7200` (2 hours each), wandb project `nemo`.
**This was the first wave with the reasoning evaluations attached.** All four runs took a SIGTERM
at 16:46, half an hour into a planned two hours and ~600 steps short of anything comparable to
earlier waves: the launching shell went away and took the whole process group with it. Not a crash
and not an OOM -- the arms were healthy (`A6` at 19.1 samples/s, 33493 MiB, loss 4.34 at step 935).
Superseded by wave 4; `run_wave.sh` (section 3.6) exists so this cannot happen again.

### 3.4b Dense baselines -- the reference that was missing

Every comparison in sections 3.2-3.3 is hybrid-against-hybrid. They answer "where should depth go
*within* a Nemotron" and never "is the hybrid worth anything at all". Two vanilla dense
transformers (pre-norm, RoPE, SwiGLU, multi-head attention, tied embeddings) close that gap,
generated from the same base config by
[generate_dense_baseline_configs.py](../../config_files/nemotron/loop_ablation/generate_dense_baseline_configs.py)
so that data, order, step profile, LR schedule, `num_target_steps` and all six evaluations are
inherited rather than re-typed.

| Baseline | Shape | Params | Matched to `A0` on | samples/s | Peak |
|---|---|---|---|---|---|
| `D1_dense_flops_matched` | 1024 / 9 / ffn 2816 / 16 heads | 225.7M | **active** params -> same FLOPs/token | 34.5 | 19.9 GiB |
| `D2_dense_param_matched` | 2048 / 21 / ffn 5504 / 16 heads | 1.111B | **total** params -> ~4.9x FLOPs/token | 7.6 | 34.6 GiB |

"Same size" is ambiguous for a sparse model and the two readings give opposite answers, so both
exist. **`D1` is the one to lead with**: it costs what `A0` costs, so if the hybrid is worth its
complexity it has to beat `D1` at matched steps. `D2` spends ~4.9x the compute per token and is
expected to win; it prices what the sparsity buys back, and is not a controlled comparison.

Depth was solved for the parameter target at a fixed SwiGLU aspect ratio (`ffn_hidden` ~ 8/3
`n_embd`), varying depth rather than width so the two differ only in scale. Both land within 0.5%.
The parameter counts are **measured by instantiation, not derived** -- a hand-written closed form
for this stack was wrong by 8-16%, and the whole value of these runs rests on that number.

Note `D1` is 1.5x the throughput of `A0` (34.5 vs 23.3 samples/s solo) at equal FLOPs: dense
attention plus a plain SwiGLU hits far better hardware utilization (MFU 0.36 vs 0.22) than a
Mamba/MoE stack. If the hybrid loses on loss *and* loses on wall-clock, that is the headline.

### 3.5 Measured throughput and memory

Waves 1 and 2, 1x A100-80GB, seq 2048, micro batch 8, grad accum 4, full AC:

| Arm | Peak | samples/s | | Arm | Peak | samples/s |
|---|---|---|---|---|---|---|
| `A0` | 33366 MiB | 23.2 | | `N1` | 34572 MiB | 16.3 |
| `A1` | 33686 MiB | 16.3 | | `N2` | 47840 MiB | 15.2 |
| `A2` | 33526 MiB | 16.0 | | `N3` | 33876 MiB | 21.0 |
| `A3` | 33558 MiB | 21.1 | | | | |
| `A4` | 33559 MiB | 16.7 | | | | |
| `A5` | 33622 MiB | 19.3 | | | | |

Each loop arm runs at its anchor's speed (16.3/16.3, 16.0/15.2, 21.1/21.0), so weight sharing is
free in wall-clock and any quality difference is attributable to sharing, not a speed handicap.
What the loop saves is memory, and only where layers are heavy: **`A2` saves 14.3 GiB over `N2`**,
while `A1` saves 0.9 and `A3` saves 0.3.

---

### 3.6 Waves 4-6: the current results

**Wave 4 was discarded.** Launched with `num_target_steps` still derived from the 2B-token training
file (30,517 steps) while the runs were stopped by wall clock at 5-11k, so the cosine never
decayed: measured LR was still 4.32e-4 (96% of `max_lr`) at step 4750 and the loss flattened at the
high-LR floor (3.90 at 2k, 3.72 at 3k, 3.63 at 4750). **This is the single largest configuration
error the study has made**, and it is worse for the ablation than for the absolute number: at a
constant peak LR every arm sits on the same noisy plateau, which is exactly the "all six arms
within 0.01" non-result of wave 1. Fixed by pinning `num_target_steps` to a literal 8500 (and
`num_target_tokens` to 8500 x 65,536 = 557,056,000; the two must agree or
`enforce_tokens_per_step_consistency` rejects the config).

**Wave 5** re-ran the same eight arms with the completed schedule, 12h, two arms per GPU on 4
A100s. **Wave 6** added the dense baselines. Both reached 8500 steps = **557M tokens**.

Final held-out LM loss (`test` split) and reasoning metrics, all at step 8500:

| Run | test | `p_hop_1` acc | `p_hop_1` nll | `p_hop_2` nll | `p_hop_3` nll | `bind_3` nll | MATH | TriviaQA |
|---|---|---|---|---|---|---|---|---|
| `N4_anchor_attention_moe` | **3.302** | **0.114** | 3.228 | 3.655 | 3.575 | 3.484 | **4.082** | **6.229** |
| `A5_loop_mamba_attention` | 3.317 | 0.091 | 3.314 | 3.690 | 3.634 | 3.560 | 4.095 | 6.492 |
| `A4_loop_mamba_moe` | 3.322 | 0.073 | 3.478 | 3.724 | 3.682 | 3.585 | 4.175 | 6.429 |
| `A6_loop_attention_moe` | 3.324 | 0.073 | 3.305 | 3.583 | 3.540 | 3.420 | 4.157 | 6.434 |
| `A6a_..._per_iteration_norm` | 3.325 | 0.108 | **3.195** | 3.568 | 3.516 | 3.428 | 4.187 | 6.465 |
| `A6c_..._norm_and_injection` | 3.330 | 0.079 | 3.302 | **3.552** | **3.514** | 3.631 | 4.210 | 6.433 |
| `A6b_..._input_injection` | 3.334 | 0.088 | 3.302 | 3.614 | 3.575 | 3.481 | 4.226 | 6.506 |
| `A0_baseline` | 3.340 | 0.086 | 3.303 | 3.638 | 3.575 | 3.463 | 4.141 | 6.436 |
| `D1_dense_flops_matched` | 3.538 | 0.010 | 6.932 | 6.970 | 6.968 | 3.509 | 4.425 | 6.740 |

Chance accuracy is 1/26 = 0.0385; ln(26) = 3.258 is the nll of a model that knows only the answer
space. Standard error on an accuracy near 0.10 over 2048 questions is ~0.007.

**Six findings.**

1. **The hybrid beats an iso-FLOP dense transformer by 0.20 nats** (`A0` 3.340 vs `D1` 3.538). This
   is the largest effect in the study and the comparison that had been missing entirely. Note the
   trade: `D1` runs at 1.5x `A0`'s throughput at equal FLOPs (34.5 vs 23.3 samples/s solo, MFU 0.36
   vs 0.22), so per unit of *wall clock* the gap is much smaller than per unit of FLOPs.
2. **Every loop arm now edges out `A0`** by 0.006-0.023 nats, where waves 1-2 had them tied. The
   completed cosine is what surfaced this. The spread is the same size as the unmeasured seed
   noise, so it is suggestive and nothing more.
3. **The anchor still wins on perplexity** (`N4` 3.302, best of everything). Fresh weights beat
   shared weights at matched FLOPs -- the published result, reproduced.
4. **Per-iteration norms are the first real signal from work item B.** `A6a` moves `p_hop_1` from
   0.073 to 0.108 (+3.9 sigma) *and* posts the best `p_hop_1` nll of any arm (3.195). Two
   independent metrics agreeing is what makes it worth chasing. Input injection does nothing
   (`A6b` 0.088), and combining the two is worse than the norm alone (`A6c` 0.079).
5. **Block-loop is worst, exactly as the literature predicts.** `A4_loop_mamba_moe` (repeat the
   stack) trails every layer/pair-loop arm on every reasoning metric.
6. **The predicted dissociation did not appear.** `minerva_math` and `triviaqa` were supposed to
   move in opposite directions if an arm buys reasoning without buying parameters. They do not --
   both track LM loss almost perfectly, with `N4` best and `D1` worst on all three. No arm is
   trading knowledge for reasoning at this budget.

**Two things that are not working and should change.**

* **`bind_3` has never produced signal in any wave.** Every arm sits at 0.021-0.041, at or below
  its own 10% format-aware floor. Drop it; it costs evaluation time and returns nothing.
* **`p_hop_*` at `prompt_length: 256` is mis-scaled.** The query symbol appears ~9.8 times in the
  prefix (255/26), so even a perfectly functioning induction head returns a mixture over ~10
  candidate successors and lands near 10%. That is precisely where the best arms sit, which means
  the ceiling being measured may be the task's, not the model's. Add a `prompt_length: 64` variant
  (~2.4 occurrences, near-unique match) as an *additional* dataloader rather than a replacement, so
  every existing number stays comparable.

**The dense `p_hop` collapse is unexplained.** `D1` is 13 sigma *below* chance (0.010 vs 0.0385)
with nll ~6.93, while scoring normally on `bind_3` -- same alphabet, same collator, same metric --
and on both real benchmarks. So it is not the eval plumbing, not the alphabet, and not target
alignment. Leading hypothesis is anti-induction: a text-trained transformer learns to down-weight
in-context repeats, and on 256 tokens of bare symbol soup with no format cues it never collapses
onto the 26 symbols present, while Mamba's recurrent state tracks the in-context distribution for
free (every hybrid lands at ~ln 26). Unconfirmed, and unconfirmable without inspecting predictions
-- see the checkpointing item in section 9.

**The `D2` learning-rate question is closed.** At matched step 3250: `max_lr` 2.25e-4 -> 3.70 (grad
norm 0.42), 4.5e-4 -> 3.70 (0.35), 9.0e-4 -> 4.47 (2.43). Halving changes nothing, doubling breaks.
The inherited 4.5e-4 is near the top of the usable range but not over it, and is not what limits
these runs.

### 3.7 The seed-noise floor -- measured, and it retracts a result

**This is the measurement section 8.5 has demanded from the start, and it should have been the
first thing run.** Four runs of `A6a` from identical configs -- the initializer seed defaults to
`None` and `run_arm.sh` passes no `--global_seed`, so weights differ per run while data order stays
fixed at sampler seed 42 -- 8500 steps each:

| metric | mean | sd | range |
|---|---|---|---|
| `test` LM loss | 3.3246 | **0.0005** | 0.0011 |
| `p_hop_1` accuracy | 0.0952 | **0.0172** | 0.0708 - 0.1079 |
| `p_hop_1` nll | 3.276 | 0.114 | 0.245 |
| `p_hop_2` nll | 3.645 | 0.112 | 0.244 |
| `p_hop_3` nll | 3.595 | 0.112 | 0.239 |

**Two consequences, pointing in opposite directions.**

1. **The reasoning metrics cannot resolve anything at this budget, and section 7.5's headline is
   withdrawn.** `A6a`'s reported 0.108 was the *top* of its own seed range, and `A6`'s 0.073 sits
   essentially at the *bottom* of it (0.0708). Difference of means 0.022 against a combined
   standard error of ~0.019 is 1.2 sigma. Per-iteration norms have **no demonstrated effect**. The
   error that produced the original claim is worth remembering: the reported +3.9 sigma used the
   *sampling* standard error over 2048 questions (0.007), which understates run-to-run noise
   (0.017) by 2.5x. Sampling error is a lower bound on measurement error, never a substitute for
   it. And more questions cannot fix it -- the variance is in the model, not the question sample.
2. **The LM-loss differences are real, and by a wide margin.** At sd 0.0005 every ordering in
   section 3.6 sits far outside noise: `N4` vs `A0` (0.038) is ~76 sd, and even the smallest gap,
   `A6b` vs `A0` (0.006), is ~12 sd. So **every loop arm genuinely beats `A0`**, the **anchor
   genuinely beats every loop arm**, and the **hybrid genuinely beats the iso-FLOP dense model by
   0.20 nats**. Held-out perplexity, not the reasoning suite, is currently the only instrument here
   with the resolution to separate arms.

Caveat: these four runs vary weight initialization only, with data order fixed. Run-to-run noise
under a varying data seed can only be larger, so treat every figure as a lower bound. Within this
study's protocol -- all arms share sampler seed 42 -- it is nonetheless the right floor.

**How many seeds would the reasoning suite need?** At sd 0.017, resolving a 0.02 difference in
`p_hop_1` accuracy at 2 sigma takes roughly 12 runs per arm. A 16-arm sweep at 12 seeds each is not
affordable, so either the token budget grows until the metric stabilizes, or the reasoning claims
give way to perplexity.

### 3.8 `D2` closes the dense-collapse question

`D2_dense_param_matched` (1.111B dense, ~4.9x `A0`'s FLOPs/token) finished at **test 3.259**,
better than every hybrid including the anchor -- which is what 4.9x the compute should buy, and is
not a controlled comparison.

The important number is elsewhere: **`D2` scores `p_hop_1` 0.075 accuracy at nll 4.06 and does not
collapse**, where `D1` (225M dense) sat at 0.010 accuracy and nll 6.93, 13 sigma *below* chance.
Same architecture, same code path, opposite behaviour -- so the collapse is **a scale-and-budget
effect, not an architecture-class property and not a bug**. A small dense transformer at 557M
tokens has not formed the in-context machinery to notice that the answer must be one of the 26
symbols present; a larger one given ~4.9x the compute has. This removes the need for the
checkpoint-and-dump diagnostic that section 3.6 called for.

`D2`'s `p_hop_1` accuracy lands inside the hybrid seed range from 3.7, while its nll (4.06) stays
well above the hybrids' ~3.2-3.3: it knows less about the answer space than any hybrid, but guesses
within it about as well.

### 3.9 `A1` / `A2` / `A3` -- IN FLIGHT locally, results not yet collected

Launched 2026-08-11, one arm per GPU on 4/6/7, 8500 steps (557M tokens), the same budget and
schedule as waves 5-6, so these slot directly into the table in section 3.6. Logs in
`results/wave7_<arm>.log`.

| Arm | Looped | K | Built/exec | samples/s | ETA |
|---|---|---|---|---|---|
| `A1_loop_mamba` | every Mamba layer | 3 | 12/22 | 16.0 | ~4.7 h |
| `A2_loop_moe` | every MoE layer | 2 | 12/17 | 15.4 | ~4.9 h |
| `A3_loop_attention` | every attention layer | 4 | 12/18 | 20.4 | ~3.7 h |

**These are the three single-layer-type loops -- the arms that answer the titular question most
directly**, and their absence is the main gap in section 3.6, where only mixed-block loops appear.
Read them against the 0.0005 seed s.d. on LM loss (differences above ~0.002 are real) and ignore
the reasoning columns, where the noise floor swamps everything.

Their anchors `N1`/`N2`/`N3` have still never run with evaluations, so the sharing-versus-fresh-
weights comparison is not available for these three at 557M; it is positions 5-7 of the cluster
queue (section 9.6).

## 4. Operational facts worth not rediscovering

**`torch.bincount` is a performance bug.** [moe.py](../../src/modalities/models/components/moe/moe.py)
computes `tokens_per_expert` with `torch.bincount`, whose output shape depends on input *data*. At
production shape this measured **4983 us/call vs 114 us** for an equivalent
`torch.zeros(E).scatter_add_(0, idx, ones)` — 44x — and the CPU-side time equals the total, meaning
it forces a **device-host sync** on every MoE layer, every forward. The replacement is verified
bit-identical (`torch.equal`). It was **reverted** to `bincount` so that all ablation arms share the
code that produced wave 1. **Reapply it once the sweep is done.**

**`torch.compile` works and is numerically sound, but is currently parked.** Graph breaks per block:
Mamba `native` 0, Mamba `fused` 11 (mamba-ssm's Triton kernels are opaque to dynamo), attention 0,
MoE 3 — all `aten.bincount`, which the `scatter_add` fix above reduces to **0**. In fp32 the
eager-vs-compiled logit difference is `1.8e-6` relative — pure bf16 rounding, not a correctness bug.
In bf16 it is 10% of logit std for the baseline and **22% for a looped MoE arm** (more executions,
more accumulated rounding), so a compiled arm is not bit-comparable to an uncompiled one.

**The fastest known configuration is ~1.56x the current one**, measured end-to-end (36.1 vs 23.2
samples/s, MFU 0.34 vs 0.22, but 61.4 GiB vs 32.6): keep `ssd_backend: fused`, add the `compiled`
model wrapper with `fullgraph: false`, drop activation checkpointing, and use micro batch 16 with
gradient accumulation 2 (preserving the global batch of 32). Full sweep, all fused+compiled:

| AC | bs 8 | bs 16 | bs 24 |
|---|---|---|---|
| full | 31.0 s/s, 33 GiB | 33.7, 46 | 34.7, 58 |
| selective f=2 | 34.8, 37 | 37.7, 52 | 38.7, 68 |
| selective f=4 | 36.6, 39 | 39.7, 56 | 40.6, 73 |
| none | 38.5, 40 | **41.6, 58** | 42.7, 77 |

Returns flatten past bs 16. **`N2_anchor_moe` (2.043B) will OOM with the fast settings** — it already
peaks at 47.8 GiB on the conservative config and the fast config adds ~29 GiB. Use `selective f=2`
at micro batch 8 for the parameter-heavy anchors.

`native` SSD is a trap: 0.61x eager, 0.79x compiled. It is the only way to compile Mamba fullgraph
but is far slower overall.

**The MoE layer is non-deterministic.** `index_add_` uses CUDA atomics, so two identical forward
passes differ (max_abs ~2.0 on logits with std 0.577). A model with no MoE layers is bit-exact.
`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes it exact.
**Any numerical comparison must include a same-code control run**, or run-to-run noise will be
mistaken for a real effect.

**The GPT-2 MFU calculator's config key is `wrapped_model`, never `model_parts`.**
`GPT2MFUCalculatorConfig` carries `@add_deprecated_alias("model_parts", "wrapped_model")`
([config.py](../../src/modalities/config/config.py)), and that decorator sets
`validation_alias=AliasChoices("wrapped_model")` *without* including the field's own name and
without `populate_by_name`. The supposedly deprecated spelling is therefore the only one pydantic
accepts. `NemotronMFUCalculatorConfig` is undecorated and takes `model_parts`, so the two
calculators disagree, and the failure lands minutes into a run rather than at parse time. The FSDP
gradient clipper configs carry the same decorator.

**`pytest` is not in the runtime install**, and installing it from the project directory fails —
`uv` tries to resolve the project's `extra-build-dependencies` (which pin `torch` with
`match-runtime`) and reports `torch ... was not found in the resolution`. Install from outside:

```bash
cd /tmp && uv pip install --python /path/to/modalities/.venv/bin/python --only-binary :all: pytest
```

**Disk was at 96% (74 GB free).** Checkpointing is disabled in the base config
(`checkpointing_interval_in_steps: 50000`, above the ~30.5k total) because a 1.1B DCP checkpoint is
~13 GiB and `N2` ~24 GiB. Lower it for real runs on a machine with room.

**Naming collision:** the MoE `experts_backend: looped` option is an expert *matmul kernel* choice,
unrelated to layer-pattern loops.

---

## 5. Literature summary

Verify headline numbers against the papers before quoting; these were read via automated extraction.

**The consensus result: looping loses on perplexity, wins on reasoning.**
[Saunshi et al.](https://arxiv.org/pdf/2502.17416) (1B, 250B tokens) found looped models had *worse*
perplexity (7.8-8.8 vs 7.4) but better reasoning (math word problems 34.3% vs 29.3%) with 50% fewer
parameters. [Looped State-Space LMs](https://arxiv.org/html/2607.10110) reproduce this for Mamba
exactly: looped Mamba crushes parameter-matched baselines (99.35% vs 7.50% on their hardest reasoning
task) but *"deeper non-looped models retain an advantage in validation perplexity under strict
iso-FLOPs comparisons."* **This is why section 6 exists** — measuring loss alone will show every loop
arm losing.

**The exchange rate is quantified.**
[How Much Is One Recurrence Worth?](https://arxiv.org/html/2604.21106) fits
`L = E + A(N_once + r^φ N_rec)^-α + B D^-β` and measures **φ = 0.46**: one recurrence buys about half
a unique block's capacity while costing a full block's FLOPs. Iso-FLOP loss gaps: r=2 → 0.03-0.06
nats, r=4 → 0.05-0.08, r=8 → 0.09-0.12. Keep **r ≤ 4**. Hyperconnections raise φ to **0.65**;
truncated backprop lowers it to **0.38** (looks like a speedup, silently costs capacity).

**What beat iso-FLOP.** [Loopie](https://arxiv.org/html/2607.16051) — a looped **MoE** family —
matches/exceeds a vanilla **30B-A3B** transformer (essentially our reference model) at equal
wall-clock with 20B-A2B, overtaking after ~600B tokens; their ablation removing the loop at fixed
compute *and* parameters degraded 8 benchmarks. [Adaptive Loops and Memory](https://arxiv.org/abs/2603.08391)
beat an iso-FLOP baseline with **3x the layers** on math. [Tied Expert Layers in MoE](https://arxiv.org/pdf/2606.16825)
confirms routers **do** re-route across iterations, and that this is what makes tied experts work.

**Layer-loop beats block-loop, specifically on MoE backbones.** Loopie: `1→1→2→2` (repeat each layer
in place) beats `1→2→1→2` (repeat the stack, what Universal Transformer and Ouro do), citing better
execution locality and parameter-sharing coherence. They also report **R=2 gives the best marginal
return**, with R=4+ diminishing. Our `A1`/`A2`/`A3`/`A6` are layer-loop or pair-loop; `A4`/`A5` are
block-loop, and the literature predicts they do worse.

**Established tricks**, roughly by evidence strength. Prerequisites for scaling past 2-3 loops:
input injection; per-iteration norms ([LayerNorm as implicit gain control](https://arxiv.org/html/2607.10681));
`1/N` residual scaling across loop count; spectral/Jacobian regularization
([STARS](https://arxiv.org/abs/2605.26733), which targets the characteristic "peak then collapse"
with more loops). Capacity/efficiency: hyperconnections; timestep/depth conditioning; adaptive depth
(Mixture-of-Recursions); random loop-count sampling for test-time scaling. Traps: truncated BPTT, and
the [readout blind spot](https://arxiv.org/pdf/2606.24898) (the readout must be trained on
intermediate hidden states).

**A cheap alternative worth remembering:** Saunshi et al. add a cosine-similarity regularizer pushing
successive blocks toward alignment (λ=10, k=4, ~0.98 cos sim). This kept baseline perplexity
(7.38-7.51) *while* improving math (36.4% vs 29.3%) — the reasoning benefit of looping without the
perplexity cost, in a normal non-looped model.

---

## 6. Work item A: reasoning evaluations — **IMPLEMENTED, NOT YET RUN**

**Why.** Every published result says the loop effect lives in reasoning, not perplexity. The first
two waves measured training loss only and therefore systematically under-reported the effect — the
result in section 3.3 ("no signal between loop arms, every anchor beats every loop arm") is what
loss alone is guaranteed to show.

**What was built.** Four held-out synthetic evaluations, attached to the base config and so to every
generated arm. Run natively in modalities rather than via an HF conversion: only GPT-2 has a
converter in this repository, and porting the hybrid Mamba/MoE/attention stack *plus* the loop
schedule to HF would be a large, correctness-critical detour whose failure mode is a silently wrong
number. Running natively also evaluates at matched steps during training, which is what this
comparison needs.

| Dataloader tag | Task | Role |
|---|---|---|
| `p_hop_1` | p-hop induction, 1 hop | Control: one lookup, no depth needed |
| `p_hop_2` | p-hop induction, 2 hops | Two serial lookups |
| `p_hop_3` | p-hop induction, 3 hops | Three serial lookups |
| `bind_3` | variable binding, 3 hops | Three lookups over *shuffled* statements, so recency does not help |
| `minerva_math` | Minerva MATH, 4-shot, NLL | Real reasoning benchmark; where looping should help |
| `triviaqa` | closed-book TriviaQA, NLL | Parametric-knowledge control; where looping should *lose* |

**Two dissociations, not six numbers.** Across the hop ladder, an arm that is simply better moves
all the synthetic tasks together while an arm that buys usable depth moves `p_hop_2`/`p_hop_3` and
leaves `p_hop_1` alone. Across the two benchmarks, `minerva_math` and `triviaqa` should move in
*opposite* directions: the claim is that weight sharing buys reasoning without buying parameters,
and closed-book recall lives in weights a loop does not add. An arm that improves both is just
training better, which is exactly what the control is there to catch. This discharges item 3 of the
original plan (a parametric-knowledge probe as control).

One caveat that changes how `bind_3` is read: its answer is always a symbol that appears as a
value but never as a variable, and there are exactly `num_distractors + 1` of those, so a
format-aware guesser reaches **0.10** without following the chain. The floor is identical across
arms and so does not bias the comparison, but an arm sitting at 10% on `bind_3` has shown
nothing. `p_hop_*` admits every symbol as an answer and therefore has no such shortcut, which is
why it carries the ladder and `bind_3` is corroboration.

**Accuracy is meaningless on the two real benchmarks** and only NLL should be read there: at 1.1B
parameters and a few billion tokens a model scores ~zero on both, so solve rate is a constant, not
a measurement. `minerva_math` is lm-evaluation-harness's `minerva_math` task -- Minerva's prompt and
verbatim 4-shot prefix over the Hendrycks MATH test split -- not plain MATH. Its SymPy answer
checking scores a *generated* answer for exact match and has no counterpart in a likelihood
evaluation, so it is unused. Both are tokenized once offline so every arm sees byte-identical
sequences; MATH's NLL is diluted ~10x by prose and LaTeX, which costs sensitivity but not validity.

Two metrics per dataloader, logged as `<tag> answer_accuracy` and `<tag> answer_nll`. `answer_nll`
excludes the MoE auxiliary loss term, whose size depends on how often the MoE layers are visited --
i.e. on exactly what the loop arms vary. Measured on a smoke run, that contamination is ~3e-4 nats,
about 1% of published effect sizes.

**Files.**

| File | What it holds |
|---|---|
| [synthetic_reasoning.py](../../src/modalities/dataloader/synthetic_reasoning.py) | `SyntheticReasoningDataset`, both task generators, `resolve_p_hop` |
| [prepared_eval.py](../../src/modalities/dataloader/prepared_eval.py) | `PreparedEvalDataset`, loading the pre-tokenized benchmarks |
| [prepare_text_evals.py](../../config_files/nemotron/loop_ablation/prepare_text_evals.py) | Builds the Minerva MATH and TriviaQA `.npz` files |
| [constants.py](../../src/modalities/constants.py) | `IGNORE_INDEX`, shared by the dataset and metric modules so neither imports the other |
| [evaluation_metrics.py](../../src/modalities/evaluation_metrics.py) | `EvaluationMetricIF`, `MaskedTokenAccuracy`, `MaskedTokenNLL` |
| [explicit_target_collator.py](../../src/modalities/dataloader/collate_fns/explicit_target_collator.py) | Passes the dataset's own masked targets through unshifted |
| [evaluator.py](../../src/modalities/evaluator.py) | Per-dataloader metric selection and exact numerator/denominator reduction |
| [derive_symbol_token_ids.py](../../config_files/nemotron/loop_ablation/derive_symbol_token_ids.py) | Regenerates the symbol alphabet's token ids from the tokenizer |
| [test_synthetic_reasoning.py](../../tests/dataloader/test_synthetic_reasoning.py), [test_evaluation_metrics.py](../../tests/test_evaluation_metrics.py), [test_reasoning_evaluation.py](../../tests/models/nemotron/test_reasoning_evaluation.py) | 30 tests, including p-hop answers against an independent brute-force reference and an end-to-end run on a looped model |

`eval_metrics` is a new optional top-level config key, defaulting to empty, so every pre-existing
config is unaffected. Two renames landed with it: `IGNORE_INDEX` moved out of
`synthetic_reasoning.py` into `constants.py` (the generic metric module was importing it from a
specific dataset module), and the config's `reasoning_collate_fn` became `masked_target_collate_fn`
(TriviaQA uses it and is a knowledge control, not a reasoning task). Both are semantics-free; wave 3
was launched from configs using the old collator name, which differ from the regenerated ones by
that name alone. The user-facing reference is
[nemotron_loops.md](nemotron_loops.md#reasoning-evaluations).

**What was deliberately not built.** GSM8K. At 1.1B parameters and ~2B tokens a model is at chance
on it, so it would cost a data dependency and return no signal. Revisit only if an arm wins on the
synthetic ladder and the result needs to be quotable against published benchmarks — at which point
the HF port becomes worth its cost.

**Verified on GPU** (`A2_loop_moe`, 1x A100, fused Mamba kernels, 2026-08-07). Two smoke runs, one
before the benchmarks were added and one after:

* Every dataloader runs, including `bind_3` at 51 tokens -- well under the Mamba `chunk_size` of
  128, which was the one thing CPU tests could not check. No special handling needed.
* Masking is correct end to end: on the synthetic dataloaders the reported loss and `answer_nll`
  agree to ~3 decimal places, which only holds if exactly the answer position is scored.
* Metrics are correctly tag-filtered: the language-modelling `test` split reports none.
* **Attaching all six evaluations costs no extra peak memory**: 33525 MiB, the same as without
  them. `minerva_math` is the only one that comes close to the language-modelling split's
  footprint, which is why its batch size is 4.

`answer_nll` after 2M tokens, i.e. what an essentially untrained model looks like:

| step | `test` loss | `p_hop_1` | `p_hop_2` | `p_hop_3` | `bind_3` | `minerva_math` | `triviaqa` |
|---|---|---|---|---|---|---|---|
| 15 | 7.649 | 8.384 | 8.352 | 8.359 | 8.014 | 9.609 | 10.843 |
| 30 | 7.303 | 8.592 | 8.559 | 8.566 | 7.859 | 9.388 | 10.739 |

The ordering is the sanity check that the masking lands on the intended spans: ordinary text (7.3)
is easiest, MATH solutions (9.4) are structured LaTeX prose, and a TriviaQA answer (10.7) is a
proper noun the model must produce from nothing. Synthetic accuracy is 0.0000 throughout, as
expected this early; `p_hop_*` near 8.6 means the model still spreads mass over the full 128k
vocabulary rather than the 26-symbol alphabet (3.26 would be uniform over it), so there is headroom
in the right direction.

**Use `answer_nll`, not the reported loss.** Two measured reasons. The configured objective is a
`weighted_sum` including the MoE auxiliary term, whose size depends on how often the MoE layers are
visited -- i.e. on exactly what the loop arms vary -- measured at ~3e-4 nats. More seriously, the
loss is a mean of per-batch means while `answer_nll` weights each scored token equally: on
`minerva_math`, where a solution may be 20 tokens or 400, the two differ by **0.21 nats** (9.397 vs
9.609 at step 15). That is roughly 7x the published iso-FLOP effect sizes, and it is the artifact
the numerator/denominator reduction exists to remove.

---

## 7. Work item B: per-iteration norms and input injection — **IMPLEMENTED**

**Why.** Both are described in the literature as near-prerequisites for scaling past 2-3 loops.
`A1` runs r=3 and `A3` runs r=4 with fully shared norms and no injection, so those arms may be
handicapped in a way that has nothing to do with the layer type — a confound in the current design.

They were deliberately left out so the first sweep isolated weight sharing alone: a meaningful
fraction of reported Universal Transformer gains come from per-iteration conditioning rather than
from sharing, and conflating the two would answer the wrong question. Now that the clean arms exist,
both are **explicit toggles**, defaulting to off so every pre-existing config describes exactly the
model it described before.

```yaml
loop_config:
  variant: simple
  per_iteration_norm: true     # own pre-norm per loop iteration
  input_injection: true        # re-inject the group's input each iteration
  injection_mode: add          # the only mode implemented; see 7.5
```

The user-facing reference is
[nemotron_loops.md](nemotron_loops.md#per-iteration-norms-and-input-injection).

### 7.1 What was built

| File | What changed |
|---|---|
| [nemotron_layers.py](../../src/modalities/models/nemotron/nemotron_layers.py) | `PerIterationNorm`; `_ResidualLayer.forward` takes an `iteration` index |
| [nemotron_layer_specs.py](../../src/modalities/models/nemotron/nemotron_layer_specs.py) | `build_norm`; every spec's `build` takes `num_norm_iterations` |
| [nemotron_model.py](../../src/modalities/models/nemotron/nemotron_model.py) | `LoopConfig` fields, `_run_loop_group`, `num_per_iteration_norm_parameters` |
| [run_wave.sh](../../config_files/nemotron/loop_ablation/run_wave.sh) | Launches several arms per GPU, detached (see 3.6) |
| [test_layer_loops.py](../../tests/models/nemotron/test_layer_loops.py) | 15 further tests, including hand-computed references for both refinements |

The model, not the spec, decides how many norms a layer gets: it already holds the schedule, so it
passes each layer the loop count of the group that layer belongs to. Layers outside a loop group
therefore get one norm and identical parameter names, which is why the flag is a no-op on `A0`.

**Input injection is applied *before* each iteration after the first, not after each iteration.**
The sketch this section originally carried added the anchor after every iteration, which would also
fire for non-looped groups — every plain layer in the model would get its input added back a second
time, and `A0_baseline` would change the moment the flag was set. Injecting beforehand makes a
single-iteration group an exact no-op, which is the property the toggle needs.

### 7.2 The trap, and what was done about it

Per-iteration norms create new parameters with new names. A naive `self.norms = nn.ModuleList(...)`
would produce `transformer.h.0.norms.0.weight`, which does **not** match the weight-decay regex
`r"\.norm\."` (the next character is `s`, not `.`).

The consequence is worse than the section originally claimed. A parameter matching no weight-decay
group is not merely decayed by accident — `_build_optimizer_groups_via_weight_decay_split` in
[optimizer_factory.py](../../src/modalities/optimizers/optimizer_factory.py) assembles both
optimizer groups *from the group members*, so an unmatched parameter is handed to the optimizer in
neither group and is **never updated at all**.

Both belts were fastened: the module is stored under the layer's existing `norm` attribute, giving
`transformer.h.<idx>.norm.norms.<iteration>.weight` which the existing regex matches, *and*
`r"\.norms\."` was added to the `layernorm` group. `test_every_parameter_lands_in_exactly_one_weight_decay_group`
asserts exactly-one-group membership for every parameter, so both failure directions (no group, two
groups — the latter makes torch reject the optimizer) are covered.

The initialization filters were checked and are a non-issue, as suspected: they match only linear
and embedding weights, and norms are initialized to ones at instantiation. Verified end to end —
after running the `scaled` Nemotron initializer over a per-iteration-norm model, every norm weight
is still 1.

### 7.3 Parameter accounting

Per-iteration norms add `(K − 1) · n_embd` per looped layer — iteration 0 reuses the norm the layer
would have had anyway. For the `A6a`/`A6c` arms that is 2 groups × 2 layers × 1 × 1024 = **4,096
parameters** of 1.105B. Reported three ways: the generated config's banner prints it,
`NemotronLLM.num_per_iteration_norm_parameters` returns it at runtime, and the measured peak memory
differs by 0.12 MiB (33493.73 vs 33493.61 MiB). Input injection with `add` adds none.

The MFU calculator charges all `K` norms on every iteration rather than the one visited,
overcounting active parameters by `K(K−1)·n_embd` per looped layer — 8,192 of 255M for `A6a`. Left
alone deliberately: it is ~0.003%, far below the accuracy of the FLOPs estimate, and a special case
here would have to be kept in sync with the layer internals.

### 7.4 Arms added — the 2x2 on `A6`, not `A2`

| Arm | `per_iteration_norm` | `input_injection` |
|---|---|---|
| `A6_loop_attention_moe` | off | off |
| `A6a_loop_attention_moe_per_iteration_norm` | on | off |
| `A6b_loop_attention_moe_input_injection` | off | on |
| `A6c_loop_attention_moe_norm_and_injection` | on | on |

`A6` rather than `A2` because `A6` loops the classical attention-then-feed-forward block — the
shape both refinements were reported for — and because it is the only loop arm with an exact
iso-FLOP anchor (`N4`), so a gain from the refinements can be read against fresh weights and not
only against `A0`. All four share `A6`'s pattern and therefore its FLOPs.

### 7.5 Result (wave 5, 557M tokens)

| Arm | `per_iteration_norm` | `input_injection` | test loss | `p_hop_1` acc | `p_hop_1` nll |
|---|---|---|---|---|---|
| `A6` | off | off | 3.324 | 0.073 | 3.305 |
| `A6a` | **on** | off | 3.325 | **0.108** | **3.195** |
| `A6b` | off | **on** | 3.334 | 0.088 | 3.302 |
| `A6c` | **on** | **on** | 3.330 | 0.079 | 3.302 |

**WITHDRAWN -- see section 3.7.** The original reading of this table was that per-iteration norms
work: `A6a` gained +3.9 sigma of `p_hop_1` accuracy over `A6` and posted the best `p_hop_1` nll in
the study. Four seed replicates of `A6a` later showed sd 0.017 on that accuracy, with 0.108 the
*top* of its own range and `A6`'s 0.073 essentially at the *bottom*. The difference is 1.2 sigma.
**Per-iteration norms have no demonstrated effect, and neither does input injection.**

What made the claim look strong was using the *sampling* standard error over 2048 questions (0.007)
in place of run-to-run noise (0.017). Note that held-out LM loss is genuinely unchanged across the
2x2 (3.324-3.334 against a seed sd of 0.0005, so those small differences are themselves real) --
what collapsed is the reasoning signal, not the perplexity one.

### 7.6 What this predicts, and how to test it

If per-iteration norms work because they let the residual stream be renormalized between
iterations, the gain must **grow with the loop count**. `A6a` is r=2. `A1` runs r=3 and `A3` runs
r=4 with fully shared norms -- the two arms section 7 was originally motivated by. Adding
`A1a`/`A3a` turns one suggestive number into a mechanism with a dose-response curve, and it is
config-only work: `generate_arm_configs.py` already takes the flag.

### 7.7 What was deliberately not built

`injection_mode: concat_proj`. Its projection is a property of the *group*, not of any one layer,
so it would have to live outside `transformer.h` — which is the one invariant
[section 2.3](#23-the-critical-design-decision) exists to protect — and would need its own entries
in the weight-decay groups and the initialization filters. At ~2·`n_embd`² per looped group (10.5M
parameters for a five-group arm, ~1% of the model) it also breaks the iso-parameter comparison the
ablation rests on. `LoopConfig` rejects it with a message rather than silently ignoring it. Add it
only if an arm's result justifies the cost.

---

## 8. Work item C: fitting φ per layer type

**Why this is the good version of the question.** "Does `A1` beat `A0`" is a single noisy comparison.
φ is a *capacity exponent*: how many unique blocks one recurrence of a given layer type is worth. It
is comparable to published numbers (φ=0.46 dense baseline, 0.65 with hyperconnections, 0.38 with
truncated BPTT), and **φ_M vs φ_E vs φ_*** answers "where is depth worth spending" in a transferable
way rather than as one loss delta.

### 8.1 The model

```
L = E + A·(N_once + r^φ·N_rec)^(-α) + B·D^(-β)
```

`N_once` = parameters executed once per token (embeddings, LM head, all non-looped layers).
`N_rec` = parameters of the looped layers. `r` = loop count. `D` = tokens.
φ=1 would mean a recurrence is worth a full unique block; φ=0 means no capacity gain.

A full joint fit needs ~20 runs across four compute budgets. The two-stage method below is much
cheaper and gives the same φ.

### 8.2 Two-stage method (recommended)

**Stage 1 — reference capacity curve.** At a *fixed* token budget `D₀`, train `K` bracket-free
(non-looped) models of varying parameter count and fit

```
L(N) = C + A·N^(-α)
```

with `C`, `A`, `α` free. `D` is fixed, so `B·D^(-β)` folds into `C`.

Vary `N` by changing **depth only** (number of layers in the bracket-free pattern), holding the
`M:E:*` ratio and every width hyperparameter fixed. Changing width would alter the per-layer cost
ratios and confound the fit. Suggested: 8, 12, 16, 20, 24 layers → 5 runs.

**Stage 2 — per-type loop runs.** For each layer type `T ∈ {M, E, *}` and each `r ∈ {2, 3, 4}`, train
the arm that loops *every* layer of type `T` with count `r`, at the same `D₀`. 9 runs.

**Stage 3 — invert to effective parameters.** For each `(T, r)`, solve `L(N_eff) = L_{T,r}`:

```
N_eff(T, r) = ((L_{T,r} - C) / A)^(-1/α)
```

**Stage 4 — fit φ per type.** With `N_once(T)` and `N_rec(T)` known exactly from the model:

```
φ_T = log( (N_eff(T, r) - N_once(T)) / N_rec(T) ) / log(r)
```

Three `r` values give three estimates per type; fit by least squares in log space and **report the
spread**. A large spread means the power-law form does not hold for that layer type, which is itself
a finding.

### 8.3 Bookkeeping decisions that change the answer

These are judgment calls. Make them explicitly, state them in the writeup, and keep them consistent
between Stage 1 and Stage 3.

1. **Embeddings.** Include in `N_once` to stay comparable to published φ. At 131M of a 1.105B model
   they compress the dynamic range, so also report the non-embedding-only variant as a sensitivity
   check.
2. **Total vs active parameters.** This matters enormously for MoE and has no obviously right answer:
   an `E` layer is 187.5M total but 11.66M active. Capacity plausibly tracks *total* (experts store
   knowledge) while compute tracks *active*. **Recommendation:** fit with total parameters
   consistently, and report the active-parameter fit as a cross-check. If φ_E differs wildly between
   the two accountings, say so — that is a real result about MoE capacity, not a nuisance.
3. **Which loss.** Held-out validation loss, not training loss.

### 8.4 Budget

Runs must be long enough that arms separate. **Wave 1 showed arms within 0.01 nats at 200M tokens —
that is below the noise floor and useless for fitting.** The published φ study used ~5e19 FLOPs
across ~20 experiments, i.e. ~2.5e18 FLOPs per run. At 225M active parameters
(6 × 225M = 1.35 GFLOP/token) that is **~1.85B tokens per run**.

At the fast configuration (~36 samples/s × 2048 tokens = 74k tokens/s) that is ~7 hours per run.
14 runs ≈ 100 GPU-hours ≈ 12.5 hours on 8 GPUs. Feasible on a cluster, not on one node.

**Cheaper alternative:** run the whole φ study at a smaller scale (e.g. `n_embd` 512, ~300M total)
where runs are 1-2 hours, accepting that φ may be scale-dependent. Given φ is a ratio, it is
plausibly more scale-stable than absolute losses, but this is an assumption to state, not a fact.

### 8.5 Establish the noise floor first

**Before any of the above**, run `A0_baseline` with 2-3 different seeds at `D₀` and measure the
spread in final validation loss. Every φ estimate inherits this noise through the inversion, and
published iso-FLOP gaps are 0.03-0.06 nats at r=2 — if seed noise is that size, the study needs
either more tokens or averaging over seeds. This is 2-3 cheap runs that determine whether the
expensive ones are worth starting.

---

## 9. Cluster handoff: 5B tokens on a new dataset

Everything so far ran on one 8x A100 node at **557M tokens** on a 2B-token FineWeb shard. The next
wave targets **4 nodes x 4 GPUs = 16 GPUs**, **5B tokens**, and **a different dataset** (to be
supplied). All three of those change things, and the token budget changes the most.

### 9.1 Do these in order

1. Confirm the new dataset's path, size (must be >= 5B tokens) and **tokenizer** -- section 9.5.
2. Regenerate the tokenizer-dependent evaluation assets -- section 9.5. **Skipping this silently
   invalidates every evaluation number.**
3. Set the budget in the base config and regenerate all arms -- section 9.3.
4. Enable checkpointing and verify a warmstart round trip on a short run -- section 9.4.
   **Mandatory at this budget**, not optional.
5. Decide on the fast configuration -- section 9.3. All arms or none.
6. Launch one arm per GPU -- section 9.2, queue in 9.6.

### 9.2 One arm per GPU, not one arm across the cluster

**Run 16 independent 1-GPU jobs. Do not shard a single arm across ranks.**

The global batch is `local_micro_batch_size 8 x gradient_accumulation_steps 4 x dp_degree x 2048` =
**65,536 tokens/step at dp_degree 1**. Sharding one arm over 16 ranks makes that 1,048,576
tokens/step: a 16x larger batch is a different optimization problem, the LR would need re-tuning
from scratch, and every number in section 3.6 would stop being comparable, because "loss at matched
steps" only holds when a step means the same thing. Dropping `gradient_accumulation_steps` to 1
does not rescue it either -- 16 ranks x micro batch 8 is still 262,144 tokens/step.

One arm per GPU keeps `dp_degree: 1`, the batch at 65,536, the LR valid, and the comparison intact.
It is also simply the right shape: the queue is a list of independent single-GPU runs.
`run_arm.sh` and `run_wave.sh` already do this; on a cluster they need a scheduler wrapper (one job
per arm at 1 GPU, or one 4-GPU job per node running four arms), not a change to the training
config.

### 9.3 Budget: 5B tokens = 76,250 steps, and why the fast config is now mandatory

Fix the **token budget**, never the wall clock: arms differ in throughput by up to 1.5x, so equal
wall clock gives unequal training and the comparison silently becomes "which arm is fastest".

```yaml
num_target_tokens: 4997120000   # 76250 * 65536
num_target_steps: 76250
```

76,250 is a multiple of `evaluation_interval_in_steps: 250`, so the final step is evaluated. The
two values must satisfy `num_target_tokens == num_target_steps * 65536` or
`enforce_tokens_per_step_consistency` rejects the config.

**5B tokens is roughly compute-optimal** for a 225M-active model (Chinchilla ~4.5B), against the
~12% of optimal that 557M represented. This is the first budget at which conclusions are not
obviously budget-limited, and it is ~9x the current one.

Measured single-GPU wall clock at 76,250 steps:

| Arm | samples/s | current config | with the 1.56x fast config |
|---|---|---|---|
| `A2_loop_moe` (slowest) | 15.4 | **44.0 h** | 28.2 h |
| `A1_loop_mamba` | 16.0 | 42.4 h | 27.2 h |
| `A3_loop_attention` | 20.4 | 33.2 h | 21.3 h |
| `A0_baseline` | 23.3 | 29.1 h | 18.6 h |
| `D1_dense_flops_matched` | 34.5 | 19.6 h | 12.6 h |

**Every arm exceeds a 24h slot, and most exceed 12h.** Two consequences that did not apply at 557M:

* **Checkpoint-and-resume is required** (9.4).
* **The fast configuration stops being a nice-to-have.** Section 4 measured 1.56x end-to-end:
  keep `ssd_backend: fused`, add the `compiled` model wrapper with `fullgraph: false`, drop
  activation checkpointing, use micro batch 16 with `gradient_accumulation_steps: 2` (which
  preserves the 65,536-token global batch, so comparability is kept). It costs memory: 61.4 GiB vs
  32.6. **`N2_anchor_moe` (2.043B, already 47.8 GiB) will OOM** -- give it `selective f=2` at micro
  batch 8. Compiled runs are not bit-comparable to uncompiled ones (bf16 rounding differs by up to
  22% of logit std on a looped MoE arm), so apply it to **every arm in the wave or none**.
* Also reapply the **`scatter_add` fix** (section 4): 44x faster than `bincount` on that op,
  removes a device-host sync per MoE layer per forward, verified bit-identical.

**Warmup is a decision to make explicitly.** `warmup_steps: 750` is 8.8% of an 8,500-step run but
1.0% of a 76,250-step one. Recommend raising it to **1,500** (2%), which is conventional, and
recording that it changed -- it is another reason 5B-token runs are not step-comparable to the 557M
ones.

### 9.4 Resumable runs -- new, and required

At 29-44 h per arm no run finishes inside a normal slot, and **no run in this study has ever saved
a checkpoint** (`checkpointing_interval_in_steps: 50000` has always exceeded the run length). That
must change first.

```yaml
checkpointing_interval_in_steps: 5000    # ~12 checkpoints over 76,250 steps
```

Modalities supports warmstart natively: `modalities warmstart --config_file_path <cfg>
--last_checkpoint_info_file_path <path>`. A warmstart config differs from a training config in that
`settings.training_progress` is *derived from the checkpoint* rather than zeroed, via the
`global_num_seen_tokens_from_checkpoint_path`, `num_seen_steps_from_checkpoint_path` and
`last_step_from_checkpoint_path` number conversions, plus
`warmstart_checkpoint_paths: ${warmstart_env:checkpoint_paths}`. See
[config_lorem_ipsum_long_fsdp2_warmstart.yaml](../../config_files/training/config_lorem_ipsum_long_fsdp2_warmstart.yaml)
for the shape. `resumable_distributed_sampler` already takes `skip_num_global_samples` from
`training_progress.num_seen_samples`, so data order is preserved across a resume.

Two things to verify on a short run **before** launching 16 long ones, because both fail silently:

* the LR schedule resumes at the right point (`last_step` feeds the scheduler), and
* the dataloader skips the right number of samples, so a resumed run does not re-train on data it
  has already seen.

Disk: a 1.1B DCP checkpoint is ~13 GiB and `N2` ~24 GiB. Twelve checkpoints x 16 arms is far beyond
what the local box had (47 GB free); either keep only the most recent checkpoint per arm or confirm
cluster quota first. `generate_arm_configs.py` will need a matching warmstart-config renderer, or a
scheduler-side wrapper that resubmits with the warmstart entrypoint.

### 9.5 Dataset swap checklist -- the silent-failure surface

The new dataset changes more than a path. Everything below is tokenizer-dependent, and **none of it
fails loudly** if forgotten -- the run trains fine and the evaluation numbers are quietly
meaningless.

| What | Where | Why it breaks |
|---|---|---|
| `train_dataset_path`, `test_dataset_path` | `settings.paths` | Obvious. The test split must come from the same corpus, or held-out loss is not comparable across arms. |
| `vocab_size` | `model_raw.config` | Must match the tokenizer exactly. A mismatch either crashes or trains a head over the wrong support. |
| **`symbol_token_ids`** (26 ids, 4 synthetic datasets) | `p_hop_{1,2,3}_dataset`, `bind_3_dataset` | Regenerate with [derive_symbol_token_ids.py](../../config_files/nemotron/loop_ablation/derive_symbol_token_ids.py). These are *token ids*, chosen because a Llama-3 tokenizer emits " A".." Z" as single tokens. Under another tokenizer the same ids are arbitrary, possibly common, possibly not single symbols -- the task still runs and the numbers become uninterpretable. |
| **`delimiter_token_ids: [2652, 284]`** | `bind_3_dataset` | Same problem: these are `;` and `=` under Llama-3 only. |
| **Pre-tokenized `.npz` for MATH and TriviaQA** | `minerva_math_dataset`, `triviaqa_dataset` | Rebuild with [prepare_text_evals.py](../../config_files/nemotron/loop_ablation/prepare_text_evals.py) using the new tokenizer. Stale files hold token ids from the old one. |
| Dataset size >= 5B tokens | -- | The old shard held 2B; at 76,250 steps the wave consumes 5.0B. A smaller corpus silently repeats data, which is a different experiment. |

**Nothing from the 5B wave will be comparable to sections 3.6-3.8**: different data, different
tokenizer, ~9x the budget and a changed schedule. Treat it as a fresh baseline and re-run the
references (`A0`, `A6`, `N4`, `D1`) inside it rather than comparing across.

### 9.6 The queue

Sixteen GPUs, one arm each. Ordered so that a truncated wave still answers the main question.

| # | Arm | Why |
|---|---|---|
| 1 | `A0_baseline` | Iso-parameter control; the reference everything is read against. |
| 2-4 | `A1_loop_mamba`, `A2_loop_moe`, `A3_loop_attention` | **The titular question.** The three single-layer-type loops: is depth worth more in the Mamba mixers, the MoE feed-forwards, or attention? Running at 557M locally now (section 3.9). |
| 5-7 | `N1_anchor_mamba`, `N2_anchor_moe`, `N3_anchor_attention` | Their iso-FLOP anchors, which turn "does looping help" into "is sharing as good as fresh weights" for each layer type. Never yet run with evaluations. `N2` needs `selective f=2`. |
| 8 | `D1_dense_flops_matched` | The iso-FLOP dense reference; the largest effect measured so far. |
| 9-11 | `A4`, `A5`, `A6` | The mixed-block loops. |
| 12 | `N4_anchor_attention_moe` | `A6`'s anchor. |
| 13 | `D2_dense_param_matched` | Memory-matched dense reference; ~4.9x FLOPs/token, so it needs the longest slot of all. |
| 14-16 | 3 seeds of whichever arm leads | **Do not end this wave with another single-seed claim.** At 557M the seed s.d. was 0.0005 on LM loss and 0.017 on `p_hop_1` accuracy; both must be re-measured at the new budget and dataset, since neither transfers. |

`A6a`/`A6b`/`A6c` are dropped: section 3.7 showed the refinements have no measurable effect, and a
dose-response ladder on an effect that does not exist is not worth a GPU. Revisit only if the 5B
budget moves the reasoning metric out of the noise.

### 9.7 Does the reasoning suite survive the budget change?

This decides whether the evaluation work is worth carrying forward. At 557M the suite could not
resolve anything (section 3.7): seed s.d. 0.017 on `p_hop_1` accuracy against a between-arm spread
of 0.04, and ~12 seeds per arm needed to see a 0.02 difference. Two things may change at 5B:

* the models may become good enough that `p_hop` leaves chance, at which point the metric has
  dynamic range it lacked; and
* run-to-run variance may shrink as runs get longer.

Both are empirical. The three seed runs at positions 14-16 answer them directly. If s.d. does not
fall materially, **drop the synthetic suite and report perplexity**, which resolved everything at
557M with s.d. 0.0005.

Two changes worth making regardless, both config-only:

* **Drop `bind_3`.** It has produced no signal in any wave; every arm sits at or below its own 10%
  format-aware floor.
* **Add a `prompt_length: 64` `p_hop_1`** alongside the 256 one. At 256 the query appears ~9.8
  times in the prefix, so even a perfect induction head returns a mixture over ~10 candidates and
  lands near 10% -- approximately where the best arms sit, meaning the measured ceiling may be the
  task's rather than the model's. At 64 it appears ~2.4 times and the match is near-unique.

### 9.8 Still open

* **The phi study (section 8)** is the high-value item and is now affordable: it wants ~1.85B
  tokens per run across ~14 runs, which is less than half of one 5B wave. Do it once a noise floor
  at the new budget exists, since every phi estimate inherits that noise through the inversion.
* **`A4` and `A5` still have no anchors.** `A4`'s would be `MEMEM*EMEMEMEMEM*E` (18 built, verified
  an exact FLOP match); `A5`'s has not been worked out.
* **Pipeline parallelism and loops remain mutually exclusive** (`NemotronStagesGenerator` parses
  with `parse_layer_pattern`, which rejects loop groups). Irrelevant at one arm per GPU.
* **FSDP2 reshards a looped block after each iteration**, so a K-iteration loop pays K all-gathers.
  Irrelevant at `dp_shard=1`; set `reshard_after_forward=False` before ever sharding an arm.
* **The `D1` sub-chance `p_hop` result** is attributed to scale (section 3.8) and no longer needs a
  checkpoint diagnostic -- but with checkpointing enabled (9.4) it becomes cheap to confirm by
  dumping top-k predictions at the answer position.
