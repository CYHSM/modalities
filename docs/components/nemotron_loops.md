# Depth-wise loops in the Nemotron hybrid stack

> Research state, literature review and the queued work (reasoning evals, per-iteration norms and
> input injection, fitting the recurrence-equivalence exponent φ per layer type) are in
> [nemotron_loops_research_plan.md](nemotron_loops_research_plan.md). This document is the
> implementation reference.

Modalities can execute a run of Nemotron layers several times while reusing one set of weights,
i.e. depth-wise weight sharing in the sense of a Universal Transformer or a recurrent-depth model.
A loop trades parameters for effective depth: the model performs more layer applications per token
without holding more weights.

Because the hybrid stack is a sequence of **single-operator** residual layers (see
[nemotron.md](nemotron.md)), a loop can be placed around any subset of layer types. That is the
point of the mechanism: it makes "where is extra depth worth spending — on the Mamba mixers, the
MoE feed-forwards, the attention layers, or a mixed block?" a one-line configuration change.

## Notation

A loop group is written `[<symbols>]^<K>` inside the layer pattern:

| Pattern | Builds | Executes |
|---------|--------|----------|
| `MEM*E` | 5 layers | 5 layer applications |
| `M[E]^3*` | 3 layers (M, E, `*`) | 5 applications; the MoE layer runs three times |
| `[ME]^3*` | 3 layers (M, E, `*`) | 7 applications; the Mamba+MoE pair runs three times |
| `[M*]^2E` | 3 layers (M, `*`, E) | 5 applications |

Brackets are mandatory, and the repeat operator is `^`, not `*`. Two reasons:

1. `*` is already the attention symbol, so `M[ME]*3*E` reads as though the `3` were multiplied by
   an attention layer. `M[ME]^3*E` is unambiguous to both the parser and the reader.
2. Requiring brackets means a bare `M^3` or `M*3` is rejected outright rather than silently
   parsed as something else.

Nesting is not supported. Loop counts must be at least 1; `[ME]^1` is legal and equivalent to `ME`,
which is convenient when sweeping `K` from a script.

**`n_layer` counts built layers, not executed ones.** A loop group's layers are built once, so
`M[ME]^3E` has `n_layer: 4`. The model validates this and the error message says so.

Do not confuse a layer-pattern loop with the MoE `experts_backend: looped` option, which is an
unrelated choice of expert matmul kernel.

## How it works

`transformer.h` stays a **flat** `nn.ModuleDict` with one entry per set of weights, exactly as
before; the loop lives in a separate execution schedule
([layer_pattern.py](../../src/modalities/models/nemotron/layer_pattern.py)):

```python
for group in self._schedule:           # list[LoopGroup]
    h = self._run_loop_group(group, h)
```

Keeping the module tree flat is what makes activation checkpointing (`layers_fqn: transformer.h`),
FSDP2 block wrapping, the weight-decay group regexes and the initialization parameter filters work
on looped models **without any changes** — parameter names are still `transformer.h.<idx>....`.

[`NemotronLLM._run_loop_group`](../../src/modalities/models/nemotron/nemotron_model.py) is the
single place where loop semantics live and the extension point for further strategies. It is
selected by `loop_config.variant`:

```yaml
model_raw:
  config:
    layer_pattern: "[ME]^3M*EMEMEM*E"
    n_layer: 12
    loop_config:
      variant: simple
```

`simple` chains the iterations: each consumes the previous one's output. `LoopConfig` accepts
unknown keys, so a future variant (for example routing the iterations and returning a weighted sum
rather than only the last one) can carry its own hyperparameters without a schema change.

## What follows the loop, and what does not

A loop changes how often parameters are *applied*, not how many exist. Everything counted per
application therefore has to follow the loop. Three places do, and getting any of them wrong
silently biases an ablation rather than raising an error:

| Quantity | Behaviour |
|----------|-----------|
| **Effective depth** (init scaling) | `NemotronLLM.n_executed_layers`. Residual variance grows with layer applications, so depth-scaled initialization must use this, **not** `n_layer`. |
| **MoE auxiliary loss** | Accumulated across visits and reset at the start of each forward pass. A looped MoE layer routes independently on each visit, so each visit's imbalance is penalized. |
| **MFU active parameters** | Execution-weighted via `NemotronLLM.get_execution_counts()`; `count_layers_by_type` counts attention *executions*, so the quadratic attention term is right for looped attention. |

For the initializer, derive the depth from the pattern rather than typing it per arm:

```yaml
num_layers:
  component_key: number_conversion
  variant_key: num_executed_layers_from_layer_pattern
  config:
    layer_pattern: ${model_raw.config.layer_pattern}
```

This is the single most important detail in the whole setup. If `num_layers` stays at `n_layer`, a
looped arm is initialized as though it were shallow, and the resulting difference in loss reads as
a loop effect when it is really an initialization effect.

The auxiliary-loss-free expert bias needed no change: `router.tokens_per_expert` already
accumulates with `+=`, so a looped MoE layer's load is counted per visit. The aux-loss
accumulation is deliberately out-of-place, so replaying the forward pass under activation
checkpointing cannot mutate the tensor already handed to the loss.

## Per-iteration norms and input injection

Two refinements the literature treats as near-prerequisites for loop counts above two or three.
Both are **off by default**, so a config that does not mention them describes exactly the model it
described before they existed, and both are toggled independently so that the loop effect and the
per-iteration conditioning effect stay separable — a meaningful share of reported Universal
Transformer gains comes from the conditioning rather than from the weight sharing.

```yaml
loop_config:
  variant: simple
  per_iteration_norm: true    # each iteration gets its own pre-norm
  input_injection: true       # the group's input is added back before each later iteration
  injection_mode: add         # the only mode implemented
```

**Per-iteration norm.** A looped layer applies one set of operator weights to a residual stream
whose scale and statistics change between iterations. With the flag on, a layer inside a `[...]^K`
group is built with `K` pre-norms and applies the one belonging to the current iteration; the
operator itself stays shared, which is the point of the loop. Layers outside a loop group are
unaffected, so the flag is a no-op on `A0_baseline`.

The norms live under the layer's existing `norm` attribute, so a parameter is named
`transformer.h.<idx>.norm.norms.<iteration>.weight`. **That name is not cosmetic.** It still
contains `.norm.` and therefore still matches the `layernorm` weight-decay group. Naming the
attribute `norms` instead would produce `transformer.h.<idx>.norms.0.weight`, which matches no
group at all — and a parameter in no weight-decay group is not merely undecayed, it is left out of
*both* optimizer groups and never updated. `test_every_parameter_lands_in_exactly_one_weight_decay_group`
pins this down.

**Input injection.** The group's input is added back to the hidden states at the start of every
iteration after the first, so no iteration is more than one residual step away from what the group
was handed. Injecting *before* an iteration rather than after one is what makes this an exact no-op
for non-looped groups (which are groups of a single iteration); injecting after each iteration
would add the input back into every plain layer of the model and quietly change `A0_baseline` the
moment the flag was set. `injection_mode: add` adds no parameters. A `concat_proj` mode is
deliberately absent: its projection belongs to the *group* rather than to any layer, so it would
have to live outside `transformer.h` and would need its own entries in the weight-decay groups and
the initialization filters — and at ~2·`n_embd`² per looped group it would break the iso-parameter
comparison the ablation rests on.

**Parameter accounting.** Per-iteration norms add `(K − 1) · n_embd` parameters per looped layer, so
such an arm is no longer *exactly* iso-parameter with the baseline. For `A6a` that is
2 groups × 2 layers × 1 × 1024 = **4,096 parameters** of 1.105B — reported rather than assumed zero.
The generated config's banner prints the figure, and
`NemotronLLM.num_per_iteration_norm_parameters` returns it at runtime. The MFU calculator charges
all `K` norms on every iteration instead of the one visited, overcounting by `K(K−1)·n_embd` per
looped layer (8,192 parameters of 255M active for `A6a`), which is far below the accuracy of the
FLOPs estimate itself.

### The 2×2

`A6_loop_attention_moe` is the cell with neither refinement, so three configs complete the design:

| Arm | `per_iteration_norm` | `input_injection` | Parameters |
|-----|----------------------|-------------------|------------|
| `A6_loop_attention_moe` | off | off | 1.105B |
| `A6a_loop_attention_moe_per_iteration_norm` | **on** | off | 1.105B + 4,096 |
| `A6b_loop_attention_moe_input_injection` | off | **on** | 1.105B |
| `A6c_loop_attention_moe_norm_and_injection` | **on** | **on** | 1.105B + 4,096 |

All four share `A6`'s pattern `MEM[*E]^2MEMEM[*E]^2` and therefore its FLOPs (1.179x `A0`) and its
iso-FLOP anchor `N4_anchor_attention_moe`. `A6` is the right arm for this question: it loops the
classical attention-then-feed-forward block, which is the shape the per-iteration-norm and
input-injection results were reported for, and it is the only loop arm whose anchor exists, so a
gain from the refinements can be read against fresh weights rather than only against `A0`.

## Research configuration

[config_research_nemotron_loops_1gpu.yaml](../../config_files/nemotron/config_research_nemotron_loops_1gpu.yaml)
is a research-scale Nemotron built for these ablations. It is a geometric scale-down of
Nemotron-3 Nano 30B-A3B that preserves the reference's **cost structure**, which is what makes
conclusions about where to spend depth transferable:

| Property | Reference 30B-A3B | Research config |
|----------|-------------------|-----------------|
| Model dimension / built layers | 2688 / 52 | 1024 / 12 |
| Parameters | 31.58B total, 3.58B active | 1.105B total, 225M active |
| Non-embedding active / total | 9.3% | 9.7% |
| Per-layer active cost `M : * : E` | 1 : 0.60 : 2.07 | 1 : 0.59 : 2.01 |
| Mamba d_inner / n_embd | 1.52 | 1.50 |
| Attention inner / n_embd | 1.52 | 1.50 |
| Expert ffn / n_embd | 0.69 | 0.69 |
| Expert sparsity | top-6 of 128 | top-6 of 128 |

The headline activation ratio (20.4% vs 11.3%) differs only because of the vocabulary tax: tied
Llama-3 embeddings are 131M, i.e. 12% of this model but 2.2% of the 30B. That term is constant
across all arms and cancels in the comparison. Retokenizing to a 32k vocab would bring it to 12.7%
at the cost of regenerating the pbin files — worth doing only if the loss curves need to be
quotable against published scaling laws.

**Why 1.1B and not 400M.** Two things are dense and always active: the embedding table and the
Mamba layers. A 400M model with a 128k vocab cannot be ~10% active, because 131M of it is an
always-on lookup table; forcing the size down instead distorts the `M : * : E` cost ratio, which
is precisely the quantity the ablation is about. The 1.1B costs little in practice: training FLOPs
scale with *active* parameters, so it trains at roughly the speed of a 225M dense model, and only
memory holds the full 1.1B.

## Ablation arms

Ready-to-run configs live in [config_files/nemotron/loop_ablation/](../../config_files/nemotron/loop_ablation/).
They are generated from the shared base config, so a change to a shared setting is made once and
propagates to every arm:

```bash
python config_files/nemotron/loop_ablation/generate_arm_configs.py   # after editing the base config
./config_files/nemotron/loop_ablation/run_arm.sh A1_loop_mamba 4     # arm name, GPU id
```

`run_arm.sh` sets `--experiment_id` to the arm name plus a timestamp, so arms are named in wandb
and re-running one never overwrites an earlier run.

All figures below are computed from the built models, not estimated. Relative FLOPs per token uses
the MFU calculator's model, `6 * active_params + 12 * attention_executions * seq_len * n_head_q *
head_dim`, at sequence length 2048.

### Loop arms — all ~1.24x baseline FLOPs, all 1.105B parameters

| Arm | `layer_pattern` | Built | Exec | Executions M/E/`*` | Active | rel. FLOPs |
|-----|-----------------|-------|------|--------------------|--------|-----------|
| `A0_baseline` | `MEM*EMEMEM*E` | 12 | 12 | 5/5/2 | 225.5M | 1.000 |
| `A1_loop_mamba` | `[M]^3E[M]^3*E[M]^3E[M]^3E[M]^3*E` | 12 | 22 | 15/5/2 | 283.6M | 1.244 |
| `A2_loop_moe` | `M[E]^2M*[E]^2M[E]^2M[E]^2M*[E]^2` | 12 | 17 | 5/10/2 | 283.9M | 1.245 |
| `A3_loop_attention` | `MEM[*]^4EMEMEM[*]^4E` | 12 | 18 | 5/5/8 | 246.0M | 1.244 |
| `A4_loop_mamba_moe` | `[ME]^2M*E[ME]^2[ME]^2M*E` | 12 | 18 | 8/8/2 | 277.9M | 1.220 |
| `A5_loop_mamba_attention` | `ME[M*]^3EMEME[M*]^3E` | 12 | 20 | 9/5/6 | 262.4M | 1.261 |
| `A6_loop_attention_moe` | `MEM[*E]^2MEMEM[*E]^2` | 12 | 16 | 5/7/4 | 255.7M | 1.179 |

`A6` loops the classical transformer block (attention then feed-forward), so it is the closest thing
in this hybrid to a Universal Transformer loop, at the loop count reported as the best marginal
return. At 1.179x it gets ~5% less compute than the other loop arms — the conservative direction, so
a win there is stronger evidence. Its exact-FLOP anchor `N4_anchor_attention_moe` now exists.

**`A4_loop_mamba_moe` still has no anchor.** For it the iso-parameter comparison against `A0` is the
only one available; an anchor would be `MEMEM*EMEMEMEMEM*E` (18 built layers, 8 Mamba / 8 MoE / 2
attention executions), which is not yet generated.

Two deliberate choices. First, the arms are matched on **FLOPs, not loop count**: an extra MoE
execution costs about twice an extra Mamba execution and an extra attention execution costs about
1.65x, so equal `K` across types would compare unequal compute budgets. Hence `[M]^3`, `[E]^2` and
`[*]^4`. Second, the loop is spread over **every** layer of the type rather than concentrated on
one, so "which layer happened to be looped" is not a confound. A4 and A5 sit 2% off the target
because pair loops do not land on the same grid; that residual is reported rather than hidden.

### Iso-FLOP anchors — same compute bought with fresh weights

| Anchor | For | `layer_pattern` | Built | Exec | Active | rel. FLOPs | Params |
|--------|-----|-----------------|-------|------|--------|-----------|--------|
| `N1_anchor_mamba` | A1 | `MMMEMMM*EMMMEMMMEMMM*E` | 22 | 22 | 283.6M | 1.244 | 1.163B |
| `N2_anchor_moe` | A2 | `MEEM*EEMEEMEEM*EE` | 17 | 17 | 283.9M | 1.245 | 2.043B |
| `N3_anchor_attention` | A3 | `MEM****EMEMEM****E` | 18 | 18 | 246.0M | 1.244 | 1.125B |
| `N4_anchor_attention_moe` | A6 | `MEM*E*EMEMEM*E*E` | 16 | 16 | 255.7M | 1.179 | 1.487B |

Each anchor matches its loop arm **exactly** — same executed count of every layer type, same active
parameters, same FLOPs — and differs only in whether those executions reuse weights or have their
own. That isolates weight sharing as the single variable.

**Why anchors are not optional.** A loop arm has the baseline's parameters but more FLOPs, so
"A1 beats A0" may only mean "more compute helps". The two comparisons answer different questions:

* **arm vs `A0_baseline`** (iso-parameter): does looping buy usable extra compute?
* **arm vs its anchor** (iso-FLOP): is weight sharing as good as fresh weights for this layer type?

The anchors also price the alternative. `N2_anchor_moe` needs **2.043B** parameters to buy what
`A2_loop_moe` gets from 1.105B — so if A2 matches N2, looping MoE layers saves nearly half the
model. `N3_anchor_attention` needs only 1.125B, so looping attention saves almost nothing in
parameters even if it works. That asymmetry is itself a result worth reporting.

## Reasoning evaluations

Measuring these arms on loss alone answers the wrong question. Every published result on depth-wise
weight sharing points the same way: looped models *lose* on validation perplexity under an iso-FLOP
comparison and *win* on reasoning. A sweep that logs only training loss will therefore show every
loop arm slightly behind its anchor and conclude that looping does not work — which is exactly what
the first two waves showed, and exactly what the literature says you should expect to see.

The base config attaches six held-out evaluations: four synthetic
([synthetic_reasoning.py](../../src/modalities/dataloader/synthetic_reasoning.py)) and two real
benchmarks ([prepared_eval.py](../../src/modalities/dataloader/prepared_eval.py)).

| Dataloader tag | Task | Prompt | What it is for |
|---|---|---|---|
| `p_hop_1` | p-hop induction, 1 hop | 256 tokens | **Control.** A single induction-head lookup; needs no depth. |
| `p_hop_2` | p-hop induction, 2 hops | 256 tokens | Two serial lookups. |
| `p_hop_3` | p-hop induction, 3 hops | 256 tokens | Three serial lookups. |
| `bind_3` | variable binding, 3 hops | 51 tokens | Three lookups over *shuffled* statements, so recency carries no signal. |
| `minerva_math` | Minerva MATH, 4-shot | ≤1997 tokens | Real reasoning benchmark. Where looping is expected to help. |
| `triviaqa` | closed-book TriviaQA | ≤142 tokens | **Parametric-knowledge control.** Where looping is expected to *lose*. |

**Two dissociations are the experiment, not the individual numbers.**

* *Across the hop ladder.* An arm that is simply better moves all the synthetic dataloaders
  together. An arm that buys usable depth moves `p_hop_2` and `p_hop_3` while leaving `p_hop_1`
  where it was.
* *Reasoning against knowledge.* The whole claim is that weight sharing buys reasoning without
  buying parameters, so `minerva_math` and `triviaqa` should move in **opposite** directions:
  closed-book recall lives in weights a loop does not add. An arm that improves both is just
  training better, and that is exactly what the control is there to catch.

`p_hop_induction` presents a random sequence of symbols and asks for the next token. The last symbol
is the query: find its most recent earlier occurrence, read off the symbol that follows it, then
repeat with that symbol as the new query, searching only further to the left. Rendered with the
Llama-3 tokenizer a two-hop prompt looks like ordinary text (shown at a shortened
`prompt_length` of 80; the configured evaluations use 256):

```
 Z Z Z J E Y C G D I N D U G U I U Z Y G O L G P X W Q W N R Y R B T T F W E U W M B E R B
 R D P O B E Z K L L N F A Q G R W L L E T S X U D U Z J E W Y Z C W M      -> answer:  T
```

`variable_binding` is the same idea in assignment form, with distractors and with the statement
order shuffled, so the chain has to be followed by content rather than by position:

```
 ; O = P ; X = W ; Z = B ; D = N ; Q = J ; I = M ; U = V ; G = L ; Y = E ; F = K ; T = R ; W = D ; X =
                                                                                   -> answer:  N
```

(`X = W`, `W = D`, `D = N`; three hops. The other nine statements are distractors, binding symbols
that never appear in the chain.)

**`bind_3` has a higher floor than the p-hop tasks, and it is not 1/26.** The answer is always a
symbol that appears as a value but never as a variable, and there are exactly `num_distractors + 1`
of those — so a model that has learned only the *format* can guess among 10 candidates without
following the chain at all. Read `bind_3` accuracy against **0.10**, not 0.0385. The floor is
identical for every arm, so it does not bias the comparison, but it does mean an arm sitting at 10%
on `bind_3` has demonstrated nothing. `SyntheticReasoningDataset.format_aware_chance_accuracy`
reports it, and a test pins the two together so a change to the generator cannot silently widen the
shortcut. `p_hop_*` has no equivalent: every symbol of the alphabet is a permissible answer there,
which is why it carries the main hop ladder and `bind_3` is corroboration rather than the headline.

### Minerva MATH and TriviaQA

These are real text, so unlike the synthetic tasks they must be tokenized. That happens **once,
offline**, so every arm is scored on byte-identical token sequences and training needs no
tokenizer — the same arrangement as the training pbin files. Build them before the first run:

```bash
python config_files/nemotron/loop_ablation/prepare_text_evals.py
```

which writes `data/prepared_evals/{minerva_math,triviaqa}.npz`. The problem statement is the prompt
and is never scored; the reference answer is scored in full.

`minerva_math` is lm-evaluation-harness's [`minerva_math`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/minerva_math)
task, **not** plain Hendrycks MATH: the same test split, but Minerva's `Problem:\n{problem}\n\nSolution:`
prompt and its verbatim 4-shot prefix, joined with the harness's default `"\n\n"` few-shot and
`" "` target delimiters. Minerva's other half — SymPy answer normalization and equivalence
checking — exists only to score a *generated* answer for exact match and has no counterpart in a
likelihood evaluation, so it is not used. 1024 problems balanced across all seven subjects, 214
scored tokens each; `triviaqa` is 8192 closed-book questions at 3.0 scored tokens each.

**Read the NLL, not the accuracy, on these two.** At 1.1B parameters and a few billion tokens a
model scores ~zero on both, so solve rate is a constant rather than a measurement. `answer_accuracy`
on these dataloaders is per-token next-token accuracy over a reference solution, reported for
completeness — it is not a benchmark score.

What MATH's NLL measures is worth being honest about: most solution tokens are prose and LaTeX
rather than reasoning, so a gain confined to the reasoning steps arrives diluted, on the order of
10x. That is a sensitivity cost, not a validity problem — the problems and tokens are identical
across arms, so there is no evaluation sampling noise and the limiting noise is the training seed
(§8.5 of the research plan).

### What gets logged

Two metrics per evaluation dataloader, as `<dataloader_tag> <metric>` in wandb:

| Metric | Meaning |
|---|---|
| `answer_accuracy` | Fraction of scored positions whose argmax is correct. **Chance is 1/26 = 0.0385 on the `p_hop_*` tasks, 1/10 = 0.10 on `bind_3`** (see above), and not a meaningful benchmark score on `minerva_math` / `triviaqa`. |
| `answer_nll` | Negative log-likelihood of the correct token. Moves while accuracy is still pinned at chance, so it is the one to watch early in a run — and the only meaningful metric on the two real benchmarks. |

Targets are masked with `-100` everywhere except the answer, so the *loss* reported on these
dataloaders is close to the answer's negative log-likelihood. **Use `answer_nll` anyway.** Two
measured reasons, both from the smoke runs:

* The configured objective is a `weighted_sum` including the MoE auxiliary loss, whose size depends
  on how often the MoE layers are visited — i.e. on precisely what the loop arms vary. Measured at
  ~3e-4 nats: small, but ~1% of published effect sizes, and biased by arm.
* More seriously, the loss is a mean of per-batch means, so it weights each *batch* equally while
  `answer_nll` weights each *scored token* equally. Where problems have similar scored-token counts
  the two agree to three decimals; on `minerva_math`, where a solution may be 20 tokens or 400, they
  differ by **0.21 nats** (9.397 vs 9.609). That is the artifact the numerator/denominator reduction
  exists to remove.

Metrics select their dataloaders by tag
([evaluation_metrics.py](../../src/modalities/evaluation_metrics.py)). This matters: on the
language-modelling `test` split no target is masked, so an "answer accuracy" there would be
ordinary next-token accuracy wearing the same name. Each metric returns a numerator and a
denominator rather than a value, and the evaluator sums both before dividing once, so an uneven
split across ranks or a short final batch cannot skew the result.

### Reproducibility

Two things must hold or the comparison between arms is void, and both are enforced rather than
assumed:

* **Same questions for every arm.** Generation is deterministic in `seed`, which the arm generator
  copies from the base config unchanged. Do not vary it per arm.
* **Exact answer position.** Symbols are configured as **token ids**, not strings, so evaluation has
  no runtime tokenizer dependency and the sequences are reproducible from the config alone. Each
  symbol must be a single token, or the answer would not occupy one position. The ids currently in
  the config are `' A'`–`' Z'` and `' ;'`, `' ='` under the Meta-Llama-3-8B-Instruct tokenizer that
  the fineweb pbin files were built with. Regenerate them if the data is ever retokenized:

```bash
python config_files/nemotron/loop_ablation/derive_symbol_token_ids.py
```

The script refuses to emit an alphabet containing a multi-token symbol, so a tokenizer change
cannot silently produce a misaligned evaluation.

### Cost

The four synthetic dataloaders hold 2048 questions each at batch size 64. Their prompts are 256
tokens (p-hop) and 51 (binding), against 2048 for the language-modelling split, so all four
together add well under a single training step's worth of compute per evaluation. The standard
error of an accuracy near 30% is about 1%, below the differences the literature reports between
looped and non-looped models.

`minerva_math` is the expensive one: 1024 problems at up to 1997 tokens, because the 4-shot prefix
alone is ~500. That is roughly the cost of the language-modelling split. The full MATH test set is
~5000 problems, but at 214 scored tokens each, 1024 already pins the mean far tighter than the
training-seed noise the comparison is limited by, so evaluating all of them would cost several
times as much for no gain. `triviaqa` is negligible at 142 tokens per question.

If evaluation cost becomes a problem, raise `evaluation_interval_in_steps` rather than shrinking
the evaluations — the arms have to be compared at matched steps, and a noisier evaluation defeats
the purpose.

## Verified runs

Single A100-SXM4-80GB, torch 2.9.1+cu128, fused Mamba kernels, sequence length 2048, micro batch 8,
gradient accumulation 4 (global batch 32 sequences), `full_activation_checkpointing`, FSDP2 with
`dp_shard=1`.

All nine arms were run for 150+ steps and train stably from initialization. These are smoke tests
of cost and stability, **not** a quality comparison — the arms are not token-matched here.

| Arm | Built / exec | Params | Peak memory | Throughput | MFU |
|-----|--------------|--------|-------------|------------|-----|
| `A0_baseline` | 12 / 12 | 1.105B | 33366 MiB | 23.2 samples/s | 0.22 |
| `A1_loop_mamba` | 12 / 22 | 1.105B | 33686 MiB | 16.3 samples/s | 0.19 |
| `A2_loop_moe` | 12 / 17 | 1.105B | 33526 MiB | 16.0 samples/s | 0.19 |
| `A3_loop_attention` | 12 / 18 | 1.105B | 33558 MiB | 21.1 samples/s | 0.25 |
| `A4_loop_mamba_moe` | 12 / 18 | 1.105B | 33559 MiB | 16.7 samples/s | 0.19 |
| `A5_loop_mamba_attention` | 12 / 20 | 1.105B | 33622 MiB | 18.8 samples/s | 0.22 |
| `N1_anchor_mamba` | 22 / 22 | 1.163B | 34572 MiB | 16.3 samples/s | 0.19 |
| `N2_anchor_moe` | 17 / 17 | 2.043B | 47840 MiB | 15.2 samples/s | 0.18 |
| `N3_anchor_attention` | 18 / 18 | 1.125B | 33876 MiB | 21.0 samples/s | 0.25 |

Two results worth having before the real sweep:

**Weight sharing is free in wall-clock.** Each loop arm runs at its anchor's speed — 16.3 vs 16.3
(A1/N1), 16.0 vs 15.2 (A2/N2), 21.1 vs 21.0 (A3/N3). Re-entering one layer costs the same as
traversing several distinct ones, so any quality difference between an arm and its anchor is
attributable to weight sharing rather than to a speed handicap.

**What the loop saves is memory, and only where the layers are heavy.** A1 saves 886 MiB over N1,
A3 saves 318 MiB over N3, but **A2 saves 14.3 GiB over N2** (33526 vs 47840). MoE layers hold
almost all the parameters, so looping them is the only arm where sharing changes the model's
footprint materially.

MFU is not constant across arms (0.18–0.25), which is why the arms are matched on modelled FLOPs
rather than on wall-clock: attention loops (A3) achieve the highest MFU because flash attention is
compute-dense, while Mamba and MoE loops sit lower.

Micro batch 8 rather than 1 with more gradient accumulation is deliberate: with 128 experts and
top-6, a micro batch of 1 at sequence length 2048 gives only ~96 tokens per expert, too few for
`grouped_mm` to be efficient.

**Measurement caveat.** Throughput was measured with three or four single-GPU runs in flight on one
machine, so the absolute numbers are contention-sensitive; the loop-vs-anchor pairs above were
measured head-to-head in the same batch and are internally comparable. An earlier measurement of
this baseline at ~10 samples/s was taken while the test suite was saturating the CPU and was
roughly 2.3x too low — when comparing arms, run them under identical load.

### Commands

One arm on one GPU (the wrapper picks an RDZV port and a timestamped experiment id):

```bash
./config_files/nemotron/loop_ablation/run_arm.sh A1_loop_mamba 4
```

The whole sweep, six arms across GPUs 0-5:

```bash
for i in "A0_baseline 0" "A1_loop_mamba 1" "A2_loop_moe 2" \
         "A3_loop_attention 3" "A4_loop_mamba_moe 4" "A5_loop_mamba_attention 5"; do
  ./config_files/nemotron/loop_ablation/run_arm.sh $i &
done; wait
```

Or the equivalent explicit invocation:

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 1 \
  $(which modalities) run \
  --experiments_root_path /home/markus_frey/Github/modalities/results \
  --experiment_id A1_loop_mamba__manual \
  --config_file_path config_files/nemotron/loop_ablation/config_A1_loop_mamba.yaml
```

Regenerate the arms after changing a shared setting in the base config:

```bash
python config_files/nemotron/loop_ablation/generate_arm_configs.py
```

Tests:

```bash
python -m pytest tests/models/nemotron/ -q
```

`pytest` is not part of the runtime install. Note that installing it from the project directory
fails, because `uv` then tries to resolve the project's `extra-build-dependencies` (which pin
`torch` with `match-runtime`) and reports `torch ... was not found in the resolution`. Install
from outside the project directory instead:

```bash
cd /tmp && uv pip install --python /path/to/modalities/.venv/bin/python --only-binary :all: pytest
```

## Limitations

* **FSDP2 resharding.** A looped layer is re-entered immediately, but
  [model_factory.py](../../src/modalities/models/model_factory.py) reshards each block after its
  forward pass, so a `K`-iteration loop pays `K` all-gathers instead of one. Irrelevant at
  `dp_shard=1`; set `reshard_after_forward=False` for looped blocks before scaling out.
* **Pipeline parallelism.** `NemotronStagesGenerator` still parses the pattern with
  `parse_layer_pattern`, which rejects loop groups. Its cost model would also need to weight layers
  by execution count. Loops and pipeline parallelism are therefore mutually exclusive for now; the
  error message points at `parse_layer_schedule`.
* **Input injection has only the `add` mode.** `concat_proj` would put parameters outside
  `transformer.h` and break the iso-parameter comparison; see the section above.
* **Loop count is fixed at training time.** Randomizing `K` per step (as in recurrent-depth models)
  would allow test-time compute scaling, and is a natural follow-up once a winning arm is known.
