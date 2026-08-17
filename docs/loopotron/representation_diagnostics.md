# Loop diagnostics: what the trained checkpoints say about looping

Two zero-compute diagnostics run on the Wave 2 step-75k checkpoints (2026-08-17), plus the code
removal they justified. Neither trained anything. Both are recorded here because both **failed to
confirm the hypothesis they were built for**, which is the most expensive kind of result to
rediscover.

Nothing here is folded into `loopotron.tex` yet — the framing is still open.

---

## 0. Background: the loss table these diagnostics are trying to explain

Wave 2, 5B tokens, iso-parameter (12 built layers each), FLOP-matched to ~1.24x baseline, n=3 seeds:

| arm | loops | LM loss | s.d. |
|---|---|---|---|
| A0 | — (baseline) | 2.5385 | — |
| **A1** | **Mamba** | **2.4951** | 0.0013 |
| A4 | Mamba+MoE | 2.5041 | 0.0050 |
| A2 | MoE | 2.5111 | 0.0015 |
| A5 | Mamba+attn | 2.5185 | 0.0010 |
| A6 | attn+MoE | 2.5216 | 0.0021 |
| A3 | attention | 2.5312 | 0.0011 |

A1 vs A3 is 0.036 nats ≈ 28 within-arm s.d. Adding attention to a working loop *hurts* (A1 → A5).

---

## 1. Removed: stabilized recurrence and injection normalization

Ablated at K ∈ {3, 6, 12} on A1's Mamba loop (Wave 3, n=1 per arm, s.d. 0.0021 borrowed from A1's
four-run replication), then deleted from the model.

- **Stabilized recurrence** (Parcae, arXiv:2604.12946), which bounds the loop's transition
  eigenvalues in `(0,1)` by construction. K=12 is the regime it was designed for, and the instability
  there was real — peak post-warmup gradient norm 85,782 for the plain loop, 104 steps above 100,
  where K=3 and K=6 never exceeded 5.1. Against that genuine instability it moved the loss by
  **−0.0003** (0.2 s.d.) and produced *more* large-gradient steps than plain looping (229 vs 104).
- **Injection normalization** (Parcae's "prelude norm"): **+0.0475 nats at K=12**, 22.6 s.d. the wrong
  way, worst gradient behaviour of any arm. It also dominated every combination — FiLM alone −0.0109,
  FiLM + recurrence + norm **+0.0186**. A single bundled arm would have hidden both the benefit and
  the larger harm; this is the wave's strongest argument for ablating components individually.

**FiLM iteration conditioning survived** and is the only per-group refinement left in the model. It
helped at every depth with the margin growing in K (−0.0001 / −0.0079 / −0.0109 at K=3/6/12) and cut
the peak gradient norm at K=12 by ~37x.

The seven configs of the removed arms are frozen, unrunnable, under
`config_files/nemotron/loop_refinements/removed_refinements/` as the provenance of Table 6.
`LoopConfig` still *declares* `variant` and `injection_norm_config` but accepts only their inert
values (`"simple"` / `None`), because `ComponentFactory` validates with `extra="forbid"` — which
overrides `LoopConfig`'s own `extra="allow"` and propagates into nested models. Deleting the keys
outright made every historical arm config unloadable, and the shipped-config schema test did not
catch it, since that test only inspects each component's *top-level* keys. Pinned by
`test_historical_loop_config_blocks_survive_the_component_factorys_strict_validation`.

---

## 2. Diagnostic: do successive loop iterations do different work?

`src/modalities/analysis/loop_updates.py`. **Hypothesis: a layer benefits from depth recurrence in
proportion to how different its successive iterations' updates are. Result: not supported.**

**Method.** Every residual layer computes `x + operator(norm(x))`, so a layer's additive contribution
is recovered exactly as `output - input` — uniformly across Mamba, attention, MoE and MLP layers,
whose operator submodules have different attribute names. Forward hooks capture it; a test asserts the
hooks do not perturb the logits, and another asserts the identity against the real layer classes.
Cosines are computed **per token then aggregated by median** (a cosine over the flattened tensor is
dominated by the highest-norm tokens, i.e. the attention sinks) and accumulated in float32.

**Result:** Spearman(LM loss, mean consecutive-update cosine) = **−0.086**. Direct counterexamples:
A5 has the most diverse iterations (0.119) and ranks 4th of 6; A2 the most redundant (0.716) and ranks
3rd.

**Also settled — no destructive interference.** Every between-member cosine in the multi-operator
groups (A4/A5/A6) lies in [−0.03, +0.16]: the two layers in a group do essentially **orthogonal**
work. "Attention cancels Mamba" is not why the hybrid arms underperform.

**Seed-stable descriptive signatures** (±0.005–0.03 across three seeds) that explain nothing about
loss: looped attention makes the smallest updates of any operator (0.19–0.22 of the group input); MoE
alignment falls monotonically with depth (0.98 → 0.55); Mamba is non-monotonic, strongly *anti*-aligned
at the first group (−0.70) then peaking mid-stack (0.75).

### What it did surface: a structural confound

Wave 2's loss ranking is almost perfectly monotonic in **where the arm's first loop group sits**:

| arm | loss | first group @ | groups |
|---|---|---|---|
| A1 | 2.4951 | 0 | 5 |
| A4 | 2.5041 | 0 | 3 |
| A2 | 2.5111 | 1 | 5 |
| A5 | 2.5185 | 2 | 2 |
| A6 | 2.5216 | 3 | 2 |
| A3 | 2.5312 | 3 | 2 |

ρ(loss, first position) = **+0.971**; ρ(loss, group count) = −0.833; ρ(loss, executed depth) = −0.493.

This is structural. The base pattern is `MEM*EMEMEM*E`, so M first occurs at 0, MoE at 1, attention at
3 — **the operator you loop determines where the loop can start.** Wave 2 deliberately spread each
loop across every layer of a type so that *which* layer was looped is not a confound, and that works;
but nothing in the design can control the *earliest* position. **Operator class and stack position are
inseparable in Wave 2 at any sample size.**

Caveat: ρ = +0.971 rests on 6 arms and 4 distinct position values.

---

## 3. Diagnostic: what are a loop group's extra iterations worth?

`src/modalities/analysis/loop_depth.py`. Overrides iteration counts at inference on a trained model,
comparing settings **paired on identical tokens** (128 sequences x 2048 = 262,144 tokens). Paired
per-token differences are used because an 8-sequence batch moved by 0.14 nats across seeds in the
diagnostic above — 8 sequences is effectively 8 samples — while the effects sought are ~0.01 nats.
Uncertainty is computed over *sequences*, not tokens, since tokens within a sequence are correlated.

**Pipeline validation from the data itself:** Δloss at each arm's own trained K came out exactly
`0.0000` for all six arms and all three seeds, confirming the baseline goes through the same override
path as the ablations and that no constant offset is hidden in the deltas.

### 3a. Per-group ablation (each group alone reduced to K=1)

Comparing groups *within* one arm holds the looped operator fixed by construction — the comparison
Wave 2 cannot make. Where it replicates cleanly across seeds it is monotone, and says **late**
iterations are worth more:

| arm | Δloss by executed position | within-arm ρ |
|---|---|---|
| A2 MoE | +0.013 (@1) → +0.064 → +0.066 → +0.144 → **+0.546** (@15) | +1.000 |
| A3 attention | +0.069 (@3) → +0.179 (@13) | +1.000 |
| A6 attn+MoE | +0.137 (@3) → +0.740 (@12) | +1.000 |
| A1 Mamba | +0.090 / +0.221 / +0.061 / +0.133 / +0.210 | +0.200 |
| A4 Mamba+MoE | +0.641 / +0.107 / +0.294 | −0.500 |
| A5 Mamba+attn | +0.395 / +0.373 | −1.000 |

Pooled over 19 groups: ρ(position, Δloss) = **+0.289**, i.e. the *opposite* sign to the position
hypothesis.

**This does not settle the position question, in either direction.** Ablating a late group leaves less
downstream depth to repair the damage, so late groups look more important regardless of what looping
buys there — a confound running opposite to the one being tested, and A2's 43x ramp is exactly the
shape it would produce. The trained position sweep remains the only clean test.

**A1/A4/A5 are seed-dominated** and cannot be ordered at all: A4's first group reads 0.38 / 1.25 / 0.29
across seeds, A1's reads 0.16 / 0.03 / 0.08. Across-seed spread runs ~20x the within-run sampling
error, so per-group importance is not a stable architectural property at this scale.

### 3b. Global depth sweep — these models do not extrapolate

Δloss against each arm's own trained K:

| arm | trained K | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|---|---|
| A1 Mamba | 3 | +1.41 | +1.18 | 0 | **+3.51** | +6.45 | +7.80 |
| A2 MoE | 2 | +0.91 | 0 | +0.26 | +0.61 | +1.39 | +2.17 |
| **A3 attention** | 4 | +0.24 | +0.02 | **−0.05** | 0 | +0.29 | +0.79 |
| A4 Mamba+MoE | 2 | +1.31 | 0 | +0.61 | +1.76 | +5.57 | +7.45 |
| A5 Mamba+attn | 3 | +0.80 | +0.10 | 0 | +0.17 | +1.37 | +4.26 |
| A6 attn+MoE | 2 | +0.76 | 0 | +0.40 | +0.90 | +2.06 | +3.04 |

**One extra iteration costs A1 3.5 nats.** Elastic depth / test-time depth scaling is not free here; it
would have to be trained for (e.g. sampling K per step). Running *shallower* is bad but survivable
(0.8–1.4 nats at K=1).

**Attention is the exception.** A3 is nearly depth-agnostic and is genuinely better at K=3 than at its
trained K=4 (−0.048, same sign in all three seeds, against a 0.001 sampling error) — trained one
iteration too deep. That independently corroborates §2's finding that looped attention makes the
smallest updates of any operator: if the iterations do little, their count matters little.

---

## 4. What is and is not established

**Established:** the update-diversity hypothesis is dead; the hybrid groups do orthogonal, not
cancelling, work; these models are locked to their trained depth, attention excepted; operator class
and loop position are inseparable in Wave 2.

**Not established:** whether loop *position* or looped *operator* drives the Wave 2 ranking. Both
diagnostics carry positional confounds, in opposite directions. Settling it needs trained arms that
hold one fixed while varying the other — e.g. one Mamba layer looped at each of positions 0/2/5/7/9,
same K, same executed depth, identical FLOPs.

**Caveat on the absolute numbers here:** the fixed evaluation set is the leading sequences of the test
file, not the full split, so baseline losses (e.g. A1 at 2.758) are not comparable to the paper's loss
table. Only the paired differences are meaningful.

---

## 5. Reproducing

```bash
# Update-diversity diagnostic: 23 runs, ~12 min on one GPU
sbatch scripts/run_loop_updates.sh
python scripts/collate_loop_updates.py          # -> docs/loopotron/loop_updates.json

# Depth diagnostic: the 18 looped runs, ~40 min on one GPU
sbatch scripts/run_loop_depth.sh
python scripts/collate_loop_depth.py            # -> docs/loopotron/loop_depth.json

pytest tests/analysis tests/models/nemotron -o addopts=""
```

Both launchers read their run list from `docs/loopotron/wave2_final_stats.json` rather than from the
experiments directory, which still holds discarded warmstart runs (e.g. `A4_loop_mamba_moe_seed2`,
superseded by `..._seed2_redo`) that the paper excludes.

| file | purpose |
|---|---|
| `src/modalities/analysis/checkpoints.py` | build `model_raw` + load the DCP `app.model` subtree, single process |
| `src/modalities/analysis/loop_updates.py` | per-layer residual-stream contributions |
| `src/modalities/analysis/loop_depth.py` | iteration-count override + paired losses |
| `docs/loopotron/loop_updates.json`, `loop_depth.json` | collated results |
