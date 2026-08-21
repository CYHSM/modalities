# The loop-position sweep: is it the operator, or where you put it?

Twelve trained arms (2026-08-17 / 2026-08-18, Slurm arrays 52604359 and 52628149), 5B tokens each,
answering the confound that Wave 2 could not: **does looping Mamba beat looping attention because of
the operator, or because the base pattern forces a Mamba loop to sit early and an attention loop to
sit late?**

Short answer: **both effects are real, they are of comparable size, and they are operator-specific.**
Position is not a property of the stack — each operator has its own, qualitatively different, curve.
Wave 2's headline survives; Wave 2's interior ordering does not.

---

## 1. The design

Within a family: fix the operator, fix the loop count at K=6, fix the executed depth, and vary only
*where* the looped layer sits. The base pattern `MEM*EMEMEM*E` gives one family per operator, with as
many arms as that operator has layers:

| Family | Operator | Built indices | Arms |
|---|---|---|---|
| `P` | Mamba (`M`) | 0, 2, 5, 7, 9 | 5 |
| `Q` | MoE (`E`) | 1, 4, 6, 8, 11 | 5 |
| `R` | attention (`*`) | 3, 10 | 2 |

Every arm in every family is **12 built / 17 executed layers and 1,025,191,272 parameters** —
verified by instantiation, not derivation. Within a family the per-type execution counts, active
parameters and FLOPs are therefore identical *by construction*, and position is the only variable.
The generator asserts this from the patterns and refuses to write configs if it fails.

Each config differs from Wave 2's `A1_loop_mamba` in exactly two lines, `experiment_id` and
`layer_pattern`, so `A0_baseline` (2.5385) and the Wave 2 arms are directly usable as references
without re-running them: same budget (76,250 steps = 4,997,120,000 tokens), same tokenizer, data,
sampler seed 42, warmup, schedule and node layout.

**n = 1 per arm.** Read against **s.d. 0.0021**, measured for this architecture from four runs of A1
(loopotron.tex, Wave 3). Every arm ran fresh in a single slot — no requeue, no `modalities warmstart`
— so none carries the tied-embedding checkpoint contamination that cost Wave 2 five runs.

**Across families, K is held fixed rather than FLOPs.** That keeps "same loop, different operator,
different place" as the contrast and maximises signal in the two weaker operators, but it means the
three families do not cost the same (1.183x / 1.369x / 1.306x baseline FLOPs). Cross-family
comparisons of *curve shape* are therefore clean; comparisons of *magnitude* need the FLOPs column in
section 3.

## 2. Results

`A0_baseline` (no loop, 12 executed) = **2.5385**. Deltas are against it, in units of the 0.0021 s.d.

### P — Mamba: large effect, optimum at index 2, monotone decay after it

| Built idx | Arm | LM loss | vs A0 |
|---|---|---|---|
| 0 | `P0_loop_mamba_at_0` | 2.5166 | −10.4 s.d. |
| **2** | **`P1_loop_mamba_at_2`** | **2.5017** | **−17.5 s.d.** |
| 5 | `P2_loop_mamba_at_5` | 2.5287 | −4.6 s.d. |
| 7 | `P3_loop_mamba_at_7` | 2.5293 | −4.4 s.d. |
| 9 | `P4_loop_mamba_at_9` | 2.5515 | **+6.2 s.d. — worse than not looping** |

Spread **0.0498 nats = 23.7 s.d.**, ρ(position, loss) = **+0.90**. Larger than Wave 2's entire
0.0361-nat operator spread. The optimum is at index 2, not 0, so it is not simply "earlier is
better". Index 9 is *worse than the unlooped baseline* despite five extra Mamba executions: depth
spent in the wrong place has a negative return, not merely a small one.

### Q — MoE: half the effect, and U-shaped

| Built idx | Arm | LM loss | vs A0 |
|---|---|---|---|
| **1** | **`Q0_loop_moe_at_1`** | **2.5246** | **−6.6 s.d.** |
| 4 | `Q1_loop_moe_at_4` | 2.5438 | +2.5 s.d. |
| 6 | `Q2_loop_moe_at_6` | 2.5344 | −1.9 s.d. |
| 8 | `Q3_loop_moe_at_8` | 2.5265 | −5.7 s.d. |
| 11 | `Q4_loop_moe_at_11` | 2.5248 | −6.5 s.d. |

Spread **0.0192 nats = 9.1 s.d.**, ρ(position, loss) = **0.00**. The zero correlation is not a null:
the curve is **U-shaped**, best at both ends of the stack (indices 1 and 11, tied within 0.1 s.d.)
and worst in the middle (index 4, which is *worse than not looping*). A monotone statistic is simply
the wrong summary for it.

### R — attention: no position effect at all

| Built idx | Arm | LM loss | vs A0 |
|---|---|---|---|
| 3 | `R0_loop_attention_at_3` | 2.5287 | −4.6 s.d. |
| 10 | `R1_loop_attention_at_10` | 2.5309 | −3.6 s.d. |

Spread **0.0022 nats = 1.0 s.d.** — indistinguishable from noise.

This is the sharpest single comparison in the wave. Over essentially the same span of the stack,
**Mamba moves 0.0498 nats (index 2 → 9) while attention moves 0.0022 (index 3 → 10)** — a factor of
**23**. Whatever makes position matter, attention does not have it.

Only two attention layers exist in this stack, so `R` is a single contrast rather than a curve. That
is a property of the base pattern, not a design choice; widening it needs a different pattern, which
would break comparability with A0 and every Wave 2 arm.

## 3. Cross-family, with compute accounted for

Relative FLOPs/token use the repo's own `NemotronMFUCalculator` model,
`6 * active_params + 12 * attention_executions * seq_len * n_head_q * head_dim` at seq 2048, with
active parameters measured by instantiation.

| Operator | Best LM | @ idx | Worst LM | Spread | rel. FLOPs | Gain vs A0 | Gain per +10% FLOPs |
|---|---|---|---|---|---|---|---|
| **Mamba** | **2.5017** | 2 | 2.5515 | 0.0498 | **1.183** | **+0.0367** | **0.0200** |
| MoE | 2.5246 | 1 | 2.5438 | 0.0192 | 1.369 | +0.0139 | 0.0038 |
| attention | 2.5287 | 3 | 2.5309 | 0.0022 | 1.306 | +0.0097 | 0.0032 |

**Mamba dominates on every axis simultaneously.** It is the cheapest family (1.183x, because an extra
Mamba execution is the cheapest kind), it has the best absolute loss at its best position, and its
return per unit of extra compute is **5-6x** either other operator. Wave 2 reported this ordering
from arms whose loops were forced into different stack positions; it survives holding position fixed,
and it survives normalising for compute.

## 4. What this settles

**Three orderings coincide, and they are the same ordering.**

| | Mamba | MoE | attention |
|---|---|---|---|
| Loop benefit (best position) | +0.0367 | +0.0139 | +0.0097 |
| Position sensitivity (spread) | 0.0498 | 0.0192 | 0.0022 |
| Residual-stream update magnitude<sup>†</sup> | largest | middle | **smallest** |

<sup>†</sup> from the position-controlled within-group `member_step_norms` comparison in
representation_diagnostics.md: in A5 (`M*`) Mamba beats attention in 6/6 group×seeds, and in A6
(`*E`) MoE beats attention in 6/6.

The unifying reading: **an operator benefits from looping, and cares where it is looped, in
proportion to how much work its iterations actually do to the residual stream.**

> **Qualified 2026-08-19 by a direct test** ([update_norm_predictor.md](update_norm_predictor.md)).
> Measured on the unlooped baseline and scored against these twelve outcomes, update magnitude
> predicts **which Mamba layer to loop** (Spearman +0.80 within the family) but **not which operator**
> (it ranks M > E > `*` where the gains rank M > `*` > E) and **not the MoE curve** (Spearman 0.000 --
> no monotone statistic can represent a U). The coincidence below was read partly against Wave 2's
> A1 < A2 < A3 ordering, which this sweep did not reproduce for the MoE/attention pair. The mechanism
> is real but local: it is about position within the recurrent operator, not about operator choice. Attention's
iterations barely move the stream, so looping it buys little and it does not matter where. This also
explains the depth diagnostic's finding that A3 is nearly depth-agnostic and is genuinely better at
K=3 than at its trained K=4 — three independent measurements of the same underlying property.

Note this rehabilitates the *magnitude* hypothesis that the *diversity* hypothesis failed at:
Spearman(loss, update **cosine**) = −0.086 (dead), while the update **norm** ordering reproduces the
loop-benefit ordering exactly.

### Wave 2's headline survives; its interior does not

* **"Looping Mamba is the best loop"** — **CONFIRMED under position control.** Mamba wins at its best
  position (2.5017 vs 2.5246 / 2.5287), wins on the family mean (2.5256 vs 2.5308 / 2.5298), and
  wins by 5-6x per unit of compute, while being the cheapest family.
* **"MoE beats attention"** — **NOT REPRODUCED HERE.** On family means they are 2.5308 vs 2.5298,
  i.e. **attention is nominally ahead** by 0.5 s.d.; on best position MoE leads by only 2.0 s.d.; per
  unit of compute they are 0.0038 vs 0.0032, a tie. Wave 2 put the gap at 5.6x its pooled s.d.
  **This is evidence against that gap, not a refutation of it**, because the sweep is not a
  decomposition of A2 and A3: those arms loop *every* layer of their type at a FLOP-matched K (MoE
  K=2 across five layers, attention K=4 across two), whereas these arms loop *one* layer at K=6. The
  two designs aggregate over position differently and give attention a different share of the budget.
  What can be said is that the single-layer sweep gives no support for an MoE-over-attention ordering
  and slightly favours the reverse. Settling it needs a FLOP-matched single-layer sweep (section 4,
  "what is not settled").
* **"Position is a stack property"** — **REFUTED.** The three curves have three different shapes
  (monotone-after-peak, U, flat) and differ 23x in amplitude.

### What is not settled

* **n = 1 per arm**, with an s.d. borrowed from a different arm's replication. The large effects
  (Mamba's 23.7 s.d. spread, its position-9 reversal, the 23x P/R amplitude ratio) are far outside
  any plausible noise floor. The MoE U-shape rests on 9.1 s.d. of spread and should be replicated
  before it is built on. The MoE-vs-attention correction is a ~0.5-2 s.d. effect and is stated here
  as "not supported", **not** as "attention beats MoE".
* **Why index 2 and not 0** for Mamba, and **why the MoE curve is U-shaped**, are unexplained.
* **K is confounded with family cost.** These curves are the K=6 curves. A FLOP-matched sweep
  (MoE K=3, attention K=9) would test whether the shapes are properties of the operator or of the
  amount of compute added.
* **One loop group, not five.** Wave 2's arms loop *every* layer of a type. These sweep a single
  layer, so they do not directly decompose a Wave 2 arm; P1 (17 exec) reaches 2.5017 against A1's
  2.4951 (22 exec), so one well-placed loop recovers most but not all of looping all five.

## 5. Reproducing

```bash
# generate (all three families; --operators M E '*' to restrict)
python config_files/nemotron/loop_ablation_position_sweep/generate_arm_configs.py
python config_files/nemotron/loop_ablation_position_sweep/generate_warmstart_configs.py

# train
sbatch config_files/nemotron/loop_ablation_position_sweep/run_position_sweep.sh                # P, array 1-5
sbatch config_files/nemotron/loop_ablation_position_sweep/run_position_sweep_moe_attention.sh  # Q+R, array 1-7

# collate straight from the offline wandb transaction logs (no sync, no network)
/leonardo_work/EUHPC_D21_101/mfrey/wandb_env/bin/python scripts/collate_offline_wandb.py \
    --arms P0_loop_mamba_at_0 ... R1_loop_attention_at_10 \
    --validate-against docs/loopotron/wave2_final_stats.json \
    --output docs/loopotron/position_sweep_stats.json
```

`--validate-against` re-derives all 33 published Wave 2 figures through the same extractor and fails
on any disagreement; it currently reproduces them at max |Δ| = 0. Keep using it: the metric keys are
not self-describing (`test WeightedSumLoss` is the held-out LM loss, while MATH and TriviaQA must be
read from `answer_nll` rather than their WeightedSumLoss), and the wrong key yields plausible numbers
that are quietly the wrong quantity.

Measured wall clock, 1 node (4x A100) per arm: P 11h02–11h14, Q 12h59–13h02, R 09h44–09h45.
