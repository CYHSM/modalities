# Can a baseline's update norms tell you where to loop?

**Verdict: partially, and not enough to replace the sweep.** The predictor recovers the study's
headline — *loop a Mamba layer near the top of the stack* — from one forward pass over the unlooped
baseline, but it picks the wrong Mamba layer, it is blind to the MoE curve entirely, and it gets the
MoE-vs-attention ordering backwards. Cost of the test: **3m53s of GPU time**, against the ~140 node-
hours of the sweep it was trying to replace.

## The idea

The position sweep ([position_sweep.md](position_sweep.md)) trained twelve arms — one layer looped at
K=6, at each of the twelve positions the base pattern offers — to find where depth pays off. Three
orderings coincided there (loop benefit, position sensitivity, update magnitude), suggesting one
underlying quantity. If that reading is right, the answer should be readable from the **unlooped
baseline alone**:

* predictor `x_i` = A0's layer-`i` update norm relative to **that layer's own input**, per-token median
* outcome `y_i` = `A0 loss − (loss of the arm that loops layer i at K=6)`

Both indexed by built layer index 0..11. A0 has exactly one layer per index; the sweep has exactly
one arm per index. `relative_to_own_input` was pre-specified as primary because it is the
normalization the earlier `member_step_norms` result used; the absolute norm is reported only as
robustness. The hypothesis is directional: bigger updates → bigger gain.

## Result

A0 baseline loss 2.5385. Predictor measured on the step-75,000 checkpoint, 8 test sequences.

| idx | type | arm | input norm | \|update\| | **upd/own** | loop LM | gain (s.d.) |
|---|---|---|---|---|---|---|---|
| 0 | M | `P0_loop_mamba_at_0` | 1.82 | 8.4 | **4.6998** | 2.5166 | +10.4 |
| 1 | E | `Q0_loop_moe_at_1` | 8.49 | 3.1 | 0.3435 | 2.5246 | +6.6 |
| 2 | M | `P1_loop_mamba_at_2` | 9.77 | 6.7 | 0.6775 | **2.5017** | **+17.5** |
| 3 | `*` | `R0_loop_attention_at_3` | 12.53 | 5.7 | 0.4869 | 2.5287 | +4.6 |
| 4 | E | `Q1_loop_moe_at_4` | 15.48 | 10.9 | 0.6590 | 2.5438 | −2.5 |
| 5 | M | `P2_loop_mamba_at_5` | 21.45 | 11.5 | 0.5267 | 2.5287 | +4.6 |
| 6 | E | `Q2_loop_moe_at_6` | 26.15 | 18.9 | 0.7012 | 2.5344 | +1.9 |
| 7 | M | `P3_loop_mamba_at_7` | 38.61 | 22.9 | 0.6144 | 2.5293 | +4.4 |
| 8 | E | `Q3_loop_moe_at_8` | 47.39 | 38.1 | 0.8180 | 2.5265 | +5.7 |
| 9 | M | `P4_loop_mamba_at_9` | 72.31 | 36.2 | 0.5025 | 2.5515 | −6.2 |
| 10 | `*` | `R1_loop_attention_at_10` | 84.49 | 52.3 | 0.6087 | 2.5309 | +3.6 |
| 11 | E | `Q4_loop_moe_at_11` | 114.33 | 199.7 | 1.7411 | 2.5248 | +6.5 |

| Statistic | Value | Reading |
|---|---|---|
| **Pooled Spearman, n=12** | **+0.294** | right sign, not significant (p≈0.35) |
| Pooled Pearson | +0.334 | — |
| **Within Mamba, n=5** | **+0.800** | works where position matters most |
| Within MoE, n=5 | **0.000** | fails completely |
| Within attention | n=2 | too few |
| Robustness: *absolute* update norm | **−0.385** | **wrong sign** |
| **Top-1 pick** | idx 0, true best idx 2 | ranks **2 of 12**; regret 0.0149 nats (7.1 s.d.) |

## What it gets right

* **The dominant axis.** Within the Mamba family — the family with the 0.0498-nat spread, i.e. the
  one worth getting right — Spearman is **+0.800**. It correctly identifies that late Mamba layers
  are bad places to loop, including that index 9 is the worst.
* **As a screen, it lands on the podium.** Given only the baseline and one pick out of twelve, it
  chooses the **second-best** arm. It would have told you "loop a Mamba layer near the top", which is
  the sweep's headline.
* **Normalization matters and the pre-specification was right.** The *absolute* update norm
  correlates **−0.385**, i.e. actively anti-predictive, because it grows monotonically with depth
  (8.4 → 199.7) along with the residual stream itself, and deeper is worse. Had the primary statistic
  been chosen after seeing the data, this is the trap.

## What it gets wrong, and why

* **It picks index 0 over index 2**, costing 0.0149 nats — 7.1 s.d., larger than most effects in this
  study. The cause is visible in the table: **layer 0's input is the bare embedding** (norm 1.82,
  against 8.5–114 everywhere else), so any update it makes looks ~7x larger in relative terms than
  the same update would elsewhere. The ratio is inflated by where the layer sits, not by what it does.
* **It is blind to the MoE U-shape.** MoE gain is best at both ends (indices 1 and 11) and worst in
  the middle — no monotone statistic can represent that, and Spearman is exactly 0.000. Worse, the
  predictor is *anti*-aligned at one end: index 1 has the **lowest** update norm of all twelve
  (0.3435) and one of the best gains (+6.6 s.d.).
* **It gets the operator ordering wrong.** Mean update norm ranks M (1.40) > E (0.85) > `*` (0.55),
  but mean gain ranks M (+0.0129) > `*` (+0.0086) > E (+0.0077). The E/`*` pair is inverted.

**This also revises the "three coincident orderings" claim** in
[representation_diagnostics.md](representation_diagnostics.md) and
[position_sweep.md](position_sweep.md). That coincidence was read partly against Wave 2's loss
ordering (A1 < A2 < A3), which the position sweep did not reproduce for the MoE/attention pair. Under
position control the orderings agree on **Mamba first** and disagree below it. The mechanism is real
but weaker and more local than the coincidence suggested: **update magnitude tracks which Mamba layer
to loop, not which operator to loop.**

## What was deliberately not done

Two normalizations would plausibly fix the layer-0 artifact — dividing by the layer's *output* norm
(bounded, and insensitive to a small input) or by a stack-wide running scale. **Neither was tried**,
because the primary statistic was pre-specified and trying variants until one correlates is exactly
the error that produced this study's retracted 3.9-sigma reasoning claim (loopotron.tex, Finding 4).
If it is worth pursuing, it should be pre-registered against a *held-out* set of positions — for
instance fit on the Mamba family and test on MoE — rather than re-scored on these twelve.

## Bottom line

The sweep is **not** replaceable by this proxy. What the proxy buys, for four minutes of GPU time, is
a decent prior on the single most important question (which operator, roughly where) and no
information at all about the finer structure. Read it as a screen, not as an answer.

## Reproducing

```bash
sbatch scripts/run_layer_profile.sh                       # A0_baseline, ~4 min on 1 GPU
python scripts/analyze_update_norm_predictor.py           # -> docs/loopotron/update_norm_predictor.json
pytest tests/analysis/test_loop_updates.py -o addopts=""  # 21 tests, incl. 3 pinning layer_profile
```

`LoopUpdateRecorder.layer_profile()` is the measurement. It differs from `stack_report()` in
normalizing each layer's update by that layer's **own** input rather than by the stack's input; a
test pins the distinction, since the stack-relative version is the one that runs 4.7 → 111 with depth
and is anti-predictive.
