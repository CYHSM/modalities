# Frozen configs: the two loop refinements that were removed

These seven configs are the exact arms that produced the `parcae`, `parcae_norm` and `all` rows of
Table 6 in `docs/loopotron/loopotron.tex`. **They no longer run.** The two mechanisms they configure
were deleted from `modalities.models.nemotron.nemotron_loop` after the ablation, and `LoopConfig`
now rejects `variant: parcae` and `injection_norm_config` explicitly rather than parsing them into
an ignored bucket — so an attempt to launch one of these fails at config validation instead of
quietly training a plain loop under a refined arm's name.

They are kept because they are the provenance of published numbers, not because they are usable.

## Why the refinements were removed

Both were ablated at K ∈ {3, 6, 12} on A1's Mamba loop, one node (two for K=12), 5B tokens, n=1 per
arm with the within-arm s.d. of 0.0021 borrowed from A1's four-run replication.

**Stabilized recurrence** (Parcae, arXiv:2604.12946) — bounded the loop's transition eigenvalues in
`(0, 1)` by construction. K=12 is the regime it was designed for and the instability there was
genuine (peak post-warmup gradient norm 85,782 for the plain loop, 104 steps above 100, where K=3
and K=6 never exceeded 5.1). Against that real instability it moved the loss by **−0.0003** — 0.2
within-arm s.d., indistinguishable from nothing — and produced *more* large-gradient steps than
plain looping (229 vs 104). It bought nothing at any depth reachable here.

**Injection normalization** (Parcae's "prelude norm") — **+0.0475 nats at K=12**, 22.6 s.d. in the
wrong direction, and the worst gradient behaviour of any arm (563 steps above 100). It also
dominated every combination it appeared in: FiLM alone was −0.0109, while FiLM + recurrence + norm
was **+0.0186**. A single bundled "all refinements" arm would have read as a mild failure and hidden
both the benefit and the larger harm — which is the wave's strongest argument for ablating
components individually.

**FiLM iteration conditioning survived** and is the one refinement still in the model. It helped at
every depth with the margin growing in K (−0.0001, −0.0079, −0.0109) and cut the peak gradient norm
at K=12 by a factor of ~37.

## Caveats attached to these numbers

Every arm is n=1. Sub-s.d. differences — the recurrence at either depth, and the whole K=3 pair —
are not results. Gradient norm here is pre-clipping and is a proxy for, not a measurement of, the
residual growth the recurrence actually bounds. The K=12 arms ran at world size 8 against 4 for
K=3/K=6, so K=6 → K=12 *loss* comparisons carry a data-order confound; the four-orders-of-magnitude
gradient difference is far past what data order explains.
