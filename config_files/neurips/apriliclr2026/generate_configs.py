#!/usr/bin/env python3
"""
Param-matched AND FLOP-matched baseline generator for asymmetric Dual-Path
(Wide + Deep loop) transformers.

Two top-level matching modes:

  1. PARAM-MATCHED (existing): you specify ffn_deep on the dual; baselines
     are sized so total parameter counts match. FLOPs end up unequal.

  2. FLOP-MATCHED (new): you specify a per-layer FLOP budget; all three
     architectures are sized to spend the same FLOPs/token/layer. Parameters
     end up unequal — and that's the point. For an architecture that
     allocates compute per token, FLOPs are the right resource to hold fixed.

For dual in FLOP-matched mode, you also specify --alpha: the fraction of the
per-layer FLOP budget assigned to the deep path. alpha=0.5 is the natural
default (50/50 split). alpha is a knob for ablation — sweep it to find where
the deep/wide tradeoff lives.

Subcommands:
  single             param-matched, one dual + baselines
  sweep              param-matched, sweep ffn_deep
  flop-single        FLOP-matched, one budget + baselines (+ optional dual α grid)
  flop-sweep         FLOP-matched, sweep budgets
  flop-alpha-sweep   FLOP-matched, fix budget, sweep dual α only

The existing param-matched code path is untouched. New code lives below the
"FLOP-MATCHED MODE" banner.
"""

import argparse
import math
import os
import re


# =====================================================================
# FLOP accounting — single source of truth
# =====================================================================
#
# Per-layer FLOPs, per token, ignoring sequence-length-dependent attention
# (n^2 * d) terms which are architecture-independent at matched d_model and
# seq_len. We count projection FLOPs only.
#
#   attn_proj  = 4 d^2 + 4 d^2 / n_rep          # Q + O full, K + V reduced
#   swiglu     = 6 d * ffn_hidden                # gate + up + down (no bias)
#   layernorm  ~ 0 (we ignore these; reviewers never quibble about LN FLOPs)
#
# Dual layer:
#   deep  = max_loops * (attn_proj + swiglu(ffn_deep))
#   wide  = attn_proj + swiglu(ffn_wide)
#   gate  = 2 d  (Linear(d, 1) per token, single-gate formulation)
#   router= max_loops * 2 * (d + 1)              # AdaptiveRouter in deep loop
#
# =====================================================================


def attn_proj_flops(d_model: int, n_rep: int = 1) -> int:
    """QKV+output projection FLOPs per token. Ignores n^2*d attention term."""
    return 4 * d_model * d_model + 4 * (d_model * d_model) // n_rep


def swiglu_flops(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU FFN FLOPs per token: gate + up + down projections."""
    return 6 * d_model * ffn_hidden


def gate_flops(d_model: int) -> int:
    """Single-gate DualPathGate: Linear(d, 1)."""
    return 2 * d_model


def router_flops(d_model: int, max_loops: int) -> int:
    """AdaptiveRouter inside deep loop: max_loops * Linear(d+1, 1)."""
    return max_loops * 2 * (d_model + 1)


def deep_per_layer_flops(d_model: int, ffn_deep: int, max_loops: int, n_rep: int = 1) -> int:
    """Deep path: attention + SwiGLU, looped max_loops times."""
    return max_loops * (attn_proj_flops(d_model, n_rep) + swiglu_flops(d_model, ffn_deep))


def wide_per_layer_flops(d_model: int, ffn_wide: int, n_rep: int = 1) -> int:
    """Wide path: attention + SwiGLU, single pass."""
    return attn_proj_flops(d_model, n_rep) + swiglu_flops(d_model, ffn_wide)


def dual_layer_flops(d_model: int, ffn_deep: int, ffn_wide: int,
                     max_loops: int, n_rep: int = 1) -> dict:
    """Break down dual layer FLOPs into components for transparent reporting."""
    deep = deep_per_layer_flops(d_model, ffn_deep, max_loops, n_rep)
    wide = wide_per_layer_flops(d_model, ffn_wide, n_rep)
    g = gate_flops(d_model)
    r = router_flops(d_model, max_loops)
    return {"deep": deep, "wide": wide, "gate": g, "router": r,
            "total": deep + wide + g + r}


# =====================================================================
# Parameter counting
# =====================================================================

def attn_params(d_model: int, n_rep: int = 1) -> int:
    """Attention projection params (no bias)."""
    return 2 * d_model * d_model + 2 * (d_model * d_model) // n_rep


def swiglu_params(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU params (no bias): gate + up + down."""
    return 3 * d_model * ffn_hidden


def block_params(d_model: int, ffn_hidden: int, n_rep: int = 1) -> int:
    """One GPT2Block: attn + SwiGLU + 2 * RMSNorm."""
    return attn_params(d_model, n_rep) + swiglu_params(d_model, ffn_hidden) + 2 * d_model


def dual_layer_params(d_model: int, ffn_deep: int, ffn_wide: int,
                      max_loops: int, n_rep: int = 1) -> int:
    """Full AdaptiveRecursiveBlock in dual mode."""
    p = 0
    p += block_params(d_model, ffn_deep, n_rep)     # deep block
    p += block_params(d_model, ffn_wide, n_rep)     # wide block
    p += (d_model + 1) + 1                          # AdaptiveRouter: Linear(d+1, 1)
    p += max_loops                                  # loop_scales
    p += 1                                          # wide_scale
    p += d_model + 1                                # DualPathGate: Linear(d, 1)
    return p


def loop_layer_params(d_model: int, ffn_loop: int, max_loops: int, n_rep: int = 1) -> int:
    """Pure-loop AdaptiveRecursiveBlock (layer_type='loop')."""
    p = block_params(d_model, ffn_loop, n_rep)
    p += (d_model + 1) + 1                          # router
    p += max_loops                                  # loop_scales
    return p


def wide_layer_params(d_model: int, ffn_wide: int, n_rep: int = 1) -> int:
    """Pure-wide AdaptiveRecursiveBlock (layer_type='wide')."""
    p = block_params(d_model, ffn_wide, n_rep)
    p += 1                                          # wide_scale
    return p


def shared_params(d_model: int, vocab_size: int, use_weight_tying: bool) -> int:
    """Embeddings + lm_head + final norm."""
    s = vocab_size * d_model                        # wte
    if not use_weight_tying:
        s += d_model * vocab_size                   # lm_head
    s += d_model                                    # lm_head_norm
    return s


def total_params(d_model: int, n_layers: int, per_layer_params: int,
                 vocab_size: int, use_weight_tying: bool) -> int:
    return n_layers * per_layer_params + shared_params(d_model, vocab_size, use_weight_tying)


# =====================================================================
# Solvers (param-matched mode — unchanged)
# =====================================================================

def solve_ffn_wide_flop_match(d_model: int, ffn_deep: int, max_loops: int,
                              wide_flop_ratio: float = 1.0,
                              ffn_round_multiple: int = 64,
                              n_rep: int = 1) -> tuple[int, int, int]:
    """
    Solve for ffn_wide such that
        wide_per_layer_flops = wide_flop_ratio * deep_per_layer_flops

    Returns (ffn_wide_rounded, deep_flops, wide_flops_at_rounded).
    """
    attn = attn_proj_flops(d_model, n_rep)
    deep_flops = deep_per_layer_flops(d_model, ffn_deep, max_loops, n_rep)
    target_wide_flops = wide_flop_ratio * deep_flops

    ffn_wide_exact = (target_wide_flops - attn) / (6.0 * d_model)
    if ffn_wide_exact <= 0:
        raise ValueError(f"ffn_wide solve: target {target_wide_flops:.0f} < attn {attn}; "
                         f"wide_flop_ratio too small or ffn_deep too small.")

    ffn_wide = max(ffn_round_multiple,
                   int(round(ffn_wide_exact / ffn_round_multiple)) * ffn_round_multiple)
    actual_wide_flops = wide_per_layer_flops(d_model, ffn_wide, n_rep)
    return ffn_wide, deep_flops, actual_wide_flops


def solve_ffn_to_match_params(target_params: int, d_model: int, n_layers: int,
                              max_loops: int, layer_type: str,
                              vocab_size: int, use_weight_tying: bool,
                              ffn_round_multiple: int = 64,
                              n_rep: int = 1) -> tuple[int, int]:
    """Param-match solve for single-path baseline FFN size."""
    shared = shared_params(d_model, vocab_size, use_weight_tying)
    attn_p = attn_params(d_model, n_rep)
    two_norms = 2 * d_model

    if layer_type == "loop":
        extras = (d_model + 1) + 1 + max_loops
    elif layer_type == "wide":
        extras = 1
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    per_layer_nonffn = attn_p + two_norms + extras
    target_per_layer = (target_params - shared) / n_layers
    ffn_exact = (target_per_layer - per_layer_nonffn) / (3.0 * d_model)

    if ffn_exact <= 0:
        raise ValueError(f"Param-match solve for '{layer_type}': target too small, "
                         f"got ffn_exact={ffn_exact:.0f}")

    ffn = max(ffn_round_multiple,
              int(round(ffn_exact / ffn_round_multiple)) * ffn_round_multiple)

    if layer_type == "loop":
        per_layer = loop_layer_params(d_model, ffn, max_loops, n_rep)
    else:
        per_layer = wide_layer_params(d_model, ffn, n_rep)
    actual = total_params(d_model, n_layers, per_layer, vocab_size, use_weight_tying)
    return ffn, actual


# =====================================================================
# Top-level config builders (param-matched — unchanged)
# =====================================================================

def build_dual(d_model: int, n_layers: int, max_loops: int, ffn_deep: int,
               vocab_size: int = 50304, use_weight_tying: bool = False,
               n_head_q: int = None, n_head_kv: int = None,
               wide_flop_ratio: float = 1.0,
               ffn_round_multiple: int = 64) -> dict:
    """Build one dual model config (param-matched mode)."""
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1

    ffn_wide, deep_flops, wide_flops = solve_ffn_wide_flop_match(
        d_model=d_model, ffn_deep=ffn_deep, max_loops=max_loops,
        wide_flop_ratio=wide_flop_ratio,
        ffn_round_multiple=ffn_round_multiple, n_rep=n_rep,
    )

    flop_breakdown = dual_layer_flops(d_model, ffn_deep, ffn_wide, max_loops, n_rep)
    per_layer_p = dual_layer_params(d_model, ffn_deep, ffn_wide, max_loops, n_rep)
    total_p = total_params(d_model, n_layers, per_layer_p, vocab_size, use_weight_tying)

    return {
        "kind": "dual",
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": max_loops,
        "ffn_deep": ffn_deep,
        "ffn_wide": ffn_wide,
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_rep": n_rep,
        "vocab_size": vocab_size,
        "use_weight_tying": use_weight_tying,
        "params_per_layer": per_layer_p,
        "total_params": total_p,
        "flops_per_layer": flop_breakdown,
        "total_flops": n_layers * flop_breakdown["total"],
        "wide_flop_ratio_actual": wide_flops / deep_flops if deep_flops > 0 else 0.0,
    }


def build_pure_loop_baseline(dual: dict, ffn_round_multiple: int = 64) -> dict:
    """Pure-loop baseline, param-matched to the dual model."""
    d_model = dual["d_model"]
    n_layers = dual["n_layers"]
    max_loops = dual["max_loops"]
    n_rep = dual["n_rep"]
    target = dual["total_params"]

    ffn_loop, actual_params = solve_ffn_to_match_params(
        target_params=target, d_model=d_model, n_layers=n_layers,
        max_loops=max_loops, layer_type="loop",
        vocab_size=dual["vocab_size"], use_weight_tying=dual["use_weight_tying"],
        ffn_round_multiple=ffn_round_multiple, n_rep=n_rep,
    )

    per_layer_flops = deep_per_layer_flops(d_model, ffn_loop, max_loops, n_rep) \
                      + router_flops(d_model, max_loops)
    mismatch_pct = (actual_params / target - 1.0) * 100

    return {
        "kind": "pure_loop",
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": max_loops,
        "ffn_loop": ffn_loop,
        "n_head_q": dual["n_head_q"],
        "n_head_kv": dual["n_head_kv"],
        "n_rep": n_rep,
        "vocab_size": dual["vocab_size"],
        "use_weight_tying": dual["use_weight_tying"],
        "total_params": actual_params,
        "target_params": target,
        "param_mismatch_pct": mismatch_pct,
        "flops_per_layer": per_layer_flops,
        "total_flops": n_layers * per_layer_flops,
    }


def build_pure_wide_baseline(dual: dict, ffn_round_multiple: int = 64) -> dict:
    """Pure-wide baseline, param-matched to the dual model. max_loops=1."""
    d_model = dual["d_model"]
    n_layers = dual["n_layers"]
    n_rep = dual["n_rep"]
    target = dual["total_params"]

    ffn_wide_bl, actual_params = solve_ffn_to_match_params(
        target_params=target, d_model=d_model, n_layers=n_layers,
        max_loops=1, layer_type="wide",
        vocab_size=dual["vocab_size"], use_weight_tying=dual["use_weight_tying"],
        ffn_round_multiple=ffn_round_multiple, n_rep=n_rep,
    )

    per_layer_flops = wide_per_layer_flops(d_model, ffn_wide_bl, n_rep)
    mismatch_pct = (actual_params / target - 1.0) * 100

    return {
        "kind": "pure_wide",
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": 1,
        "ffn_wide": ffn_wide_bl,
        "n_head_q": dual["n_head_q"],
        "n_head_kv": dual["n_head_kv"],
        "n_rep": n_rep,
        "vocab_size": dual["vocab_size"],
        "use_weight_tying": dual["use_weight_tying"],
        "total_params": actual_params,
        "target_params": target,
        "param_mismatch_pct": mismatch_pct,
        "flops_per_layer": per_layer_flops,
        "total_flops": n_layers * per_layer_flops,
    }


def build_mixed_sandwich(dual: dict, n_loop: int = None, n_wide: int = None,
                         ffn_round_multiple: int = 64) -> dict:
    """Mixed sandwich: some loop, some dual, some wide layers."""
    d_model = dual["d_model"]
    n_layers = dual["n_layers"]
    max_loops = dual["max_loops"]
    n_rep = dual["n_rep"]

    if n_loop is None:
        n_loop = n_layers // 3
    if n_wide is None:
        n_wide = n_layers // 3
    n_dual = n_layers - n_loop - n_wide

    if n_dual < 0:
        raise ValueError(f"Sandwich: n_loop+n_wide={n_loop+n_wide} > n_layers={n_layers}")

    ffn_deep = dual["ffn_deep"]
    ffn_wide = dual["ffn_wide"]

    layer_types = ["loop"] * n_loop + ["dual"] * n_dual + ["wide"] * n_wide

    p = 0
    f = 0
    for lt in layer_types:
        if lt == "loop":
            p += loop_layer_params(d_model, ffn_deep, max_loops, n_rep)
            f += deep_per_layer_flops(d_model, ffn_deep, max_loops, n_rep) \
                 + router_flops(d_model, max_loops)
        elif lt == "wide":
            p += wide_layer_params(d_model, ffn_wide, n_rep)
            f += wide_per_layer_flops(d_model, ffn_wide, n_rep)
        else:
            p += dual_layer_params(d_model, ffn_deep, ffn_wide, max_loops, n_rep)
            f += dual_layer_flops(d_model, ffn_deep, ffn_wide, max_loops, n_rep)["total"]

    total_p = p + shared_params(d_model, dual["vocab_size"], dual["use_weight_tying"])

    return {
        "kind": "sandwich",
        "d_model": d_model,
        "n_layers": n_layers,
        "n_loop_layers": n_loop,
        "n_dual_layers": n_dual,
        "n_wide_layers": n_wide,
        "layer_types": layer_types,
        "max_loops": max_loops,
        "ffn_deep": ffn_deep,
        "ffn_wide": ffn_wide,
        "n_head_q": dual["n_head_q"],
        "n_head_kv": dual["n_head_kv"],
        "n_rep": n_rep,
        "vocab_size": dual["vocab_size"],
        "use_weight_tying": dual["use_weight_tying"],
        "total_params": total_p,
        "target_params": dual["total_params"],
        "param_mismatch_pct": (total_p / dual["total_params"] - 1.0) * 100,
        "total_flops": f,
    }


# =====================================================================
# =====================================================================
# FLOP-MATCHED MODE
# =====================================================================
# =====================================================================
#
# All three architectures spend the same per-layer FLOPs/token. Parameters
# differ — pure_wide will have the most params (one wide MLP), pure_loop
# the fewest (one shared narrow MLP), dual sits between.
#
# Solver math:
#
#   pure_loop (max_loops iterations of attn + FFN, plus router):
#       F = max_loops * (attn + 6d * ffn_loop) + router_flops
#       ffn_loop = (F - router_flops - max_loops * attn) / (6d * max_loops)
#
#   pure_wide (single attn + FFN):
#       F = attn + 6d * ffn_wide
#       ffn_wide = (F - attn) / (6d)
#
#   dual at split alpha (deep gets alpha*F_arith, wide gets (1-alpha)*F_arith,
#   where F_arith = F - gate - router; gate and router are charged off the
#   top so the FFN+attn split is clean):
#       deep_target = alpha * F_arith
#       wide_target = (1-alpha) * F_arith
#       ffn_deep = (deep_target / max_loops - attn) / (6d)
#       ffn_wide = (wide_target - attn) / (6d)
#
# When alpha is too small or too large the implied FFN size goes
# non-positive — we surface that as a clear error rather than silently
# clamping. For a given budget, the feasible alpha range is
# [attn*max_loops/F_arith, 1 - attn/F_arith].
# =====================================================================


def solve_flop_matched_loop(flop_budget: int, d_model: int, max_loops: int,
                            n_rep: int = 1, ffn_round_multiple: int = 64
                            ) -> tuple[int, int]:
    """Solve ffn for pure-loop at given per-layer FLOP budget."""
    attn = attn_proj_flops(d_model, n_rep)
    router = router_flops(d_model, max_loops)
    ffn_exact = (flop_budget - router - max_loops * attn) / (6.0 * d_model * max_loops)
    if ffn_exact <= 0:
        raise ValueError(
            f"FLOP-matched pure_loop: budget {flop_budget:.2e} too small for d={d_model}, "
            f"max_loops={max_loops} (ffn_exact={ffn_exact:.0f}). "
            f"Need budget > {router + max_loops * attn:.2e} just to cover attn+router."
        )
    ffn = max(ffn_round_multiple,
              int(round(ffn_exact / ffn_round_multiple)) * ffn_round_multiple)
    actual_flops = max_loops * (attn + swiglu_flops(d_model, ffn)) + router
    return ffn, actual_flops


def solve_flop_matched_wide(flop_budget: int, d_model: int,
                            n_rep: int = 1, ffn_round_multiple: int = 64
                            ) -> tuple[int, int]:
    """Solve ffn for pure-wide at given per-layer FLOP budget."""
    attn = attn_proj_flops(d_model, n_rep)
    ffn_exact = (flop_budget - attn) / (6.0 * d_model)
    if ffn_exact <= 0:
        raise ValueError(
            f"FLOP-matched pure_wide: budget {flop_budget:.2e} too small for d={d_model} "
            f"(ffn_exact={ffn_exact:.0f}). Need budget > {attn:.2e}."
        )
    ffn = max(ffn_round_multiple,
              int(round(ffn_exact / ffn_round_multiple)) * ffn_round_multiple)
    actual_flops = attn + swiglu_flops(d_model, ffn)
    return ffn, actual_flops


def solve_flop_matched_dual(flop_budget: int, d_model: int, max_loops: int,
                            alpha: float, n_rep: int = 1,
                            ffn_round_multiple: int = 64
                            ) -> tuple[int, int, dict]:
    """Solve ffn_deep, ffn_wide for dual at given budget and split alpha."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    attn = attn_proj_flops(d_model, n_rep)
    gate = gate_flops(d_model)
    router = router_flops(d_model, max_loops)

    # Charge gate and router off the top, then split remaining FLOPs.
    f_arith = flop_budget - gate - router
    if f_arith <= 0:
        raise ValueError(f"Budget {flop_budget} insufficient even to cover gate+router "
                         f"({gate + router}).")

    deep_target = alpha * f_arith
    wide_target = (1.0 - alpha) * f_arith

    ffn_deep_exact = (deep_target / max_loops - attn) / (6.0 * d_model)
    ffn_wide_exact = (wide_target - attn) / (6.0 * d_model)

    if ffn_deep_exact <= 0:
        feasible_min = (max_loops * attn) / f_arith
        raise ValueError(
            f"FLOP-matched dual: alpha={alpha} too small at budget {flop_budget:.2e}. "
            f"Deep path can't even cover its attn cost (need ffn_deep>0, got "
            f"ffn_deep_exact={ffn_deep_exact:.0f}). Try alpha >= {feasible_min:.3f}."
        )
    if ffn_wide_exact <= 0:
        feasible_max = 1.0 - attn / f_arith
        raise ValueError(
            f"FLOP-matched dual: alpha={alpha} too large at budget {flop_budget:.2e}. "
            f"Wide path can't even cover its attn cost (need ffn_wide>0, got "
            f"ffn_wide_exact={ffn_wide_exact:.0f}). Try alpha <= {feasible_max:.3f}."
        )

    ffn_deep = max(ffn_round_multiple,
                   int(round(ffn_deep_exact / ffn_round_multiple)) * ffn_round_multiple)
    ffn_wide = max(ffn_round_multiple,
                   int(round(ffn_wide_exact / ffn_round_multiple)) * ffn_round_multiple)

    actual = dual_layer_flops(d_model, ffn_deep, ffn_wide, max_loops, n_rep)
    return ffn_deep, ffn_wide, actual


def build_flop_matched_dual(flop_budget: int, d_model: int, n_layers: int,
                            max_loops: int, alpha: float = 0.5,
                            vocab_size: int = 50304, use_weight_tying: bool = False,
                            n_head_q: int = None, n_head_kv: int = None,
                            ffn_round_multiple: int = 64) -> dict:
    """Dual at fixed per-layer FLOP budget with deep/wide split alpha."""
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1

    ffn_deep, ffn_wide, actual_flops = solve_flop_matched_dual(
        flop_budget=flop_budget, d_model=d_model, max_loops=max_loops,
        alpha=alpha, n_rep=n_rep, ffn_round_multiple=ffn_round_multiple,
    )

    per_layer_p = dual_layer_params(d_model, ffn_deep, ffn_wide, max_loops, n_rep)
    total_p = total_params(d_model, n_layers, per_layer_p, vocab_size, use_weight_tying)

    return {
        "kind": "dual",
        "match_mode": "flop",
        "flop_budget_per_layer": flop_budget,
        "alpha": alpha,
        "alpha_actual": actual_flops["deep"] / (actual_flops["deep"] + actual_flops["wide"]),
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": max_loops,
        "ffn_deep": ffn_deep,
        "ffn_wide": ffn_wide,
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_rep": n_rep,
        "vocab_size": vocab_size,
        "use_weight_tying": use_weight_tying,
        "params_per_layer": per_layer_p,
        "total_params": total_p,
        "flops_per_layer": actual_flops,
        "total_flops": n_layers * actual_flops["total"],
        "flop_match_pct": (actual_flops["total"] / flop_budget - 1.0) * 100,
    }


def build_flop_matched_loop(flop_budget: int, d_model: int, n_layers: int,
                            max_loops: int,
                            vocab_size: int = 50304, use_weight_tying: bool = False,
                            n_head_q: int = None, n_head_kv: int = None,
                            ffn_round_multiple: int = 64) -> dict:
    """Pure-loop at fixed per-layer FLOP budget."""
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1

    ffn_loop, actual_flops = solve_flop_matched_loop(
        flop_budget=flop_budget, d_model=d_model, max_loops=max_loops,
        n_rep=n_rep, ffn_round_multiple=ffn_round_multiple,
    )

    per_layer_p = loop_layer_params(d_model, ffn_loop, max_loops, n_rep)
    total_p = total_params(d_model, n_layers, per_layer_p, vocab_size, use_weight_tying)

    return {
        "kind": "pure_loop",
        "match_mode": "flop",
        "flop_budget_per_layer": flop_budget,
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": max_loops,
        "ffn_loop": ffn_loop,
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_rep": n_rep,
        "vocab_size": vocab_size,
        "use_weight_tying": use_weight_tying,
        "params_per_layer": per_layer_p,
        "total_params": total_p,
        "flops_per_layer": actual_flops,
        "total_flops": n_layers * actual_flops,
        "flop_match_pct": (actual_flops / flop_budget - 1.0) * 100,
    }


def build_flop_matched_wide(flop_budget: int, d_model: int, n_layers: int,
                            vocab_size: int = 50304, use_weight_tying: bool = False,
                            n_head_q: int = None, n_head_kv: int = None,
                            ffn_round_multiple: int = 64) -> dict:
    """Pure-wide at fixed per-layer FLOP budget. max_loops=1."""
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1

    ffn_wide, actual_flops = solve_flop_matched_wide(
        flop_budget=flop_budget, d_model=d_model,
        n_rep=n_rep, ffn_round_multiple=ffn_round_multiple,
    )

    per_layer_p = wide_layer_params(d_model, ffn_wide, n_rep)
    total_p = total_params(d_model, n_layers, per_layer_p, vocab_size, use_weight_tying)

    return {
        "kind": "pure_wide",
        "match_mode": "flop",
        "flop_budget_per_layer": flop_budget,
        "d_model": d_model,
        "n_layers": n_layers,
        "max_loops": 1,
        "ffn_wide": ffn_wide,
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_rep": n_rep,
        "vocab_size": vocab_size,
        "use_weight_tying": use_weight_tying,
        "params_per_layer": per_layer_p,
        "total_params": total_p,
        "flops_per_layer": actual_flops,
        "total_flops": n_layers * actual_flops,
        "flop_match_pct": (actual_flops / flop_budget - 1.0) * 100,
    }


# =====================================================================
# Experiment IDs
# =====================================================================

def make_exp_id(cfg: dict) -> str:
    dm = cfg["d_model"]
    L = cfg["n_layers"]
    match_tag = ""
    if cfg.get("match_mode") == "flop":
        # Embed FLOP budget in MFLOPs for sortability.
        budget_m = cfg["flop_budget_per_layer"] / 1e6
        match_tag = f"_F{budget_m:.0f}M"

    if cfg["kind"] == "dual":
        alpha_tag = ""
        if cfg.get("match_mode") == "flop":
            alpha_tag = f"_a{int(round(cfg['alpha']*100)):02d}"
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}{match_tag}"
                f"_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}_dual{alpha_tag}")
    if cfg["kind"] == "pure_loop":
        suffix = "_ipMdual" if cfg.get("match_mode") != "flop" else ""
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}{match_tag}"
                f"_ffnL{cfg['ffn_loop']}_pureloop{suffix}")
    if cfg["kind"] == "pure_wide":
        suffix = "_ipMdual" if cfg.get("match_mode") != "flop" else ""
        return f"dm{dm}_L{L}_loop1{match_tag}_ffnW{cfg['ffn_wide']}_purewide{suffix}"
    if cfg["kind"] == "sandwich":
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}"
                f"_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}"
                f"_sandwich_L{cfg['n_loop_layers']}D{cfg['n_dual_layers']}W{cfg['n_wide_layers']}")
    raise ValueError(f"Unknown kind: {cfg['kind']}")


# =====================================================================
# YAML emission (unchanged)
# =====================================================================

def _cfg_to_yaml_patches(cfg: dict) -> dict:
    """Map a config dict to (section, key) -> str patches for the template."""
    exp_id = make_exp_id(cfg)
    kind = cfg["kind"]

    patches = {
        ("model_raw", "n_layer"): str(cfg["n_layers"]),
        ("model_raw", "n_embd"): str(cfg["d_model"]),
        ("evaluation_subscriber", "experiment_id"): exp_id,
    }

    if cfg.get("n_head_q") is not None:
        patches[("model_raw", "n_head_q")] = str(cfg["n_head_q"])
    if cfg.get("n_head_kv") is not None:
        patches[("model_raw", "n_head_kv")] = str(cfg["n_head_kv"])
    if cfg.get("n_head_q") is not None:
        head_dim = cfg["d_model"] // cfg["n_head_q"]
        patches[("model_raw", "normalized_shape")] = f"{head_dim} # n_embd / n_head_q"

    if kind == "dual":
        patches[("model_raw", "ffn_hidden")] = str(cfg["ffn_deep"])
        patches[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        patches[("model_raw", "max_loops")] = str(cfg["max_loops"])
        patches[("model_raw", "enable_adaptive")] = "true"
    elif kind == "pure_loop":
        patches[("model_raw", "ffn_hidden")] = str(cfg["ffn_loop"])
        patches[("model_raw", "wide_ffn_hidden")] = "0"
        patches[("model_raw", "max_loops")] = str(cfg["max_loops"])
        patches[("model_raw", "enable_adaptive")] = "true"
    elif kind == "pure_wide":
        patches[("model_raw", "ffn_hidden")] = "64"
        patches[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        patches[("model_raw", "max_loops")] = "1"
        patches[("model_raw", "enable_adaptive")] = "true"
    elif kind == "sandwich":
        patches[("model_raw", "ffn_hidden")] = str(cfg["ffn_deep"])
        patches[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        patches[("model_raw", "max_loops")] = str(cfg["max_loops"])
        patches[("model_raw", "enable_adaptive")] = "true"

    return patches


def _patch_yaml(template_text: str, cfg: dict) -> str:
    """Walk the template, patching (section, key) lines to match the config."""
    patches = _cfg_to_yaml_patches(cfg)
    layer_types = cfg.get("layer_types")

    kv_re = re.compile(r'^(\s*)([\w_]+)(:\s+)(.+)$')
    lines = template_text.splitlines(keepends=True)
    out_lines = []
    current_section = None

    for line in lines:
        stripped = line.lstrip()

        if stripped and not stripped.startswith('#') and line[0] not in (' ', '\t', '\n', '\r'):
            colon_pos = stripped.find(':')
            if colon_pos > 0:
                current_section = stripped[:colon_pos].strip()

        m = kv_re.match(line)
        if m and current_section:
            indent, key, sep, old_val = m.group(1), m.group(2), m.group(3), m.group(4)
            if not old_val.lstrip().startswith('${'):
                patch_key = (current_section, key)
                if patch_key in patches:
                    line = f"{indent}{key}{sep}{patches[patch_key]}\n"

        out_lines.append(line)

        if (layer_types is not None
                and current_section == "model_raw"
                and m and m.group(2) == "max_loops"):
            indent = m.group(1)
            lt_str = ", ".join(f'"{t}"' for t in layer_types)
            out_lines.append(f"{indent}layer_types: [{lt_str}]\n")

    return ''.join(out_lines)


def write_yaml(template_path: str, output_dir: str, cfg: dict) -> str:
    """Load template, patch, write."""
    with open(template_path, "r") as f:
        text = f.read()

    patched = _patch_yaml(text, cfg)
    exp_id = make_exp_id(cfg)
    out_path = os.path.join(output_dir, f"{exp_id}.yaml")

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(patched)
    return out_path


# =====================================================================
# Pretty printing
# =====================================================================

def _fmt_params(p: int) -> str:
    if p >= 1e9: return f"{p/1e9:.2f}B"
    if p >= 1e6: return f"{p/1e6:.1f}M"
    if p >= 1e3: return f"{p/1e3:.1f}k"
    return str(p)


def _fmt_flops(f: int) -> str:
    if f >= 1e12: return f"{f/1e12:.2f}T"
    if f >= 1e9:  return f"{f/1e9:.2f}G"
    if f >= 1e6:  return f"{f/1e6:.1f}M"
    return f"{f/1e3:.1f}k"


def print_config_summary(dual: dict, loop: dict, wide: dict,
                         sandwich: dict = None):
    """Param-matched summary printer (unchanged)."""
    print(f"\n  {'─'*78}")
    print(f"  Dual config: d_model={dual['d_model']}  L={dual['n_layers']}  "
          f"loops={dual['max_loops']}  ffn_deep={dual['ffn_deep']}  "
          f"ffn_wide={dual['ffn_wide']}")
    print(f"  {'─'*78}")

    fb = dual["flops_per_layer"]
    ratio = dual["wide_flop_ratio_actual"]
    print(f"    Dual layer FLOPs/token (projections only):")
    print(f"      deep:   {_fmt_flops(fb['deep'])}   (attn x{dual['max_loops']} + SwiGLU x{dual['max_loops']})")
    print(f"      wide:   {_fmt_flops(fb['wide'])}   (attn + SwiGLU, single pass)")
    print(f"      gate:   {_fmt_flops(fb['gate'])}")
    print(f"      router: {_fmt_flops(fb['router'])}")
    print(f"      total:  {_fmt_flops(fb['total'])}")
    print(f"    Wide/Deep FLOP ratio (target was 1.0): {ratio:.3f}")

    print(f"\n    {'Config':<14s} {'n_layers':>9s} {'loops':>6s} "
          f"{'ffn_deep':>9s} {'ffn_wide':>9s} {'params':>9s} "
          f"{'Δparam':>8s} {'FLOPs/tok':>11s}")
    print(f"    {'-'*14} {'-'*9} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*11}")

    print(f"    {'dual':<14s} {dual['n_layers']:>9d} {dual['max_loops']:>6d} "
          f"{dual['ffn_deep']:>9d} {dual['ffn_wide']:>9d} "
          f"{_fmt_params(dual['total_params']):>9s} {'(ref)':>8s} "
          f"{_fmt_flops(dual['total_flops']):>11s}")

    dual_flops = dual['total_flops']
    for b, label in [(loop, "pure_loop"), (wide, "pure_wide")]:
        ffn_d = b.get("ffn_loop", "-") if b["kind"] == "pure_loop" else "-"
        ffn_w = "-" if b["kind"] == "pure_loop" else b.get("ffn_wide", "-")
        print(f"    {label:<14s} {b['n_layers']:>9d} {b['max_loops']:>6d} "
              f"{str(ffn_d):>9s} {str(ffn_w):>9s} "
              f"{_fmt_params(b['total_params']):>9s} "
              f"{b['param_mismatch_pct']:>+7.2f}% "
              f"{_fmt_flops(b['total_flops']):>11s}")

    if sandwich is not None:
        print(f"    {'sandwich':<14s} {sandwich['n_layers']:>9d} {sandwich['max_loops']:>6d} "
              f"{sandwich['ffn_deep']:>9d} {sandwich['ffn_wide']:>9d} "
              f"{_fmt_params(sandwich['total_params']):>9s} "
              f"{sandwich['param_mismatch_pct']:>+7.2f}% "
              f"{_fmt_flops(sandwich['total_flops']):>11s}")

    flop_overhead_loop = (dual_flops / loop['total_flops'] - 1) * 100
    flop_overhead_wide = (dual_flops / wide['total_flops'] - 1) * 100
    print(f"\n    Dual FLOP overhead vs baselines: "
          f"{flop_overhead_loop:+.0f}% vs pure_loop, "
          f"{flop_overhead_wide:+.0f}% vs pure_wide")

    for b in [loop, wide]:
        if abs(b["param_mismatch_pct"]) > 2.0:
            print(f"    ⚠  {b['kind']} param mismatch {b['param_mismatch_pct']:+.2f}% "
                  f"(>2%); consider a different ffn_round_multiple.")


def print_flop_matched_summary(budget: int, d_model: int, n_layers: int, max_loops: int,
                               loop_cfg: dict, wide_cfg: dict, dual_cfgs: list[dict]):
    """FLOP-matched summary: shows that all configs spend the same FLOPs."""
    print(f"\n  {'─'*86}")
    print(f"  FLOP-matched at {_fmt_flops(budget)}/tok per layer  "
          f"(d_model={d_model}  L={n_layers}  loops={max_loops})")
    print(f"  {'─'*86}")

    print(f"    {'Config':<22s} {'ffn_deep':>9s} {'ffn_wide':>9s} "
          f"{'params':>9s} {'FLOPs/tok':>11s} {'ΔFLOP':>8s}")
    print(f"    {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*8}")

    print(f"    {'pure_loop':<22s} "
          f"{loop_cfg['ffn_loop']:>9d} {'-':>9s} "
          f"{_fmt_params(loop_cfg['total_params']):>9s} "
          f"{_fmt_flops(loop_cfg['total_flops']):>11s} "
          f"{loop_cfg['flop_match_pct']:>+7.2f}%")

    print(f"    {'pure_wide':<22s} "
          f"{'-':>9s} {wide_cfg['ffn_wide']:>9d} "
          f"{_fmt_params(wide_cfg['total_params']):>9s} "
          f"{_fmt_flops(wide_cfg['total_flops']):>11s} "
          f"{wide_cfg['flop_match_pct']:>+7.2f}%")

    for dc in dual_cfgs:
        label = f"dual α={dc['alpha']:.2f}"
        print(f"    {label:<22s} "
              f"{dc['ffn_deep']:>9d} {dc['ffn_wide']:>9d} "
              f"{_fmt_params(dc['total_params']):>9s} "
              f"{_fmt_flops(dc['total_flops']):>11s} "
              f"{dc['flop_match_pct']:>+7.2f}%")

    # Param spread — useful context, not a failure mode in FLOP-matched mode.
    all_params = ([loop_cfg['total_params'], wide_cfg['total_params']]
                  + [dc['total_params'] for dc in dual_cfgs])
    p_min, p_max = min(all_params), max(all_params)
    spread = (p_max / p_min - 1) * 100
    print(f"\n    Param spread across configs: {_fmt_params(p_min)} → {_fmt_params(p_max)} "
          f"({spread:+.0f}%) — expected at fixed FLOPs.")

    # Sanity check: warn on large FLOP mismatch from rounding.
    for cfg in [loop_cfg, wide_cfg] + dual_cfgs:
        if abs(cfg["flop_match_pct"]) > 5.0:
            label = cfg["kind"] if cfg["kind"] != "dual" else f"dual α={cfg['alpha']:.2f}"
            print(f"    ⚠  {label} FLOP mismatch {cfg['flop_match_pct']:+.2f}% "
                  f"(>5%); consider a smaller --ffn-round-multiple.")


# =====================================================================
# CLI entry points
# =====================================================================

def cmd_single(args):
    """Param-matched: emit one dual + its baselines."""
    dual = build_dual(
        d_model=args.d_model, n_layers=args.n_layers,
        max_loops=args.max_loops, ffn_deep=args.ffn_deep,
        vocab_size=args.vocab_size, use_weight_tying=args.weight_tying,
        n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        wide_flop_ratio=args.wide_flop_ratio,
        ffn_round_multiple=args.ffn_round_multiple,
    )
    loop_bl = build_pure_loop_baseline(dual, ffn_round_multiple=args.ffn_round_multiple)
    wide_bl = build_pure_wide_baseline(dual, ffn_round_multiple=args.ffn_round_multiple)

    sandwich = None
    if args.sandwich:
        sandwich = build_mixed_sandwich(
            dual, n_loop=args.sandwich_n_loop, n_wide=args.sandwich_n_wide,
            ffn_round_multiple=args.ffn_round_multiple,
        )

    print_config_summary(dual, loop_bl, wide_bl, sandwich)

    if args.template:
        print(f"\n  Writing YAMLs to {args.output_dir}/")
        for cfg in [dual, loop_bl, wide_bl] + ([sandwich] if sandwich else []):
            p = write_yaml(args.template, args.output_dir, cfg)
            print(f"    ✓ {p}")


def cmd_sweep(args):
    """Param-matched: emit one dual + baselines per ffn_deep value."""
    print(f"\n  Sweeping ffn_deep over: {args.ffn_deep}")
    print(f"  d_model={args.d_model}  L={args.n_layers}  loops={args.max_loops}")

    all_cfgs = []
    for ffn_deep in args.ffn_deep:
        try:
            dual = build_dual(
                d_model=args.d_model, n_layers=args.n_layers,
                max_loops=args.max_loops, ffn_deep=ffn_deep,
                vocab_size=args.vocab_size, use_weight_tying=args.weight_tying,
                n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
                wide_flop_ratio=args.wide_flop_ratio,
                ffn_round_multiple=args.ffn_round_multiple,
            )
        except ValueError as e:
            print(f"\n  ⚠  Skipping ffn_deep={ffn_deep}: {e}")
            continue

        loop_bl = build_pure_loop_baseline(dual, ffn_round_multiple=args.ffn_round_multiple)
        wide_bl = build_pure_wide_baseline(dual, ffn_round_multiple=args.ffn_round_multiple)

        sandwich = None
        if args.sandwich:
            sandwich = build_mixed_sandwich(
                dual, n_loop=args.sandwich_n_loop, n_wide=args.sandwich_n_wide,
                ffn_round_multiple=args.ffn_round_multiple,
            )

        print_config_summary(dual, loop_bl, wide_bl, sandwich)
        all_cfgs.append((dual, loop_bl, wide_bl, sandwich))

        if args.template:
            for cfg in [dual, loop_bl, wide_bl] + ([sandwich] if sandwich else []):
                p = write_yaml(args.template, args.output_dir, cfg)

    if args.template:
        print(f"\n  Wrote configs to {args.output_dir}/")

    print(f"\n  {'='*90}")
    print(f"  CROSS-CONFIG SUMMARY  (one row per architecture)")
    print(f"  {'='*90}")
    print(f"  {'experiment_id':<60s} {'params':>9s} {'FLOPs/tok':>11s}")
    print(f"  {'-'*60} {'-'*9} {'-'*11}")
    for dual, loop_bl, wide_bl, sandwich in all_cfgs:
        for cfg in [dual, loop_bl, wide_bl] + ([sandwich] if sandwich else []):
            print(f"  {make_exp_id(cfg):<60s} "
                  f"{_fmt_params(cfg['total_params']):>9s} "
                  f"{_fmt_flops(cfg['total_flops']):>11s}")


# ---------- FLOP-matched commands ----------

def _build_flop_triple(args, budget: int, alpha_list: list[float]) -> tuple[dict, dict, list[dict]]:
    """Build pure_loop, pure_wide, and one or more duals at a given budget."""
    loop_cfg = build_flop_matched_loop(
        flop_budget=budget, d_model=args.d_model, n_layers=args.n_layers,
        max_loops=args.max_loops, vocab_size=args.vocab_size,
        use_weight_tying=args.weight_tying,
        n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        ffn_round_multiple=args.ffn_round_multiple,
    )
    wide_cfg = build_flop_matched_wide(
        flop_budget=budget, d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=args.vocab_size, use_weight_tying=args.weight_tying,
        n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        ffn_round_multiple=args.ffn_round_multiple,
    )
    dual_cfgs = []
    for alpha in alpha_list:
        try:
            dc = build_flop_matched_dual(
                flop_budget=budget, d_model=args.d_model, n_layers=args.n_layers,
                max_loops=args.max_loops, alpha=alpha,
                vocab_size=args.vocab_size, use_weight_tying=args.weight_tying,
                n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
                ffn_round_multiple=args.ffn_round_multiple,
            )
            dual_cfgs.append(dc)
        except ValueError as e:
            print(f"  ⚠  Skipping dual α={alpha} at budget {_fmt_flops(budget)}: {e}")
    return loop_cfg, wide_cfg, dual_cfgs


def cmd_flop_single(args):
    """FLOP-matched: one budget, optionally with a dual α grid."""
    alpha_list = args.alpha if args.alpha else [0.5]
    loop_cfg, wide_cfg, dual_cfgs = _build_flop_triple(args, args.flop_budget, alpha_list)
    print_flop_matched_summary(
        args.flop_budget, args.d_model, args.n_layers, args.max_loops,
        loop_cfg, wide_cfg, dual_cfgs,
    )

    if args.template:
        print(f"\n  Writing YAMLs to {args.output_dir}/")
        for cfg in [loop_cfg, wide_cfg] + dual_cfgs:
            p = write_yaml(args.template, args.output_dir, cfg)
            print(f"    ✓ {p}")


def cmd_flop_sweep(args):
    """FLOP-matched: sweep budgets, dual at fixed α (default 0.5)."""
    alpha_list = args.alpha if args.alpha else [0.5]
    all_groups = []
    for budget in args.flop_budget:
        loop_cfg, wide_cfg, dual_cfgs = _build_flop_triple(args, budget, alpha_list)
        print_flop_matched_summary(
            budget, args.d_model, args.n_layers, args.max_loops,
            loop_cfg, wide_cfg, dual_cfgs,
        )
        all_groups.append((budget, loop_cfg, wide_cfg, dual_cfgs))

        if args.template:
            for cfg in [loop_cfg, wide_cfg] + dual_cfgs:
                write_yaml(args.template, args.output_dir, cfg)

    if args.template:
        print(f"\n  Wrote configs to {args.output_dir}/")

    print(f"\n  {'='*92}")
    print(f"  CROSS-BUDGET SUMMARY")
    print(f"  {'='*92}")
    print(f"  {'experiment_id':<62s} {'params':>9s} {'FLOPs/tok':>11s}")
    print(f"  {'-'*62} {'-'*9} {'-'*11}")
    for budget, loop_cfg, wide_cfg, dual_cfgs in all_groups:
        for cfg in [loop_cfg, wide_cfg] + dual_cfgs:
            print(f"  {make_exp_id(cfg):<62s} "
                  f"{_fmt_params(cfg['total_params']):>9s} "
                  f"{_fmt_flops(cfg['total_flops']):>11s}")


def cmd_flop_alpha_sweep(args):
    """FLOP-matched: fix budget, sweep dual α; baselines printed once."""
    alpha_list = args.alpha
    loop_cfg, wide_cfg, dual_cfgs = _build_flop_triple(args, args.flop_budget, alpha_list)
    print_flop_matched_summary(
        args.flop_budget, args.d_model, args.n_layers, args.max_loops,
        loop_cfg, wide_cfg, dual_cfgs,
    )

    if args.template:
        print(f"\n  Writing YAMLs to {args.output_dir}/")
        for cfg in [loop_cfg, wide_cfg] + dual_cfgs:
            p = write_yaml(args.template, args.output_dir, cfg)
            print(f"    ✓ {p}")


# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Param- AND FLOP-matched baseline generator for Dual-Path transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--d-model", type=int, required=True)
        sp.add_argument("--n-layers", type=int, required=True)
        sp.add_argument("--max-loops", type=int, required=True)
        sp.add_argument("--n-head-q", type=int, default=None)
        sp.add_argument("--n-head-kv", type=int, default=None)
        sp.add_argument("--vocab-size", type=int, default=50304)
        sp.add_argument("--weight-tying", action="store_true", default=False)
        sp.add_argument("--ffn-round-multiple", type=int, default=64)
        sp.add_argument("--template", type=str, default=None,
                        help="YAML template path. If omitted, only print summary.")
        sp.add_argument("--output-dir", type=str, default="configs")

    def add_param_matched(sp):
        sp.add_argument("--wide-flop-ratio", type=float, default=1.0,
                        help="wide path FLOPs / deep path FLOPs; 1.0 = matched")
        sp.add_argument("--sandwich", action="store_true", default=False,
                        help="Also emit a mixed-sandwich variant.")
        sp.add_argument("--sandwich-n-loop", type=int, default=None)
        sp.add_argument("--sandwich-n-wide", type=int, default=None)

    # ----- Param-matched -----
    single = sub.add_parser("single", help="Param-matched: one dual + baselines.")
    add_common(single)
    add_param_matched(single)
    single.add_argument("--ffn-deep", type=int, required=True,
                        help="Deep-path FFN hidden size.")
    single.set_defaults(func=cmd_single)

    sweep = sub.add_parser("sweep", help="Param-matched: sweep ffn_deep.")
    add_common(sweep)
    add_param_matched(sweep)
    sweep.add_argument("--ffn-deep", type=int, nargs="+", required=True,
                       help="List of deep-path FFN hidden sizes to sweep.")
    sweep.set_defaults(func=cmd_sweep)

    # ----- FLOP-matched -----
    fs = sub.add_parser("flop-single", help="FLOP-matched: one budget + baselines.")
    add_common(fs)
    fs.add_argument("--flop-budget", type=int, required=True,
                    help="Per-layer FLOP budget per token (e.g. 500_000_000).")
    fs.add_argument("--alpha", type=float, nargs="*", default=None,
                    help="Dual deep-FLOP fraction(s). Default: [0.5].")
    fs.set_defaults(func=cmd_flop_single)

    fsw = sub.add_parser("flop-sweep", help="FLOP-matched: sweep budgets at fixed α.")
    add_common(fsw)
    fsw.add_argument("--flop-budget", type=int, nargs="+", required=True,
                     help="List of per-layer FLOP budgets to sweep.")
    fsw.add_argument("--alpha", type=float, nargs="*", default=None,
                     help="Dual deep-FLOP fraction(s). Default: [0.5].")
    fsw.set_defaults(func=cmd_flop_sweep)

    fas = sub.add_parser("flop-alpha-sweep",
                         help="FLOP-matched: fix budget, sweep dual α only.")
    add_common(fas)
    fas.add_argument("--flop-budget", type=int, required=True)
    fas.add_argument("--alpha", type=float, nargs="+", required=True,
                     help="Dual deep-FLOP fractions to sweep, e.g. 0.3 0.5 0.7.")
    fas.set_defaults(func=cmd_flop_alpha_sweep)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()