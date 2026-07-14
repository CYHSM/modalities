#!/usr/bin/env python3
"""
FLOP-matched config generator for the asymmetric Dual-Path
(Wide + Deep loop) transformer.

You give a per-layer FLOP budget and a list of α values (deep-path FLOP
fractions for dual). The script emits, all spending the same total FLOPs
per token:

    - pure_loop                       (single shared block, looped max_loops times)
    - pure_wide                       (single non-looped wide block, n_layers deep)
    - pure_wide expanded              (n_layers*max_loops layers, FFN scaled to fit)
    - dual α=…                        one per α you pass
    - dual min-deep                   ffn_deep at the floor, wide path absorbs rest
    - dual α=… no-cross               one per α passed via --add-no-cross-alpha
                                      (regardless of the global --no-cross setting;
                                       intended for clean per-token routing analysis)

CRITICAL — SwiGLU 2/3 scaling
=============================
The model's SwiGLU FFN does NOT use the configured ffn_hidden directly.
It rounds: h_eff = round_up(int(2 * ffn_hidden / 3), swiglu_multiple),
then uses h_eff for the three projections. So 3*d*h_eff params and
6*d*h_eff FLOPs per FFN — not 3*d*ffn_hidden / 6*d*ffn_hidden.

The default swiglu_multiple is 256 (matching the modalities project's
typical setup). Override with --swiglu-multiple if your model uses
something else.

All solving is done in h_eff space, then converted back to a configured
ffn_hidden value such that the model's rounding produces exactly the
target h_eff.
"""

import argparse
import os
import re


# =====================================================================
# Constants
# =====================================================================

DEFAULT_SWIGLU_MULTIPLE = 64
DEFAULT_FFN_ROUND_MULTIPLE = 64       # configured ffn_hidden must be % 64 == 0
DEFAULT_VOCAB_SIZE = 50304


# =====================================================================
# SwiGLU effective-hidden conversion
# =====================================================================

def swiglu_effective_hidden(ffn_hidden: int, multiple_of: int) -> int:
    """LLaMA-style SwiGLU effective hidden dim: round_up(int(2*ffn/3), multiple)."""
    h = int(2 * ffn_hidden / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def round_up(x: float, m: int) -> int:
    return max(m, m * ((int(x) + m - 1) // m))


def round_down(x: float, m: int) -> int:
    return max(m, m * (int(x) // m))


def get_ffn_hidden(h_exact: float, swiglu_m: int, ffn_round: int, mode: str = "floor") -> int:
    """
    Finds a legal ffn_hidden (multiple of ffn_round) that satisfies the budget.
    mode='ceil': Allows budget overruns (Baseline behavior).
    mode='floor': Guarantees SwiGLU effective hidden stays <= the exact budget.
    """
    if mode == "ceil":
        h = round_up(h_exact, swiglu_m)
        ffn = (3 * h + 1) // 2
        return max(ffn_round, ffn_round * ((ffn + ffn_round - 1) // ffn_round))
    else:
        # Floor mode: strict adherence to budget
        h_max = round_down(h_exact, swiglu_m)
        ffn = (3 * h_max) // 2

        # Round DOWN to nearest ffn_round
        ffn_floored = max(ffn_round, ffn_round * (ffn // ffn_round))

        # Safety loop: ensure the model's forward pass rounding doesn't push us over h_max
        while swiglu_effective_hidden(ffn_floored, swiglu_m) > h_max and ffn_floored > ffn_round:
            ffn_floored -= ffn_round

        return ffn_floored


# =====================================================================
# FLOP & param accounting (per token, projection FLOPs only — n^2*d term
# of attention is architecture-independent at matched d/seq_len)
# =====================================================================

def attn_flops(d: int, n_rep: int) -> int:
    """QKV+output projection FLOPs: 4d^2 + 4d^2/n_rep."""
    return 4 * d * d + 4 * (d * d) // n_rep


def attn_params(d: int, n_rep: int) -> int:
    """QKV+output projection params (no bias): 2d^2 + 2d^2/n_rep."""
    return 2 * d * d + 2 * (d * d) // n_rep


def swiglu_flops(d: int, ffn: int, swiglu_m: int) -> int:
    """FLOPs/token for SwiGLU FFN: 6 * d * h_eff."""
    return 6 * d * swiglu_effective_hidden(ffn, swiglu_m)


def swiglu_params(d: int, ffn: int, swiglu_m: int) -> int:
    """Params for SwiGLU FFN (no bias): 3 * d * h_eff."""
    return 3 * d * swiglu_effective_hidden(ffn, swiglu_m)


def block_params(d: int, ffn: int, n_rep: int, swiglu_m: int) -> int:
    """One GPT2Block: attn + SwiGLU + 2 RMSNorm."""
    return attn_params(d, n_rep) + swiglu_params(d, ffn, swiglu_m) + 2 * d


def shared_params(d: int, vocab: int, weight_tying: bool) -> int:
    """wte + lm_head + final norm."""
    s = vocab * d                       # wte
    if not weight_tying:
        s += d * vocab                  # lm_head
    s += d                              # lm_head_norm
    return s


# Per-layer FLOPs ------------------------------------------------------

def deep_flops(d: int, ffn_d: int, max_loops: int, n_rep: int, swiglu_m: int) -> int:
    """Deep path = max_loops × (attn + SwiGLU)."""
    return max_loops * (attn_flops(d, n_rep) + swiglu_flops(d, ffn_d, swiglu_m))


def wide_flops(d: int, ffn_w: int, n_rep: int, swiglu_m: int) -> int:
    """Wide path = attn + SwiGLU, single pass."""
    return attn_flops(d, n_rep) + swiglu_flops(d, ffn_w, swiglu_m)


def gate_flops_t(d: int, gate_mode: str) -> int:
    """DualPathGate Linear(d, 2) for two_gates, Linear(d, 1) for convex."""
    return 4 * d if gate_mode == "two_gates" else 2 * d


def router_flops_t(d: int, max_loops: int) -> int:
    """AdaptiveRouter inside deep loop: max_loops × Linear(d+1, 1)."""
    return max_loops * 2 * (d + 1)


# Per-layer params -----------------------------------------------------

def loop_layer_params(d: int, ffn_l: int, max_loops: int, n_rep: int,
                      swiglu_m: int) -> int:
    """Pure-loop AdaptiveRecursiveBlock."""
    p = block_params(d, ffn_l, n_rep, swiglu_m)
    p += (d + 1) + 1                    # router Linear(d+1, 1, bias=True)
    p += max_loops                      # loop_scales
    return p


def wide_layer_params(d: int, ffn_w: int, n_rep: int, swiglu_m: int) -> int:
    """Pure-wide AdaptiveRecursiveBlock."""
    p = block_params(d, ffn_w, n_rep, swiglu_m)
    p += 1                              # wide_scale
    return p


def dual_layer_params(d: int, ffn_d: int, ffn_w: int, max_loops: int,
                      n_rep: int, swiglu_m: int,
                      gate_mode: str, use_cross: bool) -> int:
    """Full dual AdaptiveRecursiveBlock."""
    p = 0
    p += block_params(d, ffn_d, n_rep, swiglu_m)    # deep block
    p += block_params(d, ffn_w, n_rep, swiglu_m)    # wide block
    p += (d + 1) + 1                                # router
    p += max_loops                                  # loop_scales
    p += 1                                          # wide_scale
    if gate_mode == "two_gates":
        p += 2 * d + 2                              # gate_proj Linear(d, 2)
    else:                                           # convex
        p += d + 1                                  # gate_proj Linear(d, 1)
    if use_cross:
        p += 2 * d * d + 2                          # proj_w2d + proj_d2w + 2 scalars
    return p


# =====================================================================
# Solvers — all work in h_eff space, then convert to ffn_hidden
# =====================================================================

def solve_loop(F: int, d: int, max_loops: int, n_rep: int,
               swiglu_m: int, ffn_round: int) -> tuple[int, int]:
    attn = attn_flops(d, n_rep)
    router = router_flops_t(d, max_loops)
    h_exact = (F - router - max_loops * attn) / (6.0 * d * max_loops)
    if h_exact <= 0:
        raise ValueError(f"pure_loop: budget {F:.2e} too small for d={d}, "
                         f"max_loops={max_loops}. Need >{router + max_loops*attn:.2e} "
                         f"to cover attn+router.")

    # NON-BASELINE: floor
    ffn = get_ffn_hidden(h_exact, swiglu_m, ffn_round, mode="floor")
    actual = max_loops * (attn + swiglu_flops(d, ffn, swiglu_m)) + router
    return ffn, actual


def solve_wide(F: int, d: int, n_rep: int,
               swiglu_m: int, ffn_round: int) -> tuple[int, int]:
    attn = attn_flops(d, n_rep)
    h_exact = (F - attn) / (6.0 * d)
    if h_exact <= 0:
        raise ValueError(f"pure_wide: budget {F:.2e} too small for d={d}. "
                         f"Need >{attn:.2e} to cover attn.")

    # BASELINE: ceil
    ffn = get_ffn_hidden(h_exact, swiglu_m, ffn_round, mode="ceil")
    actual = attn + swiglu_flops(d, ffn, swiglu_m)
    return ffn, actual


def solve_dual(F: int, d: int, max_loops: int, alpha: float, n_rep: int,
               swiglu_m: int, ffn_round: int, gate_mode: str
               ) -> tuple[int, int, dict]:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    attn = attn_flops(d, n_rep)
    gate = gate_flops_t(d, gate_mode)
    router = router_flops_t(d, max_loops)
    f_arith = F - gate - router
    if f_arith <= 0:
        raise ValueError(f"Budget {F} insufficient for gate+router ({gate+router}).")

    h_d_exact = (alpha * f_arith / max_loops - attn) / (6.0 * d)
    h_w_exact = ((1.0 - alpha) * f_arith - attn) / (6.0 * d)
    if h_d_exact <= 0:
        print(f"  ⚠  alpha={alpha} too small at budget {F:.2e}; deep can't cover "
              f"its attn cost. Pinned h_d_exact to minimum {swiglu_m}.")
        h_d_exact = swiglu_m
    if h_w_exact <= 0:
        print(f"  ⚠  alpha={alpha} too large at budget {F:.2e}; wide can't cover "
              f"its attn cost. Pinned h_w_exact to minimum {swiglu_m}.")
        h_w_exact = swiglu_m

    # NON-BASELINE: floor
    ffn_d = get_ffn_hidden(h_d_exact, swiglu_m, ffn_round, mode="floor")
    ffn_w = get_ffn_hidden(h_w_exact, swiglu_m, ffn_round, mode="floor")

    deep = deep_flops(d, ffn_d, max_loops, n_rep, swiglu_m)
    wide = wide_flops(d, ffn_w, n_rep, swiglu_m)
    breakdown = {"deep": deep, "wide": wide, "gate": gate, "router": router,
                 "total": deep + wide + gate + router}
    return ffn_d, ffn_w, breakdown


def solve_dual_min_deep(F: int, d: int, max_loops: int, n_rep: int,
                        swiglu_m: int, ffn_round: int, gate_mode: str
                        ) -> tuple[int, int, dict]:
    """ffn_deep pinned to the smallest legal value; wide absorbs the rest."""
    attn = attn_flops(d, n_rep)
    gate = gate_flops_t(d, gate_mode)
    router = router_flops_t(d, max_loops)
    f_arith = F - gate - router
    if f_arith <= 0:
        raise ValueError(f"Budget {F} insufficient for gate+router.")

    # Smallest legal ffn_hidden whose h_eff == swiglu_m.
    ffn_d = get_ffn_hidden(swiglu_m, swiglu_m, ffn_round, mode="ceil")
    deep = deep_flops(d, ffn_d, max_loops, n_rep, swiglu_m)
    if deep >= f_arith:
        raise ValueError(f"min-deep: floor ffn_deep={ffn_d} alone exhausts the budget "
                         f"(deep={deep} >= f_arith={f_arith}). Increase --flop-budget.")

    h_w_exact = ((f_arith - deep) - attn) / (6.0 * d)
    if h_w_exact <= 0:
        raise ValueError(f"min-deep: wide path can't cover attn cost. Increase --flop-budget.")

    # NON-BASELINE: floor
    ffn_w = get_ffn_hidden(h_w_exact, swiglu_m, ffn_round, mode="floor")

    wide = wide_flops(d, ffn_w, n_rep, swiglu_m)
    breakdown = {"deep": deep, "wide": wide, "gate": gate, "router": router,
                 "total": deep + wide + gate + router}
    return ffn_d, ffn_w, breakdown


# =====================================================================
# Config builders — return dicts that the YAML emitter and printer use
# =====================================================================

def _shared_fields(d: int, n_layers: int, n_head_q, n_head_kv, vocab: int,
                   weight_tying: bool, swiglu_m: int) -> dict:
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    return {"d_model": d, "n_layers": n_layers, "n_head_q": n_head_q,
            "n_head_kv": n_head_kv, "n_rep": n_rep, "vocab_size": vocab,
            "use_weight_tying": weight_tying, "swiglu_multiple": swiglu_m}


def build_loop(F: int, d: int, n_layers: int, max_loops: int,
               vocab: int, weight_tying: bool, n_head_q, n_head_kv,
               swiglu_m: int, ffn_round: int) -> dict:
    base = _shared_fields(d, n_layers, n_head_q, n_head_kv, vocab, weight_tying, swiglu_m)
    ffn, actual = solve_loop(F, d, max_loops, base["n_rep"], swiglu_m, ffn_round)
    per_layer_p = loop_layer_params(d, ffn, max_loops, base["n_rep"], swiglu_m)
    total_p = n_layers * per_layer_p + shared_params(d, vocab, weight_tying)
    return {**base, "kind": "pure_loop", "max_loops": max_loops, "ffn_loop": ffn,
            "ffn_loop_h_eff": swiglu_effective_hidden(ffn, swiglu_m),
            "flop_budget": F, "total_params": total_p,
            "flops_per_layer": actual, "total_flops": n_layers * actual,
            "flop_match_pct": (actual / F - 1.0) * 100}


def build_wide(F: int, d: int, n_layers: int,
               vocab: int, weight_tying: bool, n_head_q, n_head_kv,
               swiglu_m: int, ffn_round: int) -> dict:
    base = _shared_fields(d, n_layers, n_head_q, n_head_kv, vocab, weight_tying, swiglu_m)
    ffn, actual = solve_wide(F, d, base["n_rep"], swiglu_m, ffn_round)
    per_layer_p = wide_layer_params(d, ffn, base["n_rep"], swiglu_m)
    total_p = n_layers * per_layer_p + shared_params(d, vocab, weight_tying)
    return {**base, "kind": "pure_wide", "max_loops": 1, "ffn_wide": ffn,
            "ffn_wide_h_eff": swiglu_effective_hidden(ffn, swiglu_m),
            "flop_budget": F, "total_params": total_p,
            "flops_per_layer": actual, "total_flops": n_layers * actual,
            "flop_match_pct": (actual / F - 1.0) * 100}


def build_wide_expanded(F: int, d: int, n_layers: int, max_loops: int,
                        vocab: int, weight_tying: bool, n_head_q, n_head_kv,
                        swiglu_m: int, ffn_round: int) -> dict:
    """Pure-wide with n_layers*max_loops layers and per-layer budget /max_loops."""
    n_layers_exp = n_layers * max_loops
    F_exp = F // max_loops
    cfg = build_wide(F_exp, d, n_layers_exp, vocab, weight_tying,
                     n_head_q, n_head_kv, swiglu_m, ffn_round)
    cfg["variant"] = "expanded"
    cfg["flop_budget_original"] = F
    cfg["max_loops_original"] = max_loops
    cfg["n_layers_original"] = n_layers
    target_total = n_layers * F
    cfg["total_flop_match_pct"] = (cfg["total_flops"] / target_total - 1.0) * 100
    return cfg


def build_dual(F: int, d: int, n_layers: int, max_loops: int, alpha: float,
               vocab: int, weight_tying: bool, n_head_q, n_head_kv,
               swiglu_m: int, ffn_round: int,
               gate_mode: str, use_cross: bool) -> dict:
    base = _shared_fields(d, n_layers, n_head_q, n_head_kv, vocab, weight_tying, swiglu_m)
    ffn_d, ffn_w, fb = solve_dual(F, d, max_loops, alpha, base["n_rep"],
                                   swiglu_m, ffn_round, gate_mode)
    per_layer_p = dual_layer_params(d, ffn_d, ffn_w, max_loops, base["n_rep"],
                                     swiglu_m, gate_mode, use_cross)
    total_p = n_layers * per_layer_p + shared_params(d, vocab, weight_tying)
    return {**base, "kind": "dual", "max_loops": max_loops,
            "alpha": alpha,
            "alpha_actual": fb["deep"] / (fb["deep"] + fb["wide"]),
            "ffn_deep": ffn_d, "ffn_wide": ffn_w,
            "ffn_deep_h_eff": swiglu_effective_hidden(ffn_d, swiglu_m),
            "ffn_wide_h_eff": swiglu_effective_hidden(ffn_w, swiglu_m),
            "gate_mode": gate_mode, "use_cross": use_cross,
            "flop_budget": F, "total_params": total_p,
            "flops_per_layer": fb, "total_flops": n_layers * fb["total"],
            "flop_match_pct": (fb["total"] / F - 1.0) * 100}


def build_dual_min_deep(F: int, d: int, n_layers: int, max_loops: int,
                        vocab: int, weight_tying: bool, n_head_q, n_head_kv,
                        swiglu_m: int, ffn_round: int,
                        gate_mode: str, use_cross: bool) -> dict:
    base = _shared_fields(d, n_layers, n_head_q, n_head_kv, vocab, weight_tying, swiglu_m)
    ffn_d, ffn_w, fb = solve_dual_min_deep(F, d, max_loops, base["n_rep"],
                                            swiglu_m, ffn_round, gate_mode)
    per_layer_p = dual_layer_params(d, ffn_d, ffn_w, max_loops, base["n_rep"],
                                     swiglu_m, gate_mode, use_cross)
    total_p = n_layers * per_layer_p + shared_params(d, vocab, weight_tying)
    alpha_a = fb["deep"] / (fb["deep"] + fb["wide"])
    return {**base, "kind": "dual", "variant": "min_deep",
            "max_loops": max_loops, "alpha": alpha_a, "alpha_actual": alpha_a,
            "ffn_deep": ffn_d, "ffn_wide": ffn_w,
            "ffn_deep_h_eff": swiglu_effective_hidden(ffn_d, swiglu_m),
            "ffn_wide_h_eff": swiglu_effective_hidden(ffn_w, swiglu_m),
            "gate_mode": gate_mode, "use_cross": use_cross,
            "flop_budget": F, "total_params": total_p,
            "flops_per_layer": fb, "total_flops": n_layers * fb["total"],
            "flop_match_pct": (fb["total"] / F - 1.0) * 100}


def build_dual_expanded(F: int, d: int, n_layers: int, max_loops: int, alpha: float,
                        vocab: int, weight_tying: bool, n_head_q, n_head_kv,
                        swiglu_m: int, ffn_round: int,
                        gate_mode: str, use_cross: bool, loop_override: int = None) -> dict:
    """Dual-path with n_layers*max_loops layers, max_loops=loop_override (or max_loops), and per-layer budget /max_loops."""
    n_layers_exp = n_layers * max_loops
    F_exp = F // max_loops
    max_loops_to_use = loop_override if loop_override is not None else max_loops
    cfg = build_dual(F=F_exp, d=d, n_layers=n_layers_exp, max_loops=max_loops_to_use, alpha=alpha,
                     vocab=vocab, weight_tying=weight_tying, n_head_q=n_head_q, n_head_kv=n_head_kv,
                     swiglu_m=swiglu_m, ffn_round=ffn_round, gate_mode=gate_mode, use_cross=use_cross)
    cfg["variant"] = "expanded"
    cfg["flop_budget_original"] = F
    cfg["max_loops_original"] = max_loops
    cfg["n_layers_original"] = n_layers
    target_total = n_layers * F
    cfg["total_flop_match_pct"] = (cfg["total_flops"] / target_total - 1.0) * 100
    return cfg


# =====================================================================
# Experiment IDs
# =====================================================================

def make_exp_id(cfg: dict) -> str:
    d = cfg["d_model"]
    L = cfg["n_layers"]
    # For the expanded variant, use the ORIGINAL budget so it groups visually
    # with sibling configs in the output directory.
    F_tag = cfg.get("flop_budget_original", cfg["flop_budget"])
    F_m = F_tag / 1e6

    if cfg["kind"] == "dual":
        if cfg.get("variant") == "min_deep":
            alpha_tag = "_aMINdeep"
        else:
            alpha_tag = f"_a{int(round(cfg['alpha']*100)):02d}"
        # Tag no-cross variants so their YAML filenames don't collide with
        # the cross-on dual at the same α.
        cross_tag = "_nocross" if cfg.get("use_cross") is False else ""
        suffix = "_expanded" if cfg.get("variant") == "expanded" else ""
        return (f"dm{d}_L{L}_loop{cfg['max_loops']}_F{F_m:.0f}M"
                f"_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}_dual{alpha_tag}{cross_tag}{suffix}")
    if cfg["kind"] == "pure_loop":
        return (f"dm{d}_L{L}_loop{cfg['max_loops']}_F{F_m:.0f}M"
                f"_ffnL{cfg['ffn_loop']}_pureloop")
    if cfg["kind"] == "pure_wide":
        suffix = "_expanded" if cfg.get("variant") == "expanded" else ""
        return f"dm{d}_L{L}_loop1_F{F_m:.0f}M_ffnW{cfg['ffn_wide']}_purewide{suffix}"
    raise ValueError(f"Unknown kind: {cfg['kind']}")


# =====================================================================
# YAML emission
# =====================================================================

def _yaml_patches(cfg: dict) -> dict:
    """Map (section, key) -> string value to patch into the template."""
    exp_id = make_exp_id(cfg)
    n_layers = cfg["n_layers"]
    patches = {
        ("model_raw", "n_layer"): str(n_layers),
        ("model_raw", "n_embd"): str(cfg["d_model"]),
        ("evaluation_subscriber", "experiment_id"): exp_id,
    }
    if cfg.get("n_head_q") is not None:
        patches[("model_raw", "n_head_q")] = str(cfg["n_head_q"])
        patches[("model_raw", "normalized_shape")] = (
            f"{cfg['d_model'] // cfg['n_head_q']} # n_embd / n_head_q"
        )
    if cfg.get("n_head_kv") is not None:
        patches[("model_raw", "n_head_kv")] = str(cfg["n_head_kv"])

    kind = cfg["kind"]
    if kind == "dual":
        patches[("model_raw", "ffn_hidden")] = str(cfg["ffn_deep"])
        patches[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        patches[("model_raw", "max_loops")] = str(cfg["max_loops"])
        patches[("model_raw", "enable_adaptive")] = "true"
        # Patch use_cross so the YAML matches the per-config setting (this
        # is what makes --add-no-cross-alpha actually take effect at train time).
        if cfg.get("use_cross") is not None:
            patches[("model_raw", "use_cross")] = "true" if cfg["use_cross"] else "false"
        cfg.setdefault("layer_types", ["dual"] * n_layers)
    elif kind == "pure_loop":
        patches[("model_raw", "ffn_hidden")] = str(cfg["ffn_loop"])
        patches[("model_raw", "wide_ffn_hidden")] = "0"
        patches[("model_raw", "max_loops")] = str(cfg["max_loops"])
        patches[("model_raw", "enable_adaptive")] = "true"
        cfg.setdefault("layer_types", ["loop"] * n_layers)
    elif kind == "pure_wide":
        # ffn_hidden unused but must be % 64 == 0; smallest legal value is 64.
        patches[("model_raw", "ffn_hidden")] = "64"
        patches[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        patches[("model_raw", "max_loops")] = "1"
        patches[("model_raw", "enable_adaptive")] = "true"
        # CRITICAL: explicit layer_types — without this the model would build
        # 'dual' layers (because wide_ffn_hidden > 0).
        cfg.setdefault("layer_types", ["wide"] * n_layers)
    return patches


def _patch_yaml_text(text: str, cfg: dict) -> str:
    """Walk template text, patching matching (section, key) lines."""
    patches = _yaml_patches(cfg)
    layer_types = cfg.get("layer_types")
    kv_re = re.compile(r'^(\s*)([\w_]+)(:\s+)(.+)$')
    out, current_section = [], None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        # Top-level section header.
        if stripped and not stripped.startswith('#') and line[0] not in (' ', '\t', '\n', '\r'):
            colon = stripped.find(':')
            if colon > 0:
                current_section = stripped[:colon].strip()

        m = kv_re.match(line)

        # Drop pre-existing layer_types in model_raw — we write our own.
        if (layer_types is not None and current_section == "model_raw"
                and m and m.group(2) == "layer_types"):
            continue

        # Patch matching keys (skip lines whose value is a ${...} placeholder).
        if m and current_section:
            indent, key, sep, old_val = m.group(1), m.group(2), m.group(3), m.group(4)
            if not old_val.lstrip().startswith('${'):
                pk = (current_section, key)
                if pk in patches:
                    line = f"{indent}{key}{sep}{patches[pk]}\n"

        out.append(line)

        # After max_loops in model_raw, write the layer_types list.
        if (layer_types is not None and current_section == "model_raw"
                and m and m.group(2) == "max_loops"):
            indent = m.group(1)
            lt = ", ".join(f'"{t}"' for t in layer_types)
            out.append(f"{indent}layer_types: [{lt}]\n")

    return ''.join(out)


def write_yaml(template_path: str, output_dir: str, cfg: dict) -> str:
    with open(template_path, "r") as f:
        text = f.read()
    patched = _patch_yaml_text(text, cfg)
    exp_id = make_exp_id(cfg)
    out_path = os.path.join(output_dir, f"{exp_id}.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(patched)
    return out_path


# =====================================================================
# Pretty-printing
# =====================================================================

def _fmt_p(p: int) -> str:
    if p >= 1e9: return f"{p/1e9:.2f}B"
    if p >= 1e6: return f"{p/1e6:.1f}M"
    if p >= 1e3: return f"{p/1e3:.1f}k"
    return str(p)


def _fmt_f(f: int) -> str:
    if f >= 1e12: return f"{f/1e12:.2f}T"
    if f >= 1e9:  return f"{f/1e9:.2f}G"
    if f >= 1e6:  return f"{f/1e6:.1f}M"
    return f"{f/1e3:.1f}k"


def print_summary(F: int, d: int, n_layers: int, max_loops: int, swiglu_m: int,
                  configs: list[dict]):
    print(f"\n  {'─'*94}")
    print(f"  FLOP-matched at {_fmt_f(F)}/tok per layer  "
          f"(d_model={d}  L={n_layers}  loops={max_loops}  swiglu_multiple={swiglu_m})")
    print(f"  Total FLOPs/tok target: {_fmt_f(n_layers * F)}")
    print(f"  {'─'*94}")
    print(f"    {'Config':<32s} {'n_layers':>9s} {'ffn_d/h_d':>13s} {'ffn_w/h_w':>13s} "
          f"{'params':>9s} {'TotalFLOP':>11s} {'ΔFLOP':>8s}")
    print(f"    {'-'*32} {'-'*9} {'-'*13} {'-'*13} {'-'*9} {'-'*11} {'-'*8}")

    for cfg in configs:
        kind = cfg["kind"]
        if kind == "pure_loop":
            label = "pure_loop"
            d_col = f"{cfg['ffn_loop']}/{cfg['ffn_loop_h_eff']}"
            w_col = "-"
            mismatch = cfg["flop_match_pct"]
        elif kind == "pure_wide" and cfg.get("variant") == "expanded":
            label = f"pure_wide expanded (L×{cfg['max_loops_original']})"
            d_col = "-"
            w_col = f"{cfg['ffn_wide']}/{cfg['ffn_wide_h_eff']}"
            mismatch = cfg["total_flop_match_pct"]
        elif kind == "pure_wide":
            label = "pure_wide"
            d_col = "-"
            w_col = f"{cfg['ffn_wide']}/{cfg['ffn_wide_h_eff']}"
            mismatch = cfg["flop_match_pct"]
        else:                                       # dual
            if cfg.get("variant") == "min_deep":
                label = f"dual min-deep (α≈{cfg['alpha_actual']:.2f})"
            else:
                label = f"dual α={cfg['alpha']:.2f}"
            # Annotate no-cross duals so they're visually distinct in the table.
            if cfg.get("use_cross") is False:
                label += " no-cross"
            if cfg.get("variant") == "expanded":
                label += f" expanded (L×{cfg['max_loops_original']})"
                mismatch = cfg["total_flop_match_pct"]
            else:
                mismatch = cfg["flop_match_pct"]
            d_col = f"{cfg['ffn_deep']}/{cfg['ffn_deep_h_eff']}"
            w_col = f"{cfg['ffn_wide']}/{cfg['ffn_wide_h_eff']}"

        print(f"    {label:<32s} {cfg['n_layers']:>9d} {d_col:>13s} {w_col:>13s} "
              f"{_fmt_p(cfg['total_params']):>9s} "
              f"{_fmt_f(cfg['total_flops']):>11s} {mismatch:>+7.2f}%")

    all_p = [c['total_params'] for c in configs]
    p_min, p_max = min(all_p), max(all_p)
    print(f"\n    Param spread: {_fmt_p(p_min)} → {_fmt_p(p_max)} "
          f"({(p_max/p_min - 1)*100:+.0f}%)  — expected at fixed FLOPs.")
    print(f"    Note: ffn shows configured/h_eff (h_eff = round_up(int(2*ffn/3), "
          f"{swiglu_m})).")

    for cfg in configs:
        m = cfg.get("total_flop_match_pct", cfg["flop_match_pct"])
        if m > 0.0 and cfg["kind"] != "pure_wide":
            print(f"    ⚠  {cfg.get('kind')} mismatch {m:+.2f}% — this budget overrun "
                  f"should no longer happen for non-baselines with floor rounding.")


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="FLOP-matched config generator for the Dual-Path transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--max-loops", type=int, required=True)
    p.add_argument("--flop-budget", type=int, required=True,
                   help="Per-layer FLOP budget per token (e.g. 200000000).")
    p.add_argument("--alpha", type=float, nargs="+", required=True,
                   help="Dual deep-FLOP fractions, e.g. 0.95 0.5 0.05.")

    # Architecture
    p.add_argument("--n-head-q", type=int, default=None)
    p.add_argument("--n-head-kv", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--weight-tying", action="store_true", default=False)
    p.add_argument("--gate-mode", choices=["two_gates", "convex"], default="two_gates",
                   help="DualPathGate variant. Default matches the model code default.")
    p.add_argument("--no-cross", action="store_true", default=False,
                   help="Disable cross-path projections (use_cross=False) for ALL dual configs.")

    # Rounding
    p.add_argument("--swiglu-multiple", type=int, default=DEFAULT_SWIGLU_MULTIPLE,
                   help=f"SwiGLU rounding multiple (default {DEFAULT_SWIGLU_MULTIPLE}). "
                        f"Must match model's enforce_swiglu_hidden_dim_multiple_of.")
    p.add_argument("--ffn-round-multiple", type=int, default=DEFAULT_FFN_ROUND_MULTIPLE,
                   help=f"Configured ffn_hidden divisibility (default "
                        f"{DEFAULT_FFN_ROUND_MULTIPLE}, model validator requires 64).")

    # Variant toggles
    p.add_argument("--no-min-deep", action="store_true", default=False,
                   help="Skip the dual min-deep variant.")
    p.add_argument("--no-wide-expanded", action="store_true", default=False,
                   help="Skip the pure_wide expanded baseline (n_layers*max_loops layers).")
    p.add_argument("--add-no-cross-alpha", type=float, nargs="+", default=None,
                   help="Extra dual configs at these α values with use_cross=False, "
                        "regardless of the global --no-cross setting. Intended for "
                        "clean per-token routing analysis (cross-path leakage muddies "
                        "the 'which path does this token want' reading). "
                        "Example: --add-no-cross-alpha 0.5")
    p.add_argument("--add-expanded-dual-no-cross-alpha", type=float, nargs="+", default=None,
                   help="Extra expanded dual configs at these α values with use_cross=False "
                        "and n_layers*max_loops layers, max_loops=1. "
                        "Example: --add-expanded-dual-no-cross-alpha 0.25 0.5 0.75")
    p.add_argument("--add-expanded-dual-loops-a50", action="store_true", default=False,
                   help="Add expanded dual no-cross configs with alpha=0.5, 64 layers, "
                        "and loops 1, 2, 3, and 4.")

    # Output
    p.add_argument("--template", type=str, default=None,
                   help="YAML template path. If omitted, only print summary.")
    p.add_argument("--output-dir", type=str, default="configs")

    args = p.parse_args()

    use_cross = not args.no_cross
    common = dict(
        F=args.flop_budget, d=args.d_model, n_layers=args.n_layers,
        max_loops=args.max_loops, vocab=args.vocab_size,
        weight_tying=args.weight_tying,
        n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        swiglu_m=args.swiglu_multiple, ffn_round=args.ffn_round_multiple,
    )
    dual_common = dict(common, gate_mode=args.gate_mode, use_cross=use_cross)

    configs = []
    configs.append(build_loop(**common))
    # pure_wide doesn't need max_loops in its signature.
    configs.append(build_wide(F=args.flop_budget, d=args.d_model, n_layers=args.n_layers,
                              vocab=args.vocab_size, weight_tying=args.weight_tying,
                              n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
                              swiglu_m=args.swiglu_multiple, ffn_round=args.ffn_round_multiple))

    if not args.no_wide_expanded:
        try:
            configs.append(build_wide_expanded(**common))
        except ValueError as e:
            print(f"  ⚠  Skipping pure_wide expanded: {e}")

    for alpha in args.alpha:
        try:
            configs.append(build_dual(alpha=alpha, **dual_common))
        except ValueError as e:
            print(f"  ⚠  Skipping dual α={alpha}: {e}")

    if not args.no_min_deep:
        try:
            min_cfg = build_dual_min_deep(**dual_common)
            # De-dup against any α that produced the same FFN sizes.
            already = any(c.get("kind") == "dual"
                          and c.get("variant") != "min_deep"
                          and c.get("ffn_deep") == min_cfg["ffn_deep"]
                          and c.get("ffn_wide") == min_cfg["ffn_wide"]
                          for c in configs)
            if not already:
                configs.append(min_cfg)
        except ValueError as e:
            print(f"  ⚠  Skipping dual min-deep: {e}")

    # -----------------------------------------------------------------
    # Extra no-cross dual configs for per-token routing analysis.
    # These are emitted regardless of the global --no-cross setting,
    # and are tagged with `_nocross` in the experiment ID so they don't
    # collide with the cross-on dual at the same α.
    # -----------------------------------------------------------------
    if args.add_no_cross_alpha is not None:
        dual_no_cross_common = dict(common, gate_mode=args.gate_mode, use_cross=False)
        for alpha in args.add_no_cross_alpha:
            try:
                cfg_nc = build_dual(alpha=alpha, **dual_no_cross_common)
                # If the user already passed --no-cross globally AND this α is
                # in --alpha, the cross-on version above is actually also
                # cross-off, so we'd be duplicating. De-dup by (alpha, use_cross).
                already = any(
                    c.get("kind") == "dual"
                    and c.get("variant") != "min_deep"
                    and c.get("use_cross") is False
                    and abs(c.get("alpha", -1) - alpha) < 1e-9
                    for c in configs
                )
                if not already:
                    configs.append(cfg_nc)
                else:
                    print(f"  ℹ  Skipping no-cross α={alpha}: already present "
                          f"(global --no-cross + same α in --alpha).")
            except ValueError as e:
                print(f"  ⚠  Skipping no-cross dual α={alpha}: {e}")

    # -----------------------------------------------------------------
    # Extra expanded no-cross dual configs for per-token routing analysis.
    # -----------------------------------------------------------------
    if args.add_expanded_dual_no_cross_alpha is not None:
        dual_no_cross_expanded_common = dict(common, gate_mode=args.gate_mode, use_cross=False)
        for alpha in args.add_expanded_dual_no_cross_alpha:
            try:
                cfg_nc_exp = build_dual_expanded(alpha=alpha, **dual_no_cross_expanded_common)
                already = any(
                    c.get("kind") == "dual"
                    and c.get("variant") == "expanded"
                    and c.get("use_cross") is False
                    and abs(c.get("alpha", -1) - alpha) < 1e-9
                    for c in configs
                )
                if not already:
                    configs.append(cfg_nc_exp)
                else:
                    print(f"  ℹ  Skipping expanded no-cross α={alpha}: already present.")
            except ValueError as e:
                print(f"  ⚠  Skipping expanded no-cross dual α={alpha}: {e}")

    # -----------------------------------------------------------------
    # Expanded dual no-cross loops 1, 2, 3, 4 for alpha=0.5
    # -----------------------------------------------------------------
    if args.add_expanded_dual_loops_a50:
        dual_no_cross_expanded_common = dict(common, gate_mode=args.gate_mode, use_cross=False)
        for loop in [1, 2, 3, 4]:
            try:
                cfg_nc_exp = build_dual_expanded(alpha=0.5, loop_override=loop, **dual_no_cross_expanded_common)
                already = any(
                    c.get("kind") == "dual"
                    and c.get("variant") == "expanded"
                    and c.get("use_cross") is False
                    and abs(c.get("alpha", -1) - 0.5) < 1e-9
                    and c.get("max_loops") == loop
                    for c in configs
                )
                if not already:
                    configs.append(cfg_nc_exp)
                else:
                    print(f"  ℹ  Skipping loop={loop} expanded no-cross α=0.5: already present.")
            except ValueError as e:
                print(f"  ⚠  Skipping loop={loop} expanded no-cross dual α=0.5: {e}")

    print_summary(args.flop_budget, args.d_model, args.n_layers,
                  args.max_loops, args.swiglu_multiple, configs)

    if args.template:
        print(f"\n  Writing YAMLs to {args.output_dir}/")
        for cfg in configs:
            path = write_yaml(args.template, args.output_dir, cfg)
            print(f"    ✓ {path}")


if __name__ == "__main__":
    main()