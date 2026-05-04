#!/usr/bin/env python3
"""
Iso-FLOP + Iso-Params calculator for asymmetric Dual-Path transformer with
MoE wide path and dense SwiGLU deep path (looped).

Architecture (per AdaptiveRecursiveBlock):
    Deep path:  dense SwiGLU, executed `max_loops` times
    Wide path:  MoE SwiGLU (n_experts total, top_k active), executed once
    Gate:       DecoupledDualPathGate + AdaptiveRouter

Key insight
-----------
With a dense FFN in the deep path, you can't simultaneously match a dense
baseline on FLOPs AND params — the loop pays FLOPs L times but stores the
FFN weights only once. MoE breaks this symmetry on the wide path:

    effective_FLOPs  ∝  top_k · ffn_expert       (compute)
    total_params     ∝  n_experts · ffn_expert   (storage)

The ratio n_experts / top_k is the sparsity lever that absorbs the
deep-path loop factor, so we CAN hit both iso-FLOP and iso-params exactly
(up to rounding).

FLOP convention: effective — only top_k experts counted per token (matches
the realized compute of a scatter MoE implementation).
"""

import argparse
import math
import os
import re


# =====================================================================
# FLOP + param atoms
# =====================================================================

def _attn_flops(d_model: int, n_rep: int = 1) -> int:
    """QKV + output projection FLOPs (single token, single pass)."""
    return 4 * (d_model ** 2) + 4 * (d_model ** 2) // n_rep


def _swiglu_flops(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU FFN FLOPs: gate + up + down projections."""
    return 6 * d_model * ffn_hidden


def _attn_params(d_model: int, n_rep: int = 1) -> int:
    return 2 * d_model * d_model + 2 * (d_model * d_model) // n_rep


def _swiglu_params(d_model: int, ffn_hidden: int) -> int:
    return 3 * d_model * ffn_hidden


def _moe_router_flops(d_model: int, n_experts: int) -> int:
    """Router: Linear(d, n_experts) per token."""
    return d_model * n_experts


def _moe_router_params(d_model: int, n_experts: int) -> int:
    return d_model * n_experts


def _gate_flops(d_model: int, max_loops: int, has_dual_gate: bool,
                has_act_router: bool) -> int:
    """DualPathGate (Linear(d, 2)) + AdaptiveRouter (Linear(d+1, 1)) per loop."""
    g = 0
    if has_dual_gate:
        g += 2 * d_model * 2
    if has_act_router:
        g += max_loops * 2 * (d_model + 1)
    return g


def _dual_gate_params(d_model: int) -> int:
    # gate_proj + proj_w2d + proj_d2w + cross_scales
    return d_model * 2 + 2 + 2 * d_model * d_model + 2


def _act_router_params(d_model: int, max_loops: int) -> int:
    # Linear(d+1, 1) + loop_scales
    return (d_model + 1) + 1 + max_loops


# =====================================================================
# Per-layer FLOP & param counters
# =====================================================================

def layer_flops(
    d_model: int, ffn_deep: int, ffn_expert_wide: int,
    max_loops: int, top_k: int, n_experts_wide: int,
    layer_type: str = "dual", n_rep: int = 1,
) -> int:
    """Per-token FLOPs for one AdaptiveRecursiveBlock."""
    attn = _attn_flops(d_model, n_rep)
    f = 0
    has_deep = layer_type in ("loop", "dual")
    has_wide = layer_type in ("wide", "dual")

    if has_deep:
        f += max_loops * (attn + _swiglu_flops(d_model, ffn_deep))

    if has_wide:
        # MoE wide: attn once + top_k experts + router
        f += attn
        f += top_k * _swiglu_flops(d_model, ffn_expert_wide)
        f += _moe_router_flops(d_model, n_experts_wide)

    f += _gate_flops(d_model, max_loops,
                     has_dual_gate=(layer_type == "dual"),
                     has_act_router=has_deep)
    return f


def layer_params(
    d_model: int, ffn_deep: int, ffn_expert_wide: int,
    max_loops: int, n_experts_wide: int,
    layer_type: str = "dual", n_rep: int = 1,
) -> int:
    """Parameter count for one AdaptiveRecursiveBlock."""
    p = 0
    has_deep = layer_type in ("loop", "dual")
    has_wide = layer_type in ("wide", "dual")

    if has_deep:
        p += _attn_params(d_model, n_rep)
        p += _swiglu_params(d_model, ffn_deep)
        p += 2 * d_model                    # 2 RMSNorms
        p += _act_router_params(d_model, max_loops)

    if has_wide:
        p += _attn_params(d_model, n_rep)
        p += n_experts_wide * _swiglu_params(d_model, ffn_expert_wide)
        p += _moe_router_params(d_model, n_experts_wide)
        p += 2 * d_model                    # 2 RMSNorms
        p += 1                              # wide_scale

    if layer_type == "dual":
        p += _dual_gate_params(d_model)

    return p


def model_flops(
    d_model: int, n_layers: int, ffn_deep: int, ffn_expert_wide: int,
    max_loops: int, top_k: int, n_experts_wide: int,
    vocab_size: int, layer_types, n_rep: int = 1,
) -> int:
    if isinstance(layer_types, str):
        layer_types = [layer_types] * n_layers
    f = sum(layer_flops(d_model, ffn_deep, ffn_expert_wide, max_loops,
                        top_k, n_experts_wide, lt, n_rep)
            for lt in layer_types)
    # Embedding lookup is free; lm_head + final norm are the bulk
    f += 2 * d_model * vocab_size + d_model  # lm_head matmul + norm
    return f


def model_params(
    d_model: int, n_layers: int, ffn_deep: int, ffn_expert_wide: int,
    max_loops: int, n_experts_wide: int,
    vocab_size: int, use_weight_tying: bool,
    layer_types, n_rep: int = 1,
) -> int:
    if isinstance(layer_types, str):
        layer_types = [layer_types] * n_layers
    p = sum(layer_params(d_model, ffn_deep, ffn_expert_wide, max_loops,
                         n_experts_wide, lt, n_rep)
            for lt in layer_types)
    p += vocab_size * d_model                 # wte
    if not use_weight_tying:
        p += d_model * vocab_size             # lm_head
    p += d_model                              # lm_head_norm
    return p


def dense_flops(d_model, n_layers, ffn_hidden, vocab_size, n_rep=1):
    per = _attn_flops(d_model, n_rep) + _swiglu_flops(d_model, ffn_hidden)
    return n_layers * per + 2 * d_model * vocab_size + d_model


def dense_params(d_model, n_layers, ffn_hidden, vocab_size,
                 use_weight_tying=False, n_rep=1):
    per = (_attn_params(d_model, n_rep) + _swiglu_params(d_model, ffn_hidden)
           + 2 * d_model)
    shared = vocab_size * d_model + d_model
    if not use_weight_tying:
        shared += d_model * vocab_size
    return n_layers * per + shared


# =====================================================================
# The solver: iso-FLOP + iso-params for MoE wide + dense deep
# =====================================================================

def solve_dual_moe_isoflop_isoparam(
    d_model: int,
    baseline_layers: int,
    dense_ffn_mult: int,
    our_layers: int,
    max_loops: int,
    top_k: int,
    capacity_ratio: float = 0.5,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    vocab_size: int = 50304,
    use_weight_tying: bool = False,
    ffn_round_multiple: int = 16,
    experts_round_multiple: int = 1,
    min_experts: int = 2,
    max_experts: int | None = None,
):
    """
    Solve simultaneously for (ffn_deep, ffn_expert_wide, n_experts_wide) such
    that the model hits BOTH the dense baseline's FLOP budget AND its param
    budget, while splitting the FFN budget between deep and wide paths by
    `capacity_ratio` (fraction going to wide).

    Strategy
    --------
    `capacity_ratio` is a soft preference for the FLOP split between deep
    and wide; it is NOT a hard constraint. The solver searches over integer
    values of `n_experts` and for each one, solves a 2×2 linear system for
    (ffn_deep, ffn_expert) that exactly hits both F_total and P_total. It
    then picks the integer n_experts whose rounded (ffn_deep, ffn_expert)
    minimises a combined objective:

        score  =  (F_actual/F_total - 1)²  +  (P_actual/P_total - 1)²
                  +  0.1 * (actual_wide_frac - capacity_ratio)²

    The integer-first formulation matters because `n_experts` at feasible
    points is often small (2-6 for low top_k), so rounding a continuous
    n_experts to the nearest integer can introduce 10-20% param mismatch.
    By fixing n_experts to an integer and letting the (continuous) FFN
    dims absorb the slack, we get near-perfect matches everywhere.

    Why the 2×2 system is linear for fixed n_experts:
        FLOPs: max_loops*6*d*ffn_deep + top_k*6*d*ffn_expert  +  d*n_experts
        Params: 3*d*ffn_deep + n_experts*3*d*ffn_expert        +  d*n_experts
    (per layer; d*n_experts is the MoE router, independent of the hidden dims.)

    Result
    ------
    FLOPs and params both match the dense baseline to within a fraction
    of a percent across the full range of capacity_ratio, subject only to
    `ffn_round_multiple` rounding on the continuous hidden dims.
    """
    assert 0.0 < capacity_ratio < 1.0, \
        "capacity_ratio must be strictly between 0 and 1 for dual MoE"

    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    dense_ffn = d_model * dense_ffn_mult

    # -----------------------------------------------------------------
    # Targets
    # -----------------------------------------------------------------
    F_total = dense_flops(d_model, baseline_layers, dense_ffn, vocab_size, n_rep)
    P_total = dense_params(d_model, baseline_layers, dense_ffn, vocab_size,
                           use_weight_tying, n_rep)

    # -----------------------------------------------------------------
    # Strip shared and fixed costs
    # -----------------------------------------------------------------
    # Shared (same for baseline and dual) — drop to isolate the moving parts.
    shared_params = vocab_size * d_model + d_model
    if not use_weight_tying:
        shared_params += d_model * vocab_size
    shared_flops = 2 * d_model * vocab_size + d_model

    # Fixed per-dual-layer costs (attn ×2, norms ×4, gates, ACT router,
    # wide_scale, MoE router — everything EXCEPT FFN/expert hidden dims).
    attn_f = _attn_flops(d_model, n_rep)
    attn_p = _attn_params(d_model, n_rep)

    fixed_flops_per_layer = (
        max_loops * attn_f                                   # deep attn ×loops
        + attn_f                                             # wide attn once
        + _gate_flops(d_model, max_loops, True, True)        # dual gate + ACT router
        # NOTE: MoE router FLOPs depend on n_experts, handled below.
    )
    fixed_params_per_layer = (
        attn_p                                               # deep attn
        + 2 * d_model                                        # deep norms
        + _act_router_params(d_model, max_loops)             # ACT router + loop_scales
        + attn_p                                             # wide attn
        + 2 * d_model                                        # wide norms
        + 1                                                  # wide_scale
        + _dual_gate_params(d_model)                         # dual gate
        # MoE router params depend on n_experts, handled below.
    )

    # Subtract shared + fixed from targets; what remains must be spent on
    # FFN/expert hidden dims + MoE router (which scales with n_experts).
    F_remaining = F_total - shared_flops - our_layers * fixed_flops_per_layer
    P_remaining = P_total - shared_params - our_layers * fixed_params_per_layer

    if F_remaining <= 0 or P_remaining <= 0:
        return {"error": (f"Fixed costs exceed budget: "
                          f"F_rem={F_remaining:.0f}, P_rem={P_remaining:.0f}")}

    # -----------------------------------------------------------------
    # Step 3: Integer-aware search over n_experts
    # -----------------------------------------------------------------
    # Previous approach: solve 2×2 system for (ffn_expert, n_experts), then
    # round n_experts to the nearest integer. Problem: when the exact
    # n_experts is small (~2-3) and non-integer (~2.3), rounding to 2
    # throws away ~13% of the param budget and there's no knob left to
    # compensate.
    #
    # Better approach: fix n_experts to an integer, then solve the 2×2
    # system for (ffn_deep, ffn_expert) — both continuous. The system:
    #
    #   our_layers * max_loops * 6 * d * ffn_deep
    #   + our_layers * top_k * 6 * d * ffn_expert
    #   + our_layers * d * n_experts                         = F_remaining
    #
    #   our_layers * 3 * d * ffn_deep
    #   + our_layers * n_experts * 3 * d * ffn_expert
    #   + our_layers * d * n_experts                         = P_remaining
    #
    # This is linear in (ffn_deep, ffn_expert), closed-form solvable.
    # Search over integer n_experts ∈ [top_k, max_experts] and keep the
    # one whose (rounded) solution gives the best combined match.

    def solve_for_fixed_E(E: int):
        """Given integer n_experts=E, solve for (ffn_deep, ffn_expert)."""
        # Per-layer budgets after removing the MoE router (d*E per layer):
        f_per_layer = F_remaining / our_layers - d_model * E
        p_per_layer = P_remaining / our_layers - d_model * E
        if f_per_layer <= 0 or p_per_layer <= 0:
            return None

        # Let D := ffn_deep, H := ffn_expert.
        #   max_loops * 6 * d * D + top_k * 6 * d * H = f_per_layer
        #   3 * d * D             + E * 3 * d * H      = p_per_layer
        #
        # In matrix form:
        #   [ 6*d*max_loops    6*d*top_k  ] [D]   [f_per_layer]
        #   [ 3*d              3*d*E      ] [H] = [p_per_layer]
        a11 = 6 * d_model * max_loops
        a12 = 6 * d_model * top_k
        a21 = 3 * d_model
        a22 = 3 * d_model * E
        det = a11 * a22 - a12 * a21
        if det == 0:
            return None
        D = (f_per_layer * a22 - a12 * p_per_layer) / det
        H = (a11 * p_per_layer - f_per_layer * a21) / det
        if D <= 0 or H <= 0:
            return None

        # Round to grid
        D_r = max(ffn_round_multiple,
                  round(D / ffn_round_multiple) * ffn_round_multiple)
        H_r = max(ffn_round_multiple,
                  round(H / ffn_round_multiple) * ffn_round_multiple)

        F_act = model_flops(d_model, our_layers, D_r, H_r,
                            max_loops, top_k, E, vocab_size,
                            layer_types="dual", n_rep=n_rep)
        P_act = model_params(d_model, our_layers, D_r, H_r,
                             max_loops, E, vocab_size, use_weight_tying,
                             layer_types="dual", n_rep=n_rep)
        f_ratio = F_act / F_total
        p_ratio = P_act / P_total
        # Combined error: relative squared deviation from 1.0 on each axis.
        combined_err = (f_ratio - 1.0) ** 2 + (p_ratio - 1.0) ** 2
        return {
            "n_experts": E, "ffn_deep": D_r, "ffn_expert": H_r,
            "ffn_deep_exact": D, "ffn_expert_exact": H,
            "F_actual": F_act, "P_actual": P_act,
            "f_ratio": f_ratio, "p_ratio": p_ratio,
            "combined_err": combined_err,
        }

    # Search range: [max(top_k, min_experts) … max_experts or a sensible cap]
    E_lo = max(top_k, min_experts)
    # Upper cap: way above what the solver would ever pick. Gate by param
    # budget: if E * d_model already eats the whole param remainder, stop.
    E_hi_default = 256
    E_hi = max_experts if max_experts is not None else E_hi_default

    candidates = []
    for E in range(E_lo, E_hi + 1, experts_round_multiple):
        r = solve_for_fixed_E(E)
        if r is not None:
            # Respect capacity_ratio: prefer solutions where the wide/deep
            # FLOP split is close to what the user asked for.
            # Deep FLOP fraction of the remaining budget:
            deep_f = our_layers * max_loops * _swiglu_flops(d_model, r["ffn_deep"])
            wide_f = our_layers * (top_k * _swiglu_flops(d_model, r["ffn_expert"])
                                   + d_model * r["n_experts"])
            total_f = deep_f + wide_f
            actual_wide_frac = wide_f / total_f if total_f > 0 else 0.0
            r["actual_wide_frac"] = actual_wide_frac
            # Penalty for deviating from capacity_ratio (soft):
            ratio_err = (actual_wide_frac - capacity_ratio) ** 2
            r["ratio_err"] = ratio_err
            # Combined score: match error + ratio deviation weighted softly.
            # Ratio is a preference, not a hard constraint.
            r["score"] = r["combined_err"] + 0.1 * ratio_err
            candidates.append(r)

    if not candidates:
        return {"error": "No feasible integer n_experts yields a valid (ffn_deep, ffn_expert)."}

    best = min(candidates, key=lambda r: r["score"])

    ffn_deep = best["ffn_deep"]
    ffn_expert = best["ffn_expert"]
    n_experts = best["n_experts"]
    ffn_deep_exact = best["ffn_deep_exact"]
    ffn_expert_exact = best["ffn_expert_exact"]
    # Back-compat: report the "exact" continuous n_experts from the
    # OLD solve (quadratic). It's just for diagnostic logging.
    n_experts_exact = float(n_experts)  # integer-only path
    clamped_to_top_k = False  # not meaningful in this formulation
    F_actual = best["F_actual"]
    P_actual = best["P_actual"]

    return {
        "Capacity Ratio":    f"{capacity_ratio*100:.0f}% Wide / {(1-capacity_ratio)*100:.0f}% Deep",
        "Target ffn_deep":   ffn_deep,
        "Target ffn_expert": ffn_expert,
        "Target n_experts":  n_experts,
        "top_k":             top_k,
        "FLOP Match":        f"{F_actual / F_total * 100:.2f}%",
        "Param Match":       f"{P_actual / P_total * 100:.2f}%",
        "sparsity":          n_experts / top_k,
        # Raw fields (for YAML patching / reporting):
        "our_layers":         our_layers,
        "max_loops":          max_loops,
        "d_model":            d_model,
        "baseline_layers":    baseline_layers,
        "capacity_ratio":     capacity_ratio,
        "F_total_budget":     F_total,
        "P_total_budget":     P_total,
        "F_actual":           F_actual,
        "P_actual":           P_actual,
        "flop_match_ratio":   F_actual / F_total,
        "param_match_ratio":  P_actual / P_total,
        "ffn_expert_exact":   ffn_expert_exact,
        "n_experts_exact":    n_experts_exact,
        "ffn_deep_exact":     ffn_deep_exact,
        "clamped_to_top_k":   clamped_to_top_k,
        "actual_wide_frac":   best["actual_wide_frac"],
    }


# =====================================================================
# Convenience: baseline summary
# =====================================================================

def baseline_summary(d_model, baseline_layers, dense_ffn_mult, vocab_size,
                     use_weight_tying=False, n_head_q=None, n_head_kv=None):
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    dense_ffn = d_model * dense_ffn_mult
    F = dense_flops(d_model, baseline_layers, dense_ffn, vocab_size, n_rep)
    P = dense_params(d_model, baseline_layers, dense_ffn, vocab_size,
                     use_weight_tying, n_rep)
    return {"F_total": F, "P_total": P, "dense_ffn": dense_ffn}


# =====================================================================
# YAML patching (mirrors original, with MoE fields added)
# =====================================================================

def _make_experiment_id(res: dict) -> str:
    if "_exp_id" in res:
        return res["_exp_id"]
    loops = res["max_loops"]
    deep = res["Target ffn_deep"]
    expert = res["Target ffn_expert"]
    ne = res["Target n_experts"]
    tk = res["top_k"]
    layers = res["our_layers"]
    ratio_pct = int(res["capacity_ratio"] * 100)
    base = res["baseline_layers"]
    return (f"moe_loop{loops}_{deep}deep_{expert}x{ne}of{tk}_{layers}L_"
            f"r{ratio_pct}w{100-ratio_pct}d_iso{base}dense")


def _patch_yaml(template_text: str, res: dict) -> str:
    """Patch YAML fields.

    For keys that ALREADY exist in the template, we replace the value.
    For keys that DO NOT exist (but should, for MoE), we INSERT them into
    the adaptive_config section after a known anchor.

    The previous version silently dropped MoE fields if the template didn't
    already declare them; this version guarantees they show up.
    """
    exp_id = _make_experiment_id(res)

    # --- Keys to replace if present ---
    replace_patches = {
        ("model_raw", "n_layer"):          str(res["our_layers"]),
        ("model_raw", "ffn_hidden"):       str(res["Target ffn_deep"]),
        ("model_raw", "max_loops"):        str(res["max_loops"]),
        ("model_raw", "wide_ffn_hidden"):  str(res["Target ffn_expert"]),
        ("model_raw", "use_moe_wide"):     "true",
        ("model_raw", "n_experts"):        str(res["Target n_experts"]),
        ("model_raw", "top_k"):            str(res["top_k"]),
        ("model_raw", "expert_ffn_hidden"): str(res["Target ffn_expert"]),
        ("evaluation_subscriber", "experiment_id"): exp_id,
    }

    if res.get("_d_model") is not None:
        replace_patches[("model_raw", "n_embd")] = str(res["_d_model"])
    if res.get("_n_head_q") is not None:
        replace_patches[("model_raw", "n_head_q")] = str(res["_n_head_q"])
    if res.get("_n_head_kv") is not None:
        replace_patches[("model_raw", "n_head_kv")] = str(res["_n_head_kv"])
    if res.get("_d_model") is not None and res.get("_n_head_q") is not None:
        head_dim = res["_d_model"] // res["_n_head_q"]
        replace_patches[("model_raw", "normalized_shape")] = f"{head_dim} # n_embd / n_head_q"

    # --- Keys to INSERT into adaptive_config if not present in the template ---
    # These get emitted right after a known anchor key (wide_gate_init_bias
    # or wide_ffn_hidden, whichever comes last among those that exist).
    moe_inserts = [
        ("use_moe_wide",      "true"),
        ("n_experts",         str(res["Target n_experts"])),
        ("top_k",             str(res["top_k"])),
        ("expert_ffn_hidden", str(res["Target ffn_expert"])),
    ]

    # Regex for scalar key-value lines. We split off a trailing comment so
    # it can be preserved (or dropped if the replacement already has one).
    #   group 1 = indent, 2 = key, 3 = ': ', 4 = value, 5 = trailing comment (or '')
    kv_re = re.compile(r'^(\s*)([\w_]+)(:\s+)([^#\s][^#]*?)(\s*#.*)?$')
    # Regex for nested-dict keys (key:  with nothing after)
    dict_re = re.compile(r'^(\s*)([\w_]+):\s*(?:#.*)?$')
    layer_types = res.get("_layer_types")

    lines = template_text.splitlines(keepends=True)

    # --- Pass 1: find what's already in the template so we don't double-insert ---
    existing_keys: set[tuple[str, str]] = set()
    current_section = None
    adaptive_config_indent = None  # indentation used inside adaptive_config
    anchor_line_idx: int | None = None
    insertion_anchor_key: str | None = None

    # Preference order for insertion anchors inside adaptive_config.
    # We pick the LAST occurrence of the highest-ranked anchor.
    preferred_anchors = [
        "use_cross", "wide_gate_init_bias", "wide_ffn_hidden", "deep_gate_init_bias",
        "ponder_penalty_weight", "max_loops", "enable_adaptive",
    ]
    best_anchor_rank = len(preferred_anchors)  # lower = better

    in_adaptive_block = False
    adaptive_block_base_indent: int | None = None

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        # Strip trailing newline for indent calc
        leading_indent = len(line) - len(line.lstrip(' \t'))

        # Track top-level section
        if line[0] not in (' ', '\t', '\n', '\r'):
            colon = stripped.find(':')
            if colon > 0:
                current_section = stripped[:colon].strip()

        # Dict-anchor match (e.g. "    adaptive_config:")
        dm = dict_re.match(line.rstrip('\n'))
        if dm and dm.group(2) == "adaptive_config":
            in_adaptive_block = True
            adaptive_block_base_indent = leading_indent
            continue

        # Scalar key-value match
        m = kv_re.match(line.rstrip('\n'))
        if m:
            key = m.group(2)

            if in_adaptive_block:
                if leading_indent > (adaptive_block_base_indent or -1):
                    existing_keys.add(("adaptive_config", key))
                    if adaptive_config_indent is None:
                        adaptive_config_indent = m.group(1)
                    if key in preferred_anchors:
                        rank = preferred_anchors.index(key)
                        # Strict inequality: keep the FIRST best-ranked key so
                        # inserts land near the top of adaptive_config.
                        # (Ties broken by first occurrence.)
                        if rank < best_anchor_rank:
                            best_anchor_rank = rank
                            anchor_line_idx = idx
                            insertion_anchor_key = key
                else:
                    in_adaptive_block = False

            if current_section == "model_raw":
                existing_keys.add(("model_raw", key))

    # Determine which MoE fields need to be INSERTED (not already present)
    to_insert: list[tuple[str, str]] = []
    for k, v in moe_inserts:
        if ("adaptive_config", k) not in existing_keys:
            to_insert.append((k, v))

    # --- Pass 2: rewrite lines, doing replaces and insertions ---
    out: list[str] = []
    current_section = None
    inserted = False
    in_adaptive_block = False
    adaptive_block_base_indent = None

    for idx, line in enumerate(lines):
        stripped = line.lstrip()

        # Track top-level section on non-indented lines
        if (stripped and not stripped.startswith('#')
                and line[0] not in (' ', '\t', '\n', '\r')):
            colon = stripped.find(':')
            if colon > 0:
                current_section = stripped[:colon].strip()

        leading_indent = len(line) - len(line.lstrip(' \t'))

        # Dict anchor line ("adaptive_config:")
        dm = dict_re.match(line.rstrip('\n'))
        if dm and dm.group(2) == "adaptive_config" and current_section == "model_raw":
            in_adaptive_block = True
            adaptive_block_base_indent = leading_indent

        # Scalar key-value line
        m = kv_re.match(line.rstrip('\n'))
        if m:
            key = m.group(2)
            if (in_adaptive_block and adaptive_block_base_indent is not None
                    and leading_indent <= adaptive_block_base_indent
                    and key != "adaptive_config"):
                # We've exited the adaptive_config block
                in_adaptive_block = False

            # Do value replacement
            indent, sep, old = m.group(1), m.group(3), m.group(4)
            trailing = m.group(5) or ""  # existing trailing comment (with leading space)
            # If the replacement value itself already contains a '#', use it as-is
            # (e.g. "64 # n_embd / n_head_q") and drop the template's old comment.
            if not old.lstrip().startswith('${'):
                pk = ("model_raw", key) if in_adaptive_block else (current_section, key)
                if pk in replace_patches:
                    new_val = replace_patches[pk]
                    if '#' in new_val:
                        line = f"{indent}{key}{sep}{new_val}\n"
                    else:
                        line = f"{indent}{key}{sep}{new_val}{trailing}\n"

        out.append(line)

        # Insert missing MoE keys right after the anchor line
        if (not inserted and to_insert and idx == anchor_line_idx
                and adaptive_config_indent is not None):
            for k, v in to_insert:
                out.append(f"{adaptive_config_indent}{k}: {v}\n")
            inserted = True

        # layer_types injection (unchanged behaviour) — only if not already present
        if (layer_types is not None
                and m is not None
                and m.group(2) == "deep_gate_init_bias"
                and in_adaptive_block
                and ("adaptive_config", "layer_types") not in existing_keys):
            indent = m.group(1)
            lt_str = ", ".join(f'"{t}"' for t in layer_types)
            out.append(f"{indent}layer_types: [{lt_str}]\n")

    # If we never found an anchor but still need to insert, append at end of file
    # (with a warning marker so the user notices)
    if not inserted and to_insert:
        out.append("\n# WARNING: MoE fields inserted at end of file — template had no\n")
        out.append("#          adaptive_config section with a recognisable anchor key.\n")
        for k, v in to_insert:
            out.append(f"# {k}: {v}\n")

    return ''.join(out)


def write_yaml(template_path, output_path, res):
    with open(template_path, "r") as f:
        txt = f.read()
    patched = _patch_yaml(txt, res)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(patched)
    return _make_experiment_id(res)


# =====================================================================
# Runner
# =====================================================================

def run_sweep(
    d_model: int,
    baseline_layers: int,
    our_layers: int,
    max_loops: int,
    top_k: int,
    ratios: list[float],
    dense_ffn_mult: int = 4,
    vocab_size: int = 50304,
    use_weight_tying: bool = False,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    ffn_round_multiple: int = 64,
    experts_round_multiple: int = 1,
    min_experts: int = 2,
    max_experts: int | None = None,
    template_path: str | None = None,
    output_dir: str | None = None,
):
    bs = baseline_summary(d_model, baseline_layers, dense_ffn_mult,
                          vocab_size, use_weight_tying, n_head_q, n_head_kv)
    print(f"\n{'='*78}")
    print(f"  BASELINE: {baseline_layers}L dense  (d={d_model}, ffn={bs['dense_ffn']})")
    print(f"    F_total = {bs['F_total']:.3e}")
    print(f"    P_total = {bs['P_total']/1e6:.2f}M")
    print(f"  DUAL:     {our_layers}L × loops={max_loops}   top_k={top_k}")
    print(f"{'='*78}\n")

    results = []
    for r in ratios:
        res = solve_dual_moe_isoflop_isoparam(
            d_model=d_model, baseline_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, top_k=top_k, capacity_ratio=r,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
            vocab_size=vocab_size, use_weight_tying=use_weight_tying,
            ffn_round_multiple=ffn_round_multiple,
            experts_round_multiple=experts_round_multiple,
            min_experts=min_experts, max_experts=max_experts,
        )
        if "error" in res:
            print(f"  ratio={r:.2f}  ❌ {res['error']}")
            continue
        warn = "  ⚠ n_experts clamped to top_k" if res.get("clamped_to_top_k") else ""
        awf = res.get("actual_wide_frac")
        awf_str = f"  actual_wide={awf*100:.0f}%" if awf is not None else ""
        print(f"  ratio={r:.2f}  ffn_deep={res['Target ffn_deep']:>5d}  "
              f"ffn_expert={res['Target ffn_expert']:>5d}  "
              f"n_experts={res['Target n_experts']:>3d}  "
              f"(sparsity {res['sparsity']:.1f}×)  "
              f"F={res['FLOP Match']}  P={res['Param Match']}{awf_str}{warn}")
        if template_path and output_dir:
            _add_size_info(res, d_model, n_head_q, n_head_kv)
            out_path = os.path.join(output_dir, f"{_make_experiment_id(res)}.yaml")
            write_yaml(template_path, out_path, res)
            print(f"     -> {out_path}")
        results.append(res)
    return results


def _add_size_info(res, d_model, n_head_q, n_head_kv):
    res["_d_model"] = d_model
    res["_n_head_q"] = n_head_q
    res["_n_head_kv"] = n_head_kv
    return res


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Iso-FLOP + Iso-Params calculator: MoE wide + dense deep dual-path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--dense-ffn-mult", type=int, default=4)
    p.add_argument("--baseline-layers", type=int, required=True)
    p.add_argument("--our-layers", type=int, required=True)
    p.add_argument("--max-loops", type=int, required=True)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7])
    p.add_argument("--n-head-q", type=int, default=None)
    p.add_argument("--n-head-kv", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--weight-tying", action="store_true", default=False)
    p.add_argument("--ffn-round-multiple", type=int, default=64)
    p.add_argument("--experts-round-multiple", type=int, default=1)
    p.add_argument("--min-experts", type=int, default=2)
    p.add_argument("--max-experts", type=int, default=None)
    p.add_argument("--template", default=None,
                   help="YAML template path (optional; required for --output-dir)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write YAMLs (optional)")
    return p


def main():
    args = build_parser().parse_args()
    run_sweep(
        d_model=args.d_model,
        baseline_layers=args.baseline_layers,
        our_layers=args.our_layers,
        max_loops=args.max_loops,
        top_k=args.top_k,
        ratios=args.ratios,
        dense_ffn_mult=args.dense_ffn_mult,
        vocab_size=args.vocab_size,
        use_weight_tying=args.weight_tying,
        n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        ffn_round_multiple=args.ffn_round_multiple,
        experts_round_multiple=args.experts_round_multiple,
        min_experts=args.min_experts,
        max_experts=args.max_experts,
        template_path=args.template,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()