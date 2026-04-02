#!/usr/bin/env python3
"""
Iso-FLOP calculator for asymmetric Dual-Path (Wide + Deep loop) transformer.

Computes the FFN hidden sizes for Wide and Deep paths so that the total
per-token FLOPs match a dense baseline, then optionally writes out YAML
config files ready for training.

Usage examples:
  # Print FLOP tables (original behaviour)
  python flops_calculator.py table

  # Generate YAML configs for a specific scenario
  python flops_calculator.py yaml \\
      --template base.yaml \\
      --output-dir configs/ \\
      --baseline-layers 36 \\
      --our-layers 12 \\
      --max-loops 5 \\
      --ratios 0.3 0.5 0.7

  # Generate ALL configs for a NeurIPS paper (Table 1 + Table 2)
  python flops_calculator.py paper \\
      --template base.yaml \\
      --output-dir paper_configs/ \\
      --baseline-layers 36 \\
      --our-layers 12 \\
      --max-loops 5
"""

import argparse
import math
import os
import re
import sys


# =====================================================================
# Core FLOP calculation
# =====================================================================

def asymmetric_dual_config(
    d_model: int,
    target_dense_layers: int,
    dense_ffn_mult: int,
    our_layers: int,
    max_loops: int,
    capacity_ratio: float = 0.5,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    ffn_round_multiple: int = 64,
):
    """
    Calculates exact FFN sizes for an asymmetric FLOP split between Wide
    and Deep paths, ensuring total layer FLOPs match the dense baseline.
    """
    assert 0.0 <= capacity_ratio <= 1.0, "Capacity ratio must be between 0 and 1"

    if n_head_q is not None and n_head_kv is not None:
        n_rep = n_head_q // n_head_kv
    else:
        n_rep = 1
    attn_flops = 4 * (d_model ** 2) + 4 * (d_model ** 2) // n_rep

    dual_gate_flops = 2 * d_model * 2
    router_flops_total = max_loops * 2 * (d_model + 1)
    gate_flops = dual_gate_flops + router_flops_total

    dense_ffn = d_model * dense_ffn_mult
    dense_block_flops = attn_flops + (6 * d_model * dense_ffn)
    total_budget_flops = target_dense_layers * dense_block_flops

    layer_budget = total_budget_flops / our_layers
    paths_budget = layer_budget - gate_flops

    if capacity_ratio > 0:
        wide_budget = paths_budget * capacity_ratio
        ffn_wide_exact = (wide_budget - attn_flops) / (6 * d_model)
        if ffn_wide_exact < 0:
            return {"error": f"Wide budget too small to cover attention "
                    f"({wide_budget:.0f} < {attn_flops})"}
        ffn_wide = max(ffn_round_multiple,
                       math.ceil(ffn_wide_exact / ffn_round_multiple) * ffn_round_multiple)
        actual_wide_flops = attn_flops + (6 * d_model * ffn_wide)
    else:
        ffn_wide = 0
        actual_wide_flops = 0

    deep_budget = paths_budget * (1.0 - capacity_ratio)
    if deep_budget > 0 and max_loops > 0:
        budget_per_loop = deep_budget / max_loops
        ffn_deep_exact = (budget_per_loop - attn_flops) / (6 * d_model)
        if ffn_deep_exact < 0:
            return {"error": f"Deep path starved: can't afford {max_loops} loops of "
                    f"attention (need {max_loops * attn_flops:.0f}, have {deep_budget:.0f})"}
        ffn_deep = max(ffn_round_multiple,
                       math.ceil(ffn_deep_exact / ffn_round_multiple) * ffn_round_multiple)
        actual_deep_flops = max_loops * (attn_flops + (6 * d_model * ffn_deep))
    else:
        ffn_deep = 0
        actual_deep_flops = 0

    actual_total_flops = our_layers * (actual_wide_flops + actual_deep_flops + gate_flops)
    match_ratio = actual_total_flops / total_budget_flops

    return {
        "Capacity Ratio": f"{capacity_ratio*100:.0f}% Wide / {(1-capacity_ratio)*100:.0f}% Deep",
        "Target ffn_wide": ffn_wide,
        "Target ffn_deep": ffn_deep,
        "FLOP Match": f"{match_ratio * 100:.2f}%",
        "actual_total_flops": actual_total_flops,
        "total_budget_flops": total_budget_flops,
        "capacity_ratio": capacity_ratio,
        "our_layers": our_layers,
        "max_loops": max_loops,
        "d_model": d_model,
        "target_dense_layers": target_dense_layers,
        "match_ratio": match_ratio,
    }


# =====================================================================
# Parameter counting
# =====================================================================

def _attn_params(d_model: int, n_rep: int = 1) -> int:
    """Attention projection params (no bias): Q + K + V + O."""
    return 2 * d_model * d_model + 2 * (d_model * d_model) // n_rep


def _swiglu_params(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU params (no bias): gate + up + down projections.

    NOTE: uses ffn_hidden as-is.  If SwiGLU internally applies a 2/3
    scaling the absolute count will differ, but the iso-param *ratio*
    stays valid because both models use the same convention.
    """
    return 3 * d_model * ffn_hidden


def _block_params(d_model: int, ffn_hidden: int, n_rep: int = 1) -> int:
    """One GPT2Block: attn + SwiGLU + 2 × RMSNorm."""
    return _attn_params(d_model, n_rep) + _swiglu_params(d_model, ffn_hidden) + 2 * d_model


def count_dual_layer_params(
    d_model: int, ffn_deep: int, ffn_wide: int,
    max_loops: int, layer_type: str = "dual", n_rep: int = 1,
) -> int:
    """Parameter count for one AdaptiveRecursiveBlock."""
    params = 0
    if layer_type in ("loop", "dual"):
        params += _block_params(d_model, ffn_deep, n_rep)    # deep block
        params += (d_model + 1) + 1   # Router: Linear(d+1, 1) weight + bias
        params += max_loops            # loop_scales
    if layer_type in ("wide", "dual"):
        params += _block_params(d_model, ffn_wide, n_rep)    # wide block
        params += 1                    # wide_scale
    if layer_type == "dual":
        params += d_model * 2 + 2     # DualPathGate: Linear(d, 2) weight + bias
    return params


def count_model_params(
    d_model: int, n_layers: int, max_loops: int,
    ffn_deep: int, ffn_wide: int,
    vocab_size: int = 50304, use_weight_tying: bool = False,
    layer_type: str = "dual", n_rep: int = 1,
) -> int:
    """Total parameter count for the full GPT2LLM with adaptive blocks."""
    per_layer = count_dual_layer_params(d_model, ffn_deep, ffn_wide,
                                        max_loops, layer_type, n_rep)
    shared = vocab_size * d_model                      # wte
    if not use_weight_tying:
        shared += d_model * vocab_size                 # lm_head
    shared += d_model                                  # lm_head_norm
    return n_layers * per_layer + shared


def count_dense_params(
    d_model: int, n_layers: int, ffn_hidden: int,
    vocab_size: int = 50304, use_weight_tying: bool = False, n_rep: int = 1,
) -> int:
    """Total parameter count for a plain dense GPT2LLM."""
    per_layer = _block_params(d_model, ffn_hidden, n_rep)
    shared = vocab_size * d_model
    if not use_weight_tying:
        shared += d_model * vocab_size
    shared += d_model
    return n_layers * per_layer + shared


def solve_isoparam_dense_layers(
    target_params: int, d_model: int, dense_ffn_mult: int,
    vocab_size: int = 50304, use_weight_tying: bool = False, n_rep: int = 1,
) -> int:
    """Find L_dense such that count_dense_params(..., L_dense) ~ target_params."""
    dense_ffn = d_model * dense_ffn_mult
    per_layer = _block_params(d_model, dense_ffn, n_rep)
    shared = vocab_size * d_model + d_model
    if not use_weight_tying:
        shared += d_model * vocab_size
    L = (target_params - shared) / per_layer
    return max(1, round(L))


# =====================================================================
# Console printer (original behaviour)
# =====================================================================

def run_scenario(our_layers, max_loops, target_dense_layers,
                 d_model=768, dense_ffn_mult=4, ratios=None):
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*65}")
    print(f"  SCENARIO: {our_layers} Dual Layers | {max_loops} Loops")
    print(f"  Baseline: {target_dense_layers} Dense Layers "
          f"(d={d_model}, ffn={d_model*dense_ffn_mult})")
    print(f"{'='*65}")

    for r in ratios:
        res = asymmetric_dual_config(
            d_model=d_model,
            target_dense_layers=target_dense_layers,
            dense_ffn_mult=dense_ffn_mult,
            our_layers=our_layers,
            max_loops=max_loops,
            capacity_ratio=r,
        )
        print(f"\n  --- {r*100:.0f}% Wide / {(1-r)*100:.0f}% Deep ---")
        if "error" in res:
            print(f"  ❌ {res['error']}")
        else:
            print(f"  Wide FFN (wide_ffn_hidden): {res['Target ffn_wide']}")
            print(f"  Deep FFN (ffn_hidden):      {res['Target ffn_deep']}")
            print(f"  Iso-FLOP Match:             {res['FLOP Match']}")


# =====================================================================
# YAML generation — pure-text, zero dependencies
# =====================================================================

def _make_experiment_id(res: dict) -> str:
    """Build a human-readable experiment_id from a result dict."""
    # Allow override for special configs (dense baselines, etc.)
    if "_exp_id" in res:
        return res["_exp_id"]
    loops = res["max_loops"]
    deep = res["Target ffn_deep"]
    wide = res["Target ffn_wide"]
    layers = res["our_layers"]
    ratio_pct = int(res["capacity_ratio"] * 100)
    baseline = res["target_dense_layers"]
    return (f"loop{loops}_{deep}deep_{wide}wide_{layers}L_"
            f"r{ratio_pct}w{100-ratio_pct}d_iso{baseline}dense")


def _patch_yaml(template_text: str, res: dict) -> str:
    """
    Patch specific fields in the YAML template by tracking which top-level
    section we're inside and replacing only the intended key-value lines.

    Supports:
    - Replacing existing scalar values (n_layer, ffn_hidden, etc.)
    - Inserting a new layer_types list into adaptive_config (if res has "_layer_types")
    - Toggling enable_adaptive (if res has "_enable_adaptive")
    """
    exp_id = _make_experiment_id(res)

    patches = {
        ("model_raw", "n_layer"):          str(res["our_layers"]),
        ("model_raw", "ffn_hidden"):       str(res["Target ffn_deep"]),
        ("model_raw", "max_loops"):        str(res["max_loops"]),
        ("model_raw", "wide_ffn_hidden"):  str(res["Target ffn_wide"]),
        ("evaluation_subscriber", "experiment_id"): exp_id,
    }
    if "_enable_adaptive" in res:
        patches[("model_raw", "enable_adaptive")] = str(res["_enable_adaptive"]).lower()

    kv_re = re.compile(r'^(\s*)([\w_]+)(:\s+)(.+)$')

    # What to insert after deep_gate_init_bias (if needed)
    layer_types = res.get("_layer_types")   # None = don't touch, list = insert

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
            patch_key = (current_section, key)
            if patch_key in patches:
                new_val = patches[patch_key]
                line = f"{indent}{key}{sep}{new_val}\n"

        out_lines.append(line)

        # Insert layer_types after deep_gate_init_bias inside model_raw
        if (layer_types is not None
                and current_section == "model_raw"
                and m and m.group(2) == "deep_gate_init_bias"):
            indent = m.group(1)  # reuse same indentation
            lt_str = ", ".join(f'"{t}"' for t in layer_types)
            out_lines.append(f"{indent}layer_types: [{lt_str}]\n")

    return ''.join(out_lines)


def write_yaml(template_path: str, output_path: str, res: dict):
    """Load template, patch fields, write new config."""
    with open(template_path, "r") as f:
        text = f.read()

    patched = _patch_yaml(text, res)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(patched)

    return _make_experiment_id(res)


def generate_yamls(
    template_path, output_dir, baseline_layers, our_layers, max_loops,
    ratios, d_model=768, dense_ffn_mult=4, ffn_round_multiple=64,
    n_head_q=None, n_head_kv=None,
):
    """Compute configs and write one YAML per ratio."""
    results = []
    for r in ratios:
        res = asymmetric_dual_config(
            d_model=d_model, target_dense_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, capacity_ratio=r,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
            ffn_round_multiple=ffn_round_multiple,
        )
        if "error" in res:
            print(f"  ⚠  Skipping ratio {r:.0%}: {res['error']}")
            continue

        exp_id = _make_experiment_id(res)
        out_path = os.path.join(output_dir, f"{exp_id}.yaml")
        write_yaml(template_path, out_path, res)
        results.append((out_path, exp_id, res))
        print(f"  ✓  {exp_id}")
        print(f"     ffn_deep={res['Target ffn_deep']}  "
              f"ffn_wide={res['Target ffn_wide']}  "
              f"match={res['FLOP Match']}  -> {out_path}")
    return results


# =====================================================================
# Paper config generator (Table 1 + Table 2)
# =====================================================================

def _write_and_log(template_path, output_dir, res, tag, params=None):
    """Write one YAML and print a summary line."""
    exp_id = _make_experiment_id(res)
    out_path = os.path.join(output_dir, f"{exp_id}.yaml")
    write_yaml(template_path, out_path, res)
    param_str = f"  params={params/1e6:.1f}M" if params else ""
    print(f"  ✓  [{tag}]  {exp_id}")
    print(f"     L={res['our_layers']}  loops={res['max_loops']}  "
          f"ffn_deep={res['Target ffn_deep']}  ffn_wide={res['Target ffn_wide']}  "
          f"match={res['FLOP Match']}{param_str}")
    print(f"     -> {out_path}")
    return out_path, exp_id, res


def generate_paper_configs(
    template_path: str,
    output_dir: str,
    baseline_layers: int,
    our_layers: int,
    max_loops: int,
    d_model: int = 768,
    dense_ffn_mult: int = 4,
    ffn_round_multiple: int = 64,
    vocab_size: int = 50304,
    use_weight_tying: bool = False,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    table2_ratios: list[float] | None = None,
):
    """
    Generate every YAML config needed for a NeurIPS paper.

    Table 1 — Main comparison (all iso-FLOP to `baseline_layers` dense):
      1. Dense iso-FLOP          36L dense, loops=1, wide=0
      2. Dense iso-param          ≈22L dense (matched to dual-50/50 param count)
      3. Pure loop                12L, 5 loops, ratio=0  (all budget → deep)
      4. Pure wide                12L, loops=1, ratio=1  (all budget → wide)
      5. Best dual (50/50)        12L, 5 loops, ratio=0.5

    Table 2 — Ratio sweep (all iso-FLOP):
      ratio = 0.0, 0.1, 0.2, ..., 0.9, 1.0
    """
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    dense_ffn = d_model * dense_ffn_mult
    if table2_ratios is None:
        table2_ratios = [round(x * 0.1, 2) for x in range(11)]

    all_results = []

    # =================================================================
    # TABLE 1
    # =================================================================
    print(f"\n{'='*70}")
    print(f"  TABLE 1 — Main iso-FLOP comparison")
    print(f"  Baseline: {baseline_layers}L dense  (d={d_model}, ffn={dense_ffn})")
    print(f"  Dual:     {our_layers}L, {max_loops} loops")
    print(f"{'='*70}\n")

    # --- 1a. Dense iso-FLOP ---
    # baseline_layers layers, 1 loop, no wide path → ffn ≈ dense_ffn
    dense_isoflop = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=baseline_layers,
        max_loops=1, capacity_ratio=0.0, ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    dense_isoflop["_exp_id"] = f"t1_dense_isoflop_{baseline_layers}L"
    dense_params = count_dense_params(d_model, baseline_layers, dense_ffn,
                                      vocab_size, use_weight_tying, n_rep)
    all_results.append(_write_and_log(template_path, output_dir,
                                      dense_isoflop, "Dense iso-FLOP", dense_params))

    # --- 1b. Dense iso-param ---
    # First compute param count of the reference dual model (50/50)
    ref_dual = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=max_loops, capacity_ratio=0.5,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in ref_dual:
        dual_params = count_model_params(
            d_model, our_layers, max_loops,
            ref_dual["Target ffn_deep"], ref_dual["Target ffn_wide"],
            vocab_size, use_weight_tying, "dual", n_rep,
        )
        L_iso = solve_isoparam_dense_layers(dual_params, d_model, dense_ffn_mult,
                                            vocab_size, use_weight_tying, n_rep)
        # Build a result dict for the iso-param dense model
        isoparam_dense = asymmetric_dual_config(
            d_model=d_model, target_dense_layers=L_iso,
            dense_ffn_mult=dense_ffn_mult, our_layers=L_iso,
            max_loops=1, capacity_ratio=0.0,
            ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
        )
        isoparam_dense["_exp_id"] = f"t1_dense_isoparam_{L_iso}L"
        isoparam_params = count_dense_params(d_model, L_iso, dense_ffn,
                                             vocab_size, use_weight_tying, n_rep)
        all_results.append(_write_and_log(template_path, output_dir,
                                          isoparam_dense, "Dense iso-param", isoparam_params))
        print(f"     (matched to dual-50/50 with {dual_params/1e6:.1f}M params"
              f" -> {L_iso}L dense with {isoparam_params/1e6:.1f}M params)")
    else:
        print(f"  ⚠  Skipping iso-param: reference dual failed: {ref_dual['error']}")

    # --- 1c. Pure loop (ratio=0, no wide path) ---
    pure_loop = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=max_loops, capacity_ratio=0.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in pure_loop:
        pure_loop["_exp_id"] = (f"t1_pure_loop{max_loops}_"
                                f"{pure_loop['Target ffn_deep']}deep_{our_layers}L")
        loop_params = count_model_params(
            d_model, our_layers, max_loops,
            pure_loop["Target ffn_deep"], 0,
            vocab_size, use_weight_tying, "loop", n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_loop, "Pure loop", loop_params))
    else:
        print(f"  ⚠  Skipping pure loop: {pure_loop['error']}")

    # --- 1d. Pure wide (ratio=1, single pass, all budget to wide) ---
    # Use max_loops=1 since loops are irrelevant; set layer_types=["wide"]
    pure_wide = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=1, capacity_ratio=1.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in pure_wide:
        # ffn_deep=0 is invalid in config; set a dummy value (unused by "wide" layers)
        if pure_wide["Target ffn_deep"] == 0:
            pure_wide["Target ffn_deep"] = 64
        pure_wide["_exp_id"] = (f"t1_pure_wide_"
                                f"{pure_wide['Target ffn_wide']}wide_{our_layers}L")
        pure_wide["_layer_types"] = ["wide"] * our_layers
        wide_params = count_model_params(
            d_model, our_layers, 1,
            0, pure_wide["Target ffn_wide"],
            vocab_size, use_weight_tying, "wide", n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_wide, "Pure wide", wide_params))
    else:
        print(f"  ⚠  Skipping pure wide: {pure_wide['error']}")

    # --- 1e. Best dual (50/50 reference, also appears in Table 2) ---
    if "error" not in ref_dual:
        ref_dual["_exp_id"] = (f"t1_dual_{ref_dual['Target ffn_deep']}deep_"
                               f"{ref_dual['Target ffn_wide']}wide_{our_layers}L_"
                               f"{max_loops}loops")
        all_results.append(_write_and_log(template_path, output_dir,
                                          ref_dual, "Dual 50/50", dual_params))

    # =================================================================
    # TABLE 2 — Ratio sweep
    # =================================================================
    print(f"\n{'='*70}")
    print(f"  TABLE 2 — Capacity ratio sweep (all iso-FLOP to {baseline_layers}L dense)")
    print(f"  {our_layers}L, {max_loops} loops, ratios: {table2_ratios}")
    print(f"{'='*70}\n")

    for r in table2_ratios:
        # Pure wide endpoint: special handling (layer_types)
        if r == 1.0:
            res = asymmetric_dual_config(
                d_model=d_model, target_dense_layers=baseline_layers,
                dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
                max_loops=1, capacity_ratio=1.0,
                ffn_round_multiple=ffn_round_multiple,
                n_head_q=n_head_q, n_head_kv=n_head_kv,
            )
            if "error" not in res:
                if res["Target ffn_deep"] == 0:
                    res["Target ffn_deep"] = 64
                res["_layer_types"] = ["wide"] * our_layers
        elif r == 0.0:
            res = asymmetric_dual_config(
                d_model=d_model, target_dense_layers=baseline_layers,
                dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
                max_loops=max_loops, capacity_ratio=0.0,
                ffn_round_multiple=ffn_round_multiple,
                n_head_q=n_head_q, n_head_kv=n_head_kv,
            )
        else:
            res = asymmetric_dual_config(
                d_model=d_model, target_dense_layers=baseline_layers,
                dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
                max_loops=max_loops, capacity_ratio=r,
                ffn_round_multiple=ffn_round_multiple,
                n_head_q=n_head_q, n_head_kv=n_head_kv,
            )

        if "error" in res:
            print(f"  ⚠  Skipping ratio {r:.0%}: {res['error']}")
            continue

        ratio_pct = int(r * 100)
        res["_exp_id"] = (f"t2_r{ratio_pct:02d}w{100-ratio_pct:02d}d_"
                          f"{res['Target ffn_deep']}deep_{res['Target ffn_wide']}wide")
        all_results.append(_write_and_log(template_path, output_dir,
                                          res, f"Table2 {ratio_pct}w/{100-ratio_pct}d"))

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY:  {len(all_results)} configs written to {output_dir}/")
    print(f"{'='*70}\n")

    # Print a compact table for copy-paste
    print(f"  {'Experiment ID':<58s} {'L':>3s} {'Lp':>3s} "
          f"{'FFN_d':>6s} {'FFN_w':>6s} {'Match':>7s}")
    print(f"  {'-'*58} {'-'*3} {'-'*3} {'-'*6} {'-'*6} {'-'*7}")
    for _, eid, r in all_results:
        print(f"  {eid:<58s} {r['our_layers']:>3d} {r['max_loops']:>3d} "
              f"{r['Target ffn_deep']:>6d} {r['Target ffn_wide']:>6d} "
              f"{r['FLOP Match']:>7s}")

    return all_results


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Iso-FLOP calculator for Dual-Path (Wide+Deep) transformer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # Shared args helper
    def add_common(sp):
        sp.add_argument("--d-model", type=int, default=768)
        sp.add_argument("--dense-ffn-mult", type=int, default=4)
        sp.add_argument("--ffn-round-multiple", type=int, default=64)
        sp.add_argument("--n-head-q", type=int, default=None)
        sp.add_argument("--n-head-kv", type=int, default=None)

    # --- "table" ---
    t = sub.add_parser("table", help="Print FLOP tables to stdout (default)")
    add_common(t)
    t.add_argument("--baseline-layers", type=int, nargs="+", default=[60, 36])

    # --- "yaml" ---
    y = sub.add_parser("yaml", help="Generate YAML config files")
    add_common(y)
    y.add_argument("--template", required=True)
    y.add_argument("--output-dir", default="configs")
    y.add_argument("--baseline-layers", type=int, required=True)
    y.add_argument("--our-layers", type=int, required=True)
    y.add_argument("--max-loops", type=int, required=True)
    y.add_argument("--ratios", type=float, nargs="+", required=True)

    # --- "paper" ---
    pp = sub.add_parser("paper", help="Generate ALL configs for Table 1 + Table 2")
    add_common(pp)
    pp.add_argument("--template", required=True)
    pp.add_argument("--output-dir", default="paper_configs")
    pp.add_argument("--baseline-layers", type=int, required=True)
    pp.add_argument("--our-layers", type=int, required=True)
    pp.add_argument("--max-loops", type=int, required=True)
    pp.add_argument("--vocab-size", type=int, default=50304)
    pp.add_argument("--weight-tying", action="store_true", default=False)
    pp.add_argument("--table2-ratios", type=float, nargs="+", default=None,
                    help="Custom ratio list for Table 2 (default: 0.0 to 1.0 in 0.1 steps)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "table":
        d = getattr(args, "d_model", 768)
        ffn_mult = getattr(args, "dense_ffn_mult", 4)
        baselines = getattr(args, "baseline_layers", [60, 36])
        for bl in baselines:
            print(f"\n{'#'*65}")
            print(f"#  BASELINE: {bl} Dense Layers (d={d}, SwiGLU ffn={d*ffn_mult})")
            print(f"{'#'*65}")
            if bl == 60:
                scenarios = [(30,2),(20,3),(15,4),(12,5),(10,6),(12,2),(12,3),(12,4)]
            elif bl == 36:
                scenarios = [(18,2),(12,3),(9,4)]
            else:
                scenarios = [(bl//k,k) for k in [2,3,4] if bl%k==0]
            for ol, ml in scenarios:
                run_scenario(ol, ml, bl, d_model=d, dense_ffn_mult=ffn_mult)

    elif args.command == "yaml":
        print(f"\nGenerating YAML configs -> {args.output_dir}/")
        generate_yamls(
            template_path=args.template, output_dir=args.output_dir,
            baseline_layers=args.baseline_layers, our_layers=args.our_layers,
            max_loops=args.max_loops, ratios=args.ratios,
            d_model=args.d_model, dense_ffn_mult=args.dense_ffn_mult,
            ffn_round_multiple=args.ffn_round_multiple,
            n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
        )

    elif args.command == "paper":
        generate_paper_configs(
            template_path=args.template, output_dir=args.output_dir,
            baseline_layers=args.baseline_layers, our_layers=args.our_layers,
            max_loops=args.max_loops,
            d_model=args.d_model, dense_ffn_mult=args.dense_ffn_mult,
            ffn_round_multiple=args.ffn_round_multiple,
            vocab_size=args.vocab_size,
            use_weight_tying=args.weight_tying,
            n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
            table2_ratios=args.table2_ratios,
        )


if __name__ == "__main__":
    main()