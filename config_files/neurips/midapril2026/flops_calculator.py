#!/usr/bin/env python3
"""
Iso-FLOP calculator for asymmetric Dual-Path (Wide + Deep loop) transformer.

Computes the FFN hidden sizes for Wide and Deep paths so that the total
per-token FLOPs match a dense baseline, then optionally writes out YAML
config files ready for training.
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
    ffn_round_multiple: int = 16,
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


def mixed_sandwich_config(
    d_model: int,
    target_dense_layers: int,
    dense_ffn_mult: int,
    our_layers: int,
    max_loops: int,
    capacity_ratio: float = 0.5,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    ffn_round_multiple: int = 16,
):
    """
    Calculates exact FFN sizes for a mixed sandwich architecture:
    ~33% Loop layers, ~33% Dual layers, ~33% Wide layers.
    """
    assert 0.0 <= capacity_ratio <= 1.0, "Capacity ratio must be between 0 and 1"

    if n_head_q is not None and n_head_kv is not None:
        n_rep = n_head_q // n_head_kv
    else:
        n_rep = 1

    attn_flops = 4 * (d_model ** 2) + 4 * (d_model ** 2) // n_rep
    dual_gate_flops = 2 * d_model * 2
    router_flops_total = max_loops * 2 * (d_model + 1)

    dense_ffn = d_model * dense_ffn_mult
    dense_block_flops = attn_flops + (6 * d_model * dense_ffn)
    total_budget_flops = target_dense_layers * dense_block_flops

    # Calculate layer distributions
    L_loop = our_layers // 3
    L_wide = our_layers // 3
    L_dual = our_layers - L_loop - L_wide

    layer_types = (["loop"] * L_loop) + (["dual"] * L_dual) + (["wide"] * L_wide)

    # Fixed FLOPs for Attention and Gates across all layers
    loop_layer_fixed = max_loops * attn_flops + router_flops_total
    dual_layer_fixed = max_loops * attn_flops + attn_flops + dual_gate_flops + router_flops_total
    wide_layer_fixed = attn_flops

    total_fixed_flops = (L_loop * loop_layer_fixed) + (L_dual * dual_layer_fixed) + (L_wide * wide_layer_fixed)
    ffn_budget = total_budget_flops - total_fixed_flops

    if ffn_budget < 0:
        return {"error": f"Fixed FLOPs (attention+gates) exceed total budget!"}

    wide_budget = ffn_budget * capacity_ratio
    deep_budget = ffn_budget * (1.0 - capacity_ratio)

    # The fixed budget already accounts for ALL attention passes.
    # Therefore, the remaining budget strictly targets the 6*d*ffn hidden projections.
    
    # Total wide FFN executions: L_dual + L_wide
    if (L_dual + L_wide) > 0 and capacity_ratio > 0:
        wide_flops_per_exec = wide_budget / (L_dual + L_wide)
        ffn_wide_exact = wide_flops_per_exec / (6 * d_model)  # Bug fixed: removed double - attn_flops
        if ffn_wide_exact < 0: ffn_wide_exact = 0
        ffn_wide = max(ffn_round_multiple, math.ceil(ffn_wide_exact / ffn_round_multiple) * ffn_round_multiple)
    else:
        ffn_wide = 0

    # Total deep FFN executions: (L_loop + L_dual) * max_loops
    deep_execs = (L_loop + L_dual) * max_loops
    if deep_execs > 0 and (1.0 - capacity_ratio) > 0:
        deep_flops_per_exec = deep_budget / deep_execs
        ffn_deep_exact = deep_flops_per_exec / (6 * d_model)  # Bug fixed: removed double - attn_flops
        if ffn_deep_exact < 0: ffn_deep_exact = 0
        ffn_deep = max(ffn_round_multiple, math.ceil(ffn_deep_exact / ffn_round_multiple) * ffn_round_multiple)
    else:
        ffn_deep = 0

    # Recalculate actual FLOPs based on rounded FFN sizes
    actual_total_flops = total_fixed_flops + \
                         (L_dual + L_wide) * (6 * d_model * ffn_wide) + \
                         deep_execs * (6 * d_model * ffn_deep)

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
        "_layer_types": layer_types,
    }


# =====================================================================
# Parameter counting
# =====================================================================

def _attn_params(d_model: int, n_rep: int = 1) -> int:
    """Attention projection params (no bias): Q + K + V + O."""
    return 2 * d_model * d_model + 2 * (d_model * d_model) // n_rep


def _swiglu_params(d_model: int, ffn_hidden: int) -> int:
    """SwiGLU params (no bias): gate + up + down projections."""
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
    layer_types: str | list[str] = "dual", n_rep: int = 1,
) -> int:
    """Total parameter count for the full GPT2LLM with adaptive blocks."""
    if isinstance(layer_types, str):
        per_layer = count_dual_layer_params(d_model, ffn_deep, ffn_wide,
                                            max_loops, layer_types, n_rep)
        total_layers_params = n_layers * per_layer
    else:
        total_layers_params = sum(
            count_dual_layer_params(d_model, ffn_deep, ffn_wide, max_loops, lt, n_rep) 
            for lt in layer_types
        )
        
    shared = vocab_size * d_model                      # wte
    if not use_weight_tying:
        shared += d_model * vocab_size                 # lm_head
    shared += d_model                                  # lm_head_norm
    return total_layers_params + shared


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
    
    if res.get("_d_model") is not None:
        patches[("model_raw", "n_embd")] = str(res["_d_model"])
    if res.get("_n_head_q") is not None:
        patches[("model_raw", "n_head_q")] = str(res["_n_head_q"])
    if res.get("_n_head_kv") is not None:
        patches[("model_raw", "n_head_kv")] = str(res["_n_head_kv"])
    
    if res.get("_d_model") is not None and res.get("_n_head_q") is not None:
        head_dim = res["_d_model"] // res["_n_head_q"]
        patches[("model_raw", "normalized_shape")] = f"{head_dim} # n_embd / n_head_q"

    kv_re = re.compile(r'^(\s*)([\w_]+)(:\s+)(.+)$')
    layer_types = res.get("_layer_types")   

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
                    new_val = patches[patch_key]
                    line = f"{indent}{key}{sep}{new_val}\n"

        out_lines.append(line)

        if (layer_types is not None
                and current_section == "model_raw"
                and m and m.group(2) == "deep_gate_init_bias"):
            indent = m.group(1) 
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
    ratios, d_model=768, dense_ffn_mult=4, ffn_round_multiple=16,
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

def _write_and_log(template_path, output_dir, res, tag, params=None,
                   baseline_params=None):
    """Write one YAML and print a summary line."""
    exp_id = _make_experiment_id(res)
    out_path = os.path.join(output_dir, f"{exp_id}.yaml")
    write_yaml(template_path, out_path, res)
    if params is not None:
        res["_params"] = params
    param_str = ""
    if params is not None:
        param_str = f"  params={params/1e6:.1f}M"
        if baseline_params is not None and baseline_params > 0:
            delta_pct = (params / baseline_params - 1.0) * 100
            sign = "+" if delta_pct >= 0 else ""
            param_str += f" ({sign}{delta_pct:.1f}% vs baseline)"
    print(f"  ✓  [{tag}]  {exp_id}")
    print(f"     L={res['our_layers']}  loops={res['max_loops']}  "
          f"ffn_deep={res['Target ffn_deep']}  ffn_wide={res['Target ffn_wide']}  "
          f"match={res['FLOP Match']}{param_str}")
    print(f"     -> {out_path}")
    return out_path, exp_id, res


def _find_max_wide_ratio(d_model, baseline_layers, dense_ffn_mult, our_layers,
                         max_loops, ffn_round_multiple, n_head_q, n_head_kv,
                         step=0.01, start=0.99):
    """Walk down from `start` in `step` increments and return the highest
    capacity_ratio for which the deep path is still feasible."""
    r = start
    while r > 0.0:
        res = asymmetric_dual_config(
            d_model=d_model, target_dense_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, capacity_ratio=r,
            ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
        )
        if "error" not in res:
            return r, res
        r = round(r - step, 4)
    return None, None


def generate_paper_configs(
    template_path: str,
    output_dir: str,
    baseline_layers: int,
    our_layers: int,
    max_loops: int,
    d_model: int = 768,
    dense_ffn_mult: int = 4,
    ffn_round_multiple: int = 16,
    vocab_size: int = 50304,
    use_weight_tying: bool = False,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    ratios: list[float] | None = None,
    find_max_wide: bool = True,
):
    """
    Generate the experiment grid for a single (d_model, baseline_layers) setup.
    """
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    dense_ffn = d_model * dense_ffn_mult
    if ratios is None:
        ratios = [0.3, 0.5, 0.7]

    def _add_size_info(res):
        res["_d_model"] = d_model
        res["_n_head_q"] = n_head_q
        res["_n_head_kv"] = n_head_kv
        return res

    tag = f"d{d_model}"
    all_results = []

    print(f"\n{'='*70}")
    print(f"  PAPER CONFIGS")
    print(f"  d_model={d_model}  (heads q={n_head_q}, kv={n_head_kv})")
    print(f"  Baseline:  {baseline_layers}L dense, ffn={dense_ffn}")
    print(f"  Dual:      {our_layers}L × {max_loops} loops")
    print(f"  Ratios:    {ratios}    +max-wide search: {find_max_wide}")
    print(f"{'='*70}\n")

    baseline_params = None

    # --- 1. Dense iso-FLOP ---
    dense_iso = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=baseline_layers,
        max_loops=1, capacity_ratio=0.0, ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in dense_iso:
        if dense_iso["Target ffn_wide"] == 0:
            dense_iso["Target ffn_wide"] = 64
        dense_iso["_layer_types"] = ["loop"] * baseline_layers
        dense_iso["_exp_id"] = f"{tag}_dense_isoflop_{baseline_layers}L"
        _add_size_info(dense_iso)
        dense_params = count_dense_params(d_model, baseline_layers, dense_ffn,
                                          vocab_size, use_weight_tying, n_rep)
        baseline_params = dense_params
        all_results.append(_write_and_log(template_path, output_dir,
                                          dense_iso, "Dense iso-FLOP", dense_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping dense iso-FLOP: {dense_iso['error']}")

    # --- 2. Pure loop (ratio=0, no wide path) ---
    pure_loop = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=max_loops, capacity_ratio=0.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in pure_loop:
        if pure_loop["Target ffn_wide"] == 0:
            pure_loop["Target ffn_wide"] = 64
        pure_loop["_layer_types"] = ["loop"] * our_layers
        pure_loop["_exp_id"] = (f"{tag}_pure_loop{max_loops}_"
                                f"{pure_loop['Target ffn_deep']}deep_{our_layers}L")
        _add_size_info(pure_loop)
        loop_params = count_model_params(
            d_model, our_layers, max_loops,
            pure_loop["Target ffn_deep"], 0,
            vocab_size, use_weight_tying, "loop", n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_loop, "Pure loop", loop_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping pure loop: {pure_loop['error']}")

    # --- 3. Pure wide (ratio=1, single pass, all budget to wide) ---
    pure_wide = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=1, capacity_ratio=1.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in pure_wide:
        if pure_wide["Target ffn_deep"] == 0:
            pure_wide["Target ffn_deep"] = 64
        pure_wide["_layer_types"] = ["wide"] * our_layers
        pure_wide["_exp_id"] = (f"{tag}_pure_wide_"
                                f"{pure_wide['Target ffn_wide']}wide_{our_layers}L")
        _add_size_info(pure_wide)
        wide_params = count_model_params(
            d_model, our_layers, 1, 0, pure_wide["Target ffn_wide"],
            vocab_size, use_weight_tying, "wide", n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_wide, "Pure wide", wide_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping pure wide: {pure_wide['error']}")

    # --- 4. Ratio sweep ---
    for r in ratios:
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
        wp = int(round(r * 100))
        dp = 100 - wp
        res["_exp_id"] = (f"{tag}_dual_r{wp:02d}w{dp:02d}d_"
                          f"{res['Target ffn_deep']}deep_{res['Target ffn_wide']}wide_"
                          f"{our_layers}L_loop{max_loops}")
        _add_size_info(res)
        params = count_model_params(
            d_model, our_layers, max_loops,
            res["Target ffn_deep"], res["Target ffn_wide"],
            vocab_size, use_weight_tying, "dual", n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          res, f"Ratio {wp}w/{dp}d", params,
                                          baseline_params=baseline_params))

    # --- 5. Max-wide search (highest feasible wide-heavy ratio) ---
    best_r = None
    if find_max_wide:
        best_r, best_res = _find_max_wide_ratio(
            d_model=d_model, baseline_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
        )
        if best_res is not None:
            already = any(abs(best_r - rr) < 0.005 for rr in ratios)
            if already:
                print(f"\n  Max-wide ratio {best_r:.2f} already in --ratios, "
                      f"skipping duplicate.")
            else:
                wp = int(round(best_r * 100))
                dp = 100 - wp
                best_res["_exp_id"] = (
                    f"{tag}_dual_MAXWIDE_r{wp:02d}w{dp:02d}d_"
                    f"{best_res['Target ffn_deep']}deep_"
                    f"{best_res['Target ffn_wide']}wide_"
                    f"{our_layers}L_loop{max_loops}"
                )
                _add_size_info(best_res)
                params = count_model_params(
                    d_model, our_layers, max_loops,
                    best_res["Target ffn_deep"], best_res["Target ffn_wide"],
                    vocab_size, use_weight_tying, "dual", n_rep,
                )
                print(f"\n  Max-wide search: highest feasible ratio = {best_r:.2f}  "
                      f"({wp}% wide / {dp}% deep)")
                all_results.append(_write_and_log(template_path, output_dir,
                                                  best_res,
                                                  f"MAX-WIDE {wp}w/{dp}d", params,
                                                  baseline_params=baseline_params))
        else:
            print("\n  ⚠  Max-wide search found no feasible ratio")

    # --- 6. Mixed Sandwich (33% loop, 33% dual, 33% wide) ---
    sandwich_ratio = best_r if (find_max_wide and best_r is not None) else 0.5
    mixed_res = mixed_sandwich_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=max_loops, capacity_ratio=sandwich_ratio,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv,
    )
    if "error" not in mixed_res:
        wp = int(round(sandwich_ratio * 100))
        dp = 100 - wp
        mixed_res["_exp_id"] = (
            f"{tag}_mixed_sandwich_r{wp:02d}w{dp:02d}d_"
            f"{mixed_res['Target ffn_deep']}deep_{mixed_res['Target ffn_wide']}wide_"
            f"{our_layers}L_loop{max_loops}"
        )
        _add_size_info(mixed_res)
        params = count_model_params(
            d_model, our_layers, max_loops,
            mixed_res["Target ffn_deep"], mixed_res["Target ffn_wide"],
            vocab_size, use_weight_tying, mixed_res["_layer_types"], n_rep,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          mixed_res, f"Mixed Sandwich {wp}w/{dp}d", params,
                                          baseline_params=baseline_params))
    else:
        print(f"\n  ⚠  Skipping mixed sandwich: {mixed_res['error']}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY:  {len(all_results)} configs written to {output_dir}/")
    if baseline_params is not None:
        print(f"  Baseline (dense iso-FLOP) = {baseline_params/1e6:.1f}M params")
    print(f"{'='*70}\n")

    print(f"  {'Experiment ID':<58s} {'L':>3s} {'Lp':>3s} "
          f"{'FFN_d':>6s} {'FFN_w':>6s} {'Match':>7s} "
          f"{'Params':>9s} {'vs base':>9s}")
    print(f"  {'-'*58} {'-'*3} {'-'*3} {'-'*6} {'-'*6} {'-'*7} "
          f"{'-'*9} {'-'*9}")
    for _, eid, r in all_results:
        params = r.get("_params")
        if params is not None:
            params_str = f"{params/1e6:.1f}M"
            if baseline_params is not None and baseline_params > 0:
                delta_pct = (params / baseline_params - 1.0) * 100
                sign = "+" if delta_pct >= 0 else ""
                vs_str = f"{sign}{delta_pct:.1f}%"
            else:
                vs_str = "-"
        else:
            params_str = "-"
            vs_str = "-"
        print(f"  {eid:<58s} {r['our_layers']:>3d} {r['max_loops']:>3d} "
              f"{r['Target ffn_deep']:>6d} {r['Target ffn_wide']:>6d} "
              f"{r['FLOP Match']:>7s} {params_str:>9s} {vs_str:>9s}")

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

    def add_common(sp):
        sp.add_argument("--d-model", type=int, default=768)
        sp.add_argument("--dense-ffn-mult", type=int, default=4)
        sp.add_argument("--ffn-round-multiple", type=int, default=64)
        sp.add_argument("--n-head-q", type=int, default=None)
        sp.add_argument("--n-head-kv", type=int, default=None)

    t = sub.add_parser("table", help="Print FLOP tables to stdout (default)")
    add_common(t)
    t.add_argument("--baseline-layers", type=int, nargs="+", default=[60, 36])

    y = sub.add_parser("yaml", help="Generate YAML config files")
    add_common(y)
    y.add_argument("--template", required=True)
    y.add_argument("--output-dir", default="configs")
    y.add_argument("--baseline-layers", type=int, required=True)
    y.add_argument("--our-layers", type=int, required=True)
    y.add_argument("--max-loops", type=int, required=True)
    y.add_argument("--ratios", type=float, nargs="+", required=True)

    pp = sub.add_parser("paper", help="Generate ALL configs for Table 1 + Table 2")
    add_common(pp)
    pp.add_argument("--template", required=True)
    pp.add_argument("--output-dir", default="paper_configs")
    pp.add_argument("--baseline-layers", type=int, required=True)
    pp.add_argument("--our-layers", type=int, required=True)
    pp.add_argument("--max-loops", type=int, required=True)
    pp.add_argument("--vocab-size", type=int, default=50304)
    pp.add_argument("--weight-tying", action="store_true", default=False)
    pp.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="Wide-fraction ratios to sweep (default: 0.3 0.5 0.7)")
    pp.add_argument("--no-max-wide", action="store_true", default=False,
                    help="Skip the max-wide search (most extreme wide-heavy ratio).")

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
            ratios=args.ratios,
            find_max_wide=not args.no_max_wide,
        )


if __name__ == "__main__":
    main()