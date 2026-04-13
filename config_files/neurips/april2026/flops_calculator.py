#!/usr/bin/env python3
"""
Iso-FLOP calculator for asymmetric Dual-Path (Wide + Deep loop) transformer.

UPDATE: DualPathGate now has two cross-projections (proj_w2d, proj_d2w),
each a Linear(d, d). When use_cross=True they add 4*d^2 FLOPs/token/layer
to the gate cost. Only the DUAL configs subtract this from the FFN budget;
dense iso-FLOP / pure-loop / pure-wide configs are unchanged (they don't
instantiate a DualPathGate).
"""

import argparse
import math
import os
import re


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
    use_cross: bool = False,
):
    assert 0.0 <= capacity_ratio <= 1.0

    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    attn_flops = 4 * (d_model ** 2) + 4 * (d_model ** 2) // n_rep

    dual_gate_flops = 2 * (3 * d_model) * 2
    # Cross projections: proj_w2d + proj_d2w, each Linear(d, d) -> 2*d^2 FLOPs each
    cross_proj_flops = (4 * d_model * d_model) if use_cross else 0
    router_flops_total = max_loops * 2 * (d_model + 1)
    gate_flops = dual_gate_flops + cross_proj_flops + router_flops_total

    dense_ffn = d_model * dense_ffn_mult
    dense_block_flops = attn_flops + (6 * d_model * dense_ffn)
    total_budget_flops = target_dense_layers * dense_block_flops

    layer_budget = total_budget_flops / our_layers
    paths_budget = layer_budget - gate_flops

    if capacity_ratio > 0:
        wide_budget = paths_budget * capacity_ratio
        ffn_wide_exact = (wide_budget - attn_flops) / (6 * d_model)
        if ffn_wide_exact < 0:
            return {"error": f"Wide budget too small ({wide_budget:.0f} < {attn_flops})"}
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
            return {"error": f"Deep path starved (need {max_loops*attn_flops:.0f}, have {deep_budget:.0f})"}
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
        "use_cross": use_cross,
    }


# =====================================================================
# Parameter counting
# =====================================================================

def _attn_params(d_model, n_rep=1):
    return 2 * d_model * d_model + 2 * (d_model * d_model) // n_rep


def _swiglu_params(d_model, ffn_hidden):
    return 3 * d_model * ffn_hidden


def _block_params(d_model, ffn_hidden, n_rep=1):
    return _attn_params(d_model, n_rep) + _swiglu_params(d_model, ffn_hidden) + 2 * d_model


def count_dual_layer_params(d_model, ffn_deep, ffn_wide, max_loops,
                            layer_type="dual", n_rep=1, use_cross=False):
    params = 0
    if layer_type in ("loop", "dual"):
        params += _block_params(d_model, ffn_deep, n_rep)
        params += (d_model + 1) + 1
        params += max_loops
    if layer_type in ("wide", "dual"):
        params += _block_params(d_model, ffn_wide, n_rep)
        params += 1
    if layer_type == "dual":
        params += 3 * d_model * 2 + 2   # DualPathGate.gate_proj
        if use_cross:
            params += 2 * d_model * d_model   # proj_w2d + proj_d2w
            params += 2                        # cross_scale_wide + cross_scale_deep
    return params


def count_model_params(d_model, n_layers, max_loops, ffn_deep, ffn_wide,
                       vocab_size=50304, use_weight_tying=False,
                       layer_type="dual", n_rep=1, use_cross=False):
    per_layer = count_dual_layer_params(d_model, ffn_deep, ffn_wide,
                                        max_loops, layer_type, n_rep, use_cross)
    shared = vocab_size * d_model
    if not use_weight_tying:
        shared += d_model * vocab_size
    shared += d_model
    return n_layers * per_layer + shared


def count_dense_params(d_model, n_layers, ffn_hidden,
                       vocab_size=50304, use_weight_tying=False, n_rep=1):
    per_layer = _block_params(d_model, ffn_hidden, n_rep)
    shared = vocab_size * d_model + d_model
    if not use_weight_tying:
        shared += d_model * vocab_size
    return n_layers * per_layer + shared


# =====================================================================
# YAML generation
# =====================================================================

def _make_experiment_id(res):
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


def _patch_yaml(template_text, res):
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
    if "_use_cross" in res:
        patches[("model_raw", "use_cross")] = str(res["_use_cross"]).lower()
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
                    line = f"{indent}{key}{sep}{patches[patch_key]}\n"

        out_lines.append(line)

        if (layer_types is not None and current_section == "model_raw"
                and m and m.group(2) == "deep_gate_init_bias"):
            indent = m.group(1)
            lt_str = ", ".join(f'"{t}"' for t in layer_types)
            out_lines.append(f"{indent}layer_types: [{lt_str}]\n")

    return ''.join(out_lines)


def write_yaml(template_path, output_path, res):
    with open(template_path, "r") as f:
        text = f.read()
    patched = _patch_yaml(text, res)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(patched)
    return _make_experiment_id(res)


# =====================================================================
# Paper config generator
# =====================================================================

def _write_and_log(template_path, output_dir, res, tag, params=None, baseline_params=None):
    exp_id = _make_experiment_id(res)
    out_path = os.path.join(output_dir, f"{exp_id}.yaml")
    write_yaml(template_path, out_path, res)
    if params is not None:
        res["_params"] = params
    param_str = ""
    if params is not None:
        param_str = f"  params={params/1e6:.1f}M"
        if baseline_params:
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
                         use_cross, step=0.01, start=0.99):
    r = start
    while r > 0.0:
        res = asymmetric_dual_config(
            d_model=d_model, target_dense_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, capacity_ratio=r,
            ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv,
            use_cross=use_cross,
        )
        if "error" not in res:
            return r, res
        r = round(r - step, 4)
    return None, None


def generate_paper_configs(
    template_path, output_dir, baseline_layers, our_layers, max_loops,
    d_model=768, dense_ffn_mult=4, ffn_round_multiple=64,
    vocab_size=50304, use_weight_tying=False,
    n_head_q=None, n_head_kv=None,
    ratios=None, find_max_wide=True, use_cross=True,
):
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    dense_ffn = d_model * dense_ffn_mult
    if ratios is None:
        ratios = [0.3, 0.5, 0.7]

    def _add_size_info(res, include_cross=False):
        res["_d_model"] = d_model
        res["_n_head_q"] = n_head_q
        res["_n_head_kv"] = n_head_kv
        if include_cross:
            res["_use_cross"] = use_cross
        return res

    tag = f"d{d_model}"
    all_results = []

    print(f"\n{'='*70}")
    print(f"  PAPER CONFIGS   d_model={d_model}  heads q={n_head_q}/kv={n_head_kv}")
    print(f"  Baseline: {baseline_layers}L dense, ffn={dense_ffn}")
    print(f"  Dual:     {our_layers}L × {max_loops} loops   use_cross={use_cross}")
    print(f"  Ratios:   {ratios}   max-wide search: {find_max_wide}")
    if use_cross:
        extra = 4 * d_model * d_model
        print(f"  [Cross projections add {extra:,} FLOPs/token/dual-layer ({extra/1e6:.2f}M)]")
    print(f"{'='*70}\n")

    baseline_params = None

    # --- 1. Dense iso-FLOP (use_cross=False: no DualPathGate) ---
    dense_iso = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=baseline_layers,
        max_loops=1, capacity_ratio=0.0, ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv, use_cross=False,
    )
    if "error" not in dense_iso:
        if dense_iso["Target ffn_wide"] == 0:
            dense_iso["Target ffn_wide"] = 64
        dense_iso["_layer_types"] = ["loop"] * baseline_layers
        dense_iso["_exp_id"] = f"{tag}_dense_isoflop_{baseline_layers}L"
        _add_size_info(dense_iso, include_cross=False)
        dense_params = count_dense_params(d_model, baseline_layers, dense_ffn,
                                          vocab_size, use_weight_tying, n_rep)
        baseline_params = dense_params
        all_results.append(_write_and_log(template_path, output_dir,
                                          dense_iso, "Dense iso-FLOP", dense_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping dense iso-FLOP: {dense_iso['error']}")

    # --- 2. Pure loop (use_cross=False: no DualPathGate) ---
    pure_loop = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=max_loops, capacity_ratio=0.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv, use_cross=False,
    )
    if "error" not in pure_loop:
        if pure_loop["Target ffn_wide"] == 0:
            pure_loop["Target ffn_wide"] = 64
        pure_loop["_layer_types"] = ["loop"] * our_layers
        pure_loop["_exp_id"] = (f"{tag}_pure_loop{max_loops}_"
                                f"{pure_loop['Target ffn_deep']}deep_{our_layers}L")
        _add_size_info(pure_loop, include_cross=False)
        loop_params = count_model_params(
            d_model, our_layers, max_loops,
            pure_loop["Target ffn_deep"], 0,
            vocab_size, use_weight_tying, "loop", n_rep, use_cross=False,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_loop, "Pure loop", loop_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping pure loop: {pure_loop['error']}")

    # --- 3. Pure wide (use_cross=False: no DualPathGate) ---
    pure_wide = asymmetric_dual_config(
        d_model=d_model, target_dense_layers=baseline_layers,
        dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
        max_loops=1, capacity_ratio=1.0,
        ffn_round_multiple=ffn_round_multiple,
        n_head_q=n_head_q, n_head_kv=n_head_kv, use_cross=False,
    )
    if "error" not in pure_wide:
        if pure_wide["Target ffn_deep"] == 0:
            pure_wide["Target ffn_deep"] = 64
        pure_wide["_layer_types"] = ["wide"] * our_layers
        pure_wide["_exp_id"] = (f"{tag}_pure_wide_"
                                f"{pure_wide['Target ffn_wide']}wide_{our_layers}L")
        _add_size_info(pure_wide, include_cross=False)
        wide_params = count_model_params(
            d_model, our_layers, 1, 0, pure_wide["Target ffn_wide"],
            vocab_size, use_weight_tying, "wide", n_rep, use_cross=False,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          pure_wide, "Pure wide", wide_params,
                                          baseline_params=baseline_params))
    else:
        print(f"  ⚠  Skipping pure wide: {pure_wide['error']}")

    # --- 4. Ratio sweep (DUAL - uses cross projections) ---
    for r in ratios:
        res = asymmetric_dual_config(
            d_model=d_model, target_dense_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, capacity_ratio=r,
            ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv, use_cross=use_cross,
        )
        if "error" in res:
            print(f"  ⚠  Skipping ratio {r:.0%}: {res['error']}")
            continue
        wp = int(round(r * 100))
        dp = 100 - wp
        res["_exp_id"] = (f"{tag}_dual_r{wp:02d}w{dp:02d}d_"
                          f"{res['Target ffn_deep']}deep_{res['Target ffn_wide']}wide_"
                          f"{our_layers}L_loop{max_loops}")
        _add_size_info(res, include_cross=True)
        params = count_model_params(
            d_model, our_layers, max_loops,
            res["Target ffn_deep"], res["Target ffn_wide"],
            vocab_size, use_weight_tying, "dual", n_rep, use_cross=use_cross,
        )
        all_results.append(_write_and_log(template_path, output_dir,
                                          res, f"Ratio {wp}w/{dp}d", params,
                                          baseline_params=baseline_params))

    # --- 5. Max-wide search (DUAL - uses cross projections) ---
    if find_max_wide:
        best_r, best_res = _find_max_wide_ratio(
            d_model=d_model, baseline_layers=baseline_layers,
            dense_ffn_mult=dense_ffn_mult, our_layers=our_layers,
            max_loops=max_loops, ffn_round_multiple=ffn_round_multiple,
            n_head_q=n_head_q, n_head_kv=n_head_kv, use_cross=use_cross,
        )
        if best_res is not None:
            already = any(abs(best_r - rr) < 0.005 for rr in ratios)
            if already:
                print(f"\n  Max-wide ratio {best_r:.2f} already in --ratios, skipping.")
            else:
                wp = int(round(best_r * 100))
                dp = 100 - wp
                best_res["_exp_id"] = (
                    f"{tag}_dual_MAXWIDE_r{wp:02d}w{dp:02d}d_"
                    f"{best_res['Target ffn_deep']}deep_"
                    f"{best_res['Target ffn_wide']}wide_"
                    f"{our_layers}L_loop{max_loops}"
                )
                _add_size_info(best_res, include_cross=True)
                params = count_model_params(
                    d_model, our_layers, max_loops,
                    best_res["Target ffn_deep"], best_res["Target ffn_wide"],
                    vocab_size, use_weight_tying, "dual", n_rep, use_cross=use_cross,
                )
                print(f"\n  Max-wide: highest feasible ratio = {best_r:.2f}  ({wp}w/{dp}d)")
                all_results.append(_write_and_log(template_path, output_dir,
                                                  best_res, f"MAX-WIDE {wp}w/{dp}d",
                                                  params, baseline_params=baseline_params))
        else:
            print("\n  ⚠  Max-wide search found no feasible ratio")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY:  {len(all_results)} configs written to {output_dir}/")
    if baseline_params:
        print(f"  Baseline (dense iso-FLOP) = {baseline_params/1e6:.1f}M params")
    print(f"{'='*70}\n")

    print(f"  {'Experiment ID':<62s} {'L':>3s} {'Lp':>3s} "
          f"{'FFN_d':>6s} {'FFN_w':>6s} {'Match':>7s} "
          f"{'Params':>9s} {'vs base':>9s}")
    print(f"  {'-'*62} {'-'*3} {'-'*3} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*9}")
    for _, eid, r in all_results:
        params = r.get("_params")
        params_str = f"{params/1e6:.1f}M" if params else "-"
        if params and baseline_params:
            delta_pct = (params / baseline_params - 1.0) * 100
            sign = "+" if delta_pct >= 0 else ""
            vs_str = f"{sign}{delta_pct:.1f}%"
        else:
            vs_str = "-"
        print(f"  {eid:<62s} {r['our_layers']:>3d} {r['max_loops']:>3d} "
              f"{r['Target ffn_deep']:>6d} {r['Target ffn_wide']:>6d} "
              f"{r['FLOP Match']:>7s} {params_str:>9s} {vs_str:>9s}")

    return all_results


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Iso-FLOP calc for Dual-Path transformer.")
    sub = p.add_subparsers(dest="command")

    def add_common(sp):
        sp.add_argument("--d-model", type=int, default=768)
        sp.add_argument("--dense-ffn-mult", type=int, default=4)
        sp.add_argument("--ffn-round-multiple", type=int, default=64)
        sp.add_argument("--n-head-q", type=int, default=None)
        sp.add_argument("--n-head-kv", type=int, default=None)

    pp = sub.add_parser("paper", help="Generate full experiment grid")
    add_common(pp)
    pp.add_argument("--template", required=True)
    pp.add_argument("--output-dir", default="paper_configs")
    pp.add_argument("--baseline-layers", type=int, required=True)
    pp.add_argument("--our-layers", type=int, required=True)
    pp.add_argument("--max-loops", type=int, required=True)
    pp.add_argument("--vocab-size", type=int, default=50304)
    pp.add_argument("--weight-tying", action="store_true", default=False)
    pp.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    pp.add_argument("--no-max-wide", action="store_true", default=False)
    pp.add_argument("--no-cross", action="store_true", default=False,
                    help="Disable cross-projection FLOP accounting for dual configs.")

    return p


def main():
    args = build_parser().parse_args()

    if args.command == "paper":
        generate_paper_configs(
            template_path=args.template, output_dir=args.output_dir,
            baseline_layers=args.baseline_layers, our_layers=args.our_layers,
            max_loops=args.max_loops,
            d_model=args.d_model, dense_ffn_mult=args.dense_ffn_mult,
            ffn_round_multiple=args.ffn_round_multiple,
            vocab_size=args.vocab_size, use_weight_tying=args.weight_tying,
            n_head_q=args.n_head_q, n_head_kv=args.n_head_kv,
            ratios=args.ratios, find_max_wide=not args.no_max_wide,
            use_cross=not args.no_cross,
        )
    else:
        print("Use: python flops_calculator.py paper --help")


if __name__ == "__main__":
    main()