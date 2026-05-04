#!/usr/bin/env python3
"""
Param-matched baseline generator for asymmetric Dual-Path (Wide + Deep loop)
transformers.

Framing:
  - Dual model: you specify `ffn_deep`; `ffn_wide` is derived so the wide path
    FLOP-matches the deep path per layer (within the layer, not against any
    external budget).
  - Baselines: pure-loop and pure-wide variants, each param-matched to the
    dual. Same `n_layers` and `d_model`; FFN size solved in closed form.

This is the right comparison for an architectural claim ("at matched
capacity, does the mixture-of-shapes design help"). Dual will cost roughly
2x the FLOPs of either baseline per layer — this is expected and reported
openly. Reviewers care that you're honest about this.
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
# Solvers
# =====================================================================

def solve_ffn_wide_flop_match(d_model: int, ffn_deep: int, max_loops: int,
                              wide_flop_ratio: float = 1.0,
                              ffn_round_multiple: int = 64,
                              n_rep: int = 1) -> tuple[int, int, int]:
    """
    Solve for ffn_wide such that
        wide_per_layer_flops = wide_flop_ratio * deep_per_layer_flops

    Returns (ffn_wide_rounded, deep_flops, wide_flops_at_rounded).

    Derivation:
        attn + 6d * ffn_wide = ratio * max_loops * (attn + 6d * ffn_deep)
        ffn_wide = [ratio * max_loops * (attn + 6d*ffn_deep) - attn] / (6d)
                 = ratio * max_loops * ffn_deep
                   + (ratio * max_loops - 1) * attn / (6d)

    The second term is the "attention correction": deep runs attn max_loops
    times, wide once, so wide needs extra FFN width to match.
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
    """
    Solve for the FFN size of a single-path baseline such that its total
    parameter count matches `target_params`. Returns (ffn_rounded, actual_params).

    For both 'loop' and 'wide' layer types, per-layer params are linear in
    ffn_hidden (through the 3d*ffn SwiGLU term). Closed form:

        target = n_layers * (attn_params + 3*d*ffn + 2d + extras) + shared
        ffn    = [(target - shared) / n_layers - attn - 2d - extras] / (3d)
    """
    shared = shared_params(d_model, vocab_size, use_weight_tying)
    attn_p = attn_params(d_model, n_rep)
    two_norms = 2 * d_model

    if layer_type == "loop":
        extras = (d_model + 1) + 1 + max_loops   # router + loop_scales
    elif layer_type == "wide":
        extras = 1                               # wide_scale
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
# Top-level config builders
# =====================================================================

def build_dual(d_model: int, n_layers: int, max_loops: int, ffn_deep: int,
               vocab_size: int = 50304, use_weight_tying: bool = False,
               n_head_q: int = None, n_head_kv: int = None,
               wide_flop_ratio: float = 1.0,
               ffn_round_multiple: int = 64) -> dict:
    """Build one dual model config."""
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
    """
    Mixed sandwich: some layers pure-loop, some dual, some pure-wide.
    Default: ~1/3 each. Reuses the dual's ffn_deep and ffn_wide directly
    (no re-solving — the layer-internal FLOP match is already satisfied).
    Param count will differ from dual's by (n_loop_layers + n_wide_layers)
    times the per-layer swap delta; reported honestly.
    """
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
# Experiment IDs — sortable, parseable, consistent
# =====================================================================

def make_exp_id(cfg: dict) -> str:
    dm = cfg["d_model"]
    L = cfg["n_layers"]

    if cfg["kind"] == "dual":
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}"
                f"_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}_dual")
    if cfg["kind"] == "pure_loop":
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}"
                f"_ffnL{cfg['ffn_loop']}_pureloop_ipMdual")
    if cfg["kind"] == "pure_wide":
        return (f"dm{dm}_L{L}_loop1_ffnW{cfg['ffn_wide']}_purewide_ipMdual")
    if cfg["kind"] == "sandwich":
        return (f"dm{dm}_L{L}_loop{cfg['max_loops']}"
                f"_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}"
                f"_sandwich_L{cfg['n_loop_layers']}D{cfg['n_dual_layers']}W{cfg['n_wide_layers']}")
    raise ValueError(f"Unknown kind: {cfg['kind']}")


# =====================================================================
# YAML emission
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
        patches[("model_raw", "ffn_hidden")] = "64"   # unused; keep valid
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
    layer_types = cfg.get("layer_types")  # only present for sandwich

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

        # Inject layer_types for sandwich, anchored after a known key.
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
    """Print a single dual + its baselines in a consistent, reviewer-readable table."""
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


# =====================================================================
# CLI entry points
# =====================================================================

def cmd_single(args):
    """Emit one dual + its baselines."""
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
    """Emit one dual + baselines per ffn_deep value."""
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

    # Global summary table for cross-config comparison.
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


def build_parser():
    p = argparse.ArgumentParser(
        description="Param-matched baseline generator for Dual-Path (Wide+Deep) transformer.",
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
        sp.add_argument("--wide-flop-ratio", type=float, default=1.0,
                        help="wide path FLOPs / deep path FLOPs; 1.0 = matched")
        sp.add_argument("--ffn-round-multiple", type=int, default=64)
        sp.add_argument("--template", type=str, default=None,
                        help="YAML template path. If omitted, only print summary.")
        sp.add_argument("--output-dir", type=str, default="configs")
        sp.add_argument("--sandwich", action="store_true", default=False,
                        help="Also emit a mixed-sandwich variant.")
        sp.add_argument("--sandwich-n-loop", type=int, default=None)
        sp.add_argument("--sandwich-n-wide", type=int, default=None)

    single = sub.add_parser("single", help="One dual + param-matched baselines.")
    add_common(single)
    single.add_argument("--ffn-deep", type=int, required=True,
                        help="Deep-path FFN hidden size.")
    single.set_defaults(func=cmd_single)

    sweep = sub.add_parser("sweep", help="Sweep ffn_deep, one dual + baselines per value.")
    add_common(sweep)
    sweep.add_argument("--ffn-deep", type=int, nargs="+", required=True,
                       help="List of deep-path FFN hidden sizes to sweep.")
    sweep.set_defaults(func=cmd_sweep)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()