"""
Adaptive Model Calculator — Iso-FLOP Experimental Grid
======================================================
Reference: 60-layer dense transformer (768d, 12q/12kv, hd=64, ffn=3072, vocab=50304)

Architecture: Single path — each layer loops shared attention + MoE FFN.
  - "dense":    1 iteration, dense FFN (standard transformer)
  - "loop":     N iterations, dense FFN (ACT without MoE)
  - "moe":      1 iteration, MoE FFN (standard MoE, no looping)
  - "loop_moe": N iterations, MoE FFN (the new combined architecture)

Key insight: n_experts scales params without scaling FLOPs (only top_k matters
for FLOPs). So Loop+MoE uses small expert_ffn + many experts to get high
param count at iso-FLOP.

Experimental Design
-------------------
1. MAIN COMPARISON (Table 1 / Figure 1: Loss vs FLOPs)
   Five iso-FLOP budgets, each with: Dense, Loop-Only, MoE-Only, Loop+MoE.

2. EXPERT COUNT ABLATION (Table 2 / Figure 2)
   At 100% FLOP budget, sweep n_experts at fixed loop count.

3. LOOP DEPTH ABLATION (Table 3)
   Fix total FLOPs, vary physical layers vs loop count (dense FFN).

4. LOOP COUNT vs EXPERT SIZE (Table 4)
   Fix n_experts=16, sweep loop count vs expert_ffn at iso-FLOP.

Usage:
  python flops_calculator.py                    # full summary
  python flops_calculator.py <preset_name>      # detail for one config
  python flops_calculator.py --table <N>        # show just Table N
"""

import math, sys, os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Primitives
# =============================================================================

def swiglu_hidden(ffn_nominal, multiple_of=256):
    h = int(2 * ffn_nominal / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)

def swiglu_params(d, ffn_nominal, multiple_of=256):
    return 3 * d * swiglu_hidden(ffn_nominal, multiple_of)

def attn_flops(d, nq, nkv, hd):
    return 4 * d * hd * (nq + nkv)

def attn_params(d, nq, nkv, hd):
    return 2 * d * (nq * hd + nkv * hd)


# =============================================================================
# Baseline: 60L Dense
# =============================================================================

BASELINE = {
    "d": 768, "n_layers": 60, "nq": 12, "nkv": 12, "hd": 64,
    "ffn_hidden": 3072, "vocab": 50304,
}

def dense_baseline():
    b = BASELINE
    A = attn_flops(b["d"], b["nq"], b["nkv"], b["hd"])
    F = 6 * b["d"] * b["ffn_hidden"]
    ap = attn_params(b["d"], b["nq"], b["nkv"], b["hd"])
    fp = swiglu_params(b["d"], b["ffn_hidden"])
    layer_p = ap + fp + 2 * b["d"]
    embed_p = 2 * b["vocab"] * b["d"] + b["d"]
    return {
        "total_flops": b["n_layers"] * (A + F),
        "total_params": b["n_layers"] * layer_p + embed_p,
        "total_ffn_params": b["n_layers"] * fp,
    }


# =============================================================================
# Generic FLOP/Param Calculator
# =============================================================================

def calc(
    mode, n_layers, max_loops, d, vocab,
    nq, nkv, hd,
    ffn_hidden=3072,
    n_experts=1, top_k=1, expert_ffn=0,
):
    bl = dense_baseline()
    A = attn_flops(d, nq, nkv, hd)
    ap = attn_params(d, nq, nkv, hd)
    norm_p = 2 * d
    embed_p = 2 * vocab * d + d

    use_moe = mode in ["moe", "loop_moe"]
    loops = max_loops if mode in ["loop", "loop_moe"] else 1

    if use_moe:
        exp_flops = 6 * d * expert_ffn
        router_flops = 2 * d * n_experts
        ffn_flops = top_k * exp_flops + router_flops
        ffn_p = n_experts * swiglu_params(d, expert_ffn) + d * n_experts
    else:
        ffn_flops = 6 * d * ffn_hidden
        ffn_p = swiglu_params(d, ffn_hidden)

    iter_flops = A + ffn_flops
    layer_flops = loops * iter_flops

    block_p = ap + ffn_p + norm_p

    if loops > 1:
        halting_p = d + 2
        scales_p = max_loops
    else:
        halting_p = 0
        scales_p = 0

    layer_p = block_p + halting_p + scales_p

    total_f = n_layers * layer_flops
    total_p = n_layers * layer_p + embed_p
    total_ffn = n_layers * ffn_p

    attn_flops_per_layer = loops * A
    ffn_flops_per_layer = loops * ffn_flops

    return {
        "total_flops": total_f,
        "total_params": total_p,
        "total_ffn_params": total_ffn,
        "flop_pct": total_f / bl["total_flops"] * 100,
        "param_pct": total_p / bl["total_params"] * 100,
        "ffn_cap_pct": total_ffn / bl["total_ffn_params"] * 100,
        "attn_flops_per_layer": attn_flops_per_layer,
        "ffn_flops_per_layer": ffn_flops_per_layer,
        "layer_flops": layer_flops,
        "loops": loops,
        "bl": bl,
    }


# =============================================================================
# Iso-FLOP solver
# =============================================================================

def solve_expert_ffn(target_flops, mode, n_layers, max_loops, d, vocab, nq, nkv, hd,
                     n_experts, top_k, multiple_of=256):
    """Binary search for expert_ffn that lands within 2% of target_flops."""
    lo, hi = 64, 50000
    for _ in range(40):
        mid = (lo + hi) // 2
        r = calc(mode=mode, n_layers=n_layers, max_loops=max_loops, d=d, vocab=vocab,
                 nq=nq, nkv=nkv, hd=hd, n_experts=n_experts, top_k=top_k, expert_ffn=mid)
        if r["total_flops"] < target_flops:
            lo = mid + 1
        else:
            hi = mid
    return multiple_of * round(hi / multiple_of)


# =============================================================================
# Preset Helper
# =============================================================================

B = dict(vocab=50304, d=768, nq=12, nkv=12, hd=64)

def preset(label, mode, n_layers, max_loops, ffn_hidden=3072,
           n_experts=1, top_k=1, expert_ffn=0):
    return dict(
        label=label, mode=mode,
        n_layers=n_layers, max_loops=max_loops,
        ffn_hidden=ffn_hidden,
        n_experts=n_experts, top_k=top_k, expert_ffn=expert_ffn,
        **B,
    )


# =============================================================================
# EXPERIMENTAL GRID
# =============================================================================

PRESETS = {}

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Scaling Curves — Loss vs FLOPs (Figure 1)
# Four architectures at each scale: Dense, Loop, MoE, Loop+MoE
#
# Loop+MoE uses 16 experts, top_k=2, with expert_ffn solved for iso-FLOP.
# This gives >100% param count while matching FLOPs.
# ─────────────────────────────────────────────────────────────────────────────

# ── Scale 1: Dense 10L = ~0.189G ────────────────────────────────────────────
PRESETS["t1_10L_dense"] = preset(
    "Dense 10L", "dense",
    n_layers=10, max_loops=1, ffn_hidden=3072,
)
PRESETS["t1_10L_loop"] = preset(
    "Loop 5Lx2", "loop",
    n_layers=5, max_loops=2, ffn_hidden=3072,
)
PRESETS["t1_10L_moe_8e"] = preset(
    "MoE 5L 8Et2", "moe",
    n_layers=5, max_loops=1,
    n_experts=8, top_k=2, expert_ffn=3584,
)
PRESETS["t1_10L_moe_16e"] = preset(
    "MoE 5L 16Et2", "moe",
    n_layers=5, max_loops=1,
    n_experts=16, top_k=2, expert_ffn=3584,
)
PRESETS["t1_10L_loopmoe_8e"] = preset(
    "Loop+MoE 5Lx2 8Et2 e=1536", "loop_moe",
    n_layers=5, max_loops=2,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t1_10L_loopmoe_16e"] = preset(
    "Loop+MoE 5Lx2 16Et2 e=1536", "loop_moe",
    n_layers=5, max_loops=2,
    n_experts=16, top_k=2, expert_ffn=1536,
)

# ── Scale 2: Dense 20L = ~0.377G ────────────────────────────────────────────
PRESETS["t1_20L_dense"] = preset(
    "Dense 20L", "dense",
    n_layers=20, max_loops=1, ffn_hidden=3072,
)
PRESETS["t1_20L_loop"] = preset(
    "Loop 4Lx5", "loop",
    n_layers=4, max_loops=5, ffn_hidden=3072,
)
PRESETS["t1_20L_moe_8e"] = preset(
    "MoE 4L 8Et2", "moe",
    n_layers=4, max_loops=1,
    n_experts=8, top_k=2, expert_ffn=9728,
)
PRESETS["t1_20L_moe_16e"] = preset(
    "MoE 4L 16Et2", "moe",
    n_layers=4, max_loops=1,
    n_experts=16, top_k=2, expert_ffn=9728,
)
PRESETS["t1_20L_loopmoe_8e"] = preset(
    "Loop+MoE 4Lx5 8Et2 e=1536", "loop_moe",
    n_layers=4, max_loops=5,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t1_20L_loopmoe_16e"] = preset(
    "Loop+MoE 4Lx5 16Et2 e=1536", "loop_moe",
    n_layers=4, max_loops=5,
    n_experts=16, top_k=2, expert_ffn=1536,
)

# ── Scale 3: Dense 30L = ~0.566G ────────────────────────────────────────────
PRESETS["t1_30L_dense"] = preset(
    "Dense 30L", "dense",
    n_layers=30, max_loops=1, ffn_hidden=3072,
)
PRESETS["t1_30L_loop"] = preset(
    "Loop 5Lx6", "loop",
    n_layers=5, max_loops=6, ffn_hidden=3072,
)
PRESETS["t1_30L_moe_8e"] = preset(
    "MoE 5L 8Et2", "moe",
    n_layers=5, max_loops=1,
    n_experts=8, top_k=2, expert_ffn=11776,
)
PRESETS["t1_30L_moe_16e"] = preset(
    "MoE 5L 16Et2", "moe",
    n_layers=5, max_loops=1,
    n_experts=16, top_k=2, expert_ffn=11776,
)
PRESETS["t1_30L_loopmoe_8e"] = preset(
    "Loop+MoE 5Lx6 8Et2 e=1536", "loop_moe",
    n_layers=5, max_loops=6,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t1_30L_loopmoe_16e"] = preset(
    "Loop+MoE 5Lx6 16Et2 e=1536", "loop_moe",
    n_layers=5, max_loops=6,
    n_experts=16, top_k=2, expert_ffn=1536,
)

# ── Scale 4: Dense 40L = ~0.755G ────────────────────────────────────────────
PRESETS["t1_40L_dense"] = preset(
    "Dense 40L", "dense",
    n_layers=40, max_loops=1, ffn_hidden=3072,
)
PRESETS["t1_40L_loop"] = preset(
    "Loop 4Lx10", "loop",
    n_layers=4, max_loops=10, ffn_hidden=3072,
)
PRESETS["t1_40L_moe_8e"] = preset(
    "MoE 4L 8Et2", "moe",
    n_layers=4, max_loops=1,
    n_experts=8, top_k=2, expert_ffn=19968,
)
PRESETS["t1_40L_moe_16e"] = preset(
    "MoE 4L 16Et2", "moe",
    n_layers=4, max_loops=1,
    n_experts=16, top_k=2, expert_ffn=19968,
)
PRESETS["t1_40L_loopmoe_8e"] = preset(
    "Loop+MoE 4Lx10 8Et2 e=1536", "loop_moe",
    n_layers=4, max_loops=10,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t1_40L_loopmoe_16e"] = preset(
    "Loop+MoE 4Lx10 16Et2 e=1536", "loop_moe",
    n_layers=4, max_loops=10,
    n_experts=16, top_k=2, expert_ffn=1536,
)

# ── Scale 5: Dense 60L = ~1.132G ────────────────────────────────────────────
PRESETS["t1_60L_dense"] = preset(
    "Dense 60L", "dense",
    n_layers=60, max_loops=1, ffn_hidden=3072,
)
PRESETS["t1_60L_loop"] = preset(
    "Loop 12Lx5", "loop",
    n_layers=12, max_loops=5, ffn_hidden=3072,
)
PRESETS["t1_60L_moe_8e"] = preset(
    "MoE 12L 8Et2", "moe",
    n_layers=12, max_loops=1,
    n_experts=8, top_k=2, expert_ffn=9728,
)
PRESETS["t1_60L_moe_16e"] = preset(
    "MoE 12L 16Et2", "moe",
    n_layers=12, max_loops=1,
    n_experts=16, top_k=2, expert_ffn=9728,
)
PRESETS["t1_60L_loopmoe_8e"] = preset(
    "Loop+MoE 12Lx5 8Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t1_60L_loopmoe_16e"] = preset(
    "Loop+MoE 12Lx5 16Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=16, top_k=2, expert_ffn=1536,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Expert Count Ablation at ~100% FLOPs (12Lx5)
# Fix layers=12, loops=5, top_k=2, expert_ffn=1536. Sweep n_experts.
# FLOPs stay constant; params scale with n_experts.
# ─────────────────────────────────────────────────────────────────────────────

PRESETS["t2_4e"] = preset(
    "12Lx5 4Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=4, top_k=2, expert_ffn=1536,
)
PRESETS["t2_8e"] = preset(
    "12Lx5 8Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=8, top_k=2, expert_ffn=1536,
)
PRESETS["t2_16e"] = preset(
    "12Lx5 16Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=16, top_k=2, expert_ffn=1536,
)
PRESETS["t2_32e"] = preset(
    "12Lx5 32Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=32, top_k=2, expert_ffn=1536,
)
PRESETS["t2_64e"] = preset(
    "12Lx5 64Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=64, top_k=2, expert_ffn=1536,
)

# top_k=1 at same FLOP budget (expert_ffn doubled to compensate)
PRESETS["t2_8e_t1"] = preset(
    "12Lx5 8Et1 e=3072", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=8, top_k=1, expert_ffn=3072,
)
PRESETS["t2_16e_t1"] = preset(
    "12Lx5 16Et1 e=3072", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=16, top_k=1, expert_ffn=3072,
)
PRESETS["t2_32e_t1"] = preset(
    "12Lx5 32Et1 e=3072", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=32, top_k=1, expert_ffn=3072,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Loop Depth Ablation — Physical Layers vs Loop Count at ~1.13G
# Dense FFN only (isolates the looping effect)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS["t3_depth_12Lx5"] = preset(
    "12L x5 (eff=60)", "loop",
    n_layers=12, max_loops=5, ffn_hidden=3072,
)
PRESETS["t3_depth_20Lx3"] = preset(
    "20L x3 (eff=60)", "loop",
    n_layers=20, max_loops=3, ffn_hidden=3072,
)
PRESETS["t3_depth_30Lx2"] = preset(
    "30L x2 (eff=60)", "loop",
    n_layers=30, max_loops=2, ffn_hidden=3072,
)
PRESETS["t3_depth_60Lx1"] = preset(
    "60L x1 (dense baseline)", "dense",
    n_layers=60, max_loops=1, ffn_hidden=3072,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: Loop Count vs Expert Size at ~100% FLOPs (16 experts, top_k=2)
# More loops → smaller expert_ffn to stay iso-FLOP
# ─────────────────────────────────────────────────────────────────────────────

PRESETS["t4_12Lx2_16e"] = preset(
    "12Lx2  16Et2 e=4608", "loop_moe",
    n_layers=12, max_loops=2,
    n_experts=16, top_k=2, expert_ffn=4608,
)
PRESETS["t4_12Lx3_16e"] = preset(
    "12Lx3  16Et2 e=2816", "loop_moe",
    n_layers=12, max_loops=3,
    n_experts=16, top_k=2, expert_ffn=2816,
)
PRESETS["t4_12Lx5_16e"] = preset(
    "12Lx5  16Et2 e=1536", "loop_moe",
    n_layers=12, max_loops=5,
    n_experts=16, top_k=2, expert_ffn=1536,
)
PRESETS["t4_12Lx10_16e"] = preset(
    "12Lx10 16Et2 e=768", "loop_moe",
    n_layers=12, max_loops=10,
    n_experts=16, top_k=2, expert_ffn=768,
)


# =============================================================================
# YAML Output
# =============================================================================

def make_experiment_id(name, p, r):
    mode = p['mode']
    parts = [mode, f"{p['n_layers']}L"]

    if p['max_loops'] > 1:
        parts[-1] += f"x{p['max_loops']}"

    if mode in ["dense", "loop"]:
        parts.append(f"ffn{p['ffn_hidden']}")

    if mode in ["moe", "loop_moe"]:
        parts.append(f"{p['n_experts']}Et{p['top_k']}_e{p['expert_ffn']}")

    parts.append(f"{r['flop_pct']:.0f}pct")
    return "_".join(parts)


def update_model_config(d, p, experiment_id=None):
    if isinstance(d, dict):
        if 'n_layer' in d:
            d['n_layer'] = p['n_layers']

        if 'ffn_hidden' in d:
            if p['mode'] in ['moe', 'loop_moe']:
                d['ffn_hidden'] = p.get('expert_ffn', 512) if p.get('expert_ffn', 0) > 0 else 512
            else:
                d['ffn_hidden'] = p['ffn_hidden']

        if 'adaptive_config' in d:
            ac = d['adaptive_config']
            ac['max_loops'] = p['max_loops'] if p['max_loops'] > 1 else 1

            if p['mode'] in ['moe', 'loop_moe']:
                ac['n_experts'] = p['n_experts']
                ac['top_k'] = p['top_k']
                ac['expert_ffn_hidden'] = p['expert_ffn']
            else:
                ac['n_experts'] = 1
                ac['top_k'] = 1
                ac['expert_ffn_hidden'] = p['ffn_hidden']

        if experiment_id and 'experiment_id' in d:
            d['experiment_id'] = experiment_id

        for k, v in d.items():
            update_model_config(v, p, experiment_id)
    elif isinstance(d, list):
        for item in d:
            update_model_config(item, p, experiment_id)


def write_yaml(name, p, r):
    if not HAS_YAML:
        return
    template_path = "loop_moe_template.yaml"
    if not os.path.exists(template_path):
        return
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)
    experiment_id = make_experiment_id(name, p, r)
    update_model_config(config, p, experiment_id)
    os.makedirs("generated_configs", exist_ok=True)
    out_path = f"generated_configs/{name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


# =============================================================================
# Printing
# =============================================================================

def print_detail(name, p, r):
    print(f"\n{'='*72}")
    print(f"  {p['label']}   [Mode: {p['mode']}]")
    print(f"{'='*72}")
    print(f"  Layers: {p['n_layers']}, Loops: {r['loops']}")
    if p['mode'] in ['dense', 'loop']:
        print(f"  Dense FFN: {p['ffn_hidden']}")
    if p['mode'] in ['moe', 'loop_moe']:
        print(f"  MoE: {p['n_experts']}E top-{p['top_k']} expert_ffn={p['expert_ffn']}")
    print()
    print(f"  Attn FLOPs/layer:  {r['attn_flops_per_layer']/1e6:>8.1f}M  ({r['loops']} iters)")
    print(f"  FFN FLOPs/layer:   {r['ffn_flops_per_layer']/1e6:>8.1f}M  ({r['loops']} iters)")
    attn_frac = r['attn_flops_per_layer'] / r['layer_flops'] * 100 if r['layer_flops'] > 0 else 0
    print(f"  Attn/Total:        {attn_frac:.1f}%")
    print()
    print(f"  Total FLOPs:       {r['total_flops']/1e9:.3f}G  ({r['flop_pct']:.1f}% of baseline)")
    print(f"  Total Params:      {r['total_params']/1e6:.1f}M  ({r['param_pct']:.1f}%)")
    print(f"  FFN Capacity:      {r['total_ffn_params']/1e6:.1f}M  ({r['ffn_cap_pct']:.1f}%)")
    write_yaml(name, p, r)


def print_table(title, names, show_attn_frac=False):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")

    hdr = f"  {'Name':>40} │ {'FLOPs':>8} {'F%':>6} │ {'Params':>8} {'P%':>7} │ {'FFN_P':>8} {'FC%':>6} │ {'Loops':>5}"
    if show_attn_frac:
        hdr += f" │ {'Attn%':>6}"
    print(hdr)

    sep = f"  {'─'*40}─┼─{'─'*15}─┼─{'─'*16}─┼─{'─'*15}─┼─{'─'*5}"
    if show_attn_frac:
        sep += f"─┼─{'─'*6}"
    print(sep)

    for name in names:
        p = PRESETS[name]
        r = calc(**{k: v for k, v in p.items() if k != "label"})

        attn_frac = r['attn_flops_per_layer'] / r['layer_flops'] * 100 if r['layer_flops'] > 0 else 0

        row = (f"  {p['label']:>40} │ {r['total_flops']/1e9:>7.3f}G {r['flop_pct']:>5.1f}% │ "
               f"{r['total_params']/1e6:>7.1f}M {r['param_pct']:>6.1f}% │ "
               f"{r['total_ffn_params']/1e6:>7.1f}M {r['ffn_cap_pct']:>5.1f}% │ {r['loops']:>5}")

        if show_attn_frac:
            row += f" │ {attn_frac:>5.1f}%"

        print(row)
        write_yaml(name, p, r)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    bl = dense_baseline()
    print(f"Baseline (60L dense): {bl['total_flops']/1e9:.3f}G FLOPs, "
          f"{bl['total_params']/1e6:.1f}M params, {bl['total_ffn_params']/1e6:.1f}M FFN params\n")

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg and arg in PRESETS:
        p = PRESETS[arg]
        r = calc(**{k: v for k, v in p.items() if k != "label"})
        print_detail(arg, p, r)
        sys.exit(0)

    table_filter = None
    if arg and arg.startswith("--table"):
        if len(sys.argv) > 2:
            table_filter = sys.argv[2]
        else:
            table_filter = arg.replace("--table", "").strip("= ")

    # ── Table 1: Scaling curves ──
    if table_filter is None or table_filter == "1":
        for scale in ["10L", "20L", "30L", "40L", "60L"]:
            names = [f"t1_{scale}_{arch}" for arch in [
                "dense", "loop", "moe_8e", "moe_16e", "loopmoe_8e", "loopmoe_16e",
            ]]
            names = [n for n in names if n in PRESETS]
            print_table(f"TABLE 1 — Scale: Dense {scale} equivalent", names, show_attn_frac=True)

    # ── Table 2: Expert count ablation ──
    if table_filter is None or table_filter == "2":
        names_t2 = sorted([k for k in PRESETS if k.startswith("t2_") and "_t1" not in k])
        names_t1 = sorted([k for k in PRESETS if k.startswith("t2_") and "_t1" in k])
        print_table("TABLE 2a — Expert Count Ablation at ~100% FLOPs (12Lx5, top_k=2)", names_t2, show_attn_frac=True)
        if names_t1:
            print_table("TABLE 2b — Expert Count Ablation (top_k=1)", names_t1, show_attn_frac=True)

    # ── Table 3: Loop depth ablation (dense FFN) ──
    if table_filter is None or table_filter == "3":
        names = [k for k in PRESETS if k.startswith("t3_")]
        print_table("TABLE 3 — Loop Depth Ablation (Dense FFN)", names, show_attn_frac=False)

    # ── Table 4: Loop count vs expert size ──
    if table_filter is None or table_filter == "4":
        names = sorted([k for k in PRESETS if k.startswith("t4_")])
        print_table("TABLE 4 — Loop Count vs Expert Size (16Et2, ~100% FLOPs)", names, show_attn_frac=True)