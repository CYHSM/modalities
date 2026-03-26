"""
Adaptive Model Calculator — The Ultimate Iso-FLOP/Param Sweeper & YAML Generator
"""

import math, sys, os, re, argparse
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =============================================================================
# Primitives (Locked to 64-dim for Tensor Cores)
# =============================================================================

def swiglu_hidden(ffn_nominal, multiple_of=64):
    if ffn_nominal <= 0: return 0
    h = int(2 * ffn_nominal / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)

def swiglu_params(d, ffn_nominal, multiple_of=64):
    if ffn_nominal <= 0: return 0
    return 3 * d * swiglu_hidden(ffn_nominal, multiple_of)

def attn_flops(d, nq, nkv, hd):
    return 4 * d * hd * (nq + nkv)

def attn_params(d, nq, nkv, hd):
    return 2 * d * (nq * hd + nkv * hd)

# =============================================================================
# Baseline & Calculator
# =============================================================================

# Standard MHA Baseline
BASELINE = {"d": 768, "n_layers": 60, "nq": 12, "nkv": 12, "hd": 64, "ffn_hidden": 3072, "vocab": 50304}

def dense_baseline():
    b = BASELINE
    A = attn_flops(b["d"], b["nq"], b["nkv"], b["hd"])
    F = 6 * b["d"] * b["ffn_hidden"]
    ap = attn_params(b["d"], b["nq"], b["nkv"], b["hd"])
    fp = swiglu_params(b["d"], b["ffn_hidden"], multiple_of=64)
    layer_p = ap + fp + 2 * b["d"]
    return {
        "total_flops": b["n_layers"] * (A + F),
        "total_params": b["n_layers"] * layer_p + (2 * b["vocab"] * b["d"] + b["d"]),
        "total_ffn_params": b["n_layers"] * fp,
    }

def calc(n_layers, max_loops, d, vocab, nq, nkv, hd, ffn_deep, n_experts, top_k, expert_ffn):
    bl = dense_baseline()
    A = attn_flops(d, nq, nkv, hd)
    ap = attn_params(d, nq, nkv, hd)
    norm_p = 2 * d + 2 * hd
    embed_p = 2 * vocab * d + d

    # Deep Path
    deep_ffn_f = 6 * d * ffn_deep if ffn_deep > 0 else 0
    deep_ffn_p = swiglu_params(d, ffn_deep, multiple_of=64)
    deep_layer_f = max_loops * (A + deep_ffn_f)
    deep_block_p = ap + deep_ffn_p + norm_p

    # Wide Path (MoE)
    exp_f = 6 * d * expert_ffn if expert_ffn > 0 else 0
    wide_layer_f = A + top_k * exp_f + (2 * d * n_experts)
    wide_ffn_p = n_experts * swiglu_params(d, expert_ffn, multiple_of=64)
    wide_block_p = ap + wide_ffn_p + d * n_experts + norm_p

    # Dual Path Overhead
    gate_p = 2 * d * d + d * 2 + 4
    gate_f = 4 * d * d
    halting_p = d + 2
    scales_p = max_loops + 1

    total_f = n_layers * (deep_layer_f + wide_layer_f + gate_f)
    total_p = n_layers * (deep_block_p + wide_block_p + gate_p + halting_p + scales_p) + embed_p
    total_ffn = n_layers * (deep_ffn_p + wide_ffn_p)

    return {
        "total_flops": total_f,
        "total_params": total_p,
        "total_ffn_params": total_ffn,
        "flop_pct": total_f / bl["total_flops"] * 100,
        "param_pct": total_p / bl["total_params"] * 100,
        "ffn_cap_pct": total_ffn / bl["total_ffn_params"] * 100,
        "deep_layer_f": deep_layer_f,
        "wide_layer_f": wide_layer_f,
        "bl": bl,
    }

# =============================================================================
# Auto-Solver
# =============================================================================

def find_exact_iso_config(n_layers, max_loops, n_experts, top_k, param_budget=1.0, tolerance=0.8):
    bl = dense_baseline()
    d, nq, nkv, hd, vocab = BASELINE["d"], BASELINE["nq"], BASELINE["nkv"], BASELINE["hd"], BASELINE["vocab"]
    
    A = attn_flops(d, nq, nkv, hd)
    ap = attn_params(d, nq, nkv, hd)
    
    gate_f = 4 * d * d
    gate_p = 2 * d * d + d * 2 + 4
    norm_p = 2 * d + 2 * hd
    embed_p = 2 * vocab * d + d
    halting_p = d + 2
    scales_p = max_loops + 1
    router_p = d * n_experts
    router_f = 2 * d * n_experts
    
    target_f_per_layer = (bl["total_flops"] / n_layers) - (max_loops * A) - A - gate_f - router_f
    target_p_per_layer = ((bl["total_params"] * param_budget - embed_p) / n_layers) - (2*ap) - (2*norm_p) - gate_p - halting_p - scales_p - router_p

    coeff_matrix = np.array([[6 * d * max_loops, 6 * d * top_k], [2 * d, 2 * d * n_experts]])
    targets = np.array([target_f_per_layer, target_p_per_layer])
    
    try:
        x, y = np.linalg.solve(coeff_matrix, targets)
        
        # Snap to 64 to pass Pydantic validator
        ffn_deep_base = int(round(x / 64.0) * 64)
        expert_ffn_base = int(round(y / 64.0) * 64)
        
        if ffn_deep_base < 64 or expert_ffn_base < 64: return None

        best, min_error = None, float('inf')
        for dx in range(-128, 129, 64):
            for dy in range(-128, 129, 64):
                test_d, test_w = ffn_deep_base + dx, expert_ffn_base + dy
                if test_d <= 0 or test_w <= 0: continue
                    
                res = calc(n_layers, max_loops, d, vocab, nq, nkv, hd, test_d, n_experts, top_k, test_w)
                
                target_p_pct = 100.0 * param_budget
                err = abs(100 - res['flop_pct']) + abs(target_p_pct - res['param_pct'])
                
                # Use relaxed tolerance here to allow 64-dim chunks to fit
                if err < min_error and err <= tolerance: 
                    min_error = err
                    best = (test_d, test_w, res)
        return best
    except np.linalg.LinAlgError:
        return None

# =============================================================================
# YAML Output & Detail Printing
# =============================================================================

def update_model_config(d, p, experiment_id=None):
    if isinstance(d, dict):
        if 'n_layer' in d: d['n_layer'] = p['n_layers']
        if 'ffn_hidden' in d: d['ffn_hidden'] = p['ffn_deep'] if p['ffn_deep'] > 0 else 512
        if 'adaptive_config' in d:
            ac = d['adaptive_config']
            ac['path_mode'] = p['path_mode']
            ac['max_loops'] = p['max_loops'] if p['max_loops'] > 0 else 1
            ac['use_cross'] = False
            ac['wide_ffn_hidden'] = p['wide_ffn']
            ac['use_moe_wide'] = p['use_moe']
            if p['use_moe']:
                ac['n_experts'] = p['n_experts']
                ac['top_k'] = p['top_k']
                ac['expert_ffn_hidden'] = p['expert_ffn']
            if 'enable_adaptive' in ac: del ac['enable_adaptive']
        if experiment_id and 'experiment_id' in d: d['experiment_id'] = experiment_id
        for k, v in d.items(): update_model_config(v, p, experiment_id)
    elif isinstance(d, list):
        for item in d: update_model_config(item, p, experiment_id)

def write_yaml(experiment_id, p, template_path):
    if not HAS_YAML:
        print("  [!] PyYAML not installed. Cannot generate .yaml file.")
        return
    if not os.path.exists(template_path):
        print(f"  [!] Template '{template_path}' not found. Please provide a valid template using --template.")
        return
        
    with open(template_path, "r") as f:
        config = yaml.safe_load(f)
        
    update_model_config(config, p, experiment_id)
    os.makedirs("generated_configs", exist_ok=True)
    out_path = f"generated_configs/{experiment_id}.yaml"
    
    with open(out_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"  [+] Saved YAML to: {out_path}")

def print_detail_and_generate(config_id, template_path, param_budget, tolerance):
    match = re.match(r"(\d+)L_(\d+)x_(\d+)Et(\d+)", config_id)
    if not match:
        print(f"Error: Invalid config ID format '{config_id}'. Expected format like '24L_4x_2Et1'")
        sys.exit(1)
        
    n_layers, max_loops, n_experts, top_k = map(int, match.groups())
    
    found = find_exact_iso_config(n_layers, max_loops, n_experts, top_k, param_budget, tolerance)
    if not found:
        print(f"Error: Could not find a valid isolation for {config_id} within {tolerance}% tolerance.")
        sys.exit(1)
        
    d_ffn, e_ffn, r = found
    
    p = {
        'path_mode': 'dual', 'use_moe': True, 'wide_ffn': 0,
        'n_layers': n_layers, 'max_loops': max_loops, 
        'n_experts': n_experts, 'top_k': top_k, 
        'ffn_deep': d_ffn, 'expert_ffn': e_ffn
    }
    
    full_id = f"dual_{config_id}_d{d_ffn}_e{e_ffn}_{int(r['flop_pct'])}pct"

    print(f"\n{'='*72}")
    print(f"  Configuration Detail: {config_id}")
    print(f"{'='*72}")
    print(f"  Layers: {p['n_layers']}, Max Loops: {p['max_loops']}")
    print(f"  Deep FFN: {p['ffn_deep']}")
    print(f"  MoE: {p['n_experts']} Experts (Top-{p['top_k']}), Expert FFN: {p['expert_ffn']}\n")
    
    print(f"  Deep path:  {r['deep_layer_f']/1e6:>8.1f}M FLOPs/layer")
    print(f"  Wide path:  {r['wide_layer_f']/1e6:>8.1f}M FLOPs/layer")
    deep_frac = r['deep_layer_f'] / (r['deep_layer_f'] + r['wide_layer_f']) * 100
    print(f"  Deep/Total: {deep_frac:.1f}%\n")
    
    print(f"  Total FLOPs:      {r['total_flops']/1e9:.3f}G  ({r['flop_pct']:.2f}% of baseline)")
    print(f"  Total Params:     {r['total_params']/1e6:.1f}M  ({r['param_pct']:.2f}% of baseline target)")
    print(f"  FFN Capacity:     {r['total_ffn_params']/1e6:.1f}M  ({r['ffn_cap_pct']:.1f}%)\n")
    
    write_yaml(full_id, p, template_path)


# =============================================================================
# Main CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Iso-FLOP/Param Sweeper & YAML Generator")
    parser.add_argument("config_id", nargs="?", default=None, help="Specific config ID to generate (e.g., 24L_4x_2Et1)")
    parser.add_argument("--template", default="loop3_moe_4e_t2_6k.yaml", help="Path to the template YAML file")
    parser.add_argument("--param_budget", type=float, default=1.0, help="E.g., 1.5 for 150% params (Default: 1.0)")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Allowed error margin in % (Default: 0.8)")
    
    args = parser.parse_args()

    if args.config_id:
        # Mode 2: Detail & YAML Generation
        print_detail_and_generate(args.config_id, args.template, args.param_budget, args.tolerance)
        sys.exit(0)

    # Mode 1: Sweep & Print Table
    bl = dense_baseline()
    print(f"BASELINE: {bl['total_flops']/1e9:.3f}G FLOPs | {bl['total_params']/1e6:.1f}M Params")
    print(f"TARGET BUDGET: 100% FLOPs | {args.param_budget * 100:.0f}% Params (±{args.tolerance}%)\n")
    print("="*105)
    print(f"{'Ratio (Deep%)':<14} | {'Config ID':<16} | {'Deep FFN':<9} | {'Exp FFN':<9} | {'F%':<6} | {'P%':<6}")
    print("="*105)

    layers_sweep = [8, 10, 12, 16, 20, 24, 30]
    loops_sweep = [2, 3, 4, 5, 6, 8, 10]
    experts_sweep = [2, 4, 8, 16, 32]
    topk_sweep = [1, 2, 4]

    results = []

    for n in layers_sweep:
        for loops in loops_sweep:
            for exp in experts_sweep:
                for topk in topk_sweep:
                    if topk > exp: continue
                        
                    found = find_exact_iso_config(n, loops, exp, topk, param_budget=args.param_budget, tolerance=args.tolerance)
                    if found:
                        d_ffn, e_ffn, res = found
                        total_f = res['deep_layer_f'] + res['wide_layer_f']
                        ratio = (res['deep_layer_f'] / total_f) * 100 if total_f > 0 else 0
                        
                        config_id = f"{n}L_{loops}x_{exp}Et{topk}"
                        results.append({
                            'ratio': ratio, 'id': config_id,
                            'd_ffn': d_ffn, 'e_ffn': e_ffn,
                            'f_pct': res['flop_pct'], 'p_pct': res['param_pct']
                        })

    results.sort(key=lambda x: x['ratio'])
    current_bin = -1
    for r in results:
        bin_val = int(r['ratio'] // 10) * 10
        if bin_val != current_bin:
            print(f"\n--- ~{bin_val}% Deep Path ---")
            current_bin = bin_val
            
        print(f"{r['ratio']:>12.2f}% | {r['id']:<16} | {r['d_ffn']:<9} | {r['e_ffn']:<9} | {r['f_pct']:>5.2f}% | {r['p_pct']:>5.2f}%")