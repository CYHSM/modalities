import math

def reverse_solve_dual_config(
    d_model: int, 
    target_dense_layers: int, 
    dense_ffn_mult: int, 
    our_layers: int, 
    max_loops: int
):
    """
    Takes a target dense baseline and calculates the exact ffn_deep and ffn_wide 
    needed to make your dual model perfectly iso-FLOP, while keeping the 
    compute and capacity branches perfectly balanced with each other.
    """
    # 1. Base Costs per block
    attn_flops = 8 * (d_model ** 2)
    gate_flops = 4 * (d_model ** 2)  
    
    # 2. Total Budget from the Baseline
    dense_ffn = d_model * dense_ffn_mult
    dense_block_flops = attn_flops + (6 * d_model * dense_ffn)
    total_budget_flops = target_dense_layers * dense_block_flops
    
    # 3. Budget per Dual Layer
    layer_budget = total_budget_flops / our_layers
    
    # Subtract the gate cost; the rest is split 50/50 between Deep and Wide
    paths_budget = layer_budget - gate_flops
    half_budget = paths_budget / 2
    
    # 4. Solve for Wide FFN
    # half_budget = attn_flops + (6 * d_model * ffn_wide)
    ffn_wide_exact = (half_budget - attn_flops) / (6 * d_model)
    ffn_wide = max(64, math.ceil(ffn_wide_exact / 64) * 64)
    
    # 5. Solve for Deep FFN
    # half_budget = max_loops * (attn_flops + (6 * d_model * ffn_deep))
    budget_per_loop = half_budget / max_loops
    ffn_deep_exact = (budget_per_loop - attn_flops) / (6 * d_model)
    ffn_deep = max(64, math.ceil(ffn_deep_exact / 64) * 64)
    
    # 6. Verify Final FLOPs
    actual_wide_flops = attn_flops + (6 * d_model * ffn_wide)
    actual_deep_flops = max_loops * (attn_flops + (6 * d_model * ffn_deep))
    actual_total_flops = our_layers * (actual_wide_flops + actual_deep_flops + gate_flops)
    
    match_ratio = actual_total_flops / total_budget_flops
    
    return {
        "Baseline": f"{target_dense_layers} layers, d={d_model}, ffn={dense_ffn}",
        "Our Layers": our_layers,
        "Loops": max_loops,
        "Target ffn_wide": ffn_wide,
        "Target ffn_deep": ffn_deep,
        "FLOP Match": f"{match_ratio * 100:.2f}%"
    }

def print_solution(sol):
    print(f"To match a {sol['Baseline']}:")
    print(f"Build a {sol['Our Layers']}-layer model where each layer does:")
    print(f"  - {sol['Loops']} loops on the Deep Path with ffn_deep = {sol['Target ffn_deep']}")
    print(f"  - 1 pass on the Wide Path with ffn_wide = {sol['Target ffn_wide']}")
    print(f"FLOP Match Accuracy: {sol['FLOP Match']}\n")


# === The 60-Layer Scenario ===

# Scenario A: 20-layer model, 3 loops
print_solution(reverse_solve_dual_config(
    d_model=768, target_dense_layers=60, dense_ffn_mult=4, 
    our_layers=20, max_loops=3
))

# Scenario A: 15-layer model, 4 loops
print_solution(reverse_solve_dual_config(
    d_model=768, target_dense_layers=60, dense_ffn_mult=4, 
    our_layers=15, max_loops=4
))

# Scenario B: 12-layer model, 5 loops (More aggressive depth reduction)
print_solution(reverse_solve_dual_config(
    d_model=768, target_dense_layers=60, dense_ffn_mult=4, 
    our_layers=12, max_loops=5
))

# Scenario C: 10-layer model, 6 loops
print_solution(reverse_solve_dual_config(
    d_model=768, target_dense_layers=60, dense_ffn_mult=4, 
    our_layers=10, max_loops=6
))