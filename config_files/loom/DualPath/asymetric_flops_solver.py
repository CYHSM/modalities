import math

def asymmetric_dual_config(
    d_model: int, 
    target_dense_layers: int, 
    dense_ffn_mult: int, 
    our_layers: int, 
    max_loops: int,
    capacity_ratio: float = 0.5
):
    """
    Calculates exact FFN sizes for an asymmetric FLOP split between Wide and Deep paths,
    while ensuring the total layer FLOPs perfectly match the dense baseline.
    """
    assert 0.0 <= capacity_ratio <= 1.0, "Capacity ratio must be between 0 and 1"

    # 1. Base Costs
    attn_flops = 8 * (d_model ** 2)
    gate_flops = 4 * (d_model ** 2)  
    
    # 2. Total Budget from the Baseline
    dense_ffn = d_model * dense_ffn_mult
    dense_block_flops = attn_flops + (6 * d_model * dense_ffn)
    total_budget_flops = target_dense_layers * dense_block_flops
    
    # 3. Budget per Dual Layer
    layer_budget = total_budget_flops / our_layers
    paths_budget = layer_budget - gate_flops
    
    # 4. Apply the Allocation Ratio
    wide_budget = paths_budget * capacity_ratio
    deep_budget = paths_budget * (1.0 - capacity_ratio)
    
    # 5. Solve for Wide FFN
    if capacity_ratio > 0:
        ffn_wide_exact = (wide_budget - attn_flops) / (6 * d_model)
        ffn_wide = max(64, math.ceil(ffn_wide_exact / 64) * 64)
        actual_wide_flops = attn_flops + (6 * d_model * ffn_wide)
    else:
        ffn_wide = 0
        actual_wide_flops = 0
    
    # 6. Solve for Deep FFN
    if deep_budget > 0:
        budget_per_loop = deep_budget / max_loops
        ffn_deep_exact = (budget_per_loop - attn_flops) / (6 * d_model)
        
        # Check if we starved the deep path
        if ffn_deep_exact < 0:
            return {"error": f"Not enough FLOPs allocated to the Deep path to even run {max_loops} loops of Attention!"}
        
        ffn_deep = max(64, math.ceil(ffn_deep_exact / 64) * 64)
        actual_deep_flops = max_loops * (attn_flops + (6 * d_model * ffn_deep))
    else:
        ffn_deep = 0
        actual_deep_flops = 0
    
    # 7. Verify Final FLOPs Match
    actual_total_flops = our_layers * (actual_wide_flops + actual_deep_flops + gate_flops)
    match_ratio = actual_total_flops / total_budget_flops
    
    return {
        "Capacity Ratio": f"{capacity_ratio*100:.0f}% Wide / {(1-capacity_ratio)*100:.0f}% Deep",
        "Target ffn_wide": ffn_wide,
        "Target ffn_deep": ffn_deep,
        "FLOP Match": f"{match_ratio * 100:.2f}%"
    }

def run_scenario(our_layers, max_loops):
    print(f"=========================================================")
    print(f"SCENARIO: {our_layers} Dual Layers | {max_loops} Loops")
    print(f"=========================================================")
    
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 50/50, 70/30 (Wide Focus), 20/80 (Deep Focus)
    
    for r in ratios:
        res = asymmetric_dual_config(d_model=768, target_dense_layers=60, dense_ffn_mult=4, our_layers=our_layers, max_loops=max_loops, capacity_ratio=r)
        print(f"--- Ratio: {r*100:.0f}% Wide / {(1-r)*100:.0f}% Deep ---")
        
        if "error" in res:
            print(f"❌ ERROR: {res['error']}\n")
        else:
            print(f"Wide FFN (`wide_ffn_hidden`): {res['Target ffn_wide']}")
            print(f"Deep FFN (`ffn_hidden`):      {res['Target ffn_deep']}")
            print(f"Overall Iso-FLOP Match:       {res['FLOP Match']}\n")


print("TARGET BASELINE: 60 Standard Layers (d=768, ffn=3072)\n")

# Run all requested scenarios
run_scenario(our_layers=30, max_loops=2)
run_scenario(our_layers=20, max_loops=3)
run_scenario(our_layers=15, max_loops=4)
run_scenario(our_layers=12, max_loops=5)
run_scenario(our_layers=10, max_loops=6)