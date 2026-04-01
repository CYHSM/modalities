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

    FIX: gate_flops corrected from 4*d² to actual values:
      - DualPathGate: nn.Linear(d, 2)       → 2*d*2 = 4*d
      - Router:       nn.Linear(d+1, 1)     → 2*(d+1) per loop
    These are negligible vs block FLOPs, so we set them to their true values.
    """
    assert 0.0 <= capacity_ratio <= 1.0, "Capacity ratio must be between 0 and 1"

    # 1. Base Costs (projection FLOPs only — QK^T ignored, consistent across configs)
    attn_flops = 8 * (d_model ** 2)

    # FIXED: actual gate FLOPs from the code
    dual_gate_flops = 2 * d_model * 2                       # DualPathGate: Linear(d, 2)
    router_flops_total = max_loops * 2 * (d_model + 1)      # Router: Linear(d+1, 1) × max_loops
    gate_flops = dual_gate_flops + router_flops_total        # Tiny — O(d), not O(d²)

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
        if ffn_wide_exact < 0:
            return {"error": f"Wide budget too small to cover even attention ({wide_budget:.0f} < {attn_flops})"}
        ffn_wide = max(64, math.ceil(ffn_wide_exact / 64) * 64)
        actual_wide_flops = attn_flops + (6 * d_model * ffn_wide)
    else:
        ffn_wide = 0
        actual_wide_flops = 0

    # 6. Solve for Deep FFN (budget spread across max_loops iterations)
    if deep_budget > 0 and max_loops > 0:
        budget_per_loop = deep_budget / max_loops
        ffn_deep_exact = (budget_per_loop - attn_flops) / (6 * d_model)
        if ffn_deep_exact < 0:
            return {"error": f"Deep path starved: can't afford {max_loops} loops of attention "
                    f"(need {max_loops * attn_flops:.0f}, have {deep_budget:.0f})"}
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
        "FLOP Match": f"{match_ratio * 100:.2f}%",
        "actual_total_flops": actual_total_flops,
        "total_budget_flops": total_budget_flops,
    }


def run_scenario(our_layers, max_loops, target_dense_layers, d_model=768, dense_ffn_mult=4):
    print(f"\n{'='*65}")
    print(f"  SCENARIO: {our_layers} Dual Layers | {max_loops} Loops")
    print(f"  Baseline: {target_dense_layers} Dense Layers (d={d_model}, ffn={d_model*dense_ffn_mult})")
    print(f"{'='*65}")

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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


# ================================================================
# Run all scenarios against BOTH baselines
# ================================================================

for baseline_layers in [60, 36]:
    print(f"\n{'#'*65}")
    print(f"#  BASELINE: {baseline_layers} Dense Layers (d=768, SwiGLU ffn=3072)")
    print(f"{'#'*65}")

    if baseline_layers == 60:
        scenarios = [
            (30, 2), (20, 3), (15, 4), (12, 5), (10, 6), (12, 2), (12, 3), (12, 4)
        ]
    else:  # 36-layer baseline
        scenarios = [
            (18, 2), (12, 3), (9, 4),
        ]

    for our_layers, max_loops in scenarios:
        run_scenario(our_layers, max_loops, target_dense_layers=baseline_layers)