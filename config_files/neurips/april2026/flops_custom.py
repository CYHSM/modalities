def solve_custom_stack_fixed_deep(
    d_model: int,
    layer_types: list[str],
    max_loops: int,
    target_dense_layers: int,
    ffn_deep: int,                  # ← you pick this
    match: str = "flops",           # "flops" or "params"
    dense_ffn_mult: int = 4,
    vocab_size: int = 50304,
    use_weight_tying: bool = False,
    n_head_q: int | None = None,
    n_head_kv: int | None = None,
    ffn_round_multiple: int = 64,
):
    """
    Fix ffn_deep, solve for ffn_wide to match EITHER FLOPs or params exactly.
    Report the mismatch on the other axis.
    """
    n_rep = (n_head_q // n_head_kv) if (n_head_q and n_head_kv) else 1
    A = 4 * d_model**2 + 4 * d_model**2 // n_rep
    a = 2 * d_model**2 + 2 * d_model**2 // n_rep

    n_deep_blocks = sum(1 for t in layer_types if t in ("deep", "dual"))
    n_wide_blocks = sum(1 for t in layer_types if t in ("wide", "dual"))
    n_dual_gates  = sum(1 for t in layer_types if t == "dual")
    n_routers     = n_deep_blocks

    # FLOP side
    c_fd_flops = n_deep_blocks * max_loops * 6 * d_model
    c_fw_flops = n_wide_blocks * 6 * d_model
    our_fixed_flops = (
        n_deep_blocks * max_loops * A
        + n_wide_blocks * A
        + n_routers * max_loops * 2 * (d_model + 1)
        + n_dual_gates * (2 * d_model * 2)
    )
    dense_ffn = d_model * dense_ffn_mult
    target_flops = target_dense_layers * (A + 6 * d_model * dense_ffn)
    rhs_f = target_flops - our_fixed_flops

    # Param side (embeddings cancel)
    c_fd_params = n_deep_blocks * 3 * d_model
    c_fw_params = n_wide_blocks * 3 * d_model
    our_fixed_params = 0
    for t in layer_types:
        if t in ("deep", "dual"):
            our_fixed_params += a + 2 * d_model + (d_model + 1) + 1 + max_loops
        if t in ("wide", "dual"):
            our_fixed_params += a + 2 * d_model + 1
        if t == "dual":
            our_fixed_params += d_model * 2 + 2
    target_params = target_dense_layers * (a + 2 * d_model + 3 * d_model * dense_ffn)
    rhs_p = target_params - our_fixed_params

    # Solve for f_w given fixed f_d
    if match == "flops":
        f_w_exact = (rhs_f - c_fd_flops * ffn_deep) / c_fw_flops
    elif match == "params":
        f_w_exact = (rhs_p - c_fd_params * ffn_deep) / c_fw_params
    else:
        raise ValueError("match must be 'flops' or 'params'")

    if f_w_exact < 0:
        return {"error": f"ffn_deep={ffn_deep} is already over budget on {match}. "
                         f"Lower it or reduce max_loops."}

    f_w = max(ffn_round_multiple,
              round(f_w_exact / ffn_round_multiple) * ffn_round_multiple)
    f_d = ffn_deep

    # Compute both totals for reporting
    embed = vocab_size * d_model + d_model
    if not use_weight_tying:
        embed += d_model * vocab_size

    actual_flops  = our_fixed_flops  + c_fd_flops  * f_d + c_fw_flops  * f_w
    actual_params = our_fixed_params + c_fd_params * f_d + c_fw_params * f_w + embed
    target_params_total = target_params + embed

    return {
        "ffn_deep": f_d,
        "ffn_wide": f_w,
        "ffn_wide_exact": f_w_exact,
        "matched": match,
        "flop_match":  actual_flops  / target_flops,
        "param_match": actual_params / target_params_total,
        "actual_flops": actual_flops,
        "target_flops": target_flops,
        "actual_params": actual_params,
        "target_params": target_params_total,
    }


if __name__ == "__main__":
    layer_types = ["deep"]*9 + ["dual"]*2 + ["wide"]*1
    # Pick ffn_deep small — say 1024 or 1536. These are the deep blocks that
    # run max_loops times, so keeping them lean saves a lot of compute that
    # the (single-pass) wide layer can then spend generously.
    for ffn_deep in [768, 1024, 1536, 2048]:
        print(f"\n--- ffn_deep = {ffn_deep} ---")
        res = solve_custom_stack_fixed_deep(
            d_model=768,
            layer_types=layer_types,
            max_loops=3,
            target_dense_layers=36,
            ffn_deep=ffn_deep,
            match="flops",
            n_head_q=12, n_head_kv=12,
        )
        if "error" in res:
            print(f"  {res['error']}")
        else:
            print(f"  ffn_deep = {res['ffn_deep']}")
            print(f"  ffn_wide = {res['ffn_wide']}  (exact {res['ffn_wide_exact']:.0f})")
            print(f"  FLOP match:  {res['flop_match']*100:.2f}%  (matched exactly)")
            print(f"  Param match: {res['param_match']*100:.2f}%  "
                  f"({(res['actual_params']-res['target_params'])/1e6:+.1f}M vs baseline)")