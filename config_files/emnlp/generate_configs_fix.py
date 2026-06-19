#!/usr/bin/env python3
"""
FLOP-matched config generator for the asymmetric Dual-Path transformer. [FIXED]

DEFAULT = TOTAL-FLOP matched. The dual block runs attention K+1 times per layer
(K deep loops + 1 wide pass); pure_loop runs it K times; pure_wide once. The
sequence-length attention-score term (QK^T + softmax*V) therefore differs by
config. By default we now hold the *total* per-layer FLOPs fixed at the DUAL's
total (matmul + (K+1) attention passes) and size each baseline's FFN UP so that
matmul + passes*score matches that total:

    dual     : matmul budget = F                  (UNCHANGED -> no rerun needed)
    pure_loop: matmul budget = F + 1*score        (passes = K)
    pure_wide: matmul budget = F + K*score        (passes = 1)
    expanded : per-layer matmul = (F + score)/K   (KL layers, 1 pass each)

So the baselines get a larger FFN at equal total compute. Pass --matmul-match to
recover the old behavior (all configs share matmul budget F; totals then differ).

Other fix: --swiglu-multiple defaults to 256 to match the model's
enforce_swiglu_hidden_dim_multiple_of. The ACL paper backbone used 64; pass
--swiglu-multiple 64 to reproduce / match configs generated at 64.

SwiGLU: h_eff = round_up(int(2*ffn/3), swiglu_multiple); params 3*d*h_eff; FLOPs 6*d*h_eff.
Attention projections: 4d^2 + 4d^2/n_rep. Attention score/pass (per token):
~2*T*d causal (4*T*d non-causal).
"""

import argparse
import os
import re

DEFAULT_SWIGLU_MULTIPLE = 256
DEFAULT_FFN_ROUND_MULTIPLE = 64
DEFAULT_VOCAB_SIZE = 50304
DEFAULT_SEQ_LEN = 4096


# ---------------- SwiGLU ----------------
def swiglu_effective_hidden(ffn_hidden, m):
    h = int(2 * ffn_hidden / 3)
    return m * ((h + m - 1) // m)

def round_up(x, m):   return max(m, m * ((int(x) + m - 1) // m))
def round_down(x, m): return max(m, m * (int(x) // m))

def get_ffn_hidden(h_exact, swiglu_m, ffn_round, mode="floor"):
    if mode == "ceil":
        h = round_up(h_exact, swiglu_m); ffn = (3 * h + 1) // 2
        return max(ffn_round, ffn_round * ((ffn + ffn_round - 1) // ffn_round))
    h_max = round_down(h_exact, swiglu_m); ffn = (3 * h_max) // 2
    ffn_f = max(ffn_round, ffn_round * (ffn // ffn_round))
    while swiglu_effective_hidden(ffn_f, swiglu_m) > h_max and ffn_f > ffn_round:
        ffn_f -= ffn_round
    return ffn_f


# ---------------- FLOP/param ----------------
def attn_flops(d, n_rep):  return 4 * d * d + 4 * (d * d) // n_rep

def attn_score_flops_per_pass(d, seq_len, causal=True):
    if seq_len <= 0: return 0
    return (2 if causal else 4) * seq_len * d

def attn_params(d, n_rep): return 2 * d * d + 2 * (d * d) // n_rep
def swiglu_flops(d, ffn, m):  return 6 * d * swiglu_effective_hidden(ffn, m)
def swiglu_params(d, ffn, m): return 3 * d * swiglu_effective_hidden(ffn, m)
def block_params(d, ffn, n_rep, m): return attn_params(d, n_rep) + swiglu_params(d, ffn, m) + 2 * d

def shared_params(d, vocab, wt):
    s = vocab * d
    if not wt: s += d * vocab
    return s + d

def deep_flops(d, ffn_d, K, n_rep, m): return K * (attn_flops(d, n_rep) + swiglu_flops(d, ffn_d, m))
def wide_flops(d, ffn_w, n_rep, m):    return attn_flops(d, n_rep) + swiglu_flops(d, ffn_w, m)
def gate_flops_t(d, gm):   return 4 * d if gm == "two_gates" else 2 * d
def router_flops_t(d, K):  return K * 2 * (d + 1)

def loop_layer_params(d, ffn, K, n_rep, m): return block_params(d, ffn, n_rep, m) + (d + 1) + 1 + K
def wide_layer_params(d, ffn, n_rep, m):    return block_params(d, ffn, n_rep, m) + 1
def dual_layer_params(d, ffn_d, ffn_w, K, n_rep, m, gm, use_cross):
    p = block_params(d, ffn_d, n_rep, m) + block_params(d, ffn_w, n_rep, m) + (d + 1) + 1 + K + 1
    p += (2 * d + 2) if gm == "two_gates" else (d + 1)
    if use_cross: p += 2 * d * d + 2
    return p


# ---------------- solvers ----------------
def solve_loop(F, d, K, n_rep, m, fr):
    attn = attn_flops(d, n_rep); router = router_flops_t(d, K)
    h = (F - router - K * attn) / (6.0 * d * K)
    if h <= 0: raise ValueError(f"pure_loop: budget {F:.2e} too small.")
    ffn = get_ffn_hidden(h, m, fr, "floor")
    return ffn, K * (attn + swiglu_flops(d, ffn, m)) + router

def solve_wide(F, d, n_rep, m, fr):
    attn = attn_flops(d, n_rep); h = (F - attn) / (6.0 * d)
    if h <= 0: raise ValueError(f"pure_wide: budget {F:.2e} too small.")
    ffn = get_ffn_hidden(h, m, fr, "ceil")
    return ffn, attn + swiglu_flops(d, ffn, m)

def solve_dual(F, d, K, alpha, n_rep, m, fr, gm):
    if not (0.0 < alpha < 1.0): raise ValueError(f"alpha must be in (0,1), got {alpha}")
    attn = attn_flops(d, n_rep); gate = gate_flops_t(d, gm); router = router_flops_t(d, K)
    fa = F - gate - router
    if fa <= 0: raise ValueError(f"Budget {F} insufficient for gate+router.")
    hd = (alpha * fa / K - attn) / (6.0 * d)
    hw = ((1 - alpha) * fa - attn) / (6.0 * d)
    if hd <= 0: raise ValueError(f"alpha={alpha} too small; deep can't cover attn. "
                                 f"Try alpha >= {(K*attn)/fa:.3f}.")
    if hw <= 0: raise ValueError(f"alpha={alpha} too large; wide can't cover attn. "
                                 f"Try alpha <= {1 - attn/fa:.3f}.")
    ffn_d = get_ffn_hidden(hd, m, fr, "floor"); ffn_w = get_ffn_hidden(hw, m, fr, "floor")
    deep = deep_flops(d, ffn_d, K, n_rep, m); wide = wide_flops(d, ffn_w, n_rep, m)
    return ffn_d, ffn_w, {"deep": deep, "wide": wide, "gate": gate, "router": router,
                          "total": deep + wide + gate + router}

def solve_dual_min_deep(F, d, K, n_rep, m, fr, gm):
    attn = attn_flops(d, n_rep); gate = gate_flops_t(d, gm); router = router_flops_t(d, K)
    fa = F - gate - router
    if fa <= 0: raise ValueError(f"Budget {F} insufficient for gate+router.")
    ffn_d = get_ffn_hidden(m, m, fr, "ceil"); deep = deep_flops(d, ffn_d, K, n_rep, m)
    if deep >= fa: raise ValueError("min-deep: floor ffn_deep exhausts budget.")
    hw = ((fa - deep) - attn) / (6.0 * d)
    if hw <= 0: raise ValueError("min-deep: wide can't cover attn.")
    ffn_w = get_ffn_hidden(hw, m, fr, "floor"); wide = wide_flops(d, ffn_w, n_rep, m)
    return ffn_d, ffn_w, {"deep": deep, "wide": wide, "gate": gate, "router": router,
                          "total": deep + wide + gate + router}


# ---------------- total-FLOP matching ----------------
def reference_total_per_layer(F, max_loops, score):
    """Per-layer total of the DUAL: matmul F + (K+1) attention passes."""
    return F + (max_loops + 1) * score

def matmul_budget(F, passes, max_loops, score, match_total):
    """Matmul budget so that matmul + passes*score == dual total. Default."""
    if not match_total:
        return F
    ref = reference_total_per_layer(F, max_loops, score)
    F_eff = ref - passes * score
    if F_eff <= 0:
        raise ValueError(f"total-match: passes={passes} attention cost exceeds the dual total.")
    return F_eff

def _attach(cfg, n_layers, matmul_pl, passes, score, F, F_eff):
    score_pl = passes * score
    cfg.update(passes=passes, flop_budget=F, flop_budget_eff=F_eff,
               attn_score_per_layer=score_pl, matmul_per_layer=matmul_pl,
               total_per_layer=matmul_pl + score_pl,
               total_flops_with_attn=n_layers * (matmul_pl + score_pl),
               flop_match_pct=(matmul_pl / F_eff - 1.0) * 100)
    return cfg


# ---------------- builders ----------------
def _shared(d, n_layers, hq, hkv, vocab, wt, m):
    n_rep = (hq // hkv) if (hq and hkv) else 1
    return {"d_model": d, "n_layers": n_layers, "n_head_q": hq, "n_head_kv": hkv,
            "n_rep": n_rep, "vocab_size": vocab, "use_weight_tying": wt, "swiglu_multiple": m}

def build_loop(F, d, n_layers, max_loops, vocab, wt, hq, hkv, swiglu_m, fr,
               score_per_pass=0, match_total=True):
    base = _shared(d, n_layers, hq, hkv, vocab, wt, swiglu_m)
    passes = max_loops
    F_eff = matmul_budget(F, passes, max_loops, score_per_pass, match_total)
    ffn, actual = solve_loop(F_eff, d, max_loops, base["n_rep"], swiglu_m, fr)
    tp = n_layers * loop_layer_params(d, ffn, max_loops, base["n_rep"], swiglu_m) + shared_params(d, vocab, wt)
    cfg = {**base, "kind": "pure_loop", "max_loops": max_loops, "ffn_loop": ffn,
           "ffn_loop_h_eff": swiglu_effective_hidden(ffn, swiglu_m), "total_params": tp,
           "flops_per_layer": actual, "total_flops": n_layers * actual, "match_total": match_total}
    return _attach(cfg, n_layers, actual, passes, score_per_pass, F, F_eff)

def build_wide(F, d, n_layers, vocab, wt, hq, hkv, swiglu_m, fr,
               score_per_pass=0, match_total=True, max_loops_ref=1, _passes=1):
    base = _shared(d, n_layers, hq, hkv, vocab, wt, swiglu_m)
    F_eff = matmul_budget(F, _passes, max_loops_ref, score_per_pass, match_total)
    ffn, actual = solve_wide(F_eff, d, base["n_rep"], swiglu_m, fr)
    tp = n_layers * wide_layer_params(d, ffn, base["n_rep"], swiglu_m) + shared_params(d, vocab, wt)
    cfg = {**base, "kind": "pure_wide", "max_loops": 1, "ffn_wide": ffn,
           "ffn_wide_h_eff": swiglu_effective_hidden(ffn, swiglu_m), "total_params": tp,
           "flops_per_layer": actual, "total_flops": n_layers * actual, "match_total": match_total}
    return _attach(cfg, n_layers, actual, _passes, score_per_pass, F, F_eff)

def build_wide_expanded(F, d, n_layers, max_loops, vocab, wt, hq, hkv, swiglu_m, fr,
                        score_per_pass=0, match_total=True):
    n_exp = n_layers * max_loops
    if match_total:
        F_exp = int(reference_total_per_layer(F, max_loops, score_per_pass) / max_loops - score_per_pass)
        if F_exp <= 0: raise ValueError("expanded: total-match budget non-positive.")
    else:
        F_exp = F // max_loops
    cfg = build_wide(F_exp, d, n_exp, vocab, wt, hq, hkv, swiglu_m, fr,
                     score_per_pass=score_per_pass, match_total=False, _passes=1)
    cfg.update(variant="expanded", flop_budget_original=F, max_loops_original=max_loops,
               n_layers_original=n_layers, match_total=match_total)
    cfg["total_flop_match_pct"] = (cfg["total_flops"] / (n_layers * F) - 1.0) * 100
    return cfg

def build_dual(F, d, n_layers, max_loops, alpha, vocab, wt, hq, hkv, swiglu_m, fr,
               gate_mode, use_cross, score_per_pass=0, match_total=True):
    base = _shared(d, n_layers, hq, hkv, vocab, wt, swiglu_m)
    passes = max_loops + 1
    F_eff = matmul_budget(F, passes, max_loops, score_per_pass, match_total)   # == F
    ffn_d, ffn_w, fb = solve_dual(F_eff, d, max_loops, alpha, base["n_rep"], swiglu_m, fr, gate_mode)
    tp = n_layers * dual_layer_params(d, ffn_d, ffn_w, max_loops, base["n_rep"], swiglu_m, gate_mode, use_cross) \
         + shared_params(d, vocab, wt)
    cfg = {**base, "kind": "dual", "max_loops": max_loops, "alpha": alpha,
           "alpha_actual": fb["deep"] / (fb["deep"] + fb["wide"]),
           "ffn_deep": ffn_d, "ffn_wide": ffn_w,
           "ffn_deep_h_eff": swiglu_effective_hidden(ffn_d, swiglu_m),
           "ffn_wide_h_eff": swiglu_effective_hidden(ffn_w, swiglu_m),
           "gate_mode": gate_mode, "use_cross": use_cross, "total_params": tp,
           "flops_per_layer": fb, "total_flops": n_layers * fb["total"], "match_total": match_total}
    return _attach(cfg, n_layers, fb["total"], passes, score_per_pass, F, F_eff)

def build_dual_min_deep(F, d, n_layers, max_loops, vocab, wt, hq, hkv, swiglu_m, fr,
                        gate_mode, use_cross, score_per_pass=0, match_total=True):
    base = _shared(d, n_layers, hq, hkv, vocab, wt, swiglu_m)
    passes = max_loops + 1
    F_eff = matmul_budget(F, passes, max_loops, score_per_pass, match_total)   # == F
    ffn_d, ffn_w, fb = solve_dual_min_deep(F_eff, d, max_loops, base["n_rep"], swiglu_m, fr, gate_mode)
    tp = n_layers * dual_layer_params(d, ffn_d, ffn_w, max_loops, base["n_rep"], swiglu_m, gate_mode, use_cross) \
         + shared_params(d, vocab, wt)
    a = fb["deep"] / (fb["deep"] + fb["wide"])
    cfg = {**base, "kind": "dual", "variant": "min_deep", "max_loops": max_loops,
           "alpha": a, "alpha_actual": a, "ffn_deep": ffn_d, "ffn_wide": ffn_w,
           "ffn_deep_h_eff": swiglu_effective_hidden(ffn_d, swiglu_m),
           "ffn_wide_h_eff": swiglu_effective_hidden(ffn_w, swiglu_m),
           "gate_mode": gate_mode, "use_cross": use_cross, "total_params": tp,
           "flops_per_layer": fb, "total_flops": n_layers * fb["total"], "match_total": match_total}
    return _attach(cfg, n_layers, fb["total"], passes, score_per_pass, F, F_eff)


# ---------------- exp ids / YAML ----------------
def make_exp_id(cfg):
    d = cfg["d_model"]; L = cfg["n_layers"]
    F_m = cfg.get("flop_budget_original", cfg["flop_budget"]) / 1e6
    if cfg["kind"] == "dual":
        at = "_aMINdeep" if cfg.get("variant") == "min_deep" else f"_a{int(round(cfg['alpha']*100)):02d}"
        ct = "_nocross" if cfg.get("use_cross") is False else ""
        return f"dm{d}_L{L}_loop{cfg['max_loops']}_F{F_m:.0f}M_ffnD{cfg['ffn_deep']}_ffnW{cfg['ffn_wide']}_dual{at}{ct}"
    if cfg["kind"] == "pure_loop":
        return f"dm{d}_L{L}_loop{cfg['max_loops']}_F{F_m:.0f}M_ffnL{cfg['ffn_loop']}_pureloop"
    if cfg["kind"] == "pure_wide":
        sfx = "_expanded" if cfg.get("variant") == "expanded" else ""
        return f"dm{d}_L{L}_loop1_F{F_m:.0f}M_ffnW{cfg['ffn_wide']}_purewide{sfx}"
    raise ValueError(cfg["kind"])

def _yaml_patches(cfg):
    n_layers = cfg["n_layers"]
    p = {("model_raw", "n_layer"): str(n_layers), ("model_raw", "n_embd"): str(cfg["d_model"]),
         ("evaluation_subscriber", "experiment_id"): make_exp_id(cfg)}
    if cfg.get("n_head_q") is not None:
        p[("model_raw", "n_head_q")] = str(cfg["n_head_q"])
        p[("model_raw", "normalized_shape")] = f"{cfg['d_model'] // cfg['n_head_q']} # n_embd / n_head_q"
    if cfg.get("n_head_kv") is not None:
        p[("model_raw", "n_head_kv")] = str(cfg["n_head_kv"])
    k = cfg["kind"]
    if k == "dual":
        p[("model_raw", "ffn_hidden")] = str(cfg["ffn_deep"]); p[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        p[("model_raw", "max_loops")] = str(cfg["max_loops"]); p[("model_raw", "enable_adaptive")] = "true"
        if cfg.get("use_cross") is not None:
            p[("model_raw", "use_cross")] = "true" if cfg["use_cross"] else "false"
        cfg.setdefault("layer_types", ["dual"] * n_layers)
    elif k == "pure_loop":
        p[("model_raw", "ffn_hidden")] = str(cfg["ffn_loop"]); p[("model_raw", "wide_ffn_hidden")] = "0"
        p[("model_raw", "max_loops")] = str(cfg["max_loops"]); p[("model_raw", "enable_adaptive")] = "true"
        cfg.setdefault("layer_types", ["loop"] * n_layers)
    elif k == "pure_wide":
        p[("model_raw", "ffn_hidden")] = "64"; p[("model_raw", "wide_ffn_hidden")] = str(cfg["ffn_wide"])
        p[("model_raw", "max_loops")] = "1"; p[("model_raw", "enable_adaptive")] = "true"
        cfg.setdefault("layer_types", ["wide"] * n_layers)
    return p

def _patch_yaml_text(text, cfg):
    patches = _yaml_patches(cfg); layer_types = cfg.get("layer_types")
    kv = re.compile(r'^(\s*)([\w_]+)(:\s+)(.+)$'); out, sec = [], None
    for line in text.splitlines(keepends=True):
        s = line.lstrip()
        if s and not s.startswith('#') and line[0] not in (' ', '\t', '\n', '\r'):
            c = s.find(':');  sec = s[:c].strip() if c > 0 else sec
        m = kv.match(line)
        if layer_types is not None and sec == "model_raw" and m and m.group(2) == "layer_types":
            continue
        if m and sec:
            ind, key, sp, ov = m.groups()
            if not ov.lstrip().startswith('${') and (sec, key) in patches:
                line = f"{ind}{key}{sp}{patches[(sec, key)]}\n"
        out.append(line)
        if layer_types is not None and sec == "model_raw" and m and m.group(2) == "max_loops":
            out.append(f"{m.group(1)}layer_types: [{', '.join(chr(34)+t+chr(34) for t in layer_types)}]\n")
    return ''.join(out)

def write_yaml(template_path, output_dir, cfg):
    with open(template_path) as f: text = f.read()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{make_exp_id(cfg)}.yaml")
    with open(out, "w") as f: f.write(_patch_yaml_text(text, cfg))
    return out


# ---------------- printing ----------------
def _fp(p):
    if p >= 1e9: return f"{p/1e9:.2f}B"
    if p >= 1e6: return f"{p/1e6:.1f}M"
    if p >= 1e3: return f"{p/1e3:.1f}k"
    return str(p)
def _ff(f):
    if f >= 1e12: return f"{f/1e12:.2f}T"
    if f >= 1e9:  return f"{f/1e9:.2f}G"
    if f >= 1e6:  return f"{f/1e6:.1f}M"
    return f"{f/1e3:.1f}k"

def print_summary(F, d, n_layers, max_loops, swiglu_m, seq_len, causal, score, match_total, configs):
    mode = ("TOTAL-FLOP matched (baselines sized to the DUAL total; dual unchanged)"
            if match_total else "matmul/FFN-FLOP matched (totals differ by attention passes)")
    print(f"\n  {'─'*108}")
    print(f"  {mode}")
    print(f"  dual matmul budget F={_ff(F)}/tok/layer  (d={d} L={n_layers} K={max_loops} swiglu_mult={swiglu_m})")
    print(f"  seq_len={seq_len} causal={causal} attn-score/pass={_ff(score)}/tok  "
          f"(passes: wide=1, loop=K, dual=K+1)")
    if match_total:
        print(f"  Common per-layer TOTAL target = F+(K+1)*score = {_ff(reference_total_per_layer(F,max_loops,score))}/tok")
    print(f"  {'─'*108}")
    print(f"    {'Config':<30s} {'L':>4s} {'ffn_d/h_d':>12s} {'ffn_w/h_w':>12s} "
          f"{'params':>9s} {'P':>2s} {'matmul/L':>9s} {'+attn/L':>8s} {'TOT/L':>9s} {'Δmm':>7s}")
    print(f"    {'-'*30} {'-'*4} {'-'*12} {'-'*12} {'-'*9} {'-'*2} {'-'*9} {'-'*8} {'-'*9} {'-'*7}")
    for c in configs:
        k = c["kind"]
        if k == "pure_loop":
            lab, dc, wc, mm = "pure_loop", f"{c['ffn_loop']}/{c['ffn_loop_h_eff']}", "-", c["flop_match_pct"]
        elif k == "pure_wide" and c.get("variant") == "expanded":
            lab, dc, wc, mm = f"pure_wide exp(L×{c['max_loops_original']})", "-", f"{c['ffn_wide']}/{c['ffn_wide_h_eff']}", c["total_flop_match_pct"]
        elif k == "pure_wide":
            lab, dc, wc, mm = "pure_wide", "-", f"{c['ffn_wide']}/{c['ffn_wide_h_eff']}", c["flop_match_pct"]
        else:
            lab = f"dual min-deep (α≈{c['alpha_actual']:.2f})" if c.get("variant") == "min_deep" else f"dual α={c['alpha']:.2f}"
            if c.get("use_cross") is False: lab += " no-cross"
            dc, wc, mm = f"{c['ffn_deep']}/{c['ffn_deep_h_eff']}", f"{c['ffn_wide']}/{c['ffn_wide_h_eff']}", c["flop_match_pct"]
        print(f"    {lab:<30s} {c['n_layers']:>4d} {dc:>12s} {wc:>12s} {_fp(c['total_params']):>9s} "
              f"{c.get('passes',1):>2d} {_ff(c['matmul_per_layer']):>9s} {_ff(c['attn_score_per_layer']):>8s} "
              f"{_ff(c['total_per_layer']):>9s} {mm:>+6.2f}%")
    ap = [c['total_params'] for c in configs]; tot = [c['total_flops_with_attn'] for c in configs]
    print(f"\n    Param spread : {_fp(min(ap))} → {_fp(max(ap))} ({(max(ap)/min(ap)-1)*100:+.0f}%)")
    print(f"    TOTAL FLOPs/tok (incl. attention) spread: {_ff(min(tot))} → {_ff(max(tot))} "
          f"({(max(tot)/min(tot)-1)*100:+.2f}%)")
    if match_total:
        print(f"    → total FLOPs equalized to the dual; pure_wide/pure_loop FFNs grown to spend the same "
              f"total compute.\n      Your dual configs (matmul=F) are UNCHANGED. Rerun only the baselines.")
    else:
        print(f"    → matmul matched; the spread above is unbudgeted attention (dual/loop run it more). "
              f"Drop --matmul-match for total-match.")
    print(f"    Note: ffn = configured/h_eff (h_eff = round_up(int(2*ffn/3), {swiglu_m})).")


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description="FLOP-matched dual-path config generator (default: TOTAL-FLOP matched).")
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--max-loops", type=int, required=True)
    p.add_argument("--flop-budget", type=int, required=True, help="DUAL per-layer matmul (FFN+proj) budget/token.")
    p.add_argument("--alpha", type=float, nargs="+", required=True)
    p.add_argument("--n-head-q", type=int, default=None)
    p.add_argument("--n-head-kv", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--weight-tying", action="store_true", default=False)
    p.add_argument("--gate-mode", choices=["two_gates", "convex"], default="two_gates")
    p.add_argument("--no-cross", action="store_true", default=False)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="For attention-score accounting (0 disables).")
    p.add_argument("--no-causal", action="store_true", default=False, help="Full (non-causal) attention count.")
    p.add_argument("--matmul-match", action="store_true", default=False,
                   help="Revert to matching matmul/FFN FLOPs only (old behavior; totals then differ).")
    p.add_argument("--swiglu-multiple", type=int, default=DEFAULT_SWIGLU_MULTIPLE,
                   help="MUST equal the model's enforce_swiglu_hidden_dim_multiple_of. Paper used 64.")
    p.add_argument("--ffn-round-multiple", type=int, default=DEFAULT_FFN_ROUND_MULTIPLE)
    p.add_argument("--no-min-deep", action="store_true", default=False)
    p.add_argument("--no-wide-expanded", action="store_true", default=False)
    p.add_argument("--add-no-cross-alpha", type=float, nargs="+", default=None)
    p.add_argument("--template", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="configs")
    a = p.parse_args()

    causal = not a.no_causal
    score = attn_score_flops_per_pass(a.d_model, a.seq_len, causal)
    match_total = not a.matmul_match
    print(f"\n  swiglu_multiple={a.swiglu_multiple} (must match model's enforce_swiglu_hidden_dim_multiple_of)")

    use_cross = not a.no_cross
    common = dict(F=a.flop_budget, d=a.d_model, n_layers=a.n_layers, max_loops=a.max_loops,
                  vocab=a.vocab_size, wt=a.weight_tying, hq=a.n_head_q, hkv=a.n_head_kv,
                  swiglu_m=a.swiglu_multiple, fr=a.ffn_round_multiple,
                  score_per_pass=score, match_total=match_total)
    dual_common = dict(common, gate_mode=a.gate_mode, use_cross=use_cross)

    configs = [build_loop(**common)]
    configs.append(build_wide(F=a.flop_budget, d=a.d_model, n_layers=a.n_layers, vocab=a.vocab_size,
                              wt=a.weight_tying, hq=a.n_head_q, hkv=a.n_head_kv, swiglu_m=a.swiglu_multiple,
                              fr=a.ffn_round_multiple, score_per_pass=score, match_total=match_total,
                              max_loops_ref=a.max_loops, _passes=1))
    if not a.no_wide_expanded:
        try: configs.append(build_wide_expanded(**common))
        except ValueError as e: print(f"  ⚠ skip expanded: {e}")
    for al in a.alpha:
        try: configs.append(build_dual(alpha=al, **dual_common))
        except ValueError as e: print(f"  ⚠ skip dual α={al}: {e}")
    if not a.no_min_deep:
        try:
            mc = build_dual_min_deep(**dual_common)
            if not any(c.get("kind") == "dual" and c.get("variant") != "min_deep"
                       and c.get("ffn_deep") == mc["ffn_deep"] and c.get("ffn_wide") == mc["ffn_wide"]
                       for c in configs):
                configs.append(mc)
        except ValueError as e: print(f"  ⚠ skip min-deep: {e}")
    if a.add_no_cross_alpha:
        dnc = dict(common, gate_mode=a.gate_mode, use_cross=False)
        for al in a.add_no_cross_alpha:
            try:
                cnc = build_dual(alpha=al, **dnc)
                if not any(c.get("kind") == "dual" and c.get("variant") != "min_deep"
                           and c.get("use_cross") is False and abs(c.get("alpha", -1) - al) < 1e-9
                           for c in configs):
                    configs.append(cnc)
            except ValueError as e: print(f"  ⚠ skip no-cross α={al}: {e}")

    print_summary(a.flop_budget, a.d_model, a.n_layers, a.max_loops, a.swiglu_multiple,
                  a.seq_len, causal, score, match_total, configs)
    if a.template:
        print(f"\n  Writing YAMLs to {a.output_dir}/")
        for c in configs:
            print(f"    ✓ {write_yaml(a.template, a.output_dir, c)}")

if __name__ == "__main__":
    main()