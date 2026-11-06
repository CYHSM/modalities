#!/usr/bin/env python
"""Autoregressive decoding benchmark with a KV cache, for the same configurations as bench_throughput.py.

The training model has no KV cache, so this file adds an inference-only path that reuses the trained
modules (weights, norms, RoPE tables, router, gates) and caches keys/values per attention call site:
one cache per deep-path iteration (the shared block is applied K times to different hidden states, so
each iteration has its own keys/values) and one for the wide path. A dual-path layer therefore holds
K+1 caches, a PureLoop layer K and a PureWide layer 1. The path computes only the model output (none of
the training diagnostics), as a deployed model would. `--check` verifies it against the production
forward pass: prefill + token-by-token decoding must reproduce the full-sequence logits.

Timing (bf16, one GPU): prefill a random prompt of P tokens, then decode N tokens greedily; decode
tokens/s = B*N / decode time. Two modes: `eager` (plain PyTorch, one kernel launch per op, as in HF
generate) and `graph` (the decoding step captured once as a CUDA graph and replayed, as serving stacks
do; this removes the CPU launch overhead, which dominates eager decoding at this model size). Every
configuration runs on the same GPU type, so the numbers are comparable with each other.

Usage: bench_decode.py --check
       bench_decode.py --shard I --num-shards N --out results.jsonl
       bench_decode.py --table results.jsonl.*    (LaTeX rows for the paper)
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_throughput import CONFIGS, build, logits_of

MAX_SCALE = 2.0  # AdaptiveRecursiveBlock._MAX_RESIDUAL_SCALE
SETTINGS = [  # (batch, prompt length, new tokens)
    (1, 512, 128),
    (1, 3584, 128),
    (32, 512, 128),
    (32, 3584, 128),
]


class Cache:
    """Preallocated keys/values for one attention call site."""

    def __init__(self, B, H, T, hd, device, dtype):
        self.k = torch.empty(B, H, T, hd, device=device, dtype=dtype)
        self.v = torch.empty(B, H, T, hd, device=device, dtype=dtype)


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def attn_step(attn, x, cache, pos, tables, mask=None):
    """One attention call with a KV cache.

    Eager mode (mask is None): pos is an int and the cache is sliced to the current length.
    Static mode (CUDA graphs): pos is a 1-element device tensor, keys/values are written with
    index_copy_ and attention runs over the whole buffer with a boolean mask, so every step has the
    same shapes and can be captured once and replayed.
    """
    B, t, D = x.shape
    nh = attn.n_head_q
    hd = D // nh
    q, k, v = attn.projection(x)
    q = q.view(B, t, nh, hd).transpose(1, 2)
    k = k.view(B, t, -1, hd).transpose(1, 2)
    v = v.view(B, t, -1, hd).transpose(1, 2)
    if mask is None:
        cos, sin = tables[0][:, :, pos:pos + t], tables[1][:, :, pos:pos + t]
    else:
        cos, sin = tables[0].index_select(2, pos), tables[1].index_select(2, pos)
    q, k = rope(q, cos, sin), rope(k, cos, sin)  # RotaryTransform, then qk-norm, as in training
    if attn.q_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)
    if mask is None:
        cache.k[:, :, pos:pos + t] = k
        cache.v[:, :, pos:pos + t] = v
        keys, vals = cache.k[:, :, :pos + t], cache.v[:, :, :pos + t]
        if t > 1:
            assert pos == 0, "multi-token steps only for the prefill"
        y = F.scaled_dot_product_attention(q, keys, vals, is_causal=t > 1)
    else:
        cache.k.index_copy_(2, pos, k)
        cache.v.index_copy_(2, pos, v)
        y = F.scaled_dot_product_attention(q, cache.k, cache.v, attn_mask=mask)
    return attn.c_proj(y.transpose(1, 2).reshape(B, t, D))


def block_step(blk, x, scale, cache, pos, tables, mask):
    x = x + scale * attn_step(blk.attn, blk.attention_norm(x), cache, pos, tables, mask)
    return x + scale * blk.mlp(blk.ffn_norm(x))


def layer_step(lay, x, caches, pos, tables, mask):
    """AdaptiveRecursiveBlock.forward without diagnostics (ACT loop combination, two gates)."""
    if lay.has_loop_path:
        assert lay.loop_combination_mode == "act"
        K = lay.max_loops
        h, out = x, torch.zeros_like(x)
        remain = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        for s in range(K):
            scale = F.softplus(lay.loop_scales[s]).clamp(max=MAX_SCALE)
            h = block_step(lay.block, h, scale, caches[s], pos, tables, mask)
            halt = lay.router(h, step_normalized=s / max(1, K - 1), x=x)
            out = out + h * (remain * halt).unsqueeze(-1)
            remain = remain * (1.0 - halt)
        h_deep = out + h * remain.unsqueeze(-1)
    if lay.has_wide_path:
        scale = F.softplus(lay.wide_scale).clamp(max=MAX_SCALE)
        h_wide = block_step(lay.wide_block, x, scale, caches[-1], pos, tables, mask)
    if lay.layer_type == "loop":
        return h_deep
    if lay.layer_type == "wide":
        return h_wide
    g = lay.dual_gate
    assert lay.gate_mode == "two_gates" and not g.use_cross and not g.is_softmax
    gates = torch.sigmoid(g.gate_proj(x * 0 if g.token_independent else x))
    gd, gw = gates[..., 0:1], gates[..., 1:2]
    if g.sum_cap is not None:
        damp = torch.clamp((gd + gw) / g.sum_cap, min=1.0)
        gd, gw = gd / damp, gw / damp
    if g.ungated_residual:
        return x + gd * (h_deep - x) + gw * (h_wide - x)
    return gd * h_deep + gw * h_wide


def n_caches(lay):
    return (lay.max_loops if lay.has_loop_path else 0) + (1 if lay.has_wide_path else 0)


class Decoder:
    def __init__(self, model, B, T):
        self.m = model
        self.layers = [model.transformer.h[k] for k in model._layer_order]
        p = next(model.parameters())
        lay0 = self.layers[0]
        attn = (lay0.block or lay0.wide_block).attn
        hd = attn.n_embd // attn.n_head_q
        self.caches = [[Cache(B, attn.n_head_kv, T, hd, p.device, p.dtype) for _ in range(n_caches(l))] for l in self.layers]
        rot = attn.qkv_transforms[0]
        assert type(rot).__name__ == "RotaryTransform"
        self.tables = rot._update_cos_sin_tables(torch.empty(1, 1, T, hd, device=p.device, dtype=p.dtype))
        self.arange = torch.arange(T, device=p.device)
        self.kv_bytes_per_token = sum(len(c) for c in self.caches) * 2 * attn.n_head_kv * hd * p.element_size()

    def _run(self, ids, pos, mask):
        h = self.m.transformer.wte(ids)
        for lay, caches in zip(self.layers, self.caches):
            h = layer_step(lay, h, caches, pos, self.tables, mask)
        return self.m.transformer.lm_head(self.m.transformer.lm_head_norm(h))

    def step(self, ids, pos: int):
        """Eager step (prefill or one token); pos is an int."""
        return self._run(ids, pos, None)

    def step_static(self, ids, pos_t):
        """One token with static shapes; pos_t is a 1-element long tensor on the GPU."""
        return self._run(ids, pos_t, (self.arange <= pos_t).view(1, 1, 1, -1))


@torch.no_grad()
def check() -> None:
    """Cached prefill + decoding reproduces the production full-sequence logits (fp32), eager and static."""
    torch.manual_seed(0)
    for name, kind, k, dd, dw in [c for c in CONFIGS if c[0].startswith("F80M")][:5] + [CONFIGS[9]]:
        with torch.device("cuda"):
            model = build(kind, k, dd, dw, compile_blocks=False).float()
        for p in model.parameters():  # gates and router are zero-initialised; make every path token-dependent
            p.add_(0.02 * torch.randn_like(p))
        model.train()
        T, P = 64, 48
        ids = torch.randint(0, 50304, (2, T), device="cuda")
        ref = logits_of(model({"input_ids": ids})).float()
        for mode in ("eager", "static"):
            dec = Decoder(model, 2, T)
            outs = [dec.step(ids[:, :P], 0)]
            for t in range(P, T):
                if mode == "eager":
                    outs.append(dec.step(ids[:, t:t + 1], t))
                else:
                    outs.append(dec.step_static(ids[:, t:t + 1], torch.tensor([t], device="cuda")))
            got = torch.cat(outs, dim=1)
            err = (got - ref).abs().max().item() / ref.abs().max().item()
            agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
            print(f"{name:30} {mode:6} rel max err {err:.2e}  argmax agreement {agree:.4f}", flush=True)
            assert err < 1e-4, (name, mode)
        # CUDA graph replay must generate the same greedy tokens as the static path run eagerly.
        dec = Decoder(model, 2, T)
        first = dec.step(ids[:, :P], 0)[:, -1:].argmax(-1)
        toks, nxt = [], first
        for t in range(P, T):
            nxt = dec.step_static(nxt, torch.tensor([t], device="cuda"))[:, -1:].argmax(-1)
            toks.append(nxt)
        graph, ids_t, pos_t = capture_graph(dec, 2, P)
        ids_t.copy_(first)
        pos_t.fill_(P)
        gtoks = []
        for _ in range(P, T):
            graph.replay()
            gtoks.append(ids_t.clone())
        same = torch.equal(torch.cat(toks, 1), torch.cat(gtoks, 1))
        print(f"{name:30} graph  greedy tokens identical to static eager: {same}", flush=True)
        assert same, name


def time_eager(dec, prompt, P, N):
    nxt = dec.step(prompt, 0)[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(P, P + N):
        nxt = dec.step(nxt, t)[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def capture_graph(dec, B, P):
    """Capture one greedy decoding step (model + argmax + position update) as a CUDA graph."""
    ids_t = torch.zeros(B, 1, dtype=torch.long, device="cuda")
    pos_t = torch.tensor([P], dtype=torch.long, device="cuda")

    def one():
        ids_t.copy_(dec.step_static(ids_t, pos_t)[:, -1:].argmax(-1))
        pos_t.add_(1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            one()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        one()
    return graph, ids_t, pos_t


def time_graph(dec, prompt, P, N, graph, ids_t, pos_t):
    ids_t.copy_(dec.step(prompt, 0)[:, -1:].argmax(-1))
    pos_t.fill_(P)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        graph.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def bench(name, kind, k, d_deep, d_wide) -> list[dict]:
    with torch.device("cuda"):  # initialise on the GPU; CPU init of the wide FFNs is slow
        model = build(kind, k, d_deep, d_wide, compile_blocks=False)
    model.train()  # dropout is 0; train mode as in bench_throughput
    rows = []
    for B, P, N in SETTINGS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dec = Decoder(model, B, P + N)
        prompt = torch.randint(0, 50304, (B, P), device="cuda")
        graph = capture_graph(dec, B, P)
        for mode in ("eager", "graph"):
            run = (lambda: time_eager(dec, prompt, P, N)) if mode == "eager" else \
                  (lambda: time_graph(dec, prompt, P, N, *graph))
            run()  # warm-up
            dt = statistics.median(run() for _ in range(3))
            rows.append({
                "config": name, "mode": mode, "batch": B, "prompt": P, "new_tokens": N,
                "decode_tok_s": round(B * N / dt, 1), "ms_per_step": round(1e3 * dt / N, 2),
                "kv_cache_MB_per_seq": round(dec.kv_bytes_per_token * (P + N) / 1e6, 1),
                "kv_KB_per_token": round(dec.kv_bytes_per_token / 1e3, 1),
                "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            })
        del dec, graph
    return rows


def table(paths: list[str]) -> None:
    """LaTeX rows of the paper's decoding table (CUDA-graph mode): KV cache per token and decode
    tokens/s relative to PureWide, at B=1/P=512 (latency) and B=32/P=3584 (throughput). Dual rows use
    alpha=50; the script prints the largest alpha spread so the caption can state it."""
    rows = [json.loads(l) for path in paths for l in open(path) if "error" not in l]
    by = {(r["config"], r["mode"], r["batch"], r["prompt"]): r for r in rows}
    cols = [("F80M", 1, 512), ("F80M", 32, 3584), ("F160M", 1, 512), ("F160M", 32, 3584)]
    names = [("PureWide", r"\textsc{PureWide}"), ("PureWide compute-matched", r"\textsc{PureWide} (compute-matched)"),
             ("PureLoop K2", r"\textsc{PureLoop} $K{=}2$"), ("PureLoop K3", r"\textsc{PureLoop} $K{=}3$"),
             ("PureLoop K4", r"\textsc{PureLoop} $K{=}4$"), ("PureLoop K4 compute-matched", r"\textsc{PureLoop} $K{=}4$ (compute-matched)"),
             ("Dual a50 K2", r"\textsc{Dual} $K{=}2$"), ("Dual a50 K3", r"\textsc{Dual} $K{=}3$"), ("Dual a50 K4", r"\textsc{Dual} $K{=}4$")]
    for n, label in names:
        cells = []
        for bud, b, pr in cols:
            v = by[(f"{bud} {n}", "graph", b, pr)]["decode_tok_s"]
            base = by[(f"{bud} PureWide", "graph", b, pr)]["decode_tok_s"]
            cells.append(f"{v:,.0f} ({v / base:.2f}$\\times$)".replace(",", "{,}"))
        kv = by[(f"F80M {n}", "graph", 1, 512)]["kv_KB_per_token"]
        print(f"{label} & {kv:.0f} & " + " & ".join(cells) + r" \\")
    spread = max(abs(by[(f"{bud} Dual a{a} K{k}", m, b, pr)]["decode_tok_s"] / by[(f"{bud} Dual a50 K{k}", m, b, pr)]["decode_tok_s"] - 1)
                 for bud in ("F80M", "F160M") for a in (25, 75) for k in (2, 3, 4) for m in ("graph", "eager")
                 for b, pr in [(1, 512), (1, 3584), (32, 512), (32, 3584)])
    print(f"% largest alpha=25/75 deviation from alpha=50: {100 * spread:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--table", nargs="+", metavar="JSONL")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.table:
        table(args.table)
        return
    done = set()
    if args.out and Path(args.out).exists():  # resume: skip configs that already have all settings
        rows = [json.loads(l) for l in open(args.out)]
        done = {c for c in {r["config"] for r in rows} if sum(r["config"] == c and "error" not in r for r in rows) == 2 * len(SETTINGS)}
    for i, (name, kind, k, dd, dw) in enumerate(CONFIGS):
        if i % args.num_shards != args.shard or name in done:
            continue
        try:
            rows = bench(name, kind, k, dd, dw)
        except Exception as e:  # keep going; one failing config must not lose the others
            rows = [{"config": name, "error": f"{type(e).__name__}: {e}"[:300]}]
        with open(args.out, "a") as f:
            for row in rows:
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
