#!/usr/bin/env python
"""Controlled throughput benchmark for every L=16 dual-path / single-axis configuration.

Builds each model with the production code (variant `dualpath`) from a janus config, compiles each
AdaptiveRecursiveBlock as in training, and on ONE GPU measures, in bf16 at sequence length 4096:
  train_tok_s     forward + backward of one sequence (no optimizer step), tokens per second
  prefill_ms      no-grad forward of one sequence, milliseconds
Every configuration runs on the same GPU type in the same job, so the numbers are comparable;
the cluster wall-clock in the paper's Table 8 is not (shared nodes, varying contention).

Usage: bench_throughput.py --shard I --num-shards N --out results.jsonl
"""

import argparse
import copy
import json
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from modalities.models.utils import ModelTypeEnum, get_model_from_config

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "config_files/janus/l16_sweep/8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross.yaml"
L = 16
T = 4096

# (name, kind, K, d_deep, d_wide): widths of Table 8, plus the compute-matched baselines.
CONFIGS = []
for budget, rows in {
    "F80M": [("PureWide", "wide", 1, 64, 24576), ("PureLoop K2", "loop", 2, 11392, 0), ("PureLoop K3", "loop", 3, 7104, 0),
             ("PureLoop K4", "loop", 4, 4864, 0), ("Dual a25 K2", "dual", 2, 1600, 17920), ("Dual a25 K3", "dual", 3, 576, 17920),
             ("Dual a25 K4", "dual", 4, 64, 17920), ("Dual a50 K2", "dual", 2, 4864, 11392), ("Dual a50 K3", "dual", 3, 2752, 11392),
             ("Dual a50 K4", "dual", 4, 1600, 11392), ("Dual a75 K2", "dual", 2, 8128, 4864), ("Dual a75 K3", "dual", 3, 4864, 4864),
             ("Dual a75 K4", "dual", 4, 3264, 4864), ("PureWide compute-matched", "wide", 1, 64, 32704),
             ("PureLoop K4 compute-matched", "loop", 4, 5504, 0)],
    "F160M": [("PureWide", "wide", 1, 64, 50624), ("PureLoop K2", "loop", 2, 24448, 0), ("PureLoop K3", "loop", 3, 15744, 0),
              ("PureLoop K4", "loop", 4, 11392, 0), ("Dual a25 K2", "dual", 2, 4864, 37440), ("Dual a25 K3", "dual", 3, 2752, 37440),
              ("Dual a25 K4", "dual", 4, 1600, 37440), ("Dual a50 K2", "dual", 2, 11392, 24448), ("Dual a50 K3", "dual", 3, 7104, 24448),
              ("Dual a50 K4", "dual", 4, 4864, 24448), ("Dual a75 K2", "dual", 2, 17920, 11392), ("Dual a75 K3", "dual", 3, 11392, 11392),
              ("Dual a75 K4", "dual", 4, 8128, 11392), ("PureWide compute-matched", "wide", 1, 64, 58432),
              ("PureLoop K4 compute-matched", "loop", 4, 11968, 0)],
}.items():
    CONFIGS += [(f"{budget} {n}", kind, k, dd, dw) for n, kind, k, dd, dw in rows]


def resolve(node, values: dict):
    """Replaces the few ${...} interpolations the model_raw block uses with concrete values."""
    if isinstance(node, dict):
        return {k: resolve(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve(v, values) for v in node]
    if isinstance(node, str):
        m = re.fullmatch(r"\$\{([^}]+)\}", node)
        if m:
            return values[m.group(1)]
    return node


def build(kind: str, k: int, d_deep: int, d_wide: int) -> torch.nn.Module:
    raw = yaml.safe_load(BASE_CONFIG.read_text())["model_raw"]
    raw = resolve(raw, {
        "settings.referencing_keys.sample_key": "input_ids",
        "settings.step_profile.sequence_length": T,
        "loss_fn.config.prediction_key": "logits",
        "model_raw.config.n_embd": raw["config"]["n_embd"],
        "model_raw.config.n_head_q": raw["config"]["n_head_q"],
    })
    cfg = copy.deepcopy(raw)
    cfg["config"]["use_meta_device"] = False
    cfg["config"]["ffn_hidden"] = d_deep
    ac = cfg["config"]["adaptive_config"]
    ac["max_loops"] = k
    ac["layer_types"] = [kind] * L
    ac["wide_ffn_hidden"] = d_wide
    model = get_model_from_config(config={"model": cfg}, model_type=ModelTypeEnum.MODEL)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    # Compile per block, as the training configs do (compiled_model, block_names: AdaptiveRecursiveBlock).
    for module in model.modules():
        if type(module).__name__ == "AdaptiveRecursiveBlock":
            module.compile()
    return model


def logits_of(out):
    while isinstance(out, dict):
        out = out["logits"]
    return out


def timed(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def bench(name, kind, k, d_deep, d_wide) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build(kind, k, d_deep, d_wide)
    params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, 50304, (1, T), device="cuda")
    model.train()  # train-mode forward in both measurements: eval mode attaches per-token diagnostics

    def train_step():
        loss = F.cross_entropy(logits_of(model({"input_ids": ids})).float().view(-1, 50304), ids.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    def prefill():
        with torch.no_grad():
            logits_of(model({"input_ids": ids}))

    train = timed(train_step, warmup=5, iters=20)
    pre = timed(prefill, warmup=3, iters=20)
    return {
        "config": name, "params_M": round(params / 1e6, 1),
        "train_tok_s": round(T / statistics.median(train)), "train_step_ms": round(1e3 * statistics.median(train), 1),
        "prefill_ms": round(1e3 * statistics.median(pre), 1),
        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for i, (name, kind, k, dd, dw) in enumerate(CONFIGS):
        if i % args.num_shards != args.shard:
            continue
        try:
            row = bench(name, kind, k, dd, dw)
        except Exception as e:  # keep going; one failing config must not lose the others
            row = {"config": name, "error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
