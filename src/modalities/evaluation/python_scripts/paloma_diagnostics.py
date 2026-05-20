#!/usr/bin/env python
"""
Run a checkpoint over a small set of eval sources and dump per-token
diagnostics to parquet.

Sources (intentionally minimal — three regimes to compare):
    wikitext_103  : general English text (Paloma's natural-language baseline)
    gsm8k         : math word problems with chain-of-thought (reasoning-heavy)
    triviaqa      : Q/A pairs with no context (world-knowledge recall)

Usage:
    python paloma_diagnostics.py \
        --ckpt /path/to/hf/checkpoint \
        --out-dir /path/to/output \
        --max-docs-per-source 200

One parquet file per source is produced under <out-dir>/paloma_<source>.parquet,
plus a static.json with per-layer parameter values that don't vary by chunk.

See the original docstring for the full parquet schema. It's unchanged.
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default sources. Wikitext is loaded via Paloma; gsm8k & triviaqa from
# their own datasets so we don't depend on them being part of Paloma val.
DEFAULT_SOURCES = ["wikitext_103", "gsm8k", "triviaqa"]


def load_source_texts(source: str):
    """Return a list of (text, native_id) tuples for the given source.

    Raises on failure (caller logs traceback). No silent fallbacks here —
    silent fallbacks are why gsm8k/triviaqa were missing from the viewer.
    """
    if source == "gsm8k":
        # Test split: ~1300 examples. Each has a question and a chain-of-thought
        # answer ending in "#### <number>".
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            (f"Question: {ex['question']}\nAnswer: {ex['answer']}", f"gsm8k-{i}")
            for i, ex in enumerate(ds)
        ]

    if source == "triviaqa":
        # rc.nocontext: pure Q/A pairs, isolating "world knowledge recall"
        # without reading-comprehension confound.
        #ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        out = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            a = ex["answer"]["value"] if isinstance(ex["answer"], dict) else str(ex["answer"])
            out.append((f"Question: {q}\nAnswer: {a}", f"triviaqa-{i}"))
        return out

    # Default: native Paloma source (wikitext_103 is the main one we still use).
    ds = load_dataset("allenai/paloma", source, split="val")
    return [(ex.get("text", ""), str(i)) for i, ex in enumerate(ds)]


def chunk_tokens(ids: torch.Tensor, chunk_size: int):
    """Yield non-overlapping chunks of (1, chunk_size). Last chunk may be shorter."""
    T = ids.size(1)
    for start in range(0, T, chunk_size):
        yield ids[:, start:start + chunk_size]


def to_nested_list(t: torch.Tensor) -> list:
    """Convert a 2D tensor (L, T) or (L, max_loops) to list-of-lists for parquet."""
    return t.float().numpy().tolist()


def run_source(model, tokenizer, source: str, max_docs: int, chunk_size: int,
               out_path: Path, device: str, batch_size: int = 32):
    """Load one source, run model, stream rows to parquet in batches."""
    print(f"\n[{source}] loading dataset...", flush=True)
    try:
        docs = load_source_texts(source)
    except Exception:
        # Print full traceback so we actually see WHY it failed (offline cache miss,
        # wrong dataset name, etc.) instead of silently skipping.
        print(f"[{source}] FAILED to load:", flush=True)
        traceback.print_exc()
        return

    n_docs = min(len(docs), max_docs) if max_docs > 0 else len(docs)
    print(f"[{source}] {n_docs} docs (of {len(docs)} available), "
          f"chunk_size={chunk_size}", flush=True)

    writer = None
    pending = []
    total_written = 0

    def flush():
        nonlocal pending, total_written, writer
        if not pending:
            return
        table = pa.Table.from_pylist(pending)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        total_written += len(pending)
        pending = []

    try:
        for doc_idx in range(n_docs):
            text, native_id = docs[doc_idx]
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            if ids.size(1) < 2:
                continue
            for chunk_idx, chunk in enumerate(chunk_tokens(ids, chunk_size)):
                if chunk.size(1) < 2:
                    continue
                with torch.no_grad():
                    model(chunk)
                d = model.last_diagnostics
                T = d["tokens"].size(1)
                pending.append({
                    "source": source,
                    "doc": int(doc_idx),
                    "native_id": native_id,
                    "chunk": int(chunk_idx),
                    "n_tokens": int(T),
                    "tokens": d["tokens"][0].numpy().tolist(),
                    "loss": d["loss"][0].float().numpy().tolist(),
                    "gate_deep": to_nested_list(d["gate_deep"][:, 0]),
                    "gate_wide": to_nested_list(d["gate_wide"][:, 0]),
                    "expected_steps": to_nested_list(d["expected_steps"][:, 0]),
                    "delta_deep_norm": to_nested_list(d["delta_deep_norm"][:, 0]),
                    "delta_wide_norm": to_nested_list(d["delta_wide_norm"][:, 0]),
                    "delta_cos_sim": to_nested_list(d["delta_cos_sim"][:, 0]),
                    "cross_w2d_norm": to_nested_list(d["cross_w2d_norm"][:, 0]),
                    "cross_d2w_norm": to_nested_list(d["cross_d2w_norm"][:, 0]),
                    "loop_displacement": to_nested_list(d["loop_displacement"][:, 0]),
                    "step_halt_probs": to_nested_list(d["step_halt_probs"]),
                    "step_loop_scales": to_nested_list(d["step_loop_scales"]),
                    "step_displacement": to_nested_list(d["step_displacement"]),
                })
                model.last_diagnostics = None
                if len(pending) >= batch_size:
                    flush()
            if (doc_idx + 1) % 50 == 0:
                print(f"[{source}]   {doc_idx + 1}/{n_docs} docs, "
                      f"{total_written + len(pending)} chunks", flush=True)
        flush()
    finally:
        if writer is not None:
            writer.close()
        del docs

    if total_written == 0:
        print(f"[{source}] WARNING: no rows produced.", flush=True)
        return
    print(f"[{source}] wrote {total_written} rows to {out_path}", flush=True)


def dump_static(model, out_path: Path):
    """Dump model-level constants that don't change across the run."""
    static = {
        "n_layer": model.config.n_layer,
        "max_loops": model.config.max_loops,
        "gate_mode": model.config.gate_mode,
        "use_cross": model.config.use_cross,
        "adaptive_layer_types": model.config.adaptive_layer_types,
        "wide_ffn_hidden": model.config.wide_ffn_hidden,
        "per_layer": [],
    }
    for key in model.transformer._layer_order:
        layer = model.transformer.h[key]
        entry = {"layer_idx": int(key), "layer_type": getattr(layer, "layer_type", "plain")}
        if hasattr(layer, "wide_scale"):
            entry["wide_scale_raw"] = float(layer.wide_scale.detach().squeeze().cpu())
            entry["wide_scale_softplus"] = float(
                torch.nn.functional.softplus(layer.wide_scale.detach().squeeze().cpu())
            )
        if hasattr(layer, "loop_scales"):
            entry["loop_scales_raw"] = layer.loop_scales.detach().cpu().tolist()
            entry["loop_scales_softplus"] = (
                torch.nn.functional.softplus(layer.loop_scales.detach().cpu()).tolist()
            )
        if hasattr(layer, "dual_gate") and getattr(layer.dual_gate, "use_cross", False):
            entry["cross_scale_deep_raw"] = float(layer.dual_gate.cross_scale_deep.detach().squeeze().cpu())
            entry["cross_scale_wide_raw"] = float(layer.dual_gate.cross_scale_wide.detach().squeeze().cpu())
            entry["cross_scale_deep_softplus"] = float(
                torch.nn.functional.softplus(layer.dual_gate.cross_scale_deep.detach().squeeze().cpu())
            )
            entry["cross_scale_wide_softplus"] = float(
                torch.nn.functional.softplus(layer.dual_gate.cross_scale_wide.detach().squeeze().cpu())
            )
        static["per_layer"].append(entry)
    with open(out_path, "w") as f:
        json.dump(static, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir or hub id")
    ap.add_argument("--out-dir", required=True, help="Where to write parquets")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                    help="Sources to evaluate (default: wikitext_103 gsm8k triviaqa)")
    ap.add_argument("--max-docs-per-source", type=int, default=200,
                    help="Cap docs per source (0 = all)")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="Token chunk size (defaults to model.config.sequence_length)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    chunk_size = args.chunk_size or model.config.sequence_length
    print(f"Using chunk_size={chunk_size}")
    print(f"Sources: {args.sources}")

    dump_static(model, out_dir / "static.json")
    model.set_record_diagnostics(True)

    for source in args.sources:
        out_path = out_dir / f"paloma_{source}.parquet"
        if out_path.exists():
            print(f"[{source}] {out_path} exists, skipping")
            continue
        run_source(model, tokenizer, source, args.max_docs_per_source,
                   chunk_size, out_path, args.device)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()