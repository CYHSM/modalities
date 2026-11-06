#!/usr/bin/env python
"""
Paloma diagnostics: runner + analysis library in one file.

Two CLI subcommands:

    python paloma_diagnostics.py run \
        --ckpt /path/to/hf_checkpoint --out-dir RUN/

    python paloma_diagnostics.py tag-pos \
        --in-dir RUN/ --ckpt /path/to/hf_checkpoint

`run` produces, for each source:
    paloma_<source>.parquet     per-token, per-layer diagnostics
    static.json                 model-level constants & learned scales

`tag-pos` produces sidecars (requires spaCy + en_core_web_sm):
    paloma_<source>_pos.parquet UPOS tag per token, aligned 1:1 with `tokens`

Importable analysis API (used by paloma_figures.py and paloma_viewer.py):
    load_source(in_dir, source)         dict of stacked (L, T) arrays
    load_static(in_dir)                 static.json as dict
    load_pos(in_dir, source)            flat list[str], aligned to tokens
    update_pref(rec) / gate_pref(rec)   (L, T) deep-share arrays
    decisive_layer(pref)                per-token decisive layer
    pos_layer_matrix(pref, pos)         (n_tags, L) mean by POS
    pos_decisive_summary(pref, pos)     per-POS decisive-layer summary
    aligned_around_anchor(...)          event-aligned trajectories

Sources by default: wikitext_103, gsm8k, triviaqa.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SOURCES = ["wikitext_103", "gsm8k", "triviaqa"]
EPS = 1e-6

VALID_UPOS = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
})


# ============================================================================
# `run` subcommand: model pass -> parquets + static.json
# ============================================================================

def load_source_texts(source: str):
    """Return list of (text, native_id). Raises on failure (no silent fallbacks)."""
    from datasets import load_dataset

    if source == "synthetic":
        return [
            ("The capital city of France is Paris", "cap_paris"),
            ("The chemical symbol for Gold is Au", "cap_gold"),
            ("The author of the novel 1984 is George Orwell", "cap_orwell"),
            ("The word 'HELLO' reversed is O L L E H", "comp_reverse"),
            ("Box A has 5. Box B has Box A. Box C has Box B. What is in Box C? 5", "comp_pointers"),
            ("Start with 2. Add 3. Multiply by 2. The result is 10", "comp_math"),
        ]
    if source == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [(f"Question: {ex['question']}\nAnswer: {ex['answer']}", f"gsm8k-{i}")
                for i, ex in enumerate(ds)]
    if source == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        out = []
        for i, ex in enumerate(ds):
            q = ex["question"]
            a = ex["answer"]["value"] if isinstance(ex["answer"], dict) else str(ex["answer"])
            out.append((f"Question: {q}\nAnswer: {a}", f"triviaqa-{i}"))
        return out
    ds = load_dataset("allenai/paloma", source, split="val")
    return [(ex.get("text", ""), str(i)) for i, ex in enumerate(ds)]


def _chunk_tokens(ids, chunk_size: int):
    T = ids.size(1)
    for start in range(0, T, chunk_size):
        yield ids[:, start:start + chunk_size]


def _to_nested(t) -> list:
    return t.float().numpy().tolist()


def run_source(model, tokenizer, source, max_docs, chunk_size, out_path, device,
               batch_size=32):
    import torch

    print(f"\n[{source}] loading dataset...", flush=True)
    try:
        docs = load_source_texts(source)
    except Exception:
        print(f"[{source}] FAILED to load:", flush=True)
        traceback.print_exc()
        return

    n_docs = min(len(docs), max_docs) if max_docs > 0 else len(docs)
    print(f"[{source}] {n_docs}/{len(docs)} docs, chunk_size={chunk_size}", flush=True)

    writer, pending, total_written = None, [], 0

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
            for chunk_idx, chunk in enumerate(_chunk_tokens(ids, chunk_size)):
                if chunk.size(1) < 2:
                    continue
                with torch.no_grad():
                    model(chunk)
                d = model.last_diagnostics
                pending.append({
                    "source": source,
                    "doc": int(doc_idx),
                    "native_id": native_id,
                    "chunk": int(chunk_idx),
                    "n_tokens": int(d["tokens"].size(1)),
                    "tokens": d["tokens"][0].numpy().tolist(),
                    "loss": d["loss"][0].float().numpy().tolist(),
                    "gate_deep": _to_nested(d["gate_deep"][:, 0]),
                    "gate_wide": _to_nested(d["gate_wide"][:, 0]),
                    "expected_steps": _to_nested(d["expected_steps"][:, 0]),
                    "delta_deep_norm": _to_nested(d["delta_deep_norm"][:, 0]),
                    "delta_wide_norm": _to_nested(d["delta_wide_norm"][:, 0]),
                    "delta_cos_sim": _to_nested(d["delta_cos_sim"][:, 0]),
                    "cross_w2d_norm": _to_nested(d["cross_w2d_norm"][:, 0]),
                    "cross_d2w_norm": _to_nested(d["cross_d2w_norm"][:, 0]),
                    "loop_displacement": _to_nested(d["loop_displacement"][:, 0]),
                    "step_halt_probs": _to_nested(d["step_halt_probs"]),
                    "step_loop_scales": _to_nested(d["step_loop_scales"]),
                    "step_displacement": _to_nested(d["step_displacement"]),
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

    if total_written == 0:
        print(f"[{source}] WARNING: no rows produced.", flush=True)
        return
    print(f"[{source}] wrote {total_written} rows to {out_path}", flush=True)


def dump_static(model, out_path: Path):
    import torch
    sp = torch.nn.functional.softplus
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
        entry = {"layer_idx": int(key),
                 "layer_type": getattr(layer, "layer_type", "plain")}
        if hasattr(layer, "wide_scale"):
            v = layer.wide_scale.detach().squeeze().cpu()
            entry["wide_scale_raw"] = float(v)
            entry["wide_scale_softplus"] = float(sp(v))
        if hasattr(layer, "loop_scales"):
            v = layer.loop_scales.detach().cpu()
            entry["loop_scales_raw"] = v.tolist()
            entry["loop_scales_softplus"] = sp(v).tolist()
        if hasattr(layer, "dual_gate") and getattr(layer.dual_gate, "use_cross", False):
            for side in ("deep", "wide"):
                v = getattr(layer.dual_gate, f"cross_scale_{side}").detach().squeeze().cpu()
                entry[f"cross_scale_{side}_raw"] = float(v)
                entry[f"cross_scale_{side}_softplus"] = float(sp(v))
        static["per_layer"].append(entry)
    with open(out_path, "w") as f:
        json.dump(static, f, indent=2)


def cmd_run(args):
    """Subcommand: model pass -> per-source parquets + static.json."""
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    chunk_size = args.chunk_size or model.config.sequence_length
    print(f"Using chunk_size={chunk_size}, sources: {args.sources}")

    dump_static(model, out_dir / "static.json")
    model.set_record_diagnostics(True)

    for source in args.sources:
        out_path = out_dir / f"paloma_{source}.parquet"
        if out_path.exists():
            print(f"[{source}] {out_path} exists, skipping")
            continue
        run_source(model, tokenizer, source, args.max_docs_per_source,
                   chunk_size, out_path, args.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


# ============================================================================
# Analysis library: loading + derived metrics
# ============================================================================

def load_source(in_dir, source: str) -> dict:
    """Load one source's parquet and stack arrays.

    Returns dict with:
        tokens          (T,)      int
        loss            (T,)      float
        gate_deep, gate_wide, expected_steps, loop_displacement,
        delta_deep_norm, delta_wide_norm, delta_cos_sim,
        cross_w2d_norm, cross_d2w_norm                       (L, T)
        step_halt_probs, step_loop_scales, step_displacement (n_chunks, L, K)
        chunk_offsets   (n_chunks+1,)
        chunk_ids       list[list[int]]
    """
    path = Path(in_dir) / f"paloma_{source}.parquet"
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"empty parquet: {path}")

    def stack3(key):
        return np.concatenate([np.asarray(r[key], dtype=np.float32) for r in rows], axis=1)

    def flat1(key):
        return np.concatenate([np.asarray(r[key], dtype=np.float32) for r in rows])

    out = {
        "source": source,
        "tokens": np.concatenate([np.asarray(r["tokens"]) for r in rows]),
        "loss": flat1("loss"),
        "gate_deep": stack3("gate_deep"),
        "gate_wide": stack3("gate_wide"),
        "expected_steps": stack3("expected_steps"),
        "loop_displacement": stack3("loop_displacement"),
        "delta_deep_norm": stack3("delta_deep_norm"),
        "delta_wide_norm": stack3("delta_wide_norm"),
        "delta_cos_sim": stack3("delta_cos_sim"),
        "cross_w2d_norm": stack3("cross_w2d_norm"),
        "cross_d2w_norm": stack3("cross_d2w_norm"),
        "step_halt_probs": np.stack([np.asarray(r["step_halt_probs"], dtype=np.float32) for r in rows]),
        "step_loop_scales": np.stack([np.asarray(r["step_loop_scales"], dtype=np.float32) for r in rows]),
        "step_displacement": np.stack([np.asarray(r["step_displacement"], dtype=np.float32) for r in rows]),
        "chunk_lens": np.array([r["n_tokens"] for r in rows]),
        "chunk_ids": [r["tokens"] for r in rows],
        "chunk_meta": [(r["doc"], r["chunk"], r.get("native_id", "")) for r in rows],
    }
    out["chunk_offsets"] = np.concatenate([[0], np.cumsum(out["chunk_lens"])])
    return out


def load_static(in_dir) -> dict:
    with open(Path(in_dir) / "static.json") as f:
        return json.load(f)


def load_pos(in_dir, source: str) -> list[str]:
    """Flat list of POS tags aligned 1:1 with tokens. Raises if sidecar missing."""
    path = Path(in_dir) / f"paloma_{source}_pos.parquet"
    rows = pq.read_table(path).to_pylist()
    out = []
    for r in rows:
        out.extend(r["pos"])
    return out


def update_pref(rec: dict) -> np.ndarray:
    """(L, T) actual update share of deep path: gd*dd / (gd*dd + gw*dw)."""
    dc = rec["gate_deep"] * rec["delta_deep_norm"]
    wc = rec["gate_wide"] * rec["delta_wide_norm"]
    denom = np.where(dc + wc < EPS, EPS, dc + wc)
    return dc / denom


def gate_pref(rec: dict) -> np.ndarray:
    """(L, T) gate-only deep share: gd / (gd + gw)."""
    gd, gw = rec["gate_deep"], rec["gate_wide"]
    denom = np.where(gd + gw < EPS, EPS, gd + gw)
    return gd / denom


def decisive_layer(pref: np.ndarray):
    """For each token, the layer maximizing |pref - 0.5|.

    Returns (layer_idx (T,), signed_pref_at_decisive_layer (T,)).
    """
    commitment = np.abs(pref - 0.5)
    L, T = pref.shape
    layer_idx = np.argmax(commitment, axis=0)
    signed = pref[layer_idx, np.arange(T)]
    return layer_idx, signed


def pos_layer_matrix(pref: np.ndarray, pos, tags=None, min_count: int = 10):
    """Mean preference per (tag, layer). Returns (M (n_tags, L), tags, counts)."""
    L, T = pref.shape
    assert len(pos) == T, f"pos len {len(pos)} != T {T}"
    pos_arr = np.array(pos)
    if tags is None:
        ctr = Counter(p for p in pos if p in VALID_UPOS)
        tags = [t for t, c in ctr.most_common() if c >= min_count]
    M = np.full((len(tags), L), np.nan, dtype=np.float32)
    counts = np.zeros(len(tags), dtype=np.int64)
    for i, t in enumerate(tags):
        mask = pos_arr == t
        n = int(mask.sum())
        counts[i] = n
        if n:
            M[i] = pref[:, mask].mean(axis=1)
    return M, tags, counts


def pos_decisive_summary(pref: np.ndarray, pos, tags=None, min_count: int = 10):
    """Per-POS: mean signed pref at decisive layer, mean decisive-layer index."""
    pos_arr = np.array(pos)
    if tags is None:
        ctr = Counter(p for p in pos if p in VALID_UPOS)
        tags = [t for t, c in ctr.most_common() if c >= min_count]
    dec_layer, signed = decisive_layer(pref)
    mean_signed = np.full(len(tags), np.nan, dtype=np.float32)
    mean_layer = np.full(len(tags), np.nan, dtype=np.float32)
    counts = np.zeros(len(tags), dtype=np.int64)
    for i, t in enumerate(tags):
        mask = pos_arr == t
        n = int(mask.sum())
        counts[i] = n
        if n:
            mean_signed[i] = signed[mask].mean()
            mean_layer[i] = dec_layer[mask].mean()
    return tags, mean_signed, mean_layer, counts


def aligned_around_anchor(rec: dict, pref: np.ndarray, tokenizer, anchor: str,
                          window: int = 25) -> np.ndarray:
    """(L, 2W+1) mean preference aligned to first token containing `anchor`."""
    L = pref.shape[0]
    W = 2 * window + 1
    sums = np.zeros((L, W), dtype=np.float64)
    counts = np.zeros(W, dtype=np.int64)
    t_cur = 0
    for chunk_ids, T in zip(rec["chunk_ids"], rec["chunk_lens"]):
        anchor_i = -1
        for i, tid in enumerate(chunk_ids):
            try:
                if anchor in tokenizer.decode([tid]):
                    anchor_i = i
                    break
            except Exception:
                pass
        if anchor_i >= 0:
            seg = pref[:, t_cur:t_cur + T]
            for i in range(T):
                offset = i - anchor_i
                if -window <= offset <= window:
                    sums[:, offset + window] += seg[:, i]
                    counts[offset + window] += 1
        t_cur += T
    out = np.full((L, W), np.nan, dtype=np.float32)
    nz = counts > 0
    out[:, nz] = (sums[:, nz] / counts[nz]).astype(np.float32)
    return out


# ============================================================================
# `tag-pos` subcommand: produce POS sidecar parquets
# ============================================================================

MATH_PATTERNS = [
    (re.compile(r"####"), "PUNCT"),
    (re.compile(r"<<"), "SYM"),
    (re.compile(r">>"), "SYM"),
    (re.compile(r"-?\d+(?:\.\d+)?%?"), "NUM"),
    (re.compile(r"[+\-*/^]"), "SYM"),
    (re.compile(r"==|<=|>=|!=|=|<|>"), "SYM"),
]


def find_math_spans(text: str):
    raw = []
    for pat, tag in MATH_PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), m.end(), tag))
    if not raw:
        return []
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last_end = [], -1
    for s, e, t in raw:
        if s < last_end:
            continue
        kept.append((s, e, t))
        last_end = e
    return kept


def _mask_for_spacy(text: str, spans):
    if not spans:
        return text
    chars = list(text)
    for s, e, _ in spans:
        for i in range(s, e):
            chars[i] = " "
    return "".join(chars)


def _decode_with_offsets(input_ids, tokenizer):
    text_parts, offsets, cur = [], [], 0
    special_ids = set(tokenizer.all_special_ids or [])
    for tid in input_ids:
        if tid in special_ids:
            offsets.append((cur, cur))
            continue
        piece = tokenizer.decode([tid], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        if piece == "":
            offsets.append((cur, cur))
            continue
        start, end = cur, cur + len(piece)
        text_parts.append(piece)
        offsets.append((start, end))
        cur = end
    return "".join(text_parts), offsets


def align_tokens_to_pos(input_ids, tokenizer, nlp, use_math_override=True):
    text, offsets = _decode_with_offsets(input_ids, tokenizer)
    if not text.strip():
        return ["SPECIAL" if s == e else "UNKNOWN" for (s, e) in offsets]

    math_spans = find_math_spans(text) if use_math_override else []
    masked = _mask_for_spacy(text, math_spans)
    doc = nlp(masked)
    word_spans = [(t.idx, t.idx + len(t), t.pos_ or "UNKNOWN")
                  for t in doc if (t.pos_ or "") in VALID_UPOS]

    out, sp_idx = [], 0
    for start, end in offsets:
        if start == end:
            out.append("SPECIAL")
            continue
        tag, best = None, 0
        for s, e, mtag in math_spans:
            if e <= start:
                continue
            if s >= end:
                break
            ov = min(end, e) - max(start, s)
            if ov > best:
                best, tag = ov, mtag
        if tag is not None:
            out.append(tag)
            continue
        while sp_idx < len(word_spans) and word_spans[sp_idx][1] <= start:
            sp_idx += 1
        best, j = 0, sp_idx
        while j < len(word_spans) and word_spans[j][0] < end:
            s, e, stag = word_spans[j]
            ov = min(end, e) - max(start, s)
            if ov > best:
                best, tag = ov, stag
            j += 1
        out.append(tag if tag is not None else "UNKNOWN")
    return out


def cmd_tag_pos(args):
    """Subcommand: produce paloma_<source>_pos.parquet sidecars."""
    from transformers import AutoTokenizer
    import spacy

    in_dir = Path(args.in_dir)
    print("Loading tokenizer and spaCy...")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True, use_fast=True)
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = 5_000_000

    parquets = sorted(p for p in in_dir.glob("paloma_*.parquet")
                      if not p.stem.endswith("_pos"))
    if args.sources:
        wanted = set(args.sources)
        parquets = [p for p in parquets if p.stem.replace("paloma_", "") in wanted]
    if not parquets:
        raise SystemExit(f"No paloma_*.parquet files in {in_dir}")

    for p in parquets:
        out_path = p.with_name(p.stem + "_pos.parquet")
        if out_path.exists():
            print(f"[{p.name}] sidecar exists, skipping")
            continue
        print(f"\n[{p.name}] tagging...")
        rows = pq.read_table(p).to_pylist()
        if args.max_chunks_per_source > 0:
            rows = rows[:args.max_chunks_per_source]

        writer, pending = None, []
        total_real, total_tokens = 0, 0
        for i, row in enumerate(rows):
            ids = row["tokens"]
            try:
                pos = align_tokens_to_pos(ids, tokenizer, nlp,
                                          use_math_override=not args.no_math_override)
            except Exception as e:
                print(f"  chunk {i}: {e}")
                pos = ["UNKNOWN"] * len(ids)
            total_real += sum(1 for x in pos if x in VALID_UPOS)
            total_tokens += len(pos)
            pending.append({"doc": int(row["doc"]),
                            "chunk": int(row["chunk"]),
                            "pos": pos})
            if len(pending) >= 32:
                t = pa.Table.from_pylist(pending)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
                writer.write_table(t)
                pending = []
            if (i + 1) % 50 == 0:
                cov = 100 * total_real / max(total_tokens, 1)
                print(f"  {i+1}/{len(rows)} chunks, coverage {cov:.1f}%")
        if pending:
            t = pa.Table.from_pylist(pending)
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
            writer.write_table(t)
        if writer is not None:
            writer.close()
        print(f"[{p.name}] coverage {100*total_real/max(total_tokens,1):.1f}%")

    print("\ntag-pos done.")


# ============================================================================
# CLI dispatch
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Paloma diagnostics: runner + analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run model and dump per-token diagnostics")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    r.add_argument("--max-docs-per-source", type=int, default=200)
    r.add_argument("--chunk-size", type=int, default=None)
    r.add_argument("--device", default="cuda")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("tag-pos", help="Produce POS sidecar parquets")
    t.add_argument("--in-dir", required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--spacy-model", default="en_core_web_sm")
    t.add_argument("--sources", nargs="*", default=None)
    t.add_argument("--max-chunks-per-source", type=int, default=200)
    t.add_argument("--no-math-override", action="store_true")
    t.set_defaults(func=cmd_tag_pos)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()