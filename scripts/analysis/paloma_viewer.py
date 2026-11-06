#!/usr/bin/env python
"""
Generate a single-file interactive HTML viewer from paloma_diagnostics parquets.

This version includes:
  - Event-Aligned Trajectories (§1b) with dynamic anchor selection.
  - Cross-Source Aligned Difference (§1c).
  - Before vs. After Pivot Slopegraphs (§1d) - Summary and Per-Layer.
  - The "Sequence Routing Grid" (§4b) heatmap.
  - The cross-layer "mean" view in the token explorer.
"""

import argparse
import base64
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pyarrow.parquet as pq


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def _stack_3d(rows, key):
    if not rows:
        return None
    arrs = [np.asarray(r[key], dtype=np.float32) for r in rows]
    return np.concatenate(arrs, axis=1)


def _flat_1d(rows, key):
    if not rows:
        return None
    arrs = [np.asarray(r[key], dtype=np.float32) for r in rows]
    return np.concatenate(arrs)


def _safe_mean(arr, axis=None):
    if arr is None or arr.size == 0:
        return None
    m = np.nanmean(arr, axis=axis)
    if isinstance(m, np.ndarray):
        return m.tolist()
    return float(m)


def _hist(arr, bins, range_):
    if arr is None or arr.size == 0:
        return {"counts": [], "edges": []}
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=bins, range=range_)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def aggregate_source(rows, n_layer, max_loops, pos_map=None, source_name="?", tokenizer=None, align_strings=None):
    """Build the per-source aggregate payload."""
    if align_strings is None:
        align_strings = ["Answer"]

    gd = _stack_3d(rows, "gate_deep")        # (L, total_T)
    gw = _stack_3d(rows, "gate_wide")
    es = _stack_3d(rows, "expected_steps")
    dd = _stack_3d(rows, "delta_deep_norm")
    dw = _stack_3d(rows, "delta_wide_norm")
    ld = _stack_3d(rows, "loop_displacement")
    loss = _flat_1d(rows, "loss")

    if gd is not None and gw is not None:
        denom = (gd + gw)
        denom = np.where(denom < 1e-6, 1e-6, denom)
        gate_pref = gd / denom
    else:
        gate_pref = None

    if gd is not None and dd is not None:
        d_c = gd * dd
        w_c = gw * dw
        denom = d_c + w_c
        denom = np.where(denom < 1e-6, 1e-6, denom)
        update_pref = d_c / denom
    else:
        update_pref = None

    halt = np.stack([np.array(r["step_halt_probs"], dtype=np.float32) for r in rows])
    halt_mean = halt.mean(axis=0)
    step_disp = np.stack([np.array(r["step_displacement"], dtype=np.float32) for r in rows]).mean(axis=0)

    summary = {
        "n_chunks": len(rows),
        "n_tokens": int(gd.shape[1]) if gd is not None else 0,
        "mean_gate_deep": _safe_mean(gd),
        "mean_gate_wide": _safe_mean(gw),
        "mean_gate_pref": _safe_mean(gate_pref),
        "mean_update_pref": _safe_mean(update_pref),
        "mean_expected_steps": _safe_mean(es),
        "mean_loop_displacement": _safe_mean(ld),
        "mean_loss": float(np.nanmean(loss)) if loss is not None else None,
    }

    per_layer = {
        "gate_deep": _safe_mean(gd, axis=1),
        "gate_wide": _safe_mean(gw, axis=1),
        "gate_pref": _safe_mean(gate_pref, axis=1),
        "update_pref": _safe_mean(update_pref, axis=1),
        "expected_steps": _safe_mean(es, axis=1),
        "loop_displacement": _safe_mean(ld, axis=1),
        "halt_probs": halt_mean.tolist(),
        "step_displacement": step_disp.tolist(),
    }

    hist = {
        "gate_pref":   _hist(gate_pref.ravel()   if gate_pref   is not None else None, 40, (0.0, 1.0)),
        "update_pref": _hist(update_pref.ravel() if update_pref is not None else None, 40, (0.0, 1.0)),
        "gate_deep":   _hist(gd.ravel() if gd is not None else None, 40, (0.0, 1.0)),
        "gate_wide":   _hist(gw.ravel() if gw is not None else None, 40, (0.0, 1.0)),
        "expected_steps": _hist(es.ravel() if es is not None else None, 40, (0.0, float(max_loops))),
        "loss": _hist(loss, 40, (0.0, 15.0)),
    }

    # ---- Event-Aligned Trajectories (Tracking Tokens + Trajectories) ----
    aligned_data = {}
    if tokenizer is not None and update_pref is not None:
        WINDOW = 25 # Look 25 tokens before and after
        
        for anchor in align_strings:
            aligned_sums = {l: {offset: 0.0 for offset in range(-WINDOW, WINDOW+1)} for l in range(n_layer)}
            aligned_counts = {l: {offset: 0 for offset in range(-WINDOW, WINDOW+1)} for l in range(n_layer)}
            aligned_samples = {l: [] for l in range(n_layer)}
            token_counters = {offset: Counter() for offset in range(-WINDOW, WINDOW+1)}
            
            MAX_SAMPLES = 50
            current_t = 0
            
            for chunk_idx, r in enumerate(rows):
                ids = r["tokens"]
                T = len(ids)
                
                # Find the first token containing the anchor string
                anchor_idx = -1
                for i, tid in enumerate(ids):
                    try:
                        token_str = tokenizer.decode([tid])
                        if anchor in token_str:
                            anchor_idx = i
                            break
                    except:
                        pass
                
                if anchor_idx != -1:
                    chunk_upref = update_pref[:, current_t : current_t + T]
                    
                    # Track actual string values at each offset
                    for i, tid in enumerate(ids):
                        offset = i - anchor_idx
                        if -WINDOW <= offset <= WINDOW:
                            try:
                                tstr = tokenizer.decode([tid]).replace('\n', '↵')
                                token_counters[offset][tstr] += 1
                            except:
                                pass
                            
                    # Track numerical trajectories
                    for l in range(n_layer):
                        sample_arr = [None] * (2 * WINDOW + 1)
                        has_data = False
                        for i in range(T):
                            offset = i - anchor_idx
                            if -WINDOW <= offset <= WINDOW:
                                val = float(chunk_upref[l, i])
                                aligned_sums[l][offset] += val
                                aligned_counts[l][offset] += 1
                                sample_arr[offset + WINDOW] = val
                                has_data = True
                        if has_data and len(aligned_samples[l]) < MAX_SAMPLES:
                            aligned_samples[l].append([round(v, 3) if v is not None else None for v in sample_arr])
                current_t += T
                
            # Finalize means
            aligned_means = []
            for l in range(n_layer):
                layer_means = []
                for offset in range(-WINDOW, WINDOW+1):
                    c = aligned_counts[l][offset]
                    if c >= 3:
                        layer_means.append(round(aligned_sums[l][offset] / c, 3))
                    else:
                        layer_means.append(None)
                aligned_means.append(layer_means)
                
            # Finalize token frequency strings for hover
            top_tokens_list = []
            for offset in range(-WINDOW, WINDOW+1):
                c = token_counters[offset]
                if not c:
                    top_tokens_list.append("n/a")
                else:
                    total = sum(c.values())
                    top = c.most_common(4)
                    top_str = ", ".join([f"'{k}' ({int(100*v/total)}%)" for k, v in top])
                    top_tokens_list.append(top_str)
                    
            aligned_data[anchor] = {
                "means": aligned_means,
                "samples": aligned_samples,
                "top_tokens": top_tokens_list
            }

    # ---- POS-grouped aggregates ----
    pos_agg = None
    if pos_map is not None:
        pos_flat = []
        for r in rows:
            key = (int(r["doc"]), int(r["chunk"]))
            tags = pos_map.get(key)
            T = r["n_tokens"]
            if tags is None or len(tags) != T:
                pos_flat.extend(["UNKNOWN"] * T)
            else:
                pos_flat.extend(tags)
        pos_arr = np.array(pos_flat)

        pos_for_split = pos_arr.copy()
        punct_mask = np.isin(pos_for_split, ["PUNCT", "SYM", "SPECIAL"])
        if len(pos_for_split) > 0:
            change = np.empty(len(pos_for_split), dtype=bool)
            change[0] = True
            change[1:] = (pos_for_split[1:] != pos_for_split[:-1]) | punct_mask[1:] | punct_mask[:-1]
            starts_arr = np.flatnonzero(change)
            ends_arr = np.empty_like(starts_arr)
            ends_arr[:-1] = starts_arr[1:]
            ends_arr[-1] = len(pos_for_split)
            tags_per_group = pos_for_split[starts_arr]
            keep = (tags_per_group != "SPECIAL")
            starts_arr = starts_arr[keep]
            ends_arr = ends_arr[keep]
            tags_per_group = tags_per_group[keep]
        else:
            starts_arr = np.array([], dtype=np.int64)
            ends_arr = np.array([], dtype=np.int64)
            tags_per_group = np.array([], dtype=object)

        layered_metrics_full = {
            "gate_pref":         gate_pref,
            "update_pref":       update_pref,
            "expected_steps":    es,
            "loop_displacement": ld,
            "gate_deep":         gd,
            "gate_wide":         gw,
        }

        n_layers_actual = gd.shape[0] if gd is not None else n_layer
        lengths = (ends_arr - starts_arr).astype(np.float32) if len(starts_arr) > 0 else np.array([])

        def per_word_reduce(flat_1d):
            if len(starts_arr) == 0:
                return np.array([]), np.array([])
            safe = np.where(np.isfinite(flat_1d), flat_1d, 0.0)
            sums = np.add.reduceat(safe, starts_arr)
            means = sums / np.maximum(lengths, 1.0)
            neg_inf = np.where(np.isfinite(flat_1d), flat_1d, -np.inf)
            maxes = np.maximum.reduceat(neg_inf, starts_arr)
            maxes = np.where(np.isinf(maxes), np.nan, maxes)
            return means.astype(np.float32), maxes.astype(np.float32)

        unique_tags = sorted(set(tags_per_group.tolist())) if len(tags_per_group) else []
        by_pos_layer_mean = {}
        by_pos_layer_max = {}
        for metric_name, full_arr in layered_metrics_full.items():
            per_tag_layers_mean = {tag: [None] * n_layers_actual for tag in unique_tags}
            per_tag_layers_max  = {tag: [None] * n_layers_actual for tag in unique_tags}
            for li in range(n_layers_actual):
                layer_vals = full_arr[li]
                word_means, word_maxes = per_word_reduce(layer_vals)
                for tag in unique_tags:
                    mask = (tags_per_group == tag)
                    if mask.sum() == 0:
                        continue
                    with np.errstate(all="ignore"):
                        per_tag_layers_mean[tag][li] = float(np.nanmean(word_means[mask]))
                        per_tag_layers_max [tag][li] = float(np.nanmean(word_maxes[mask]))
            by_pos_layer_mean[metric_name] = per_tag_layers_mean
            by_pos_layer_max [metric_name] = per_tag_layers_max

        decisive = {} 
        if update_pref is not None and len(starts_arr) > 0:
            per_word_per_layer = np.empty((n_layers_actual, len(starts_arr)), dtype=np.float32)
            for li in range(n_layers_actual):
                wm, _ = per_word_reduce(update_pref[li])
                per_word_per_layer[li] = wm
            score = np.abs(per_word_per_layer - 0.5)
            score = np.where(np.isfinite(score), score, -np.inf)
            decisive_layer_per_word = score.argmax(axis=0) 
            wcols = np.arange(per_word_per_layer.shape[1])
            pref_at_decisive = per_word_per_layer[decisive_layer_per_word, wcols]
            for tag in unique_tags:
                mask = (tags_per_group == tag)
                n = int(mask.sum())
                if n == 0:
                    continue
                with np.errstate(all="ignore"):
                    decisive[tag] = {
                        "mean_pref_at_decisive": float(np.nanmean(pref_at_decisive[mask])),
                        "mean_decisive_layer":   float(np.nanmean(decisive_layer_per_word[mask])),
                        "n_words": n,
                    }

        coverage = float(np.mean(np.isin(pos_arr, ["UNKNOWN", "SPECIAL"], invert=True)))

        n_words_per_pos = {}
        if len(tags_per_group) > 0:
            for tag in unique_tags:
                n_words_per_pos[tag] = int((tags_per_group == tag).sum())

        pos_agg = {
            "coverage": coverage,
            "by_pos_layer_mean": by_pos_layer_mean,
            "by_pos_layer_max":  by_pos_layer_max,
            "n_words_per_pos": n_words_per_pos,
            "decisive": decisive,
        }

    return {
        "summary": summary,
        "per_layer": per_layer,
        "hist": hist,
        "pos": pos_agg,
        "aligned": aligned_data,
    }


def sample_chunks_for_explorer(rows, n_sample, rng):
    if len(rows) <= n_sample:
        return rows
    idxs = rng.sample(range(len(rows)), n_sample)
    return [rows[i] for i in idxs]


def decode_tokens(token_ids, tokenizer):
    if tokenizer is None:
        return [str(t) for t in token_ids]
    out = []
    for tid in token_ids:
        try:
            s = tokenizer.decode([tid])
        except Exception:
            s = f"<{tid}>"
        s = s.replace("\n", "↵\n").replace("\t", "→")
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-explorer-chunks", type=int, default=1)
    ap.add_argument("--n-explorer-tokens", type=int, default=128)
    ap.add_argument("--skip-sources", nargs="*", default=None)
    ap.add_argument("--max-chunks-per-source", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    
    # Allow user to define alignment strings from command line
    ap.add_argument("--align-strings", nargs="+", default=["Answer", "Question", ":", "####"])
    
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    static_path = in_dir / "static.json"
    if not static_path.exists():
        raise SystemExit(f"static.json not found at {static_path}")
    static = json.loads(static_path.read_text())

    n_layer = static["n_layer"]
    max_loops = static["max_loops"]
    gate_mode = static["gate_mode"]
    use_cross = static["use_cross"]

    tokenizer = None
    if args.ckpt:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: couldn't load tokenizer from {args.ckpt}: {e}")

    rng = random.Random(args.seed)

    sources_data = {}
    parquet_files = sorted(in_dir.glob("paloma_*.parquet"))
    parquet_files = [p for p in parquet_files if not p.stem.endswith("_pos")]
    if args.skip_sources:
        skip = set(args.skip_sources)
        parquet_files = [p for p in parquet_files if p.stem.replace("paloma_", "") not in skip]
    if not parquet_files:
        raise SystemExit(f"No paloma_*.parquet files in {in_dir}")

    import time

    for pq_path in parquet_files:
        source = pq_path.stem.replace("paloma_", "")
        t_src = time.time()
        print(f"[{source}] loading parquet...", flush=True)
        table = pq.read_table(pq_path)
        rows = table.to_pylist()
        if args.max_chunks_per_source > 0 and len(rows) > args.max_chunks_per_source:
            rows = rows[:args.max_chunks_per_source]
        print(f"[{source}]   read {len(rows)} chunks", flush=True)
        if not rows:
            continue

        pos_path = pq_path.with_name(pq_path.stem + "_pos.parquet")
        pos_map = {}
        if pos_path.exists():
            pos_table = pq.read_table(pos_path).to_pylist()
            for r in pos_table:
                pos_map[(int(r["doc"]), int(r["chunk"]))] = r["pos"]
            print(f"[{source}]   loaded POS sidecar ({len(pos_map)} entries)", flush=True)
        else:
            print(f"[{source}]   no POS sidecar found", flush=True)

        agg = aggregate_source(rows, n_layer, max_loops, pos_map=pos_map,
                               source_name=source, tokenizer=tokenizer, 
                               align_strings=args.align_strings)

        sample = sample_chunks_for_explorer(rows, args.n_explorer_chunks, rng)

        N_TOK = args.n_explorer_tokens
        per_layer_keys = ("gate_deep", "gate_wide", "expected_steps",
                          "delta_deep_norm", "delta_wide_norm",
                          "loop_displacement")
        per_tok_keys = ("loss", "tokens")

        for chunk in sample:
            T = chunk["n_tokens"]
            if T <= N_TOK:
                start = 0
                end = T
            else:
                start = rng.randint(0, T - N_TOK)
                end = start + N_TOK
            chunk["n_tokens"] = end - start
            chunk["window_start"] = start
            chunk["window_end"] = end
            for k in per_tok_keys:
                chunk[k] = chunk[k][start:end]
            for k in per_layer_keys:
                chunk[k] = [row[start:end] for row in chunk[k]]

            chunk["token_strs"] = decode_tokens(chunk["tokens"], tokenizer)

            key = (int(chunk["doc"]), int(chunk["chunk"]))
            if key in pos_map:
                full_pos = pos_map[key]
                chunk["pos"] = full_pos[start:end]

            for k in ("loss",) + per_layer_keys:
                val = chunk[k]
                if isinstance(val[0], list):
                    chunk[k] = [[round(x, 3) if x is not None and np.isfinite(x) else None for x in row] for row in val]
                else:
                    chunk[k] = [round(x, 3) if x is not None and np.isfinite(x) else None for x in val]
            
            chunk.pop("step_halt_probs", None)
            chunk.pop("step_loop_scales", None)
            chunk.pop("step_displacement", None)
            chunk.pop("cross_w2d_norm", None)
            chunk.pop("cross_d2w_norm", None)
            chunk.pop("delta_cos_sim", None)

        sources_data[source] = {
            "agg": agg,
            "chunks": sample,
            "has_pos": bool(pos_map),
        }

        del rows, pos_map
        import gc; gc.collect()

        print(f"[{source}] total {time.time()-t_src:.1f}s", flush=True)

    print("\nAll sources processed. Serializing payload...", flush=True)

    payload = {
        "static": static,
        "sources": sources_data,
        "meta": {
            "n_layer": n_layer,
            "max_loops": max_loops,
            "gate_mode": gate_mode,
            "use_cross": use_cross,
            "source_order": sorted(sources_data.keys()),
        },
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    print(f"Payload size: {len(payload_json) / 1e6:.1f} MB", flush=True)
    payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
    html = HTML_TEMPLATE.replace("__PAYLOAD_B64__", payload_b64)
    Path(args.out).write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1e6:.1f} MB)", flush=True)


# ----------------------------------------------------------------------------
# HTML template
# ----------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Paloma diagnostics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #f6f3ec;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --line: #c8c0ad;
  --accent: #b6533a;
  --deep: #2a4d6e;
  --wide: #b6533a;
  --panel: #efeadd;
  --hover: #e3dcc8;
  --mono: "IBM Plex Mono", "JetBrains Mono", ui-monospace, Menlo, monospace;
  --serif: "Iowan Old Style", "Source Serif Pro", Georgia, serif;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--serif);
  font-size: 15px;
  line-height: 1.5;
}
header {
  border-bottom: 1px solid var(--line);
  padding: 18px 32px 12px;
  display: flex; align-items: baseline; justify-content: space-between;
  flex-wrap: wrap; gap: 16px;
}
header h1 {
  font-family: var(--mono);
  font-weight: 600;
  font-size: 18px;
  margin: 0;
  letter-spacing: -0.01em;
}
header .meta {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
}
header .meta span { margin-right: 16px; }
main { padding: 0 32px 64px; max-width: 1400px; margin: 0 auto; }
section {
  border-bottom: 1px dashed var(--line);
  padding: 28px 0;
}
.subsection {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px dotted var(--line);
}
.subsection:first-of-type { border-top: none; margin-top: 12px; padding-top: 0; }
section h2 {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 4px;
}
section h2 .num { color: var(--accent); margin-right: 8px; }
section h3 {
  font-family: var(--serif);
  font-weight: 400;
  font-style: italic;
  font-size: 22px;
  margin: 0 0 20px;
  letter-spacing: -0.01em;
}
.subsection h4 {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin: 0 0 6px;
}
.subsection h5 {
  font-family: var(--serif);
  font-weight: 400;
  font-style: italic;
  font-size: 18px;
  margin: 0 0 14px;
}
.controls {
  display: flex; gap: 16px; align-items: center;
  margin: 12px 0 20px;
  font-family: var(--mono);
  font-size: 13px;
  flex-wrap: wrap;
}
.controls label { color: var(--muted); }
.controls select, .controls input {
  font-family: var(--mono);
  font-size: 13px;
  padding: 4px 8px;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--fg);
  border-radius: 0;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.chart { height: 320px; }
.chart-tall { height: 420px; }
.chart-heatmap { height: 520px; }
.token-text {
  font-family: var(--mono);
  font-size: 14px;
  line-height: 2.0;
  padding: 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 480px;
  overflow-y: auto;
}
.tok { padding: 1px 1px; border-radius: 2px; cursor: pointer; transition: outline 0.05s; }
.tok:hover, .tok.selected { outline: 1.5px solid var(--accent); outline-offset: 0px; }
.detail-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 16px;
  font-family: var(--mono);
  font-size: 12px;
  min-height: 200px;
}
.detail-panel h4 {
  margin: 0 0 8px; font-size: 12px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
}
.detail-panel table { width: 100%; border-collapse: collapse; font-size: 11px; }
.detail-panel th, .detail-panel td {
  text-align: right; padding: 2px 6px;
  border-bottom: 1px solid rgba(0,0,0,0.05);
}
.detail-panel th { color: var(--muted); font-weight: 500; }
.detail-panel td:first-child, .detail-panel th:first-child {
  text-align: left; color: var(--muted);
}
.legend {
  display: inline-flex; gap: 16px; align-items: center;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  margin-left: 20px;
}
.legend .swatch {
  display: inline-block; width: 60px; height: 10px; vertical-align: middle;
  border: 1px solid var(--line);
}
.kpis {
  display: flex; gap: 36px; flex-wrap: wrap;
  font-family: var(--mono);
  margin: 0 0 12px;
}
.kpi { display: flex; flex-direction: column; }
.kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .val { font-size: 22px; font-weight: 500; }
.kpi .sub { color: var(--muted); font-size: 11px; }
.token-explorer-layout {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 20px;
}
@media (max-width: 1000px) {
  .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
  .token-explorer-layout { grid-template-columns: 1fr; }
}
.note { font-style: italic; color: var(--muted); font-size: 13px; margin: 4px 0 16px; }
.glossary {
  background: var(--panel); border-left: 3px solid var(--accent);
  padding: 12px 16px; font-size: 13px; margin: 0 0 24px;
}
.glossary code { font-family: var(--mono); font-size: 12px; }
</style>
</head>
<body>

<header>
  <h1>paloma diagnostics</h1>
  <div class="meta" id="model-meta"></div>
</header>

<main>

<div class="glossary">
  <strong>Two preference metrics:</strong>
  <code>gate_pref = g_d / (g_d + g_w)</code> &nbsp;— what the router intended (gate-only).
  <code>update_pref = g_d·‖Δd‖ / (g_d·‖Δd‖ + g_w·‖Δw‖)</code> &nbsp;— what the residual
  stream actually got (accounts for both gate openness and path output magnitude).
  Both range [0, 1]; 0 = wide, 1 = deep, 0.5 = balanced.
</div>

<section>
  <h2><span class="num">§1</span> cross-source overview</h2>
  <h3>How does the model route across Paloma sources?</h3>
  <p class="note">Each bar is the mean over all tokens in that source, all layers, all chunks.</p>
  <div class="grid-3">
    <div id="overview-update-pref" class="chart"></div>
    <div id="overview-steps" class="chart"></div>
    <div id="overview-loss" class="chart"></div>
  </div>
</section>

<section>
  <h2><span class="num">§1b</span> event-aligned trajectories</h2>
  <h3>How does routing shift before, during, and after a specific string?</h3>
  <div class="controls">
    <label>source</label> <select id="source-select-aligned"></select>
    <label>align to</label> <select id="anchor-select"></select>
    <label><input type="checkbox" id="show-samples-toggle"> show individual sample lines</label>
  </div>
  <div id="aligned-plot" class="chart-tall"></div>
  <p class="note">X-axis is relative position to the alignment token (0 = match). Hover over any point on the line to see the <strong>actual tokens</strong> most frequently processed at that offset.</p>
</section>

<section>
  <h2><span class="num">§1c</span> cross-source aligned difference</h2>
  <h3>Isolating cognitive bottlenecks by subtracting trajectories (A - B).</h3>
  <div class="controls">
    <label>source A (positive)</label> <select id="source-diff-a"></select>
    <label>source B (negative)</label> <select id="source-diff-b"></select>
    <label>align to</label> <select id="anchor-select-diff"></select>
  </div>
  <div id="aligned-diff-plot" class="chart-tall"></div>
  <p class="note">Y-axis shows <code>update_pref(A) - update_pref(B)</code>. >0 means Source A uses more Deep path. <0 means Source B uses more Deep path. This perfectly isolates task-specific routing by cancelling out shared structural attention spikes.</p>
</section>

<section>
  <h2><span class="num">§1d</span> before vs. after pivot</h2>
  <h3>How does the cognitive load shift pre- and post-answer across all sources?</h3>
  <div class="controls">
    <label>align to</label> <select id="anchor-select-bva"></select>
    <label>window size</label>
    <select id="bva-window">
      <option value="3">3 tokens</option>
      <option value="5" selected>5 tokens</option>
      <option value="10">10 tokens</option>
    </select>
  </div>
  <p class="note">Averages the selected window of tokens immediately before the pivot, and the window immediately after the pivot. Slope indicates shift in cognitive load.</p>
  
  <div class="subsection">
    <h4>Summary Panels</h4>
    <div id="bva-summary-grid" class="grid-4">
       <div id="bva-sum-overall" class="chart"></div>
       <div id="bva-sum-early" class="chart"></div>
       <div id="bva-sum-mid" class="chart"></div>
       <div id="bva-sum-late" class="chart"></div>
    </div>
  </div>
  
  <div class="subsection">
    <h4>Per-Layer Panels</h4>
    <div id="bva-layer-grid" class="grid-4" style="margin-top: 16px;">
       </div>
  </div>
</section>

<section>
  <h2><span class="num">§2</span> source deep-dive</h2>
  <h3>Distributions and aggregates for a single source.</h3>
  <div class="controls">
    <label>source</label>
    <select id="source-select-2"></select>
  </div>
  <div class="kpis" id="source-kpis"></div>
  <div class="grid-2">
    <div id="hist-update-pref" class="chart"></div>
    <div id="hist-gate-pref" class="chart"></div>
  </div>
  <div class="grid-3">
    <div id="hist-gate-deep" class="chart"></div>
    <div id="hist-gate-wide" class="chart"></div>
    <div id="hist-steps" class="chart"></div>
  </div>
  <div id="hist-loss" class="chart"></div>
</section>

<section>
  <h2><span class="num">§3</span> layer dynamics</h2>
  <h3>How do routing and adaptive computation evolve through the stack?</h3>
  <div class="controls">
    <label>source</label>
    <select id="source-select-3"></select>
  </div>
  <div class="grid-2">
    <div id="layer-gate" class="chart"></div>
    <div id="layer-preference" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="layer-steps" class="chart"></div>
    <div id="layer-loop-disp" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="halt-curves" class="chart-tall"></div>
    <div id="step-disp-curves" class="chart-tall"></div>
  </div>
</section>

<section>
  <h2><span class="num">§4</span> token explorer</h2>
  <h3>Which tokens go which way? Hover or click a token for the full per-layer breakdown.</h3>
  <div class="controls">
    <label>source</label> <select id="source-select-4"></select>
    <label>chunk</label> <select id="chunk-select"></select>
    <label>color by</label>
    <select id="color-metric">
      <optgroup label="preference (diverging deep ↔ wide)">
        <option value="update_pref" selected>update_pref (actual update share)</option>
        <option value="gate_pref">gate_pref (gate-only)</option>
      </optgroup>
      <optgroup label="raw gates (sequential)">
        <option value="gate_deep">gate_deep</option>
        <option value="gate_wide">gate_wide</option>
      </optgroup>
      <optgroup label="compute / loop">
        <option value="expected_steps">expected_steps</option>
        <option value="loop_displacement">loop_displacement</option>
      </optgroup>
      <optgroup label="loss">
        <option value="loss">loss</option>
      </optgroup>
      <optgroup label="POS (categorical)">
        <option value="pos">pos</option>
      </optgroup>
    </select>
    <label>layer</label>
    <select id="layer-select"></select>
    <label><input type="checkbox" id="show-pos-toggle"> show POS subscripts</label>
    <span class="legend" id="color-legend"></span>
  </div>
  <div class="token-explorer-layout">
    <div class="token-text" id="token-text"></div>
    <div class="detail-panel" id="detail-panel">
      <h4>token detail</h4>
      <div style="color: var(--muted);">Hover or click a token to see per-layer values.</div>
    </div>
  </div>
</section>

<section>
  <h2><span class="num">§4b</span> sequence routing grid</h2>
  <h3>How does routing evolve across layers for the selected chunk?</h3>
  <div class="controls">
    <label>metric</label>
    <select id="grid-metric">
      <option value="update_pref" selected>update_pref (actual update share)</option>
      <option value="gate_pref">gate_pref (gate-only)</option>
    </select>
  </div>
  <div id="routing-grid" class="chart" style="height: 400px;"></div>
  <p class="note">X-axis: Tokens from the chunk selected in §4. Y-axis: Layer depth. Colors map to the diagram: <strong style="color: #2a4d6e;">Teal = Deep</strong>, <strong style="color: #b6533a;">Rust = Wide</strong>.</p>
</section>

<section>
  <h2><span class="num">§5</span> POS analysis</h2>
  <h3>Function vs content words — does the router actually care about syntax?</h3>
  <p class="note">UPOS tags from spaCy (with a math-aware override for gsm8k),
    aligned to subword tokens. Multi-token words can be aggregated by
    <em>mean</em> (average compute the word received) or <em>max</em> (peak
    compute triggered by any subword). For non-English or heavily-symbolic
    sources, POS tags are unreliable — check coverage below.</p>
  <div class="controls">
    <label>source</label> <select id="source-select-5"></select>
    <label>aggregation</label>
    <select id="pos-agg">
      <option value="mean">mean over subwords</option>
      <option value="max">max over subwords</option>
    </select>
    <span style="margin-left:24px; color: var(--muted); font-family: var(--mono); font-size: 12px;"
          id="pos-coverage"></span>
  </div>

  <div class="subsection">
    <h4>§5a — POS × layer heatmap</h4>
    <h5>For each POS tag, the chosen metric at every layer. Layer-averaging
      hides most of the signal — this view doesn't average.</h5>
    <div class="controls">
      <label>metric</label>
      <select id="pos-heatmap-metric">
        <option value="update_pref" selected>update_pref</option>
        <option value="gate_pref">gate_pref</option>
        <option value="gate_deep">gate_deep</option>
        <option value="gate_wide">gate_wide</option>
        <option value="expected_steps">expected_steps</option>
        <option value="loop_displacement">loop_displacement</option>
      </select>
      <span class="legend" id="pos-heatmap-legend"></span>
    </div>
    <div id="pos-heatmap" class="chart-heatmap"></div>
    <p class="note">Cell value = mean over all words tagged that POS, at that layer.
      Tags with fewer than 10 words in the source are dropped.</p>
  </div>

  <div class="subsection">
    <h4>§5c — single layer</h4>
    <h5>Drill into a single layer. Use this to confirm a row in the §5a heatmap.</h5>
    <div class="controls">
      <label>layer</label>
      <select id="pos-single-layer"></select>
    </div>
    <div class="grid-2">
      <div id="pos-single-update-pref" class="chart-tall"></div>
      <div id="pos-single-gate-pref" class="chart-tall"></div>
    </div>
    <div class="grid-2">
      <div id="pos-single-steps" class="chart-tall"></div>
      <div id="pos-single-loopdisp" class="chart-tall"></div>
    </div>
  </div>

  <div class="subsection">
    <h4>§5d — per-POS lines across layers</h4>
    <h5>One line per POS tag. Shows the trajectory of each tag through the
      stack — much easier to read than a heatmap row when comparing tags.</h5>
    <div class="controls">
      <label>metric</label>
      <select id="pos-lines-metric">
        <option value="update_pref" selected>update_pref</option>
        <option value="gate_pref">gate_pref</option>
        <option value="gate_deep">gate_deep</option>
        <option value="gate_wide">gate_wide</option>
        <option value="expected_steps">expected_steps</option>
        <option value="loop_displacement">loop_displacement</option>
      </select>
      <span style="color: var(--muted); font-size: 12px;">
        Click POS names in the legend to toggle.
      </span>
    </div>
    <div id="pos-lines" class="chart-tall"></div>
  </div>

  <div class="subsection">
    <h4>§5e — decisive-layer commitment</h4>
    <h5>For each word, find the layer where the model commits hardest
      (|update_pref − 0.5| is maximal). Mean over words of that POS tells you
      <em>which way it commits when it commits hardest</em>, without weak-gating
      layers dragging the mean toward 0.5.</h5>
    <div class="grid-2">
      <div id="decisive-pref" class="chart-tall"></div>
      <div id="decisive-layer" class="chart-tall"></div>
    </div>
    <p class="note">Left: mean signed <code>update_pref</code> at each POS's decisive layer
      (red = wide, blue = deep, grey horizontal line = 0.5 = balanced).
      Right: mean layer index at which that decisive moment occurs.</p>
  </div>
</section>

</main>

<script>
// ---------------------------------------------------------------------------
// Decode embedded payload
// ---------------------------------------------------------------------------
const PAYLOAD_B64 = "__PAYLOAD_B64__";
const PAYLOAD = JSON.parse(atob(PAYLOAD_B64));
const META = PAYLOAD.meta;
const SOURCES = PAYLOAD.sources;
const STATIC = PAYLOAD.static;

const PLOT_FONT = { family: "IBM Plex Mono, ui-monospace, Menlo, monospace", size: 11, color: "#1a1a1a" };
const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 50, r: 20, t: 30, b: 50 },
  font: PLOT_FONT,
  xaxis: { gridcolor: "#d8d2bd", zerolinecolor: "#b8b09a", linecolor: "#1a1a1a" },
  yaxis: { gridcolor: "#d8d2bd", zerolinecolor: "#b8b09a", linecolor: "#1a1a1a" },
  showlegend: false,
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };
const COLOR_DEEP = "#2a4d6e";
const COLOR_WIDE = "#b6533a";
const COLOR_NEUTRAL = "#6b6b6b";
const RGB_DEEP    = [42, 77, 110];
const RGB_WIDE    = [182, 83, 58];
const RGB_NEUTRAL = [246, 243, 236];

function layoutCopy(extra) {
  return Object.assign({}, JSON.parse(JSON.stringify(PLOT_LAYOUT)), extra || {});
}

function fillMeta() {
  const m = document.getElementById("model-meta");
  m.innerHTML = `
    <span>layers: ${META.n_layer}</span>
    <span>max_loops: ${META.max_loops}</span>
    <span>gate_mode: ${META.gate_mode}</span>
    <span>use_cross: ${META.use_cross}</span>
    <span>sources: ${META.source_order.length}</span>
  `;
}

// ---------------------------------------------------------------------------
// UI Setup and Plumping
// ---------------------------------------------------------------------------
function fillSourceSelectors() {
  for (const id of ["source-select-2", "source-select-3", "source-select-4", "source-select-5", "source-select-aligned", "source-diff-a", "source-diff-b"]) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    sel.innerHTML = "";
    for (const s of META.source_order) {
      const o = document.createElement("option");
      o.value = s; o.textContent = s;
      sel.appendChild(o);
    }
  }
}

function updateAnchorSelector(source) {
  const sel = document.getElementById("anchor-select");
  if (!sel) return;
  sel.innerHTML = "";
  const anchors = Object.keys(SOURCES[source].agg.aligned || {});
  if (anchors.length === 0) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "No alignment data";
    sel.appendChild(o);
    return;
  }
  for (const a of anchors) {
    const o = document.createElement("option");
    o.value = a; o.textContent = `"${a}"`;
    sel.appendChild(o);
  }
}

function updateDiffAnchorSelector() {
  const srcA = document.getElementById("source-diff-a").value;
  const srcB = document.getElementById("source-diff-b").value;
  const sel = document.getElementById("anchor-select-diff");
  if (!sel) return;
  sel.innerHTML = "";
  
  if (!srcA || !srcB) return;
  const anchorsA = Object.keys(SOURCES[srcA].agg.aligned || {});
  const anchorsB = Object.keys(SOURCES[srcB].agg.aligned || {});
  // Find common anchors available in both sources
  const common = anchorsA.filter(a => anchorsB.includes(a));
  
  if (common.length === 0) {
    const o = document.createElement("option"); o.value = ""; o.textContent = "No shared anchors"; sel.appendChild(o);
    return;
  }
  for (const a of common) {
    const o = document.createElement("option"); o.value = a; o.textContent = `"${a}"`; sel.appendChild(o);
  }
}

function updateBVAAnchorSelector() {
  const sel = document.getElementById("anchor-select-bva");
  if (!sel) return;
  sel.innerHTML = "";
  const firstSource = META.source_order[0];
  const anchors = Object.keys(SOURCES[firstSource].agg.aligned || {});
  for (const a of anchors) {
    const o = document.createElement("option"); o.value = a; o.textContent = `"${a}"`; sel.appendChild(o);
  }
}

function fillLayerSelector() {
  const sel = document.getElementById("layer-select");
  const meanOpt = document.createElement("option");
  meanOpt.value = "mean";
  meanOpt.textContent = "mean (all layers)";
  sel.appendChild(meanOpt);

  for (let i = 0; i < META.n_layer; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = "layer " + i;
    sel.appendChild(o);
  }
  sel.value = "mean";

  const sel5c = document.getElementById("pos-single-layer");
  if (sel5c) {
    for (let i = 0; i < META.n_layer; i++) {
      const o = document.createElement("option");
      o.value = i; o.textContent = "layer " + i;
      sel5c.appendChild(o);
    }
    sel5c.value = Math.floor(META.n_layer / 2);
  }
}

// ---------------------------------------------------------------------------
// §1 Overview bars
// ---------------------------------------------------------------------------
function drawOverview() {
  const sources = META.source_order;
  const upref = sources.map(s => SOURCES[s].agg.summary.mean_update_pref);
  const es    = sources.map(s => SOURCES[s].agg.summary.mean_expected_steps);
  const loss  = sources.map(s => SOURCES[s].agg.summary.mean_loss);

  const baseBarLayout = (title, range) => layoutCopy({
    title: { text: title, font: { size: 12 } },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, range ? { range: range } : {}),
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 9 } }),
  });

  const upref_colors = upref.map(v => v >= 0.5
    ? `rgba(${RGB_DEEP[0]},${RGB_DEEP[1]},${RGB_DEEP[2]},${0.4 + (v - 0.5)})`
    : `rgba(${RGB_WIDE[0]},${RGB_WIDE[1]},${RGB_WIDE[2]},${0.4 + (0.5 - v)})`);
  Plotly.newPlot("overview-update-pref", [{
    type: "bar", x: sources, y: upref,
    marker: { color: upref_colors },
    hovertemplate: "%{x}<br>update_pref: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean update_pref per source  (0.5 = balanced)", [0, 1]), PLOT_CONFIG);

  Plotly.newPlot("overview-steps", [{
    type: "bar", x: sources, y: es,
    marker: { color: COLOR_NEUTRAL },
    hovertemplate: "%{x}<br>expected_steps: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean expected_steps per source", [0, META.max_loops]), PLOT_CONFIG);

  Plotly.newPlot("overview-loss", [{
    type: "bar", x: sources, y: loss,
    marker: { color: COLOR_WIDE },
    hovertemplate: "%{x}<br>loss: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean per-token loss per source", null), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §1b Aligned Trajectories 
// ---------------------------------------------------------------------------
function drawAlignedTrajectories(source) {
  const anchor = document.getElementById("anchor-select").value;
  const aggData = (SOURCES[source].agg.aligned || {})[anchor];
  const plotDiv = document.getElementById("aligned-plot");
  
  if (!aggData || aggData.means.length === 0 || aggData.means[0].every(x => x === null)) {
    plotDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);">No matches for anchor token in this source.</div>';
    return;
  }
  plotDiv.innerHTML = '';

  const WINDOW = (aggData.means[0].length - 1) / 2;
  const xVals = Array.from({length: WINDOW * 2 + 1}, (_, i) => i - WINDOW);
  const nLayer = META.n_layer;
  const showSamples = document.getElementById("show-samples-toggle").checked;

  const traces = [];
  function getLayerColor(l, alpha) {
    const intensity = l / (nLayer - 1);
    const r = Math.round(196 - intensity * (196 - 26));
    const g = Math.round(215 - intensity * (215 - 60));
    const b = Math.round(232 - intensity * (232 - 94));
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  // Draw faint background sample lines
  if (showSamples && aggData.samples) {
    for (let l = 0; l < nLayer; l++) {
      const colorStr = getLayerColor(l, 0.15); 
      const samples = aggData.samples[l];
      if (!samples) continue;
      for (let s = 0; s < samples.length; s++) {
        traces.push({
          type: "scatter", mode: "lines", x: xVals, y: samples[s],
          line: { color: colorStr, width: 1 },
          hoverinfo: "skip", showlegend: false
        });
      }
    }
  }

  // Draw thick layer average lines
  const topTokens = aggData.top_tokens;
  for (let l = 0; l < nLayer; l++) {
    traces.push({
      type: "scatter", mode: "lines", x: xVals, y: aggData.means[l],
      name: `Layer ${l}`,
      customdata: topTokens,
      line: { color: getLayerColor(l, 1.0), width: l === nLayer - 1 ? 3 : 2 },
      hovertemplate: `Offset: %{x}<br>Layer ${l}<br>update_pref: %{y:.3f}<br><br><b>Most frequent tokens here:</b><br>%{customdata}<extra></extra>`
    });
  }

  const layout = layoutCopy({
    title: { text: `Layer Trajectories Aligned to "${anchor}" — ${source}`, font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "v", x: 1.02, y: 1 },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: `Distance from '${anchor}' match`, zeroline: true, zerolinewidth: 2, zerolinecolor: "#b6533a" }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { title: "update_pref", range: [0, 1] }),
    margin: { l: 60, r: 100, t: 40, b: 50 },
    shapes: [{ type: "line", x0: -WINDOW, x1: WINDOW, y0: 0.5, y1: 0.5, line: { color: "rgba(0,0,0,0.3)", width: 1, dash: "dot" } }]
  });
  Plotly.newPlot("aligned-plot", traces, layout, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §1c Cross-Source Aligned Difference
// ---------------------------------------------------------------------------
function drawAlignedDiff() {
  const srcA = document.getElementById("source-diff-a").value;
  const srcB = document.getElementById("source-diff-b").value;
  const anchor = document.getElementById("anchor-select-diff").value;
  const plotDiv = document.getElementById("aligned-diff-plot");

  if (!srcA || !srcB || srcA === srcB) {
    plotDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);">Select two different sources to compare.</div>';
    return;
  }
  if (!anchor) {
    plotDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);">No shared anchor strings found between these sources.</div>';
    return;
  }

  const dataA = (SOURCES[srcA].agg.aligned || {})[anchor];
  const dataB = (SOURCES[srcB].agg.aligned || {})[anchor];

  if (!dataA || !dataB || dataA.means.length === 0 || dataB.means.length === 0) {
    plotDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);">Missing alignment data for one or both sources.</div>';
    return;
  }

  const nLayer = META.n_layer;
  const WINDOW = (dataA.means[0].length - 1) / 2;
  const xVals = Array.from({length: WINDOW * 2 + 1}, (_, i) => i - WINDOW);
  
  const traces = [];
  for (let l = 0; l < nLayer; l++) {
    const diffs = [];
    for (let i = 0; i < xVals.length; i++) {
      const vA = dataA.means[l][i];
      const vB = dataB.means[l][i];
      if (vA !== null && vB !== null) {
        diffs.push(vA - vB);
      } else {
        diffs.push(null);
      }
    }
    
    // Sequential color map for layers
    const intensity = l / (nLayer - 1);
    const r = Math.round(196 - intensity * (196 - 26));
    const g = Math.round(215 - intensity * (215 - 60));
    const b = Math.round(232 - intensity * (232 - 94));
    
    traces.push({
      type: "scatter", mode: "lines", x: xVals, y: diffs,
      name: `Layer ${l}`,
      line: { color: `rgb(${r}, ${g}, ${b})`, width: l === nLayer - 1 ? 3 : 2 },
      hovertemplate: `Offset: %{x}<br>Layer ${l}<br>Δ update_pref: %{y:.3f}<extra></extra>`
    });
  }

  const layout = layoutCopy({
    title: { text: `Difference: (${srcA}) - (${srcB}) Aligned to "${anchor}"`, font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "v", x: 1.02, y: 1 },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: `Distance from '${anchor}' match`, zeroline: true, zerolinewidth: 2, zerolinecolor: "#b6533a" }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { title: "Δ update_pref (A - B)", range: [-1, 1] }),
    margin: { l: 60, r: 100, t: 40, b: 50 },
    shapes: [{ type: "line", x0: -WINDOW, x1: WINDOW, y0: 0, y1: 0, line: { color: "rgba(0,0,0,0.3)", width: 1, dash: "dot" } }]
  });
  Plotly.newPlot("aligned-diff-plot", traces, layout, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §1d Before vs After Pivot Slopegraphs
// ---------------------------------------------------------------------------
function drawBeforeVsAfter() {
  const anchor = document.getElementById("anchor-select-bva").value;
  if (!anchor) return;
  const W = parseInt(document.getElementById("bva-window").value);
  const L = META.n_layer;

  const L_early = [0, Math.floor(L/3)];
  const L_mid = [Math.floor(L/3), Math.floor(2*L/3)];
  const L_late = [Math.floor(2*L/3), L];

  const sources = META.source_order.filter(s => SOURCES[s].agg.aligned && SOURCES[s].agg.aligned[anchor]);
  const palette = ["#2a4d6e", "#b6533a", "#6b8e6e", "#d9905a", "#7b5fa8"];

  function getMean(source, layers, offsets) {
    let sum = 0; let count = 0;
    const data = SOURCES[source].agg.aligned[anchor].means;
    if (!data || data.length === 0) return null;
    const WINDOW = (data[0].length - 1) / 2;
    for (let l = layers[0]; l < layers[1]; l++) {
      for (let o of offsets) {
        const val = data[l][WINDOW + o];
        if (val !== null && isFinite(val)) { sum += val; count++; }
      }
    }
    return count > 0 ? sum / count : null;
  }

  // -W to -1 (Before)
  const offsets_before = Array.from({length: W}, (_, i) => -W + i);
  // 1 to W (After). Skipping offset 0 (the pivot itself) to isolate true before/after states.
  const offsets_after = Array.from({length: W}, (_, i) => i + 1);   

  function makePlot(elId, title, layers) {
    const traces = sources.map((s, i) => {
      const b = getMean(s, layers, offsets_before);
      const a = getMean(s, layers, offsets_after);
      return {
        type: "scatter", mode: "lines+markers",
        x: ["Before", "After"], y: [b, a],
        name: s, legendgroup: s,
        line: { color: palette[i % palette.length], width: 2.5 },
        marker: { size: 8 },
        hovertemplate: s + "<br>%{x}: %{y:.3f}<extra></extra>"
      };
    });
    
    const layout = layoutCopy({
      title: { text: title, font: { size: 12 } },
      margin: { l: 45, r: 10, t: 40, b: 30 },
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1], dtick: 0.2, title: "update_pref" }),
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { showgrid: false }),
      showlegend: elId === "bva-sum-overall"
    });
    
    if (elId === "bva-sum-overall") {
       layout.legend = { orientation: "h", y: -0.2, x: 0, font: { size: 10 } };
       layout.margin.b = 60; // Make room for legend
    }
    
    Plotly.newPlot(elId, traces, layout, PLOT_CONFIG);
  }

  // Render Summaries
  makePlot("bva-sum-overall", "Overall Average", [0, L]);
  makePlot("bva-sum-early", `Early Layers (0 - ${L_early[1]-1})`, L_early);
  makePlot("bva-sum-mid", `Mid Layers (${L_mid[0]} - ${L_mid[1]-1})`, L_mid);
  makePlot("bva-sum-late", `Late Layers (${L_late[0]} - ${L_late[1]-1})`, L_late);

  // Render Per-Layer Grid
  const layerGrid = document.getElementById("bva-layer-grid");
  layerGrid.innerHTML = ""; // Clear existing
  for (let l = 0; l < L; l++) {
    const div = document.createElement("div");
    div.id = `bva-layer-${l}`;
    div.className = "chart";
    div.style.height = "240px";
    layerGrid.appendChild(div);
    makePlot(div.id, `Layer ${l}`, [l, l+1]);
  }
}

// ---------------------------------------------------------------------------
// §2 Source deep-dive
// ---------------------------------------------------------------------------
function histTrace(h, color, name) {
  const edges = h.edges;
  const counts = h.counts;
  const centers = [];
  for (let i = 0; i < counts.length; i++) centers.push((edges[i] + edges[i + 1]) / 2);
  return {
    type: "bar", x: centers, y: counts,
    marker: { color: color }, name: name,
    hovertemplate: name + "<br>bin: %{x:.3f}<br>count: %{y}<extra></extra>",
  };
}

function drawSourceDeepDive(source) {
  const agg = SOURCES[source].agg;
  const s = agg.summary;
  document.getElementById("source-kpis").innerHTML = `
    <div class="kpi"><span class="label">tokens</span><span class="val">${s.n_tokens.toLocaleString()}</span><span class="sub">${s.n_chunks} chunks</span></div>
    <div class="kpi"><span class="label">mean update_pref</span><span class="val">${s.mean_update_pref.toFixed(3)}</span><span class="sub">actual update share</span></div>
    <div class="kpi"><span class="label">mean gate_pref</span><span class="val">${s.mean_gate_pref.toFixed(3)}</span><span class="sub">gate-only</span></div>
    <div class="kpi"><span class="label">mean gate_deep</span><span class="val">${s.mean_gate_deep.toFixed(3)}</span><span class="sub">raw</span></div>
    <div class="kpi"><span class="label">mean gate_wide</span><span class="val">${s.mean_gate_wide.toFixed(3)}</span><span class="sub">raw</span></div>
    <div class="kpi"><span class="label">mean expected_steps</span><span class="val">${s.mean_expected_steps.toFixed(3)}</span><span class="sub">out of ${META.max_loops}</span></div>
    <div class="kpi"><span class="label">mean loss</span><span class="val">${s.mean_loss.toFixed(3)}</span></div>
  `;
  Plotly.newPlot("hist-update-pref", [histTrace(agg.hist.update_pref, COLOR_DEEP, "update_pref")], layoutCopy({ title: { text: "update_pref — actual update share", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-gate-pref", [histTrace(agg.hist.gate_pref, COLOR_DEEP, "gate_pref")], layoutCopy({ title: { text: "gate_pref — gate-only", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-gate-deep", [histTrace(agg.hist.gate_deep, COLOR_DEEP, "gate_deep")], layoutCopy({ title: { text: "gate_deep (raw)", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-gate-wide", [histTrace(agg.hist.gate_wide, COLOR_WIDE, "gate_wide")], layoutCopy({ title: { text: "gate_wide (raw)", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-steps", [histTrace(agg.hist.expected_steps, COLOR_NEUTRAL, "expected_steps")], layoutCopy({ title: { text: "expected_steps (all layers)", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-loss", [histTrace(agg.hist.loss, COLOR_WIDE, "loss")], layoutCopy({ title: { text: "per-token loss", font: { size: 12 } } }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §3 Layer dynamics
// ---------------------------------------------------------------------------
function lineTrace(y, color, name) {
  const x = y.map((_, i) => i);
  return { type: "scatter", mode: "lines+markers", x: x, y: y,
           marker: { color: color, size: 6 }, line: { color: color, width: 1.5 },
           name: name, hovertemplate: name + "<br>layer %{x}: %{y:.3f}<extra></extra>" };
}

function drawLayerDynamics(source) {
  const pl = SOURCES[source].agg.per_layer;
  Plotly.newPlot("layer-gate", [
    lineTrace(pl.gate_deep, COLOR_DEEP, "gate_deep"),
    lineTrace(pl.gate_wide, COLOR_WIDE, "gate_wide"),
  ], layoutCopy({
    title: { text: "mean raw gate per layer", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "h", y: 1.1 },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1] }),
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-preference", [
    lineTrace(pl.gate_pref, COLOR_DEEP, "gate_pref"),
    lineTrace(pl.update_pref, COLOR_WIDE, "update_pref"),
  ], layoutCopy({
    title: { text: "preference per layer (0.5 = balanced)", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "h", y: 1.1 },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1] }),
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-steps", [
    lineTrace(pl.expected_steps, COLOR_NEUTRAL, "expected_steps"),
  ], layoutCopy({
    title: { text: "mean expected_steps per layer", font: { size: 12 } },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, META.max_loops] }),
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-loop-disp", [
    lineTrace(pl.loop_displacement, COLOR_DEEP, "loop_displacement"),
  ], layoutCopy({
    title: { text: "mean loop_displacement per layer", font: { size: 12 } },
  }), PLOT_CONFIG);

  const halts = pl.halt_probs;
  const halt_traces = halts.map((row, li) => ({
    type: "scatter", mode: "lines", x: row.map((_, i) => i), y: row,
    name: "layer " + li,
    line: { color: `hsl(${(li * 360 / META.n_layer) | 0}, 50%, 45%)`, width: 1.2 },
    hovertemplate: `layer ${li}<br>step %{x}: halt_prob=%{y:.3f}<extra></extra>`,
  }));
  Plotly.newPlot("halt-curves", halt_traces, layoutCopy({
    title: { text: "mean halt probability per (layer, step)", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 9 }, orientation: "v", x: 1.02, y: 1 },
    margin: { l: 50, r: 100, t: 30, b: 50 },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1] }),
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "step" }),
  }), PLOT_CONFIG);

  const sd = pl.step_displacement;
  const sd_traces = sd.map((row, li) => ({
    type: "scatter", mode: "lines", x: row.map((_, i) => i), y: row,
    name: "layer " + li,
    line: { color: `hsl(${(li * 360 / META.n_layer) | 0}, 50%, 45%)`, width: 1.2 },
    hovertemplate: `layer ${li}<br>step %{x}: ‖h[s+1]−h[s]‖=%{y:.3f}<extra></extra>`,
  }));
  Plotly.newPlot("step-disp-curves", sd_traces, layoutCopy({
    title: { text: "step displacement ‖h[s+1] − h[s]‖ per (layer, step)", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 9 }, orientation: "v", x: 1.02, y: 1 },
    margin: { l: 50, r: 100, t: 30, b: 50 },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "step" }),
  }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §4 Token explorer
// ---------------------------------------------------------------------------
const METRIC_RANGES = {
  gate_pref:         { range: [0, 1], cmap: "deepwide" },
  update_pref:       { range: [0, 1], cmap: "deepwide" },
  gate_deep:         { range: [0, 1], cmap: "sequential_deep" },
  gate_wide:         { range: [0, 1], cmap: "sequential_wide" },
  expected_steps:    { range: [0, META.max_loops], cmap: "viridis" },
  loop_displacement: { range: [0, 1], cmap: "viridis" },
  loss:              { range: [0, 10], cmap: "ylorrd" },
};
const POS_PALETTE = {
  NOUN:  "#2a4d6e", PROPN: "#3d6b94", VERB:  "#1f6f64", ADJ:   "#6b8e6e",
  ADV:   "#a3a661", NUM:   "#7b5fa8", DET:   "#d9905a", PRON:  "#c47054", ADP:   "#b6533a",
  CCONJ: "#a35a3f", SCONJ: "#a35a3f", PART:  "#9c5b4a", AUX:   "#bb6f4e",
  PUNCT: "#d4c2a8", SYM:   "#d4c2a8", INTJ:  "#9c9c9c", X:     "#9c9c9c", SPACE: "#cccccc",
  UNKNOWN: "#bdbdbd", SPECIAL: "#eeeeee",
};
const POS_ORDER = [
  "NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM",
  "DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART", "AUX", "PUNCT", "SYM",
  "INTJ", "X", "SPACE", "UNKNOWN",
];

function _mix(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function colorFor(val, metric) {
  if (metric === "pos") {
    if (!val) return "transparent";
    const hex = POS_PALETTE[val] || POS_PALETTE.UNKNOWN;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, 0.55)`;
  }
  if (val === null || val === undefined || !isFinite(val)) return "transparent";
  const cfg = METRIC_RANGES[metric];
  let lo = cfg.range ? cfg.range[0] : 0;
  let hi = cfg.range ? cfg.range[1] : 5;
  const t = Math.max(0, Math.min(1, (val - lo) / (hi - lo)));
  const alpha = 0.6;
  if (cfg.cmap === "deepwide") {
    let rgb;
    if (t >= 0.5) rgb = _mix(RGB_NEUTRAL, RGB_DEEP, (t - 0.5) * 2);
    else          rgb = _mix(RGB_WIDE, RGB_NEUTRAL, t * 2);
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }
  if (cfg.cmap === "sequential_deep") {
    const rgb = _mix(RGB_NEUTRAL, RGB_DEEP, t);
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }
  if (cfg.cmap === "sequential_wide") {
    const rgb = _mix(RGB_NEUTRAL, RGB_WIDE, t);
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }
  if (cfg.cmap === "viridis") {
    const r = Math.round(255 * Math.pow(t, 1.5));
    const g = Math.round(255 * t);
    const b = Math.round(255 * (1 - t) * 0.7);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  if (cfg.cmap === "ylorrd") {
    const r = 255;
    const g = Math.round(255 * (1 - t));
    const b = Math.round(150 * (1 - t));
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return "rgba(100,100,100,0.5)";
}

function fillChunkSelector(source) {
  const sel = document.getElementById("chunk-select");
  sel.innerHTML = "";
  const chunks = SOURCES[source].chunks;
  chunks.forEach((c, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `doc ${c.doc}, chunk ${c.chunk} (${c.n_tokens} tok)`;
    sel.appendChild(o);
  });
}
function getChunk() {
  const source = document.getElementById("source-select-4").value;
  const idx = parseInt(document.getElementById("chunk-select").value);
  return SOURCES[source].chunks[idx];
}

function getMetricArrayForChunk(chunk, metric, layer) {
  if (metric === "loss") return chunk.loss;
  if (metric === "pos") return chunk.pos || chunk.tokens.map(() => "UNKNOWN");

  const isMean = (layer === "mean");
  const singleLayer = isMean ? 0 : parseInt(layer);
  const L = chunk.gate_deep.length;
  const numToks = chunk.tokens.length;

  if (metric === "gate_pref") {
    if (isMean) {
      return Array.from({length: numToks}, (_, i) => {
        let sum = 0;
        for (let l=0; l<L; l++) {
          const gd = chunk.gate_deep[l][i];
          const gw = chunk.gate_wide[l][i];
          sum += (gd + gw) > 1e-6 ? gd / (gd + gw) : 0.5;
        }
        return sum / L;
      });
    }
    const gd = chunk.gate_deep[singleLayer];
    const gw = chunk.gate_wide[singleLayer];
    return gd.map((v, i) => {
      const denom = (v + gw[i]);
      return denom > 1e-6 ? v / denom : 0.5;
    });
  }

  if (metric === "update_pref") {
    if (isMean) {
      return Array.from({length: numToks}, (_, i) => {
        let sum = 0;
        for (let l=0; l<L; l++) {
          const gd = chunk.gate_deep[l][i];
          const gw = chunk.gate_wide[l][i];
          const dd = chunk.delta_deep_norm[l][i];
          const dw = chunk.delta_wide_norm[l][i];
          const dc = gd * dd;
          const wc = gw * dw;
          const denom = dc + wc;
          sum += (denom > 1e-6) ? dc / denom : 0.5;
        }
        return sum / L;
      });
    }
    const gd = chunk.gate_deep[singleLayer];
    const gw = chunk.gate_wide[singleLayer];
    const dd = chunk.delta_deep_norm[singleLayer];
    const dw = chunk.delta_wide_norm[singleLayer];
    return gd.map((g, i) => {
      const dc = g * dd[i];
      const wc = gw[i] * dw[i];
      const denom = dc + wc;
      return denom > 1e-6 ? dc / denom : 0.5;
    });
  }

  if (isMean) {
    return Array.from({length: numToks}, (_, i) => {
      let sum = 0;
      for (let l=0; l<L; l++) sum += chunk[metric][l][i];
      return sum / L;
    });
  }

  return chunk[metric][singleLayer];
}

function drawRoutingGrid() {
  const chunk = getChunk();
  if (!chunk) return;
  const metric = document.getElementById("grid-metric").value;
  const L = META.n_layer;
  const numToks = chunk.tokens.length;
  
  const xLabels = chunk.token_strs.map((t, i) => 
    `[${i}] ` + t.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "↵")
  );

  const z = [];
  for (let l = 0; l < L; l++) {
    const row = [];
    for (let i = 0; i < numToks; i++) {
      const gd = chunk.gate_deep[l][i];
      const gw = chunk.gate_wide[l][i];
      
      if (metric === "gate_pref") {
        row.push((gd + gw) > 1e-6 ? gd / (gd + gw) : 0.5);
      } else { 
        const dd = chunk.delta_deep_norm[l][i];
        const dw = chunk.delta_wide_norm[l][i];
        const dc = gd * dd;
        const wc = gw * dw;
        row.push((dc + wc) > 1e-6 ? dc / (dc + wc) : 0.5);
      }
    }
    z.push(row);
  }

  const yLabels = Array.from({length: L}, (_, i) => `Layer ${i}`);

  const trace = {
    type: "heatmap", z: z, x: xLabels, y: yLabels,
    colorscale: [[0.0, "#b6533a"], [0.5, "#f6f3ec"], [1.0, "#2a4d6e"]],
    zmin: 0, zmax: 1, hovertemplate: "Token: %{x}<br>%{y}<br>" + metric + ": %{z:.3f}<extra></extra>",
    colorbar: { thickness: 12, len: 0.9, tickfont: { size: 10 } }
  };

  const layout = layoutCopy({
    title: { text: "Network Depth vs. Sequence Position", font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 11, family: "IBM Plex Mono, monospace" } }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { title: "Network Depth", tickfont: { size: 11 } }),
    margin: { l: 60, r: 20, t: 40, b: 100 }
  });

  Plotly.newPlot("routing-grid", [trace], layout, PLOT_CONFIG);
}

function renderTokens() {
  const chunk = getChunk();
  if (!chunk) return;
  const metric = document.getElementById("color-metric").value;
  
  let layer = document.getElementById("layer-select").value;
  if (layer !== "mean") layer = parseInt(layer);

  const isPerLayer = (metric !== "loss" && metric !== "pos");
  const arr = getMetricArrayForChunk(chunk, metric, layer);
  const showPos = document.getElementById("show-pos-toggle").checked && chunk.pos;
  const container = document.getElementById("token-text");
  container.innerHTML = "";
  const tokens = chunk.token_strs;
  for (let i = 0; i < tokens.length; i++) {
    const span = document.createElement("span");
    span.className = "tok";
    span.dataset.idx = i;
    const v = arr[i];
    span.style.backgroundColor = colorFor(v, metric);
    let tokenContent;
    if (tokens[i].includes("\n")) {
      tokenContent = tokens[i].split("\n").map(s => s || " ").join("<br>");
    } else {
      tokenContent = tokens[i].replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    if (showPos && chunk.pos[i]) {
      const tag = chunk.pos[i];
      if (tag !== "SPECIAL" && tag !== "UNKNOWN") {
        tokenContent += `<sub style="color:var(--muted); font-size:9px; margin-left:1px;">${tag}</sub>`;
      }
    }
    span.innerHTML = tokenContent;
    const valStr = (typeof v === "number") ? v.toFixed(3) : (v ?? "n/a");
    const posStr = chunk.pos ? ` pos=${chunk.pos[i]}` : "";
    span.title = `[${i}] ${metric}=${valStr}${posStr}`;
    span.addEventListener("mouseenter", () => showTokenDetail(chunk, i));
    span.addEventListener("click", () => showTokenDetail(chunk, i, true));
    container.appendChild(span);
  }
  updateLegend(metric, isPerLayer ? layer : null);
  drawRoutingGrid();
}

function updateLegend(metric, layer) {
  if (metric === "pos") {
    const items = POS_ORDER.filter(t => POS_PALETTE[t]).map(t => {
      const hex = POS_PALETTE[t];
      return `<span style="display:inline-block; padding:0 6px; margin: 0 4px 4px 0; background:${hex}33; border-left:3px solid ${hex}; font-family:var(--mono); font-size:10px;">${t}</span>`;
    }).join("");
    document.getElementById("color-legend").innerHTML = items;
    return;
  }
  const cfg = METRIC_RANGES[metric];
  const lo = cfg.range ? cfg.range[0] : 0;
  const hi = cfg.range ? cfg.range[1] : 5;
  
  let layerStr = " (no layer)";
  if (layer !== null) {
      layerStr = (layer === "mean") ? " (mean across all layers)" : ` @ layer ${layer}`;
  }

  const stops = [];
  const N = 24;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    stops.push(colorFor(lo + t * (hi - lo), metric));
  }
  const grad = `linear-gradient(to right, ${stops.join(",")})`;
  document.getElementById("color-legend").innerHTML =
    `${metric}${layerStr} &nbsp; ${lo.toFixed(1)} <span class="swatch" style="background:${grad}"></span> ${hi.toFixed(1)}`;
}

let lockedToken = null;
function showTokenDetail(chunk, idx, lock) {
  if (lockedToken !== null && !lock) return;
  if (lock) {
    lockedToken = (lockedToken === idx) ? null : idx;
    document.querySelectorAll("#token-text .tok.selected").forEach(e => e.classList.remove("selected"));
    if (lockedToken !== null) {
      const el = document.querySelector(`#token-text .tok[data-idx="${idx}"]`);
      if (el) el.classList.add("selected");
    } else return;
  }
  const tokStr = chunk.token_strs[idx];
  const tokId = chunk.tokens[idx];
  const loss = chunk.loss[idx];
  const L = chunk.gate_deep.length;
  let rows = "";
  for (let li = 0; li < L; li++) {
    const gd = chunk.gate_deep[li][idx];
    const gw = chunk.gate_wide[li][idx];
    const dd = chunk.delta_deep_norm[li][idx];
    const dw = chunk.delta_wide_norm[li][idx];
    const gpref = (gd + gw) > 1e-6 ? gd / (gd + gw) : 0.5;
    const upref = (gd * dd + gw * dw) > 1e-6 ? (gd * dd) / (gd * dd + gw * dw) : 0.5;
    rows += `<tr>
      <td>${li}</td><td>${gd.toFixed(3)}</td><td>${gw.toFixed(3)}</td>
      <td><strong>${gpref.toFixed(2)}</strong></td><td><strong>${upref.toFixed(2)}</strong></td>
      <td>${chunk.expected_steps[li][idx].toFixed(2)}</td><td>${chunk.loop_displacement[li][idx].toFixed(2)}</td>
      <td>${dd.toFixed(2)}</td><td>${dw.toFixed(2)}</td>
    </tr>`;
  }
  const displayTok = tokStr.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "↵");
  document.getElementById("detail-panel").innerHTML = `
    <h4>token [${idx}] ${lockedToken !== null ? "(locked, click again to unlock)" : ""}</h4>
    <div style="font-family:var(--mono); font-size: 14px; margin-bottom: 6px;">
      "<span style="background:#fff3a8;padding:1px 3px;">${displayTok}</span>"
      <span style="color:var(--muted); margin-left:8px;">id=${tokId}</span>
    </div>
    <div style="margin-bottom: 10px;">loss (predict next) = <strong>${loss !== null ? loss.toFixed(3) : "n/a"}</strong></div>
    <table>
      <tr><th>l</th><th>g_d</th><th>g_w</th><th>g_pref</th><th>u_pref</th><th>steps</th><th>loop-d</th><th>‖Δd‖</th><th>‖Δw‖</th></tr>
      ${rows}
    </table>
    <div style="color: var(--muted); margin-top: 8px; font-size: 10px;">
      g_pref = gate_pref, u_pref = update_pref, loop-d = loop_displacement
    </div>
  `;
}

// ---------------------------------------------------------------------------
// §5 POS analysis
// ---------------------------------------------------------------------------
const POS_METRIC_CFG = {
  gate_pref:         { range: [0, 1], diverging: true,  mid: 0.5, label: "gate_pref" },
  update_pref:       { range: [0, 1], diverging: true,  mid: 0.5, label: "update_pref" },
  expected_steps:    { range: [0, META.max_loops], diverging: false, label: "expected_steps" },
  loop_displacement: { range: null,   diverging: false, label: "loop_displacement" },
  gate_deep:         { range: [0, 1], diverging: false, label: "gate_deep" },
  gate_wide:         { range: [0, 1], diverging: false, label: "gate_wide" },
};

const DIVERGING_SCALE = [
  [0.0, "#b6533a"], [0.25, "#d99d8a"], [0.5, "#f6f3ec"],
  [0.75, "#7a98b8"], [1.0, "#2a4d6e"],
];
const SEQUENTIAL_SCALE = [
  [0.0, "#f6f3ec"], [0.5, "#d8c98c"], [1.0, "#2a4d6e"],
];

const MIN_WORDS_PER_CELL = 10;

function getPOSTagsPresent(pos, minWords) {
  return POS_ORDER.filter(t => (pos.n_words_per_pos[t] || 0) >= minWords);
}

function _layerArray(pos, metric, agg, tag) {
  const src = (agg === "max") ? pos.by_pos_layer_max : pos.by_pos_layer_mean;
  if (!src[metric] || !src[metric][tag]) return null;
  return src[metric][tag];
}

function drawPOSAll(source) {
  const data = SOURCES[source];
  const cov = data.agg.pos ? `POS coverage: ${(data.agg.pos.coverage * 100).toFixed(1)}%` : "no POS data";
  document.getElementById("pos-coverage").textContent = cov;
  if (!data.agg.pos) {
    for (const id of ["pos-heatmap", "pos-lines", "decisive-pref", "decisive-layer",
                      "pos-single-update-pref", "pos-single-gate-pref",
                      "pos-single-steps", "pos-single-loopdisp"]) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">no POS sidecar for this source</div>';
    }
    return;
  }
  drawPOSHeatmap(source);
  drawPOSSingleLayer(source);
  drawPOSLines(source);
  drawDecisive(source);
}

function drawPOSHeatmap(source) {
  const pos = SOURCES[source].agg.pos;
  const agg = document.getElementById("pos-agg").value;
  const metric = document.getElementById("pos-heatmap-metric").value;
  const cfg = POS_METRIC_CFG[metric];
  const tags = getPOSTagsPresent(pos, MIN_WORDS_PER_CELL);
  const nLayer = META.n_layer;
  const src = (agg === "max") ? pos.by_pos_layer_max : pos.by_pos_layer_mean;
  const data = src[metric] || {};

  const z = tags.map(t => {
    const row = data[t] || [];
    return Array.from({length: nLayer}, (_, li) => {
      const v = row[li];
      return (v === undefined || v === null || !isFinite(v)) ? null : v;
    });
  });

  let zmin, zmax, colorscale;
  if (cfg.diverging) {
    zmin = cfg.range[0]; zmax = cfg.range[1]; colorscale = DIVERGING_SCALE;
  } else {
    const flat = z.flat().filter(v => v !== null && isFinite(v));
    zmin = cfg.range ? cfg.range[0] : Math.min(...flat);
    zmax = cfg.range ? cfg.range[1] : Math.max(...flat);
    colorscale = SEQUENTIAL_SCALE;
  }

  const wordCounts = tags.map(t => pos.n_words_per_pos[t] || 0);
  const customdata = tags.map((_, ti) =>
    Array.from({length: nLayer}, () => wordCounts[ti])
  );
  const trace = {
    type: "heatmap",
    z: z,
    x: Array.from({length: nLayer}, (_, i) => "L" + i),
    y: tags,
    colorscale: colorscale,
    zmin: zmin, zmax: zmax,
    customdata: customdata,
    hovertemplate: "POS=%{y}<br>layer=%{x}<br>" + cfg.label +
                   "=%{z:.3f}<br>n_words=%{customdata}<extra></extra>",
    colorbar: { thickness: 12, len: 0.9, tickfont: { size: 10 } },
  };
  const layout = layoutCopy({
    title: { text: `${cfg.label} — ${source} (${agg})`, font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, {
      title: "layer", tickfont: { size: 10 }, type: "category",
    }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, {
      title: "POS", tickfont: { size: 11 }, autorange: "reversed",
    }),
    margin: { l: 65, r: 60, t: 35, b: 45 },
  });
  Plotly.newPlot("pos-heatmap", [trace], layout, PLOT_CONFIG);

  document.getElementById("pos-heatmap-legend").innerHTML =
    cfg.diverging
      ? `${cfg.range[0].toFixed(1)} <span class="swatch" style="background: linear-gradient(to right, #b6533a, #f6f3ec, #2a4d6e);"></span> ${cfg.range[1].toFixed(1)}  (wide ↔ deep)`
      : `${zmin.toFixed(2)} <span class="swatch" style="background: linear-gradient(to right, #f6f3ec, #d8c98c, #2a4d6e);"></span> ${zmax.toFixed(2)}`;
}

function drawPOSSingleLayer(source) {
  const pos = SOURCES[source].agg.pos;
  const agg = document.getElementById("pos-agg").value;
  const layer = parseInt(document.getElementById("pos-single-layer").value);
  const tags = getPOSTagsPresent(pos, MIN_WORDS_PER_CELL);
  const wordCounts = tags.map(t => pos.n_words_per_pos[t] || 0);

  function _bar(elId, metric, color, refLine) {
    const ys = tags.map(t => {
      const arr = _layerArray(pos, metric, agg, t);
      return (arr && arr[layer] !== undefined && arr[layer] !== null) ? arr[layer] : null;
    });
    let colors = color;
    if (metric === "update_pref" || metric === "gate_pref") {
      colors = ys.map(v => v === null ? "rgba(150,150,150,0.4)" :
        (v >= 0.5
          ? `rgba(${RGB_DEEP[0]},${RGB_DEEP[1]},${RGB_DEEP[2]},${0.4 + (v - 0.5)})`
          : `rgba(${RGB_WIDE[0]},${RGB_WIDE[1]},${RGB_WIDE[2]},${0.4 + (0.5 - v)})`));
    }
    const layout = layoutCopy({
      title: { text: `${POS_METRIC_CFG[metric].label} @ layer ${layer}`, font: { size: 12 } },
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 10 } }),
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis,
        POS_METRIC_CFG[metric].range ? { range: POS_METRIC_CFG[metric].range } : {}),
      shapes: refLine ? [{
        type: "line", x0: -0.5, x1: tags.length - 0.5, y0: refLine, y1: refLine,
        line: { color: "rgba(0,0,0,0.4)", width: 1, dash: "dot" },
      }] : [],
    });
    Plotly.newPlot(elId, [{
      type: "bar", x: tags, y: ys,
      marker: { color: colors },
      customdata: wordCounts,
      hovertemplate: "%{x}<br>" + POS_METRIC_CFG[metric].label +
                     " = %{y:.3f}<br>n_words = %{customdata}<extra></extra>",
    }], layout, PLOT_CONFIG);
  }

  _bar("pos-single-update-pref", "update_pref",       COLOR_DEEP, 0.5);
  _bar("pos-single-gate-pref",   "gate_pref",         COLOR_DEEP, 0.5);
  _bar("pos-single-steps",       "expected_steps",    COLOR_NEUTRAL, null);
  _bar("pos-single-loopdisp",    "loop_displacement", COLOR_DEEP, null);
}

function drawPOSLines(source) {
  const pos = SOURCES[source].agg.pos;
  const agg = document.getElementById("pos-agg").value;
  const metric = document.getElementById("pos-lines-metric").value;
  const cfg = POS_METRIC_CFG[metric];
  const tags = getPOSTagsPresent(pos, MIN_WORDS_PER_CELL);

  const traces = tags.map(tag => {
    const ys = _layerArray(pos, metric, agg, tag);
    const color = POS_PALETTE[tag] || "#888";
    return {
      type: "scatter", mode: "lines+markers",
      x: ys ? ys.map((_, i) => i) : [],
      y: ys || [],
      name: tag + ` (n=${pos.n_words_per_pos[tag]})`,
      line: { color: color, width: 1.5 },
      marker: { color: color, size: 5 },
      hovertemplate: tag + "<br>layer %{x}: " + cfg.label + "=%{y:.3f}<extra></extra>",
    };
  });

  const shapes = (metric === "update_pref" || metric === "gate_pref") ? [{
    type: "line", x0: 0, x1: META.n_layer - 1, y0: 0.5, y1: 0.5,
    line: { color: "rgba(0,0,0,0.4)", width: 1, dash: "dot" },
  }] : [];

  const layout = layoutCopy({
    title: { text: `${cfg.label} across layers, per POS — ${source}`, font: { size: 12 } },
    showlegend: true,
    legend: { font: { size: 10 }, orientation: "v", x: 1.02, y: 1 },
    margin: { l: 50, r: 140, t: 30, b: 50 },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "layer", dtick: 1 }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis,
      cfg.range ? { range: cfg.range } : {}),
    shapes: shapes,
  });
  Plotly.newPlot("pos-lines", traces, layout, PLOT_CONFIG);
}

function drawDecisive(source) {
  const pos = SOURCES[source].agg.pos;
  if (!pos || !pos.decisive || Object.keys(pos.decisive).length === 0) {
    for (const id of ["decisive-pref", "decisive-layer"]) {
      document.getElementById(id).innerHTML =
        '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">no decisive-layer data</div>';
    }
    return;
  }
  const tags = POS_ORDER.filter(t => pos.decisive[t] && pos.decisive[t].n_words >= MIN_WORDS_PER_CELL);
  const prefs = tags.map(t => pos.decisive[t].mean_pref_at_decisive);
  const layers = tags.map(t => pos.decisive[t].mean_decisive_layer);
  const counts = tags.map(t => pos.decisive[t].n_words);

  const prefColors = prefs.map(v => v >= 0.5
    ? `rgba(${RGB_DEEP[0]},${RGB_DEEP[1]},${RGB_DEEP[2]},${0.4 + (v - 0.5)})`
    : `rgba(${RGB_WIDE[0]},${RGB_WIDE[1]},${RGB_WIDE[2]},${0.4 + (0.5 - v)})`);
  Plotly.newPlot("decisive-pref", [{
    type: "bar", x: tags, y: prefs,
    marker: { color: prefColors },
    customdata: counts,
    hovertemplate: "%{x}<br>update_pref at decisive layer = %{y:.3f}<br>n_words = %{customdata}<extra></extra>",
  }], layoutCopy({
    title: { text: "update_pref at the decisive layer, per POS", font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 10 } }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1] }),
    shapes: [{
      type: "line", x0: -0.5, x1: tags.length - 0.5, y0: 0.5, y1: 0.5,
      line: { color: "rgba(0,0,0,0.4)", width: 1, dash: "dot" },
    }],
  }), PLOT_CONFIG);

  Plotly.newPlot("decisive-layer", [{
    type: "bar", x: tags, y: layers,
    marker: { color: COLOR_NEUTRAL },
    customdata: counts,
    hovertemplate: "%{x}<br>mean decisive layer = %{y:.2f}<br>n_words = %{customdata}<extra></extra>",
  }], layoutCopy({
    title: { text: "mean decisive layer index, per POS", font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 10 } }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, META.n_layer - 1] }),
  }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function init() {
  fillMeta();
  fillSourceSelectors();
  fillLayerSelector();
  drawOverview();

  // Pick intelligent defaults for A and B
  let srcA = META.source_order.find(s => s.includes("gsm")) || META.source_order[0];
  let srcB = META.source_order.find(s => s.includes("trivia") || s.includes("cap")) || 
             (META.source_order.length > 1 ? META.source_order[1] : META.source_order[0]);

  document.getElementById("source-select-2").value = srcA;
  document.getElementById("source-select-3").value = srcA;
  document.getElementById("source-select-4").value = srcA;
  document.getElementById("source-select-aligned").value = srcA;
  
  document.getElementById("source-diff-a").value = srcA;
  document.getElementById("source-diff-b").value = srcB;

  const ss5 = document.getElementById("source-select-5");
  if (ss5) ss5.value = srcA;

  drawSourceDeepDive(srcA);
  drawLayerDynamics(srcA);
  fillChunkSelector(srcA);
  renderTokens();
  drawPOSAll(srcA);
  
  // Init Event-Aligned Plots
  updateAnchorSelector(srcA);
  drawAlignedTrajectories(srcA);
  updateDiffAnchorSelector();
  drawAlignedDiff();
  updateBVAAnchorSelector();
  drawBeforeVsAfter();

  // Wiring Event Listeners
  document.getElementById("source-select-2").addEventListener("change", e => drawSourceDeepDive(e.target.value));
  document.getElementById("source-select-3").addEventListener("change", e => drawLayerDynamics(e.target.value));
  document.getElementById("source-select-4").addEventListener("change", e => {
    fillChunkSelector(e.target.value);
    lockedToken = null;
    renderTokens();
  });
  
  document.getElementById("source-select-aligned").addEventListener("change", e => {
    updateAnchorSelector(e.target.value);
    drawAlignedTrajectories(e.target.value);
  });
  document.getElementById("anchor-select").addEventListener("change", () => {
    drawAlignedTrajectories(document.getElementById("source-select-aligned").value);
  });
  document.getElementById("show-samples-toggle").addEventListener("change", () => {
    drawAlignedTrajectories(document.getElementById("source-select-aligned").value);
  });
  
  document.getElementById("source-diff-a").addEventListener("change", () => {
    updateDiffAnchorSelector(); drawAlignedDiff();
  });
  document.getElementById("source-diff-b").addEventListener("change", () => {
    updateDiffAnchorSelector(); drawAlignedDiff();
  });
  document.getElementById("anchor-select-diff").addEventListener("change", drawAlignedDiff);

  document.getElementById("anchor-select-bva").addEventListener("change", drawBeforeVsAfter);
  document.getElementById("bva-window").addEventListener("change", drawBeforeVsAfter);

  document.getElementById("chunk-select").addEventListener("change", () => {
    lockedToken = null;
    renderTokens();
  });
  document.getElementById("color-metric").addEventListener("change", renderTokens);
  document.getElementById("layer-select").addEventListener("change", renderTokens);
  document.getElementById("show-pos-toggle").addEventListener("change", renderTokens);
  document.getElementById("grid-metric").addEventListener("change", drawRoutingGrid);

  if (ss5) {
    ss5.addEventListener("change", e => drawPOSAll(e.target.value));
    document.getElementById("pos-agg").addEventListener("change", () => {
      drawPOSAll(document.getElementById("source-select-5").value);
    });
    document.getElementById("pos-heatmap-metric").addEventListener("change", () => {
      drawPOSHeatmap(document.getElementById("source-select-5").value);
    });
    document.getElementById("pos-single-layer").addEventListener("change", () => {
      drawPOSSingleLayer(document.getElementById("source-select-5").value);
    });
    document.getElementById("pos-lines-metric").addEventListener("change", () => {
      drawPOSLines(document.getElementById("source-select-5").value);
    });
  }
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()