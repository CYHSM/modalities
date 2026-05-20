#!/usr/bin/env python
"""
Generate a single-file interactive HTML viewer from paloma_diagnostics parquets.

Section 5 (POS analysis) now has three subsections:
  5a — POS × layer heatmap with a metric dropdown (the main "is there
       structure?" view). Layer-collapsed averages hide most of the signal,
       so this is what you want to look at first.
  5b — POS-only bars averaged across ALL layers (the original §5 view).
       Useful as a contrast to show what's lost by collapsing layers.
  5c — POS-only bars for a SINGLE selected layer, with a layer dropdown.
       Useful for drilling into a layer flagged by the heatmap.

Per-layer POS aggregates are stored as:
    pos.by_pos_layer_mean[metric][tag] = [n_layers values]
    pos.by_pos_layer_max [metric][tag] = [n_layers values]
plus pos.n_words_per_pos[tag] for sample size.
"""

import argparse
import base64
import json
import random
from collections import Counter
from pathlib import Path

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


def aggregate_source(rows, n_layer, max_loops, pos_map=None, source_name="?"):
    gd = _stack_3d(rows, "gate_deep")            # (L, total_T)
    gw = _stack_3d(rows, "gate_wide")
    es = _stack_3d(rows, "expected_steps")
    dd = _stack_3d(rows, "delta_deep_norm")
    dw = _stack_3d(rows, "delta_wide_norm")
    dc = _stack_3d(rows, "delta_cos_sim")
    cwd = _stack_3d(rows, "cross_w2d_norm")
    cdw = _stack_3d(rows, "cross_d2w_norm")
    ld = _stack_3d(rows, "loop_displacement")
    loss = _flat_1d(rows, "loss")

    if gd is not None and gw is not None:
        denom = (gd + gw)
        denom = np.where(denom < 1e-6, 1e-6, denom)
        gdr = gd / denom
    else:
        gdr = None

    if gd is not None and dd is not None:
        d_c = gd * dd
        w_c = gw * dw
        denom = d_c + w_c
        denom = np.where(denom < 1e-6, 1e-6, denom)
        dcs = d_c / denom
    else:
        dcs = None

    halt = np.stack([np.array(r["step_halt_probs"], dtype=np.float32) for r in rows])
    halt_mean = halt.mean(axis=0)
    loop_scales = np.stack([np.array(r["step_loop_scales"], dtype=np.float32) for r in rows]).mean(axis=0)
    step_disp = np.stack([np.array(r["step_displacement"], dtype=np.float32) for r in rows]).mean(axis=0)

    summary = {
        "n_chunks": len(rows),
        "n_tokens": int(gd.shape[1]) if gd is not None else 0,
        "mean_gate_deep": _safe_mean(gd),
        "mean_gate_wide": _safe_mean(gw),
        "mean_gate_deep_relative": _safe_mean(gdr),
        "mean_deep_contribution_share": _safe_mean(dcs),
        "mean_expected_steps": _safe_mean(es),
        "mean_loop_displacement": _safe_mean(ld),
        "mean_loss": float(np.nanmean(loss)) if loss is not None else None,
        "mean_delta_cos_sim": _safe_mean(dc),
    }

    per_layer = {
        "gate_deep": _safe_mean(gd, axis=1),
        "gate_wide": _safe_mean(gw, axis=1),
        "gate_deep_relative": _safe_mean(gdr, axis=1),
        "deep_contribution_share": _safe_mean(dcs, axis=1),
        "expected_steps": _safe_mean(es, axis=1),
        "loop_displacement": _safe_mean(ld, axis=1),
        "delta_deep_norm": _safe_mean(dd, axis=1),
        "delta_wide_norm": _safe_mean(dw, axis=1),
        "delta_cos_sim": _safe_mean(dc, axis=1),
        "cross_w2d_norm": _safe_mean(cwd, axis=1),
        "cross_d2w_norm": _safe_mean(cdw, axis=1),
        "halt_probs": halt_mean.tolist(),
        "loop_scales": loop_scales.tolist(),
        "step_displacement": step_disp.tolist(),
    }

    hist = {
        "gate_deep": _hist(gd.ravel() if gd is not None else None, 40, (0.0, 1.0)),
        "gate_wide": _hist(gw.ravel() if gw is not None else None, 40, (0.0, 1.0)),
        "gate_deep_relative": _hist(gdr.ravel() if gdr is not None else None, 40, (0.0, 1.0)),
        "deep_contribution_share": _hist(dcs.ravel() if dcs is not None else None, 40, (0.0, 1.0)),
        "expected_steps": _hist(es.ravel() if es is not None else None, 40, (0.0, float(max_loops))),
        "loop_displacement": _hist(ld.ravel() if ld is not None else None, 40, (0.0, 2.0)),
        "delta_cos_sim": _hist(dc.ravel() if dc is not None else None, 40, (-1.0, 1.0)),
        "loss": _hist(loss, 40, (0.0, 15.0)),
    }

    per_layer_hist_gate = []
    per_layer_hist_steps = []
    if gd is not None:
        for li in range(gd.shape[0]):
            per_layer_hist_gate.append(_hist(gd[li], 30, (0.0, 1.0)))
            per_layer_hist_steps.append(_hist(es[li], 30, (0.0, float(max_loops))))

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

        # Build word groups (consecutive same-tag tokens become one word,
        # except PUNCT/SYM/SPECIAL which stay one-per-word).
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

        # Metrics that DO have a layer axis -> per-layer-per-POS aggregates.
        # We compute per-word mean and max separately at each layer; the word
        # group boundaries are identical across layers because POS doesn't
        # change with layer, so the same starts/ends array applies.
        #
        # The 'per-layer' metrics here are exactly the ones in `per_layer`
        # above that have an L axis. Loss has no L axis so it stays
        # POS-only (handled below).
        layered_metrics_full = {
            "gate_deep_relative":      gdr,   # (L, total_T)
            "deep_contribution_share": dcs,
            "expected_steps":          es,
            "loop_displacement":       ld,
            "delta_cos_sim":           dc,
            "delta_deep_norm":         dd,
            "delta_wide_norm":         dw,
            "gate_deep":               gd,
            "gate_wide":               gw,
        }

        # ---- Per-word aggregates at each layer ----
        # For each layer, compute per-word mean and max via reduceat, just
        # like the existing single-layer code does. We end up with
        # by_pos_layer_<agg>[metric][tag] = list of n_layer values.
        n_layers_actual = gd.shape[0] if gd is not None else n_layer
        lengths = (ends_arr - starts_arr).astype(np.float32) if len(starts_arr) > 0 else np.array([])

        def per_word_reduce(flat_1d):
            """Given (total_T,) array, return per-word (mean, max) over word groups."""
            if len(starts_arr) == 0:
                return np.array([]), np.array([])
            safe = np.where(np.isfinite(flat_1d), flat_1d, 0.0)
            sums = np.add.reduceat(safe, starts_arr)
            means = sums / np.maximum(lengths, 1.0)
            neg_inf = np.where(np.isfinite(flat_1d), flat_1d, -np.inf)
            maxes = np.maximum.reduceat(neg_inf, starts_arr)
            maxes = np.where(np.isinf(maxes), np.nan, maxes)
            return means.astype(np.float32), maxes.astype(np.float32)

        # Also need loss handled per-POS (single value per token, no layer axis).
        loss_means_per_word = loss_maxes_per_word = None
        if loss is not None and len(starts_arr) > 0:
            loss_means_per_word, loss_maxes_per_word = per_word_reduce(loss)

        # Token-collapsed-over-layers version for the layer-averaged §5b view.
        # We compute this per-word by averaging the metric across layers FIRST
        # (giving one value per token), then aggregating per-word. That matches
        # how the original §5 worked.
        if gd is not None and len(starts_arr) > 0:
            buckets_mean = {}   # for §5b
            buckets_max = {}

            # Per-token average across layers, for each metric.
            tok_collapsed = {
                "gate_deep_relative":      np.nanmean(gdr, axis=0),
                "deep_contribution_share": np.nanmean(dcs, axis=0),
                "expected_steps":          np.nanmean(es,  axis=0),
                "loop_displacement":       np.nanmean(ld,  axis=0),
                "delta_cos_sim":           np.nanmean(dc,  axis=0),
                "delta_deep_norm":         np.nanmean(dd,  axis=0),
                "delta_wide_norm":         np.nanmean(dw,  axis=0),
            }
            collapsed_means = {}
            collapsed_maxes = {}
            for name, tok_vals in tok_collapsed.items():
                m, mx = per_word_reduce(tok_vals)
                collapsed_means[name] = m
                collapsed_maxes[name] = mx

            unique_tags = sorted(set(tags_per_group.tolist()))
            for tag in unique_tags:
                mask = (tags_per_group == tag)
                n = int(mask.sum())
                with np.errstate(all="ignore"):
                    buckets_mean[tag] = {
                        "n_words": n,
                        "gate_deep_relative":      float(np.nanmean(collapsed_means["gate_deep_relative"][mask])),
                        "deep_contribution_share": float(np.nanmean(collapsed_means["deep_contribution_share"][mask])),
                        "expected_steps":          float(np.nanmean(collapsed_means["expected_steps"][mask])),
                        "loop_displacement":       float(np.nanmean(collapsed_means["loop_displacement"][mask])),
                        "delta_cos_sim":           float(np.nanmean(collapsed_means["delta_cos_sim"][mask])),
                        "delta_deep_norm":         float(np.nanmean(collapsed_means["delta_deep_norm"][mask])),
                        "delta_wide_norm":         float(np.nanmean(collapsed_means["delta_wide_norm"][mask])),
                        "loss": float(np.nanmean(loss_means_per_word[mask])) if loss_means_per_word is not None else None,
                    }
                    buckets_max[tag] = {
                        "n_words": n,
                        "gate_deep_relative":      float(np.nanmean(collapsed_maxes["gate_deep_relative"][mask])),
                        "deep_contribution_share": float(np.nanmean(collapsed_maxes["deep_contribution_share"][mask])),
                        "expected_steps":          float(np.nanmean(collapsed_maxes["expected_steps"][mask])),
                        "loop_displacement":       float(np.nanmean(collapsed_maxes["loop_displacement"][mask])),
                        "delta_cos_sim":           float(np.nanmean(collapsed_maxes["delta_cos_sim"][mask])),
                        "delta_deep_norm":         float(np.nanmean(collapsed_maxes["delta_deep_norm"][mask])),
                        "delta_wide_norm":         float(np.nanmean(collapsed_maxes["delta_wide_norm"][mask])),
                        "loss": float(np.nanmean(loss_maxes_per_word[mask])) if loss_maxes_per_word is not None else None,
                    }

            # ---- Per-layer-per-POS aggregates (§5a heatmap & §5c bars) ----
            # For each (metric, layer), compute per-word reductions then group
            # by tag. End up with:
            #   by_pos_layer_mean[metric][tag] = [val_layer_0, ..., val_layer_L-1]
            # This is the dataset for the heatmap.
            by_pos_layer_mean = {}
            by_pos_layer_max  = {}
            for metric_name, full_arr in layered_metrics_full.items():
                # full_arr shape (L, total_T)
                per_tag_layers_mean = {tag: [None] * n_layers_actual for tag in unique_tags}
                per_tag_layers_max  = {tag: [None] * n_layers_actual for tag in unique_tags}
                for li in range(n_layers_actual):
                    layer_vals = full_arr[li]  # (total_T,)
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
        else:
            buckets_mean = {}
            buckets_max = {}
            by_pos_layer_mean = {}
            by_pos_layer_max = {}

        coverage = float(np.mean(np.isin(pos_arr, ["UNKNOWN", "SPECIAL"], invert=True)))

        # Word counts per POS (sample size; used to grey out sparse cells).
        n_words_per_pos = {}
        if len(tags_per_group) > 0:
            for tag in sorted(set(tags_per_group.tolist())):
                n_words_per_pos[tag] = int((tags_per_group == tag).sum())

        pos_agg = {
            "coverage": coverage,
            "by_pos_mean": buckets_mean,                 # §5b (layer-averaged)
            "by_pos_max":  buckets_max,
            "by_pos_layer_mean": by_pos_layer_mean,      # §5a, §5c (per-layer)
            "by_pos_layer_max":  by_pos_layer_max,
            "n_words_per_pos": n_words_per_pos,
        }

    return {
        "summary": summary,
        "per_layer": per_layer,
        "hist": hist,
        "per_layer_hist_gate": per_layer_hist_gate,
        "per_layer_hist_steps": per_layer_hist_steps,
        "pos": pos_agg,
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
                               source_name=source)

        sample = sample_chunks_for_explorer(rows, args.n_explorer_chunks, rng)

        N_TOK = args.n_explorer_tokens
        per_layer_keys = ("gate_deep", "gate_wide", "expected_steps",
                          "delta_deep_norm", "delta_wide_norm", "delta_cos_sim",
                          "cross_w2d_norm", "cross_d2w_norm", "loop_displacement")
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
section h2 .num {
  color: var(--accent);
  margin-right: 8px;
}
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
.chart { height: 320px; }
.chart-tall { height: 420px; }
.chart-heatmap { height: 520px; }
.chart-small { height: 240px; }
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
.tok {
  padding: 1px 1px;
  border-radius: 2px;
  cursor: pointer;
  transition: outline 0.05s;
}
.tok:hover, .tok.selected {
  outline: 1.5px solid var(--accent);
  outline-offset: 0px;
}
.detail-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 16px;
  font-family: var(--mono);
  font-size: 12px;
  min-height: 200px;
}
.detail-panel h4 {
  margin: 0 0 8px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
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
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
  .token-explorer-layout { grid-template-columns: 1fr; }
}
.note {
  font-style: italic; color: var(--muted); font-size: 13px; margin: 4px 0 16px;
}
</style>
</head>
<body>

<header>
  <h1>paloma diagnostics</h1>
  <div class="meta" id="model-meta"></div>
</header>

<main>

<section>
  <h2><span class="num">§1</span> cross-source overview</h2>
  <h3>How does the model route across Paloma sources?</h3>
  <p class="note">Each bar is the mean over all tokens in that source, aggregated across all layers and all chunks.</p>
  <div class="grid-3">
    <div id="overview-gate" class="chart"></div>
    <div id="overview-steps" class="chart"></div>
    <div id="overview-loss" class="chart"></div>
  </div>
  <div class="grid-3">
    <div id="overview-loop-disp" class="chart"></div>
    <div></div>
    <div></div>
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
  <div class="grid-3">
    <div id="hist-gate" class="chart"></div>
    <div id="hist-steps" class="chart"></div>
    <div id="hist-cos" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="hist-loss" class="chart"></div>
    <div id="hist-gate-wide" class="chart"></div>
  </div>
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
    <div id="layer-steps" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="layer-loop-disp" class="chart"></div>
    <div id="layer-preference" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="layer-delta" class="chart"></div>
    <div id="layer-cos" class="chart"></div>
  </div>
  <div class="grid-2">
    <div id="layer-cross" class="chart"></div>
    <div id="halt-curves" class="chart-tall"></div>
  </div>
  <div class="grid-2">
    <div id="step-disp-curves" class="chart-tall"></div>
    <div></div>
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
      <optgroup label="preference (diverging deep↔wide)">
        <option value="gate_deep_relative" selected>gate_deep_relative</option>
        <option value="deep_contribution_share">deep_contribution_share</option>
        <option value="delta_cos_sim">delta_cos_sim (path agreement)</option>
      </optgroup>
      <optgroup label="raw gates (sequential)">
        <option value="gate_deep">gate_deep</option>
        <option value="gate_wide">gate_wide</option>
      </optgroup>
      <optgroup label="compute / loop">
        <option value="expected_steps">expected_steps</option>
        <option value="loop_displacement">loop_displacement (did loop work?)</option>
      </optgroup>
      <optgroup label="updates / loss">
        <option value="loss">loss</option>
        <option value="delta_deep_norm">delta_deep_norm</option>
        <option value="delta_wide_norm">delta_wide_norm</option>
      </optgroup>
      <optgroup label="POS (categorical)">
        <option value="pos">pos (categorical color)</option>
      </optgroup>
    </select>
    <label>layer</label>
    <select id="layer-select"></select>
    <label><input type="checkbox" id="show-pos-toggle"> show POS</label>
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
  <h2><span class="num">§5</span> POS analysis</h2>
  <h3>Function vs content words — does the router actually care about syntax?</h3>
  <p class="note">UPOS tags from spaCy, aligned to subword tokens. Multi-token
    words can be aggregated by <em>mean</em> (average compute the word received)
    or <em>max</em> (peak compute triggered by any subword). For non-English
    or code-heavy sources, POS tags are unreliable — see coverage below.</p>
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

  <!-- §5a: POS × layer heatmap with metric dropdown -->
  <div class="subsection">
    <h4>§5a — POS × layer heatmap</h4>
    <h5>The averaging across layers in §5b hides most of the signal. Here it isn't averaged.</h5>
    <div class="controls">
      <label>metric</label>
      <select id="pos-heatmap-metric">
        <option value="gate_deep_relative" selected>gate_deep_relative</option>
        <option value="deep_contribution_share">deep_contribution_share</option>
        <option value="expected_steps">expected_steps</option>
        <option value="loop_displacement">loop_displacement</option>
        <option value="delta_cos_sim">delta_cos_sim</option>
        <option value="delta_deep_norm">delta_deep_norm</option>
        <option value="delta_wide_norm">delta_wide_norm</option>
        <option value="gate_deep">gate_deep</option>
        <option value="gate_wide">gate_wide</option>
      </select>
      <span class="legend" id="pos-heatmap-legend"></span>
    </div>
    <div id="pos-heatmap" class="chart-heatmap"></div>
    <p class="note">Cell value = mean over all words tagged that POS, at that layer.
      Hover for exact value and word count. Tags with fewer than 3 words are dropped.</p>
  </div>

  <!-- §5b: averaged-across-layers bars (the original §5 view) -->
  <div class="subsection">
    <h4>§5b — averaged across all layers</h4>
    <h5>What the §5a heatmap collapses to if you average over the layer axis.
      Useful only as a contrast — most POS effects are layer-specific.</h5>
    <div class="grid-2">
      <div id="pos-bar-dcs" class="chart-tall"></div>
      <div id="pos-bar-loopdisp" class="chart-tall"></div>
    </div>
    <div class="grid-2">
      <div id="pos-bar-steps" class="chart-tall"></div>
      <div id="pos-bar-loss" class="chart-tall"></div>
    </div>
  </div>

  <!-- §5c: single-layer bars -->
  <div class="subsection">
    <h4>§5c — single layer</h4>
    <h5>Drill into a single layer. Use this to confirm a row in the §5a heatmap.</h5>
    <div class="controls">
      <label>layer</label>
      <select id="pos-single-layer"></select>
    </div>
    <div class="grid-2">
      <div id="pos-single-dcs" class="chart-tall"></div>
      <div id="pos-single-loopdisp" class="chart-tall"></div>
    </div>
    <div class="grid-2">
      <div id="pos-single-steps" class="chart-tall"></div>
      <div id="pos-single-gdr" class="chart-tall"></div>
    </div>
  </div>
</section>

<section>
  <h2><span class="num">§6</span> cross-source comparison</h2>
  <h3>Same heatmap, different sources. Is the routing pattern data-specific or model-specific?</h3>
  <p class="note">Shared colorscale across all panels so cells are directly comparable.
    Cells with fewer than 10 words for a (POS, source) are dropped — appears as a grey gap.</p>
  <div class="controls">
    <label>metric</label>
    <select id="pos-compare-metric">
      <option value="gate_deep_relative" selected>gate_deep_relative</option>
      <option value="deep_contribution_share">deep_contribution_share</option>
      <option value="expected_steps">expected_steps</option>
      <option value="loop_displacement">loop_displacement</option>
      <option value="delta_cos_sim">delta_cos_sim</option>
      <option value="delta_deep_norm">delta_deep_norm</option>
      <option value="delta_wide_norm">delta_wide_norm</option>
      <option value="gate_deep">gate_deep</option>
      <option value="gate_wide">gate_wide</option>
    </select>
    <label>aggregation</label>
    <select id="pos-compare-agg">
      <option value="mean" selected>mean over subwords</option>
      <option value="max">max over subwords</option>
    </select>
  </div>

  <!-- §6a: side-by-side, one heatmap per source -->
  <div class="subsection">
    <h4>§6a — side-by-side per source</h4>
    <h5>If the layer pattern is identical across sources, the model has a fixed
      routing schedule. If POS rows light up differently per source, the model
      adapts routing to content.</h5>
    <div id="pos-compare-panels"></div>
  </div>

  <!-- §6b: difference heatmap -->
  <div class="subsection">
    <h4>§6b — difference (A − B)</h4>
    <h5>Subtraction makes data-specific routing visible. Blue cells: source A
      routes more deep than B. Red cells: A routes more wide than B.</h5>
    <div class="controls">
      <label>source A</label>
      <select id="pos-diff-a"></select>
      <label>−</label>
      <label>source B</label>
      <select id="pos-diff-b"></select>
    </div>
    <div id="pos-diff-heatmap" class="chart-heatmap"></div>
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
// §1 Overview bars
// ---------------------------------------------------------------------------
const RGB_DEEP    = [42, 77, 110];
const RGB_WIDE    = [182, 83, 58];
const RGB_NEUTRAL = [246, 243, 236];

function drawOverview() {
  const sources = META.source_order;
  const dcs = sources.map(s => SOURCES[s].agg.summary.mean_deep_contribution_share);
  const es = sources.map(s => SOURCES[s].agg.summary.mean_expected_steps);
  const ld = sources.map(s => SOURCES[s].agg.summary.mean_loop_displacement);
  const loss = sources.map(s => SOURCES[s].agg.summary.mean_loss);

  const baseBarLayout = (title, range) => layoutCopy({
    title: { text: title, font: { size: 12 } },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, range ? { range: range } : {}),
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 9 } }),
  });

  const dcs_colors = dcs.map(v => v >= 0.5
    ? `rgba(${RGB_DEEP[0]},${RGB_DEEP[1]},${RGB_DEEP[2]},${0.4 + (v - 0.5)})`
    : `rgba(${RGB_WIDE[0]},${RGB_WIDE[1]},${RGB_WIDE[2]},${0.4 + (0.5 - v)})`);
  Plotly.newPlot("overview-gate", [{
    type: "bar", x: sources, y: dcs,
    marker: { color: dcs_colors },
    hovertemplate: "%{x}<br>deep contrib share: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean deep_contribution_share per source (0.5 = balanced)", [0, 1]),
     PLOT_CONFIG);

  Plotly.newPlot("overview-steps", [{
    type: "bar", x: sources, y: es,
    marker: { color: COLOR_NEUTRAL }, hovertemplate: "%{x}<br>expected_steps: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean expected_steps per source", [0, META.max_loops]), PLOT_CONFIG);

  Plotly.newPlot("overview-loss", [{
    type: "bar", x: sources, y: loss,
    marker: { color: COLOR_WIDE }, hovertemplate: "%{x}<br>loss: %{y:.3f}<extra></extra>",
  }], baseBarLayout("mean per-token loss per source", null), PLOT_CONFIG);

  const ldEl = document.getElementById("overview-loop-disp");
  if (ldEl) {
    Plotly.newPlot("overview-loop-disp", [{
      type: "bar", x: sources, y: ld,
      marker: { color: COLOR_DEEP },
      hovertemplate: "%{x}<br>loop_displacement: %{y:.3f}<extra></extra>",
    }], baseBarLayout("mean loop_displacement per source — did the loop work?", null),
       PLOT_CONFIG);
  }
}

function fillSourceSelectors() {
  for (const id of ["source-select-2", "source-select-3", "source-select-4", "source-select-5"]) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    for (const s of META.source_order) {
      const o = document.createElement("option");
      o.value = s; o.textContent = s;
      sel.appendChild(o);
    }
  }
}

function fillLayerSelector() {
  const sel = document.getElementById("layer-select");
  for (let i = 0; i < META.n_layer; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = "layer " + i;
    sel.appendChild(o);
  }
  sel.value = Math.floor(META.n_layer / 2);
  // Also fill the §5c per-layer selector with the same options.
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
    <div class="kpi"><span class="label">mean gate_deep_rel</span><span class="val">${s.mean_gate_deep_relative.toFixed(3)}</span><span class="sub">g_d / (g_d + g_w)</span></div>
    <div class="kpi"><span class="label">mean deep contrib</span><span class="val">${s.mean_deep_contribution_share.toFixed(3)}</span><span class="sub">share of update</span></div>
    <div class="kpi"><span class="label">mean expected_steps</span><span class="val">${s.mean_expected_steps.toFixed(3)}</span><span class="sub">out of ${META.max_loops}</span></div>
    <div class="kpi"><span class="label">mean loop_disp</span><span class="val">${s.mean_loop_displacement.toFixed(3)}</span><span class="sub">||h_final − h_step1||/||h_step1||</span></div>
    <div class="kpi"><span class="label">mean loss</span><span class="val">${s.mean_loss.toFixed(3)}</span></div>
    <div class="kpi"><span class="label">mean cos(Δd,Δw)</span><span class="val">${s.mean_delta_cos_sim.toFixed(3)}</span></div>
  `;
  Plotly.newPlot("hist-gate", [histTrace(agg.hist.gate_deep_relative, COLOR_DEEP, "gate_deep_relative")],
    layoutCopy({ title: { text: "gate_deep_relative — g_d / (g_d + g_w)", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-gate-wide", [histTrace(agg.hist.deep_contribution_share, COLOR_DEEP, "deep_contribution_share")],
    layoutCopy({ title: { text: "deep_contribution_share — share of update from deep", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-steps", [histTrace(agg.hist.expected_steps, COLOR_NEUTRAL, "expected_steps")],
    layoutCopy({ title: { text: "expected_steps distribution (all layers)", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-cos", [histTrace(agg.hist.delta_cos_sim, COLOR_NEUTRAL, "delta_cos_sim")],
    layoutCopy({ title: { text: "cosine(Δdeep, Δwide) — do paths do different things?", font: { size: 12 } } }), PLOT_CONFIG);
  Plotly.newPlot("hist-loss", [histTrace(agg.hist.loop_displacement, COLOR_DEEP, "loop_displacement")],
    layoutCopy({ title: { text: "loop_displacement — did loop work past step 1?", font: { size: 12 } } }), PLOT_CONFIG);
}

function lineTrace(y, color, name) {
  const x = y.map((_, i) => i);
  return { type: "scatter", mode: "lines+markers", x: x, y: y,
           marker: { color: color, size: 6 }, line: { color: color, width: 1.5 },
           name: name,
           hovertemplate: name + "<br>layer %{x}: %{y:.3f}<extra></extra>" };
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
  Plotly.newPlot("layer-steps", [
    lineTrace(pl.expected_steps, COLOR_NEUTRAL, "expected_steps"),
  ], layoutCopy({
    title: { text: "mean expected_steps per layer", font: { size: 12 } },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, META.max_loops] }),
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-loop-disp", [
    lineTrace(pl.loop_displacement, COLOR_DEEP, "loop_displacement"),
  ], layoutCopy({
    title: { text: "mean loop_displacement per layer — did the loop work?", font: { size: 12 } },
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-preference", [
    lineTrace(pl.gate_deep_relative, COLOR_DEEP, "gate_deep_relative"),
    lineTrace(pl.deep_contribution_share, COLOR_WIDE, "deep_contribution_share"),
  ], layoutCopy({
    title: { text: "deep preference per layer (0.5 = balanced)", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "h", y: 1.1 },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [0, 1] }),
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-delta", [
    lineTrace(pl.delta_deep_norm, COLOR_DEEP, "‖Δdeep‖"),
    lineTrace(pl.delta_wide_norm, COLOR_WIDE, "‖Δwide‖"),
  ], layoutCopy({
    title: { text: "mean update magnitude per layer", font: { size: 12 } },
    showlegend: true, legend: { font: { size: 10 }, orientation: "h", y: 1.1 },
  }), PLOT_CONFIG);
  Plotly.newPlot("layer-cos", [
    lineTrace(pl.delta_cos_sim, COLOR_NEUTRAL, "cos(Δd,Δw)"),
  ], layoutCopy({
    title: { text: "cosine similarity of path updates per layer", font: { size: 12 } },
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, { range: [-1, 1] }),
  }), PLOT_CONFIG);
  if (META.use_cross) {
    Plotly.newPlot("layer-cross", [
      lineTrace(pl.cross_w2d_norm, COLOR_DEEP, "w→d"),
      lineTrace(pl.cross_d2w_norm, COLOR_WIDE, "d→w"),
    ], layoutCopy({
      title: { text: "cross-path contamination magnitude per layer", font: { size: 12 } },
      showlegend: true, legend: { font: { size: 10 }, orientation: "h", y: 1.1 },
    }), PLOT_CONFIG);
  } else {
    document.getElementById("layer-cross").innerHTML =
      '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">use_cross=False — no cross-path data</div>';
  }
  const halts = pl.halt_probs;
  const halt_traces = halts.map((row, li) => ({
    type: "scatter", mode: "lines", x: row.map((_, i) => i), y: row,
    name: "layer " + li,
    line: { color: `hsl(${(li * 360 / META.n_layer) | 0}, 50%, 45%)`, width: 1.2 },
    hovertemplate: `layer ${li}<br>step %{x}: halt_prob=%{y:.3f}<extra></extra>`,
  }));
  Plotly.newPlot("halt-curves", halt_traces, layoutCopy({
    title: { text: "mean halt probability per (layer, step)", font: { size: 12 } },
    showlegend: true,
    legend: { font: { size: 9 }, orientation: "v", x: 1.02, y: 1 },
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
    showlegend: true,
    legend: { font: { size: 9 }, orientation: "v", x: 1.02, y: 1 },
    margin: { l: 50, r: 100, t: 30, b: 50 },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { title: "step" }),
  }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// §4 Token explorer (unchanged from previous version)
// ---------------------------------------------------------------------------
const METRIC_RANGES = {
  gate_deep_relative:      { range: [0, 1], cmap: "deepwide" },
  deep_contribution_share: { range: [0, 1], cmap: "deepwide" },
  delta_cos_sim:           { range: [-1, 1], cmap: "diverging_cos" },
  gate_deep: { range: [0, 1], cmap: "sequential_deep" },
  gate_wide: { range: [0, 1], cmap: "sequential_wide" },
  expected_steps:    { range: [0, META.max_loops], cmap: "viridis" },
  loop_displacement: { range: [0, 1], cmap: "viridis" },
  loss:            { range: [0, 10], cmap: "ylorrd" },
  delta_deep_norm: { range: null, cmap: "viridis" },
  delta_wide_norm: { range: null, cmap: "viridis" },
};
const POS_PALETTE = {
  NOUN:  "#2a4d6e", PROPN: "#3d6b94", VERB:  "#1f6f64", ADJ:   "#6b8e6e",
  ADV:   "#a3a661", NUM:   "#7b5fa8",
  DET:   "#d9905a", PRON:  "#c47054", ADP:   "#b6533a",
  CCONJ: "#a35a3f", SCONJ: "#a35a3f", PART:  "#9c5b4a", AUX:   "#bb6f4e",
  PUNCT: "#d4c2a8", SYM:   "#d4c2a8",
  INTJ:  "#9c9c9c", X:     "#9c9c9c", SPACE: "#cccccc",
  UNKNOWN: "#bdbdbd", SPECIAL: "#eeeeee",
};
const POS_CONTENT = new Set(["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"]);
const POS_FUNCTION = new Set(["DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART", "AUX", "PUNCT", "SYM"]);
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
  if (cfg.cmap === "deepwide" || cfg.cmap === "diverging_cos") {
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
  if (metric === "gate_deep_relative") {
    const gd = chunk.gate_deep[layer];
    const gw = chunk.gate_wide[layer];
    return gd.map((v, i) => {
      const denom = (v + gw[i]);
      return denom > 1e-6 ? v / denom : 0.5;
    });
  }
  if (metric === "deep_contribution_share") {
    const gd = chunk.gate_deep[layer];
    const gw = chunk.gate_wide[layer];
    const dd = chunk.delta_deep_norm[layer];
    const dw = chunk.delta_wide_norm[layer];
    return gd.map((g, i) => {
      const dc = g * dd[i];
      const wc = gw[i] * dw[i];
      const denom = dc + wc;
      return denom > 1e-6 ? dc / denom : 0.5;
    });
  }
  return chunk[metric][layer];
}

function renderTokens() {
  const chunk = getChunk();
  if (!chunk) return;
  const metric = document.getElementById("color-metric").value;
  const layer = parseInt(document.getElementById("layer-select").value);
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
  const layerStr = layer !== null ? ` @ layer ${layer}` : " (single value, no layer)";
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
    const gdr = (gd + gw) > 1e-6 ? gd / (gd + gw) : 0.5;
    const dcShare = (gd * dd + gw * dw) > 1e-6 ? (gd * dd) / (gd * dd + gw * dw) : 0.5;
    rows += `<tr>
      <td>${li}</td><td>${gd.toFixed(3)}</td><td>${gw.toFixed(3)}</td>
      <td><strong>${gdr.toFixed(2)}</strong></td><td><strong>${dcShare.toFixed(2)}</strong></td>
      <td>${chunk.expected_steps[li][idx].toFixed(2)}</td><td>${chunk.loop_displacement[li][idx].toFixed(2)}</td>
      <td>${dd.toFixed(2)}</td><td>${dw.toFixed(2)}</td><td>${chunk.delta_cos_sim[li][idx].toFixed(2)}</td>
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
      <tr><th>l</th><th>g_d</th><th>g_w</th><th>g_d/Σ</th><th>Δd-sh</th><th>steps</th><th>loop-d</th><th>‖Δd‖</th><th>‖Δw‖</th><th>cos</th></tr>
      ${rows}
    </table>
    <div style="color: var(--muted); margin-top: 8px; font-size: 10px;">
      g_d/Σ = gate_deep_relative, Δd-sh = deep_contribution_share, loop-d = loop_displacement
    </div>
  `;
}

// ---------------------------------------------------------------------------
// §5 POS analysis — new heatmap + single-layer + averaged-bars
// ---------------------------------------------------------------------------
// Metric -> {range, diverging, mid}. Used by both heatmap and single-layer.
// 'diverging' means colorscale should be centered at `mid`. Otherwise sequential.
const POS_METRIC_CFG = {
  gate_deep_relative:      { range: [0, 1], diverging: true,  mid: 0.5,
                              label: "gate_deep_relative" },
  deep_contribution_share: { range: [0, 1], diverging: true,  mid: 0.5,
                              label: "deep_contribution_share" },
  expected_steps:          { range: [0, META.max_loops], diverging: false,
                              label: "expected_steps" },
  loop_displacement:       { range: null,   diverging: false,
                              label: "loop_displacement" },
  delta_cos_sim:           { range: [-1, 1], diverging: true,  mid: 0,
                              label: "cos(Δdeep, Δwide)" },
  delta_deep_norm:         { range: null,   diverging: false,
                              label: "‖Δdeep‖" },
  delta_wide_norm:         { range: null,   diverging: false,
                              label: "‖Δwide‖" },
  gate_deep:               { range: [0, 1], diverging: false, label: "gate_deep" },
  gate_wide:               { range: [0, 1], diverging: false, label: "gate_wide" },
};

// Plotly colorscale used for diverging metrics (wide → neutral → deep).
const DIVERGING_SCALE = [
  [0.0, `rgb(${RGB_WIDE[0]},${RGB_WIDE[1]},${RGB_WIDE[2]})`],
  [0.5, `rgb(${RGB_NEUTRAL[0]},${RGB_NEUTRAL[1]},${RGB_NEUTRAL[2]})`],
  [1.0, `rgb(${RGB_DEEP[0]},${RGB_DEEP[1]},${RGB_DEEP[2]})`],
];
const SEQUENTIAL_SCALE = "Viridis";

// Minimum word count per (POS, source) cell. Below this, cells are nulled
// out so noisy under-sampled tags (e.g. SPACE in triviaqa with ~5 examples)
// don't dominate the colorscale.
const MIN_WORDS_PER_CELL = 10;

function getPOSTagsPresent(pos, minWords) {
  // Return POS tags present in this source with at least minWords words,
  // in canonical POS_ORDER.
  return POS_ORDER.filter(t => {
    const n = pos.n_words_per_pos[t] || 0;
    return n >= minWords;
  });
}

function drawPOSHeatmap(source) {
  const pos = SOURCES[source].agg.pos;
  if (!pos) {
    document.getElementById("pos-heatmap").innerHTML =
      '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">no POS data</div>';
    document.getElementById("pos-heatmap-legend").innerHTML = "";
    return;
  }
  const metric = document.getElementById("pos-heatmap-metric").value;
  const aggKey = document.getElementById("pos-agg").value === "max"
                 ? "by_pos_layer_max" : "by_pos_layer_mean";
  const data = pos[aggKey][metric];   // tag -> [val per layer] or undefined
  if (!data) {
    document.getElementById("pos-heatmap").innerHTML =
      '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">metric not available</div>';
    return;
  }

  // Tags as Y (POS), layers as X (0..N-1). Plotly heatmap z is rows by columns.
  const tags = getPOSTagsPresent(pos, MIN_WORDS_PER_CELL);
  const nLayer = META.n_layer;
  // Build z: row = tag, col = layer. Use null for missing cells; Plotly renders
  // them as gaps with `connectgaps:false`.
  const z = tags.map(tag => {
    const row = data[tag] || [];
    return Array.from({length: nLayer}, (_, li) => {
      const v = row[li];
      return (v === null || v === undefined) ? null : v;
    });
  });
  // Color scaling. For diverging metrics with known bounds (e.g. [0,1]),
  // we lock the colorscale to the full range so the midpoint always maps
  // to neutral and cross-source comparison is honest. Auto-zooming made
  // tiny variations look dramatic and broke cross-source comparability.
  const cfg = POS_METRIC_CFG[metric];
  let zmin, zmax, colorscale;
  if (cfg.diverging && cfg.range !== null) {
    zmin = cfg.range[0];
    zmax = cfg.range[1];
    colorscale = DIVERGING_SCALE;
  } else if (cfg.range !== null) {
    zmin = cfg.range[0];
    zmax = cfg.range[1];
    colorscale = SEQUENTIAL_SCALE;
  } else {
    // auto-range for unbounded metrics (delta norms, etc.)
    const flat = z.flat().filter(v => v !== null && isFinite(v));
    if (flat.length === 0) { zmin = 0; zmax = 1; }
    else { zmin = Math.min(...flat); zmax = Math.max(...flat); }
    colorscale = SEQUENTIAL_SCALE;
  }
  // Build customdata for hover: word counts.
  const wordCounts = tags.map(t => pos.n_words_per_pos[t] || 0);
  // Replicate so plotly can show one count per cell.
  const customdata = tags.map((t, ti) =>
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
    title: { text: cfg.label + " — POS × layer (" +
                   (aggKey === "by_pos_layer_max" ? "max" : "mean") +
                   " over subwords)",
             font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, {
      title: "layer", tickfont: { size: 10 }, type: "category",
    }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, {
      title: "POS", tickfont: { size: 11 }, autorange: "reversed",
    }),
    margin: { l: 65, r: 60, t: 40, b: 50 },
  });
  Plotly.newPlot("pos-heatmap", [trace], layout, PLOT_CONFIG);

  // Small inline legend / sample size note.
  const totalWords = tags.reduce((s, t) => s + (pos.n_words_per_pos[t] || 0), 0);
  document.getElementById("pos-heatmap-legend").innerHTML =
    `${tags.length} tags, ${totalWords.toLocaleString()} words total ` +
    `(tags with <3 words excluded)`;
}

function drawPOSAveragedBars(source) {
  // §5b — the original §5 view, kept as contrast.
  const pos = SOURCES[source].agg.pos;
  const covEl = document.getElementById("pos-coverage");
  if (!pos) {
    covEl.textContent = "no POS sidecar for this source";
    for (const id of ["pos-bar-dcs", "pos-bar-loopdisp", "pos-bar-steps", "pos-bar-loss"]) {
      document.getElementById(id).innerHTML =
        '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">no POS data</div>';
    }
    return;
  }
  const cov = (pos.coverage * 100).toFixed(1);
  covEl.textContent = `coverage: ${cov}% of tokens tagged. ` +
                      (pos.coverage < 0.5 ? "⚠ low — POS unreliable for this source." : "");

  const aggMode = document.getElementById("pos-agg").value === "max"
                  ? "by_pos_max" : "by_pos_mean";
  const byPos = pos[aggMode] || {};
  const tags = POS_ORDER.filter(t => byPos[t] && byPos[t].n_words >= 3);

  function makeBar(elId, metricKey, title, range) {
    const xs = tags;
    const ys = tags.map(t => byPos[t][metricKey]);
    const ns = tags.map(t => byPos[t].n_words);
    const colors = tags.map(t => POS_PALETTE[t] || "#888");
    Plotly.newPlot(elId, [{
      type: "bar", x: xs, y: ys, customdata: ns,
      marker: { color: colors },
      hovertemplate: "%{x}<br>" + metricKey + ": %{y:.3f}<br>n_words: %{customdata}<extra></extra>",
    }], layoutCopy({
      title: { text: title, font: { size: 12 } },
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, range ? { range: range } : {}),
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 10 } }),
    }), PLOT_CONFIG);
  }
  makeBar("pos-bar-dcs",      "deep_contribution_share",
          "deep_contribution_share per POS (avg across all layers)", [0, 1]);
  makeBar("pos-bar-loopdisp", "loop_displacement",
          "loop_displacement per POS (avg across all layers)", null);
  makeBar("pos-bar-steps",    "expected_steps",
          "expected_steps per POS (avg across all layers)", [0, META.max_loops]);
  makeBar("pos-bar-loss",     "loss",
          "loss per POS — which words are hard to predict?", null);
}

function drawPOSSingleLayer(source) {
  // §5c — bars for ONE layer at a time.
  const pos = SOURCES[source].agg.pos;
  if (!pos) {
    for (const id of ["pos-single-dcs", "pos-single-loopdisp", "pos-single-steps", "pos-single-gdr"]) {
      document.getElementById(id).innerHTML =
        '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">no POS data</div>';
    }
    return;
  }
  const aggKey = document.getElementById("pos-agg").value === "max"
                 ? "by_pos_layer_max" : "by_pos_layer_mean";
  const layer = parseInt(document.getElementById("pos-single-layer").value);
  const tags = getPOSTagsPresent(pos, MIN_WORDS_PER_CELL);

  function makeBar(elId, metricKey, title, range) {
    const data = pos[aggKey][metricKey] || {};
    const xs = tags;
    const ys = tags.map(t => (data[t] || [])[layer]);
    const ns = tags.map(t => pos.n_words_per_pos[t] || 0);
    const colors = tags.map(t => POS_PALETTE[t] || "#888");
    Plotly.newPlot(elId, [{
      type: "bar", x: xs, y: ys, customdata: ns,
      marker: { color: colors },
      hovertemplate: "%{x}<br>" + metricKey + " @ L" + layer +
                     ": %{y:.3f}<br>n_words: %{customdata}<extra></extra>",
    }], layoutCopy({
      title: { text: title + " — layer " + layer, font: { size: 12 } },
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, range ? { range: range } : {}),
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, { tickangle: -45, tickfont: { size: 10 } }),
    }), PLOT_CONFIG);
  }
  makeBar("pos-single-dcs",       "deep_contribution_share",
          "deep_contribution_share per POS", [0, 1]);
  makeBar("pos-single-loopdisp",  "loop_displacement",
          "loop_displacement per POS", null);
  makeBar("pos-single-steps",     "expected_steps",
          "expected_steps per POS", [0, META.max_loops]);
  makeBar("pos-single-gdr",       "gate_deep_relative",
          "gate_deep_relative per POS", [0, 1]);
}

function drawPOSAll(source) {
  drawPOSAveragedBars(source);
  drawPOSHeatmap(source);
  drawPOSSingleLayer(source);
}

// ---------------------------------------------------------------------------
// §6 — cross-source comparison
// ---------------------------------------------------------------------------

/** Build z-matrix [tags × layers] for one (source, metric, agg). Null cells
 *  for under-sampled tags. */
function _buildSourceZ(source, metric, agg, tags) {
  const pos = SOURCES[source].agg.pos;
  if (!pos) return null;
  const aggKey = agg === "max" ? "by_pos_layer_max" : "by_pos_layer_mean";
  const data = pos[aggKey][metric];
  if (!data) return null;
  const nLayer = META.n_layer;
  // null-out cells where this source's word count for this tag is too low.
  return tags.map(tag => {
    const n = pos.n_words_per_pos[tag] || 0;
    if (n < MIN_WORDS_PER_CELL) {
      return Array(nLayer).fill(null);
    }
    const row = data[tag] || [];
    return Array.from({length: nLayer}, (_, li) => {
      const v = row[li];
      return (v === null || v === undefined) ? null : v;
    });
  });
}

/** Union of POS tags present (with >=MIN_WORDS_PER_CELL words) in ANY of the
 *  given sources, ordered by canonical POS_ORDER. */
function _unionTagsAcrossSources(sources) {
  const present = new Set();
  for (const s of sources) {
    const pos = SOURCES[s].agg.pos;
    if (!pos) continue;
    for (const tag of POS_ORDER) {
      if ((pos.n_words_per_pos[tag] || 0) >= MIN_WORDS_PER_CELL) {
        present.add(tag);
      }
    }
  }
  return POS_ORDER.filter(t => present.has(t));
}

function drawPOSCompareSideBySide() {
  const metric = document.getElementById("pos-compare-metric").value;
  const agg = document.getElementById("pos-compare-agg").value;
  const cfg = POS_METRIC_CFG[metric];
  const container = document.getElementById("pos-compare-panels");
  container.innerHTML = "";

  const sources = META.source_order.filter(s => SOURCES[s].agg.pos);
  if (sources.length === 0) {
    container.innerHTML = '<div style="color:var(--muted); font-family:var(--mono); font-size:12px;">no source has POS data</div>';
    return;
  }

  // Union of tags so every panel has the same y-axis.
  const tags = _unionTagsAcrossSources(sources);
  const nLayer = META.n_layer;

  // Decide shared zmin/zmax/colorscale BEFORE drawing, so all panels use
  // the same color mapping. Same rule as §5a: locked range for diverging
  // metrics with known bounds, auto for unbounded ones.
  let zmin, zmax, colorscale;
  if (cfg.diverging && cfg.range !== null) {
    zmin = cfg.range[0]; zmax = cfg.range[1]; colorscale = DIVERGING_SCALE;
  } else if (cfg.range !== null) {
    zmin = cfg.range[0]; zmax = cfg.range[1]; colorscale = SEQUENTIAL_SCALE;
  } else {
    // auto: take min/max across ALL sources to keep panels comparable.
    let all = [];
    for (const s of sources) {
      const z = _buildSourceZ(s, metric, agg, tags);
      if (z) all = all.concat(z.flat().filter(v => v !== null && isFinite(v)));
    }
    if (all.length === 0) { zmin = 0; zmax = 1; }
    else { zmin = Math.min(...all); zmax = Math.max(...all); }
    colorscale = SEQUENTIAL_SCALE;
  }

  // One heatmap per source, stacked vertically.
  for (const source of sources) {
    const wrap = document.createElement("div");
    wrap.className = "chart-heatmap";
    wrap.style.marginBottom = "8px";
    const elId = `pos-compare-${source.replace(/[^a-z0-9_]/gi, "_")}`;
    wrap.id = elId;
    container.appendChild(wrap);

    const z = _buildSourceZ(source, metric, agg, tags);
    if (!z) {
      wrap.innerHTML = `<div style="color:var(--muted); font-family:var(--mono); font-size:12px;">${source}: no POS data</div>`;
      continue;
    }
    const wordCounts = tags.map(t => SOURCES[source].agg.pos.n_words_per_pos[t] || 0);
    const customdata = tags.map((t, ti) =>
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
      hovertemplate: source + "<br>POS=%{y}<br>layer=%{x}<br>" + cfg.label +
                     "=%{z:.3f}<br>n_words=%{customdata}<extra></extra>",
      colorbar: { thickness: 12, len: 0.9, tickfont: { size: 10 } },
    };
    const layout = layoutCopy({
      title: { text: `${source} — ${cfg.label}`, font: { size: 12 } },
      xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, {
        title: "layer", tickfont: { size: 10 }, type: "category",
      }),
      yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, {
        title: "POS", tickfont: { size: 11 }, autorange: "reversed",
      }),
      margin: { l: 65, r: 60, t: 35, b: 45 },
    });
    Plotly.newPlot(elId, [trace], layout, PLOT_CONFIG);
  }
}

function drawPOSDiffHeatmap() {
  const metric = document.getElementById("pos-compare-metric").value;
  const agg = document.getElementById("pos-compare-agg").value;
  const cfg = POS_METRIC_CFG[metric];
  const a = document.getElementById("pos-diff-a").value;
  const b = document.getElementById("pos-diff-b").value;
  const el = document.getElementById("pos-diff-heatmap");

  if (!a || !b || !SOURCES[a] || !SOURCES[b] || !SOURCES[a].agg.pos || !SOURCES[b].agg.pos) {
    el.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">pick two sources with POS data</div>';
    return;
  }
  if (a === b) {
    el.innerHTML = '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:12px;">pick two different sources</div>';
    return;
  }

  // Use intersection of well-sampled tags, so every cell in the diff exists.
  // Union'd tags with one source missing would produce all-null rows.
  const tags = POS_ORDER.filter(t =>
    (SOURCES[a].agg.pos.n_words_per_pos[t] || 0) >= MIN_WORDS_PER_CELL &&
    (SOURCES[b].agg.pos.n_words_per_pos[t] || 0) >= MIN_WORDS_PER_CELL
  );
  const nLayer = META.n_layer;
  const zA = _buildSourceZ(a, metric, agg, tags);
  const zB = _buildSourceZ(b, metric, agg, tags);

  // Subtract element-wise. Null if either side is null.
  const z = zA.map((rowA, ti) => {
    const rowB = zB[ti];
    return rowA.map((va, li) => {
      const vb = rowB[li];
      if (va === null || vb === null) return null;
      return va - vb;
    });
  });

  // Symmetric colorscale around 0 for the diff. Magnitude = max |diff|.
  const flat = z.flat().filter(v => v !== null && isFinite(v));
  const maxAbs = flat.length ? Math.max(...flat.map(Math.abs)) : 0.1;
  const zmin = -maxAbs;
  const zmax = +maxAbs;

  // Hover shows A, B, and the diff.
  const aData = SOURCES[a].agg.pos[agg === "max" ? "by_pos_layer_max" : "by_pos_layer_mean"][metric] || {};
  const bData = SOURCES[b].agg.pos[agg === "max" ? "by_pos_layer_max" : "by_pos_layer_mean"][metric] || {};
  const customdata = tags.map((t, ti) =>
    Array.from({length: nLayer}, (_, li) => {
      const va = (aData[t] || [])[li];
      const vb = (bData[t] || [])[li];
      return [
        (va === undefined || va === null) ? NaN : va,
        (vb === undefined || vb === null) ? NaN : vb,
      ];
    })
  );

  const trace = {
    type: "heatmap",
    z: z,
    x: Array.from({length: nLayer}, (_, i) => "L" + i),
    y: tags,
    colorscale: DIVERGING_SCALE,
    zmin: zmin, zmax: zmax,
    customdata: customdata,
    hovertemplate: `POS=%{y}<br>layer=%{x}<br>` +
                   `${a}: %{customdata[0]:.3f}<br>` +
                   `${b}: %{customdata[1]:.3f}<br>` +
                   `diff: %{z:.3f}<extra></extra>`,
    colorbar: { thickness: 12, len: 0.9, tickfont: { size: 10 } },
  };
  const layout = layoutCopy({
    title: { text: `${cfg.label}: ${a} − ${b}`, font: { size: 12 } },
    xaxis: Object.assign({}, PLOT_LAYOUT.xaxis, {
      title: "layer", tickfont: { size: 10 }, type: "category",
    }),
    yaxis: Object.assign({}, PLOT_LAYOUT.yaxis, {
      title: "POS", tickfont: { size: 11 }, autorange: "reversed",
    }),
    margin: { l: 65, r: 60, t: 40, b: 50 },
  });
  Plotly.newPlot("pos-diff-heatmap", [trace], layout, PLOT_CONFIG);
}

function drawPOSCompareAll() {
  drawPOSCompareSideBySide();
  drawPOSDiffHeatmap();
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function init() {
  fillMeta();
  fillSourceSelectors();
  fillLayerSelector();
  drawOverview();

  const defaultSource = META.source_order[0];
  document.getElementById("source-select-2").value = defaultSource;
  document.getElementById("source-select-3").value = defaultSource;
  document.getElementById("source-select-4").value = defaultSource;
  const ss5 = document.getElementById("source-select-5");
  if (ss5) ss5.value = defaultSource;

  drawSourceDeepDive(defaultSource);
  drawLayerDynamics(defaultSource);
  fillChunkSelector(defaultSource);
  renderTokens();
  drawPOSAll(defaultSource);

  document.getElementById("source-select-2").addEventListener("change", e => drawSourceDeepDive(e.target.value));
  document.getElementById("source-select-3").addEventListener("change", e => drawLayerDynamics(e.target.value));
  document.getElementById("source-select-4").addEventListener("change", e => {
    fillChunkSelector(e.target.value);
    lockedToken = null;
    renderTokens();
  });
  document.getElementById("chunk-select").addEventListener("change", () => {
    lockedToken = null;
    renderTokens();
  });
  document.getElementById("color-metric").addEventListener("change", renderTokens);
  document.getElementById("layer-select").addEventListener("change", renderTokens);
  document.getElementById("show-pos-toggle").addEventListener("change", renderTokens);
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
  }

  // §6 wiring. Fill the source pickers for the difference heatmap with the
  // available sources, default to first vs second so something shows on load.
  const diffA = document.getElementById("pos-diff-a");
  const diffB = document.getElementById("pos-diff-b");
  if (diffA && diffB) {
    for (const s of META.source_order) {
      const o1 = document.createElement("option"); o1.value = s; o1.textContent = s; diffA.appendChild(o1);
      const o2 = document.createElement("option"); o2.value = s; o2.textContent = s; diffB.appendChild(o2);
    }
    diffA.value = META.source_order[0];
    diffB.value = META.source_order[1] || META.source_order[0];
    drawPOSCompareAll();
    document.getElementById("pos-compare-metric").addEventListener("change", drawPOSCompareAll);
    document.getElementById("pos-compare-agg").addEventListener("change", drawPOSCompareAll);
    diffA.addEventListener("change", drawPOSDiffHeatmap);
    diffB.addEventListener("change", drawPOSDiffHeatmap);
  }
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()