#!/usr/bin/env python3
"""Collate OLMES metrics-all.jsonl across checkpoints of one or more runs.

Usage: collate_olmes.py <label>=<experiment_dir> [<label>=<experiment_dir> ...]

Emits a tidy CSV on stdout: run,step,task,primary_metric,value
"""
import json, sys, csv
from pathlib import Path


def primary_value(row):
    """OLMES names the headline metric in task_config.primary_metric; that name
    is sometimes a bare key ('macro') and sometimes a full metric key."""
    tc = row.get("task_config") or {}
    pm = tc.get("primary_metric")
    m = row.get("metrics") or {}
    if pm in m:
        return pm, m[pm]
    # 'macro'/'micro' are suffix forms, e.g. primary='macro' -> 'bits_per_byte_corr_macro'
    for k in m:
        if k.endswith("_" + str(pm)) and "bits_per_byte" in k:
            return k, m[k]
    for cand in (f"bits_per_byte_corr_{pm}", f"acc_{pm}", "primary_score"):
        if cand in m:
            return cand, m[cand]
    return pm, None


def alias(row):
    tc = row.get("task_config") or {}
    return (tc.get("metadata") or {}).get("alias") or row.get("task_name")


w = csv.writer(sys.stdout)
w.writerow(["run", "step", "task", "metric", "value"])
for spec in sys.argv[1:]:
    label, d = spec.split("=", 1)
    for f in sorted(Path(d).glob("eid_*/hf_checkpoint/olmes_eval_*/metrics-all.jsonl")):
        step = int(str(f).split("seen_steps_")[1].split("-")[0])
        for line in f.open():
            row = json.loads(line)
            k, v = primary_value(row)
            w.writerow([label, step, alias(row), k, v])
