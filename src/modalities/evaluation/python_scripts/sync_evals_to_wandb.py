#!/usr/bin/env python3
"""Match olmes eval results to wandb runs by yaml+timestamp, then sync to wandb."""
import os, re, json
from pathlib import Path
from datetime import datetime, timezone

WANDB_DIR = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments/wandb/wandb")
EXPERIMENTS_DIR = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments")
WANDB_ENTITY = "cyhsm"
WANDB_PROJECT = "widedeep"

# ---------- parse wandb offline runs ----------
def parse_wandb_runs():
    runs = []
    for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
        run_id = run_dir.name.split("-")[-1]
        meta_file = run_dir / "files" / "wandb-metadata.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        args = meta.get("args", [])
        yaml_name = None
        for i, a in enumerate(args):
            if a == "--config_file_path" and i + 1 < len(args):
                yaml_name = Path(args[i + 1]).name
                break
        if not yaml_name:
            continue
        started = meta.get("started_at")
        try:
            ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
        except Exception:
            continue
        runs.append({"run_id": run_id, "yaml": yaml_name, "started": ts})
    return runs

# ---------- parse experiment folders ----------
FOLDER_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})_")

def parse_experiment_folders():
    exps = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir() or d.name == "wandb":
            continue
        m = FOLDER_RE.match(d.name)
        if not m:
            continue
        # Folder timestamp is LOCAL (CEST = UTC+2 in April). Treat as UTC+2, convert to UTC.
        y, mo, da, h, mi, s = map(int, m.groups())
        # Leonardo is Europe/Rome; April = CEST = UTC+2
        local = datetime(y, mo, da, h, mi, s)
        ts_utc = local.replace(tzinfo=timezone.utc) - __import__("datetime").timedelta(hours=2)
        yamls = [y.name for y in d.glob("*.yaml") if not y.name.endswith(".resolved")]
        if not yamls:
            continue
        exps.append({"folder": d, "yaml": yamls[0], "ts": ts_utc})
    return exps

# ---------- matching ----------
def match(runs, exps):
    by_yaml_runs = {}
    for r in runs:
        by_yaml_runs.setdefault(r["yaml"], []).append(r)
    matches = {}  # folder -> run_id
    for e in exps:
        candidates = by_yaml_runs.get(e["yaml"], [])
        if not candidates:
            print(f"!! no wandb run for yaml {e['yaml']} (folder {e['folder'].name})")
            continue
        # Pick the wandb run with started_at closest to (and >=) the folder timestamp
        best = min(candidates, key=lambda r: abs((r["started"] - e["ts"]).total_seconds()))
        delta = (best["started"] - e["ts"]).total_seconds()
        print(f"-> {e['folder'].name} → {best['run_id']} (yaml={e['yaml']}, Δ={delta:.0f}s)")
        matches[e["folder"]] = best["run_id"]
    return matches

# ---------- collect metrics from olmes eval output ----------
STEP_RE = re.compile(r"seen_steps_(\d+)")

def collect_metrics(folder):
    """Return list of (step, metrics_dict) sorted by step."""
    out = []
    for ckpt in folder.glob("eid_*"):
        m = STEP_RE.search(ckpt.name)
        if not m:
            continue
        step = int(m.group(1))
        metrics_file = ckpt / "hf_checkpoint" / "olmes_eval" / "metrics-all.jsonl"
        if not metrics_file.exists():
            continue
        flat = {}
        with open(metrics_file) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                alias = obj.get("task_config", {}).get("metadata", {}).get("alias") or obj.get("task_name")
                score = obj.get("metrics", {}).get("primary_score")
                if alias and score is not None:
                    flat[f"eval/{alias}"] = score
        if flat:
            flat["seen_steps"] = step
            out.append((step, flat))
    out.sort(key=lambda x: x[0])
    return [m for _, m in out]

# ---------- main ----------
def main():
    import wandb
    runs = parse_wandb_runs()
    exps = parse_experiment_folders()
    print(f"Found {len(runs)} wandb runs, {len(exps)} experiment folders")
    matches = match(runs, exps)
    print(f"\nMatched {len(matches)} folders")

    for folder, run_id in matches.items():
        if run_id != "a1l1s3xo":
            print(f"!! skipping {folder.name} → {run_id} (not the target run)")
            continue
        metrics_list = collect_metrics(folder)
        if not metrics_list:
            print(f"   [{run_id}] no metrics yet, skip")
            continue
        print(f"\n== Syncing {len(metrics_list)} steps to {run_id} ({folder.name}) ==")
        run = wandb.init(id=run_id, project=WANDB_PROJECT, entity=WANDB_ENTITY,
                         resume="allow", reinit=True)
        run.define_metric("seen_steps")
        run.define_metric("eval/*", step_metric="seen_steps")
        for m in metrics_list:
            run.log(m)
        run.finish()
        print(f"   done")

if __name__ == "__main__":
    main()