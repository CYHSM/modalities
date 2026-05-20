#!/usr/bin/env python3
"""Match olmes eval results to wandb runs by yaml+timestamp, then sync to wandb."""
import os, re, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ---------- Updated Paths for Clariden / Alps ----------
EXPERIMENTS_DIR = Path("/capstor/scratch/cscs/markusfrey/experiments")
WANDB_DIR = EXPERIMENTS_DIR / "wandb" / "wandb"

WANDB_ENTITY = "cyhsm"
WANDB_PROJECT = "onegate"

# ---------- parse wandb offline runs ----------
def parse_wandb_runs():
    runs = []
    if not WANDB_DIR.exists():
        print(f"Warning: WANDB_DIR {WANDB_DIR} does not exist.")
        return runs
        
    print(f"Scanning {WANDB_DIR} for offline runs...")
    for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
        run_id = run_dir.name.split("-")[-1]
        meta_file = run_dir / "files" / "wandb-metadata.json"
        if not meta_file.exists():
            print(f"  [DEBUG] Skipping {run_dir.name}: missing files/wandb-metadata.json")
            continue
            
        try:
            meta = json.loads(meta_file.read_text())
        except Exception as e:
            print(f"  [DEBUG] Skipping {run_dir.name}: invalid JSON - {e}")
            continue
            
        args = meta.get("args", [])
        yaml_name = None
        
        # Robust argument parsing (handles both '--arg val' and '--arg=val')
        for i, a in enumerate(args):
            if a == "--config_file_path" and i + 1 < len(args):
                yaml_name = Path(args[i + 1]).name
                break
            elif a.startswith("--config_file_path="):
                yaml_name = Path(a.split("=", 1)[1]).name
                break
                
        if not yaml_name:
            print(f"  [DEBUG] Skipping {run_dir.name}: '--config_file_path' not found in args: {args}")
            continue
            
        # Check both possible wandb timestamp keys
        started = meta.get("startedAt") or meta.get("started_at")
        if not started:
            print(f"  [DEBUG] Skipping {run_dir.name}: missing startedAt/started_at timestamp in metadata")
            continue
            
        try:
            ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
        except Exception as e:
            print(f"  [DEBUG] Skipping {run_dir.name}: timestamp parse error '{started}' - {e}")
            continue
            
        runs.append({"run_id": run_id, "yaml": yaml_name, "started": ts})
        print(f"  [OK] Found valid run: {run_dir.name} (yaml={yaml_name})")
        
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
        
        # Folder timestamp is LOCAL. Clariden (Switzerland) is Europe/Zurich.
        # April = CEST = UTC+2. 
        y, mo, da, h, mi, s = map(int, m.groups())
        local = datetime(y, mo, da, h, mi, s)
        ts_utc = local.replace(tzinfo=timezone.utc) - timedelta(hours=2)
        
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
    print(f"\nFound {len(runs)} wandb offline runs, {len(exps)} experiment folders")
    matches = match(runs, exps)
    print(f"\nMatched {len(matches)} folders")

    for folder, run_id in matches.items():
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