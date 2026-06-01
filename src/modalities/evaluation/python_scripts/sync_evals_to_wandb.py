#!/usr/bin/env python3
"""Match olmes eval results to wandb runs by yaml + start-timestamp, then sync to wandb.

Matching design notes
---------------------
There is no shared unique ID between an offline wandb run and an experiment
folder:
  - wandb stores only the *unresolved* ${modalities_env:experiment_id} in
    config.yaml, plus a yaml-stem "experiment_id" that is NOT unique per run.
  - the resolved experiment_id (== the experiment folder name) appears nowhere
    in the wandb run files.
  - no SLURM job id is written into the experiment folder's resolved yaml.

So the only reliable shared signals are (yaml name, precise start time).
We match on those, but defensively:
  - wandb's startedAt is ISO-8601 UTC with sub-second precision -> trust it.
  - the experiment folder name timestamp is LOCAL cluster time -> convert it
    via a real timezone so DST is handled correctly.
  - the nearest wandb run must be within MATCH_TOLERANCE, and each run may be
    claimed by at most one folder. Otherwise we raise instead of silently
    syncing to the wrong run.
"""
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ---------- Paths ----------
EXPERIMENTS_DIR = Path("/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp_revisions")
WANDB_DIR = EXPERIMENTS_DIR / "wandb" / "wandb"

WANDB_ENTITY = "cyhsm"
WANDB_PROJECT = "dualfull"

# Leonardo (CINECA, Bologna, Italy). Experiment-folder names are in this local
# time; zoneinfo handles CET/CEST automatically so we don't hardcode +1/+2.
CLUSTER_TZ = ZoneInfo("Europe/Rome")

# A wandb run is only considered the same job as a folder if their start times
# agree within this window. The observed gap is ~25s (folder name is stamped at
# job launch, wandb startedAt a moment later). Keep this tight: it is the guard
# against matching the wrong run.
MATCH_TOLERANCE = timedelta(minutes=3)


# ---------- parse wandb offline runs ----------
def parse_wandb_runs():
    """Return list of {run_id, yaml, started(UTC aware)} for valid offline runs."""
    runs = []
    if not WANDB_DIR.exists():
        print(f"Warning: WANDB_DIR {WANDB_DIR} does not exist.")
        return runs

    print(f"Scanning {WANDB_DIR} for offline runs...")
    for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
        run_id = run_dir.name.split("-")[-1]
        meta_file = run_dir / "files" / "wandb-metadata.json"
        if not meta_file.exists():
            print(f"  [skip] {run_dir.name}: missing files/wandb-metadata.json")
            continue

        try:
            meta = json.loads(meta_file.read_text())
        except Exception as e:
            print(f"  [skip] {run_dir.name}: invalid metadata JSON - {e}")
            continue

        # --- yaml name from args (handles '--arg val' and '--arg=val') ---
        args = meta.get("args", [])
        yaml_name = None
        for i, a in enumerate(args):
            if a == "--config_file_path" and i + 1 < len(args):
                yaml_name = Path(args[i + 1]).name
                break
            if a.startswith("--config_file_path="):
                yaml_name = Path(a.split("=", 1)[1]).name
                break
        if not yaml_name:
            print(f"  [skip] {run_dir.name}: no --config_file_path in args")
            continue

        # --- start time: wandb writes ISO-8601 UTC, just trust it ---
        started = meta.get("startedAt") or meta.get("started_at")
        if not started:
            print(f"  [skip] {run_dir.name}: no startedAt in metadata")
            continue
        try:
            ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
        except Exception as e:
            print(f"  [skip] {run_dir.name}: bad startedAt '{started}' - {e}")
            continue

        runs.append({"run_id": run_id, "yaml": yaml_name, "started": ts})
        print(f"  [ok] {run_dir.name}  yaml={yaml_name}  started={ts.isoformat()}")

    return runs


# ---------- parse experiment folders ----------
FOLDER_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})_")


def parse_experiment_folders():
    """Return list of {folder, yaml, ts(UTC aware)} for experiment folders."""
    exps = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir() or d.name == "wandb":
            continue
        m = FOLDER_RE.match(d.name)
        if not m:
            continue

        # Folder name timestamp is cluster-LOCAL. Attach the real cluster tz so
        # DST is correct, then convert to UTC for comparison with wandb.
        y, mo, da, h, mi, s = map(int, m.groups())
        local = datetime(y, mo, da, h, mi, s, tzinfo=CLUSTER_TZ)
        ts_utc = local.astimezone(ZoneInfo("UTC"))

        yamls = [f.name for f in d.glob("*.yaml") if not f.name.endswith(".resolved")]
        if not yamls:
            print(f"  [skip] {d.name}: no .yaml config in folder")
            continue
        if len(yamls) > 1:
            print(f"  [warn] {d.name}: multiple yamls {yamls}, using {yamls[0]}")

        exps.append({"folder": d, "yaml": yamls[0], "ts": ts_utc})
    return exps


# ---------- matching ----------
def match(runs, exps):
    """Match each experiment folder to exactly one wandb run.

    Match key is (yaml name, start time within MATCH_TOLERANCE). The result is
    enforced to be an injection: a wandb run is claimed by at most one folder.
    Anything ambiguous or out-of-tolerance raises instead of guessing.
    """
    by_yaml = {}
    for r in runs:
        by_yaml.setdefault(r["yaml"], []).append(r)

    matches = {}        # folder -> run_id
    claimed = {}        # run_id -> folder  (enforce one-to-one)
    problems = []

    for e in exps:
        candidates = by_yaml.get(e["yaml"], [])
        if not candidates:
            problems.append(f"no wandb run with yaml {e['yaml']} (folder {e['folder'].name})")
            continue

        best = min(candidates, key=lambda r: abs(r["started"] - e["ts"]))
        delta = abs(best["started"] - e["ts"])

        if delta > MATCH_TOLERANCE:
            problems.append(
                f"{e['folder'].name}: nearest run {best['run_id']} is {delta} away "
                f"(> tolerance {MATCH_TOLERANCE}); yaml={e['yaml']}"
            )
            continue

        if best["run_id"] in claimed:
            problems.append(
                f"run {best['run_id']} matched by two folders: "
                f"{claimed[best['run_id']].name} and {e['folder'].name}"
            )
            continue

        matches[e["folder"]] = best["run_id"]
        claimed[best["run_id"]] = e["folder"]
        signed = (best["started"] - e["ts"]).total_seconds()
        print(f"-> {e['folder'].name}  ->  {best['run_id']}  "
              f"(yaml={e['yaml']}, Delta={signed:+.0f}s)")

    if problems:
        raise RuntimeError(
            "Run matching failed; refusing to sync to avoid wrong-run logging:\n  - "
            + "\n  - ".join(problems)
        )

    return matches


# ---------- collect metrics from olmes eval output ----------
STEP_RE = re.compile(r"seen_steps_(\d+)")


def collect_metrics(folder):
    """Return list of metrics dicts (one per checkpoint step), sorted by step."""
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
                alias = (obj.get("task_config", {}).get("metadata", {}).get("alias")
                         or obj.get("task_name"))
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
    print(f"\nMatched {len(matches)} folders (all within tolerance, one-to-one)")

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