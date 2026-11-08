#!/usr/bin/env python3
"""Training throughput of the 64-GPU (16 nodes x 4 A100, FSDP) runs, read from their Slurm logs.

Per run: median `train samples/s` over the logged steps after a warm-up (compile, first checkpoint),
converted to tokens/s per GPU (samples x 4096 tokens / 64 GPUs). Per arm (seeds pooled): mean and
range over runs. Use only runs that were measured in the same period when comparing arms, since
node placement and file-system load vary between weeks (the log date is printed for that reason).

  cluster_throughput.py [LOG_GLOB] [--since YYYY-MM-DD]

The default glob takes array-job logs only; debug jobs (nan-repro, nocompile, smoketest) ran on fewer
nodes or without compile and are not comparable.
"""
import argparse
import glob
import os
import re
import statistics as st
import time
from collections import defaultdict

SEQ, GPUS = 4096, 64
CFG = re.compile(r"(\d+e7_loops\d+__dm768_L\d+_loop\d+_F\d+M_[A-Za-z0-9_]+?)(?:_s(4[234]))?(?:_WARMSTART)?(?:_smoketest)?\.yaml")
STEP = re.compile(r"step: (\d+) \| train samples/s: ([0-9.]+)")


def run_rate(path: str, warmup: int = 300):
    """(config stem, seed, median samples/s, steps used) of one log, or None."""
    text = open(path, errors="replace").read()
    m = CFG.search(text)
    if not m or "smoketest" in m.group(0):
        return None
    rates = [float(r) for s, r in STEP.findall(text) if int(s) > warmup]
    if len(rates) < 20:
        return None
    return m.group(1), m.group(2) or "-", st.median(rates), len(rates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="?", default="logs/janus-[0-9]*_*.log")  # array runs only, no debug jobs
    ap.add_argument("--since", help="only logs modified on or after this date")
    a = ap.parse_args()
    since = time.mktime(time.strptime(a.since, "%Y-%m-%d")) if a.since else 0
    arms = defaultdict(list)
    for f in sorted(glob.glob(a.logs)):
        if os.path.getmtime(f) < since:
            continue
        r = run_rate(f)
        if r:
            arms[r[0]].append((r[1], r[2], r[3], time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f))), os.path.basename(f)))
    print(f"{'arm':78} {'seeds':9} {'samples/s':>9} {'tok/s/GPU':>9}  range")
    for arm in sorted(arms):
        runs = arms[arm]
        rates = [r[1] for r in runs]
        tok = [x * SEQ / GPUS for x in rates]
        print(f"{arm:78} {','.join(r[0] for r in runs):9} {st.mean(rates):9.1f} {st.mean(tok) / 1e3:8.1f}k  "
              f"{min(tok) / 1e3:.1f}-{max(tok) / 1e3:.1f}k  ({', '.join(sorted({r[3] for r in runs}))})")


if __name__ == "__main__":
    main()
