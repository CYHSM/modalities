#!/usr/bin/env python3
"""Fill the review-round tables (compute-matched, gate controls, capped sweep) from finished evals.

Runs are found by config name, the same way queue_eval_after_training.sh finds them: the newest
experiment dir that holds the config (or its _WARMSTART twin) and a checkpoint at the target step.
Metric conventions are those of render_paper_tables.py (limit=1024 dir, each task's primary_score).
Cells are mean +- std over the seeds found so far; a row with no finished seed stays \\ph.

  render_review_tables.py                  summary of every review arm (per seed and mean +- std)
  render_review_tables.py --apply MAIN.tex  also replace the \\ph rows of tab:compute-matched and
                                            tab:gate-controls in MAIN.tex (rows still missing a
                                            seed are filled with what exists and reported)
"""
import argparse
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_paper_tables import BUD, summary  # noqa: E402

REV = "dm768_L16_loop{K}_{B}M_{W}"
SEEDS = (42, 43, 44)
# Seed-43 runs that already exist from September (Tables softcap and depth-sweep); not retrained.
SEP_CAPPED_S43 = {
    "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_softcap1": "2026-09-14__01-44-11_4767e14501931d3a",
    "16e7_loops4__dm768_L16_loop4_F160M_ffnD4864_ffnW24448_dual_a50_nocross_softcap1": "2026-09-13__03-57-19_7baca097a6a8a0d2",
    # L=32 depth-sweep runs of Table 3 (seed 43); seeds 42/44 added in the review round
    "8e7_loops4__dm768_L32_loop4_F80M_dual_a50_nocross_softcap1": "2026-09-13__03-54-08_e2578e41d2128d01",
    "8e7_loops2__dm768_L32_loop1_F80M_ffnW24576_purewide": "2026-09-10__05-47-34_4c2694a69d16531d",
}

# arm key -> config stem (without _s<seed>.yaml)
ARMS = {
    "pw_cm_F80": "8e7_loops2__dm768_L16_loop1_F80M_ffnW32704_purewide_attnmatched",
    "pl_cm_F80": "8e7_loops4__dm768_L16_loop4_F80M_ffnL5504_pureloop_attnmatched",
    "cap_a25_F80": "8e7_loops4__dm768_L16_loop4_F80M_ffnD64_ffnW17920_dual_a25_nocross_softcap1",
    "cap_a50_F80": "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_softcap1",
    "cap_a75_F80": "8e7_loops4__dm768_L16_loop4_F80M_ffnD3264_ffnW4864_dual_a75_nocross_softcap1",
    "pw_cm_F160": "16e7_loops2__dm768_L16_loop1_F160M_ffnW58432_purewide_attnmatched",
    "pl_cm_F160": "16e7_loops4__dm768_L16_loop4_F160M_ffnL11968_pureloop_attnmatched",
    "cap_a25_F160": "16e7_loops4__dm768_L16_loop4_F160M_ffnD1600_ffnW37440_dual_a25_nocross_softcap1",
    "cap_a50_F160": "16e7_loops4__dm768_L16_loop4_F160M_ffnD4864_ffnW24448_dual_a50_nocross_softcap1",
    "cap_a75_F160": "16e7_loops4__dm768_L16_loop4_F160M_ffnD8128_ffnW11392_dual_a75_nocross_softcap1",
    "gate_tokenwise": "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_tokenwise",
    "l32_dual_cap": "8e7_loops4__dm768_L32_loop4_F80M_dual_a50_nocross_softcap1",
    "l32_purewide": "8e7_loops2__dm768_L32_loop1_F80M_ffnW24576_purewide",
    "gate_fixed": "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_gatefixed",
    "gate_perlayer": "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_gateperlayer",
    "gate_ungated": "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross_gateungated",
}
METRICS = [("c4", 3), ("wiki", 3), ("cs_acc", 4), ("cs_bpb", 4), ("gsm8k", 4), ("math_bpb", 4)]

# (table label, block index split at \midrule, row prefix in main.tex) -> arm key
TABLE_ROWS = {
    ("tab:compute-matched", 0, r"\textsc{PureWide} (compute-matched)"): "pw_cm_F80",
    ("tab:compute-matched", 0, r"\textsc{PureLoop} $K{=}4$ (compute-matched)"): "pl_cm_F80",
    ("tab:compute-matched", 0, r"\textsc{Dual} $\alpha{=}25$, $K{=}4$, cap $c{=}1$"): "cap_a25_F80",
    ("tab:compute-matched", 0, r"\textsc{Dual} $\alpha{=}50$, $K{=}4$, cap $c{=}1$"): "cap_a50_F80",
    ("tab:compute-matched", 0, r"\textsc{Dual} $\alpha{=}75$, $K{=}4$, cap $c{=}1$"): "cap_a75_F80",
    ("tab:compute-matched", 1, r"\textsc{PureWide} (compute-matched)"): "pw_cm_F160",
    ("tab:compute-matched", 1, r"\textsc{PureLoop} $K{=}4$ (compute-matched)"): "pl_cm_F160",
    ("tab:compute-matched", 1, r"\textsc{Dual} $\alpha{=}25$, $K{=}4$, cap $c{=}1$"): "cap_a25_F160",
    ("tab:compute-matched", 1, r"\textsc{Dual} $\alpha{=}50$, $K{=}4$, cap $c{=}1$"): "cap_a50_F160",
    ("tab:compute-matched", 1, r"\textsc{Dual} $\alpha{=}75$, $K{=}4$, cap $c{=}1$"): "cap_a75_F160",
    ("tab:gate-controls", 0, r"Token-wise, Eq.~\ref{eq:two-gates}"): "gate_tokenwise",
    ("tab:gate-controls", 0, r"Static $0.5/0.5$"): "gate_fixed",
    ("tab:gate-controls", 0, r"Per-layer, token-independent"): "gate_perlayer",
    ("tab:gate-controls", 0, r"Update-only (ungated residual)"): "gate_ungated",
}


def find_run(config_name: str):
    """Newest experiment dir holding the config (or its WARMSTART twin) with a finished checkpoint."""
    twin = config_name[: -len(".yaml")] + "_WARMSTART.yaml"
    for d in sorted(glob.glob(f"{BUD}/*/"), key=os.path.getmtime, reverse=True):
        if not (os.path.exists(d + config_name) or os.path.exists(d + twin)):
            continue
        for c in glob.glob(d + "eid_*"):
            m = re.search(r"seen_steps_(\d+)-.*target_steps_(\d+)", c)
            if m and m.group(1) == m.group(2):
                return os.path.basename(d.rstrip("/"))
    return None


def arm_runs(stem: str) -> dict:
    """seed -> summary dict for every seed whose final eval exists."""
    out = {}
    for s in SEEDS:
        exp = SEP_CAPPED_S43.get(stem) if s == 43 and stem in SEP_CAPPED_S43 else find_run(f"{stem}_s{s}.yaml")
        if exp:
            res = summary(BUD, exp)
            if res and all(res.get(k) is not None for k, _ in METRICS):
                out[s] = res
    return out


def cell(vals: list, nd: int) -> str:
    if not vals:
        return r"\ph"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    sd = st.stdev(vals)
    return f"{st.mean(vals):.{nd}f}{{\\tiny$\\pm${sd:.{nd}f}}}"


def latex_cells(runs: dict) -> str:
    return " & ".join(cell([r[k] for r in runs.values()], nd) for k, nd in METRICS)


LOWER_IS_BETTER = [True, True, False, True, False, True]  # C4, Wiki, CS acc, CS BPB, GSM8K, math BPB


def _mean_of(c: str):
    c = re.sub(r"\\textbf\{(.*)\}$", r"\1", c.strip())
    m = re.match(r"([0-9.]+)", c)
    return float(m.group(1)) if m else None


def bold_best(lines: list, start: int, end: int) -> None:
    """Bold the best mean of each metric column within every \\midrule block of a table."""
    blocks, cur = [], []
    for i in range(start, end):
        l = lines[i].strip()
        if l.startswith("\\midrule") or l.startswith("\\bottomrule"):
            if cur:
                blocks.append(cur)
            cur = []
        elif " & " in l and l.endswith("\\\\") and cur is not None and any(l.startswith(p) for p in ("\\textsc", "\\quad", "Token", "Static", "Per-layer", "Update")):
            cur.append(i)
    if cur:
        blocks.append(cur)
    for rows in blocks:
        cells = {i: lines[i].strip()[:-2].strip().split(" & ") for i in rows}
        for j, low in enumerate(LOWER_IS_BETTER):
            col = -6 + j
            vals = {i: _mean_of(c[col]) for i, c in cells.items()}
            vals = {i: v for i, v in vals.items() if v is not None}
            if not vals:
                continue
            best = min(vals.values()) if low else max(vals.values())
            for i, c in cells.items():
                raw = re.sub(r"^\\textbf\{(.*)\}$", r"\1", c[col].strip())
                c[col] = f"\\textbf{{{raw}}}" if vals.get(i) == best else raw
        for i, c in cells.items():
            lines[i] = " & ".join(c) + r" \\"


def apply(tex_path: str, data: dict) -> None:
    lines = open(tex_path).read().split("\n")
    changed = 0
    for label in {k[0] for k in TABLE_ROWS}:
        start = next(i for i, l in enumerate(lines) if f"\\label{{{label}}}" in l)
        end = next(i for i in range(start, len(lines)) if "\\end{tabular}" in lines[i])
        block = -1
        for i in range(start, end):
            l = lines[i].strip()
            if l.startswith("\\midrule"):
                block += 1
                continue
            for (lab, b, prefix), arm in TABLE_ROWS.items():
                if lab != label or b != block or not l.startswith(prefix + " &"):
                    continue
                if not data[arm]:
                    continue
                cells = l.split(" & ")
                # Table 11 rows: label & params & gflops & 6 metrics; Table 12 rows: label & 6 metrics
                keep = cells[:-6]
                new = " & ".join(keep) + " & " + latex_cells(data[arm]) + r" \\"
                if re.sub(r"\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", new) != re.sub(r"\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", l):
                    lines[i] = new
                    changed += 1
        bold_best(lines, start, end)
    open(tex_path, "w").write("\n".join(lines))
    print(f"\n{tex_path}: {changed} rows updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", metavar="MAIN_TEX")
    a = ap.parse_args()
    data = {arm: arm_runs(stem) for arm, stem in ARMS.items()}
    hdr = "  ".join(f"{k:>8}" for k, _ in METRICS)
    print(f"{'arm':14} {'seeds':10} {hdr}")
    for arm, runs in data.items():
        seeds = ",".join(str(s) for s in runs) or "-"
        if runs:
            means = "  ".join(f"{st.mean(r[k] for r in runs.values()):8.4f}" for k, _ in METRICS)
            print(f"{arm:14} {seeds:10} {means}")
            if len(runs) > 1:
                sds = "  ".join(f"{st.stdev([r[k] for r in runs.values()]):8.4f}" for k, _ in METRICS)
                print(f"{'':14} {'sd':10} {sds}")
        else:
            print(f"{arm:14} {seeds:10} (no finished eval)")
    if a.apply:
        apply(a.apply, data)


if __name__ == "__main__":
    main()
