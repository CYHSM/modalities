#!/usr/bin/env python3
"""Render the softcap / depth-sweep / unrolled appendix tables for main.tex.

Re-run this as evaluations land; cells whose run or eval is not finished come
out as \\ph (a TBD placeholder) instead of failing.

Metric conventions are pinned to reproduce the published Tables
appendix-f80m / appendix-f160m exactly (verified against 18 known values):
  * the limit=1024 eval dir, NOT olmes_eval_128
  * each task's own `primary_score`, selected via task_config.primary_metric
    (so ARC-c Acc is acc_uncond=0.4512, not acc_raw=0.3643)
  * Commonsense = mean over ARC-c, ARC-e, HellaSwag, PIQA, SIQA, WinoGrande
  * Math BPB   = olmo3:base_easy:math_bpb, primary_score_macro
  * C4 / Wiki  = paloma_c4_en / paloma_wikitext_103 BPB

Newer runs dropped gsm8k from their task list, so their Math Acc is evaluated
separately into olmes_eval_<step>_gsm8k; both dirs are merged here.

Usage: render_paper_tables.py [-o OUT.tex]
"""
import argparse, glob, json, os

BUD = "/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest"
EMN = "/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
CS = ["arc_challenge", "arc_easy", "hellaswag", "piqa", "socialiqa", "winogrande"]


def _rows(evaldir):
    """Rows of one eval dir, limit=1024 only (skips smoke tests such as olmes_eval_<step>_envtest)."""
    f = os.path.join(evaldir, "metrics-all.jsonl")
    if not os.path.exists(f):
        return []
    rows = [json.loads(l) for l in open(f)]
    return [r for r in rows if r.get("task_config", {}).get("limit") == 1024]


def _collect(root, exp_hash, step=18360):
    """All metric rows for a run's final checkpoint, main eval + gsm8k side-car."""
    base = glob.glob(f"{root}/{exp_hash}/eid_*seen_steps_{step}-*/hf_checkpoint")
    if not base:
        return []
    hc, out = base[0], []
    # 'olmes_eval' (older, complete incl. gsm8k) sorts before 'olmes_eval_<step>'
    for d in sorted(glob.glob(hc + "/olmes_eval*")):
        if d.endswith("_128"):
            continue  # limit=128 variant; the paper uses limit=1024
        out += _rows(d)
    return out


def _score(rows, name, want):
    for r in rows:
        if r["task_name"] != name:
            continue
        is_bpb = "bits_per_byte" in str(r.get("task_config", {}).get("primary_metric", ""))
        if (want == "bpb") == is_bpb:
            m = r["metrics"]
            return m.get("primary_score", m.get("primary_score_macro"))
    return None


def summary(root, exp_hash):
    rows = _collect(root, exp_hash)
    if not rows:
        return {}
    acc = [_score(rows, t, "acc") for t in CS]
    bpb = [_score(rows, t, "bpb") for t in CS]
    em = [r for r in rows if r["task_name"] == "olmo3:base_easy:math_bpb"]
    return {
        "cs_acc": sum(acc) / len(acc) if all(a is not None for a in acc) else None,
        "cs_bpb": sum(bpb) / len(bpb) if all(b is not None for b in bpb) else None,
        "gsm8k": _score(rows, "gsm8k", "acc"),
        "math_bpb": em[0]["metrics"].get("primary_score_macro") if em else None,
        "c4": _score(rows, "paloma_c4_en", "bpb"),
        "wiki": _score(rows, "paloma_wikitext_103", "bpb"),
    }


# label -> (root, experiment dir, params in M, total GFLOPs/token)
RUNS = {
    "pw_L8_F80":    (BUD, "2026-09-14__01-40-27_bfe620344dd90688", 398, 0.64),
    "du_L8_F80":    (BUD, "2026-09-14__01-39-55_a60bd94f927862fe", 280, 0.63),
    "pw_L16_F80":   (EMN, "2026-05-13__02-00-20_7ad77be5d81edda7", 719, 1.28),
    "du_L16_F80":   (EMN, "2026-05-13__05-25-03_e3f7d3d2ad461681", 483, 1.26),
    "sc_L16_F80":   (BUD, "2026-09-14__01-44-11_4767e14501931d3a", 483, 1.26),
    "pw_L32_F80":   (BUD, "2026-09-10__05-47-34_4c2694a69d16531d", 1361, 2.57),
    "sc_L32_F80":   (BUD, "2026-09-13__03-54-08_e2578e41d2128d01", 889, 2.52),
    "pw_L64_F80":   (BUD, "2026-09-14__02-07-17_dbeab5ff2dc0b63d", 2644, 5.13),
    "sc_L64_F80":   (BUD, "2026-09-14__01-47-51_dc320e1a594ebeee", 1700, 5.04),
    "pw_L16_F160":  (EMN, "2026-05-12__06-08-00_e2024d4668888064", 1361, 2.57),
    "du_L16_F160":  (EMN, "2026-05-12__06-22-54_3719e6b84f11c9c1", 880, 2.54),
    "sc_L16_F160":  (BUD, "2026-09-13__03-57-19_7baca097a6a8a0d2", 880, 2.54),
    "exp_K2_F80":   (EMN, "2026-05-18__19-29-00_09776dacfbd02318", 719, 1.28),
    "exp_K3_F80":   (EMN, "2026-05-18__19-28-58_b0f6ff56da44e1d2", 729, 1.30),
    "exp_K4_F80":   (EMN, "2026-05-18__19-28-56_35921c8a7d8e2634", 719, 1.28),
    "exp_K2_F160":  (EMN, "2026-05-18__19-28-59_78bcbfab173bc55b", 1361, 2.57),
    "exp_K3_F160":  (EMN, "2026-05-18__19-28-58_b701cc1f92e023b7", 1366, 2.58),
    "exp_K4_F160":  (EMN, "2026-05-18__22-41-23_d8e95812e2a31843", 1361, 2.57),
}

PH = r"\ph"


def fmt(v, n=4):
    return PH if v is None else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="appendix_softcap_tables.tex")
    a = ap.parse_args()

    D = {k: summary(r, h) for k, (r, h, _, _) in RUNS.items()}
    P = {k: (p, g) for k, (_, _, p, g) in RUNS.items()}

    def row(label, key, budget, indent=""):
        s, (p, _) = D.get(key, {}), P[key]
        return (f"{indent}{label} & {p}M & {budget} & {fmt(s.get('c4'),3)} & {fmt(s.get('wiki'),3)} & "
                f"{fmt(s.get('cs_acc'))} & {fmt(s.get('cs_bpb'))} & {fmt(s.get('gsm8k'))} & {fmt(s.get('math_bpb'))} \\\\")

    def drow(label, key):
        s, (p, g) = D.get(key, {}), P[key]
        return (f" & {label} & {p}M & {g:.2f} & {fmt(s.get('c4'),3)} & {fmt(s.get('wiki'),3)} & "
                f"{fmt(s.get('cs_acc'))} & {fmt(s.get('cs_bpb'))} & {fmt(s.get('gsm8k'))} & {fmt(s.get('math_bpb'))} \\\\")

    L = []
    L.append(r"\newcommand{\ph}{\textcolor{gray}{\textsc{tbd}}}")
    L.append("")

    # ---- Table 1: soft cap ablation ----
    L += [r"\begin{table*}[t]", r"\centering",
          r"\caption{\textbf{Gate soft cap.} The two-gate residual gain $g_d+g_w$ lies in $(0,2)$ and is free",
          r"to sit above $1$, so it compounds with depth. Capping the \emph{sum} at $c=1$ (dividing both gates",
          r"by $\max(1,(g_d+g_w)/c)$, which preserves the routing ratio $g_d\!:\!g_w$ exactly and leaves tokens",
          r"below the cap untouched) removes the divergence seen at $L\!\ge\!32$ and does not cost accuracy at",
          r"$L=16$. All rows are $\alpha=50$, \texttt{loop=4}, FLOP-matched to \textsc{PureWide}. Single runs.}",
          r"\label{tab:softcap}", r"\small", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{l c c cc cc cc}", r"\toprule",
          r" & & & \multicolumn{2}{c}{Language modelling} & \multicolumn{2}{c}{Commonsense} & \multicolumn{2}{c}{Math} \\",
          r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
          r"Configuration & Params & FLOPs & C4 $\downarrow$ & Wiki $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ \\",
          r"\midrule"]
    L.append(row(r"\textsc{PureWide}", "pw_L16_F80", "80M"))
    L.append(row(r"\textsc{DualPath} $\alpha=50$", "du_L16_F80", "80M"))
    L.append(r"\rowcolor{oursrow}")
    L.append(row(r"\quad + gate cap $c=1$", "sc_L16_F80", "80M"))
    L.append(r"\midrule")
    L.append(row(r"\textsc{PureWide}", "pw_L16_F160", "160M"))
    L.append(row(r"\textsc{DualPath} $\alpha=50$", "du_L16_F160", "160M"))
    L.append(r"\rowcolor{oursrow}")
    L.append(row(r"\quad + gate cap $c=1$", "sc_L16_F160", "160M"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    # ---- Table 2: depth sweep ----
    L += [r"\begin{table*}[t]", r"\centering",
          r"\caption{\textbf{Depth sweep at the F80M budget.} $F_M$ is a \emph{per-layer} budget, so the block",
          r"is identical at every depth and only $L$ changes; total FLOPs/token and parameters scale linearly",
          r"with $L$. \textsc{DualPath} rows use $\alpha=50$, \texttt{loop=4} and the gate cap $c=1$: without",
          r"the cap the $L=32$ run diverges to \texttt{NaN} at step ${\sim}1.3$k, so the cap is what makes this",
          r"sweep trainable. Global batch is held constant across depths. Single runs.}",
          r"\label{tab:depth-sweep}", r"\small", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{ll c c cc cc cc}", r"\toprule",
          r" & & & & \multicolumn{2}{c}{Language modelling} & \multicolumn{2}{c}{Commonsense} & \multicolumn{2}{c}{Math} \\",
          r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
          r"$L$ & Configuration & Params & GFLOPs & C4 $\downarrow$ & Wiki $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ \\",
          r"\midrule"]
    for depth, pw, sc in [(8, "pw_L8_F80", "du_L8_F80"), (16, "pw_L16_F80", "sc_L16_F80"),
                          (32, "pw_L32_F80", "sc_L32_F80"), (64, "pw_L64_F80", "sc_L64_F80")]:
        L.append(r"\multirow{2}{*}{%d}" % depth)
        L.append(drow(r"\textsc{PureWide}", pw))
        L.append(r"\rowcolor{oursrow}")
        L.append(drow(r"\textsc{DualPath} + cap", sc))
        L.append(r"\midrule" if depth != 64 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}", ""]

    # ---- Table 3: unrolled baseline ----
    L += [r"\begin{table*}[t]", r"\centering",
          r"\caption{\textbf{Unrolled (non-shared) depth baseline.} \textsc{PureWide-Exp} replaces the $K$ shared",
          r"applications of a block with $L\!\cdot\!K$ \emph{distinct} layers at $1/K$ the per-layer budget,",
          r"matching total FLOPs/token. This is the ``deeper unshared stack'' comparison of",
          r"Section~\ref{sec:related}, which buys capacity at the same compute. Backbone $L=16$, $d=768$.}",
          r"\label{tab:unrolled}", r"\small", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{l c c cc cc cc}", r"\toprule",
          r" & & & \multicolumn{2}{c}{Language modelling} & \multicolumn{2}{c}{Commonsense} & \multicolumn{2}{c}{Math} \\",
          r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
          r"Configuration & Params & FLOPs & C4 $\downarrow$ & Wiki $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ & Acc. $\uparrow$ & BPB $\downarrow$ \\",
          r"\midrule"]
    for budget, tag in [("80M", "F80"), ("160M", "F160")]:
        L.append(r"\multicolumn{9}{l}{\textit{%s budget}} \\" % budget)
        for K in (2, 3, 4):
            L.append(row(r"\quad \textsc{PureWide-Exp} $K$=%d ($L$=%d)" % (K, 16 * K),
                         f"exp_K{K}_{tag}", budget))
        if tag == "F80":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    open(a.out, "w").write("\n".join(L) + "\n")
    missing = sum(l.count(PH) for l in L)
    print(f"wrote {a.out}  ({missing} placeholder cells remaining)")
    for k in RUNS:
        s = D.get(k, {})
        if not s:
            print(f"  no eval yet: {k}")
        elif s.get("gsm8k") is None:
            print(f"  missing gsm8k: {k}")


if __name__ == "__main__":
    main()
