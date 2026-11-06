#!/usr/bin/env python
"""Aggregate the review-round gate interventions over the Table 4 checkpoints.

Per checkpoint and source (gsm8k, triviaqa, wikitext_103), loss increase over the learned gates:
  ratio_shuffle / ratio_mean   ratio g_d:g_w changed, each token's g_d+g_w kept    (path mixture)
  sum_shuffle / sum_mean       g_d+g_w changed, ratio kept                        (residual gain)
and the class-specific excess of the Table 4 overrides over two random controls: the original one
(same number of tokens) and the perturbation-matched one (same distribution of |delta g|).
Excess p-values are paired Wilcoxon signed-rank tests across checkpoints, as in Table 4.

Usage: aggregate_review.py [results_dir]   (needs scipy; use the olmes_uv venv)
"""

import json
import statistics as st
import sys
from pathlib import Path

from scipy.stats import wilcoxon

RES = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
SOURCES = ["gsm8k", "triviaqa", "wikitext_103"]
MIX = ["ratio_shuffle", "ratio_mean", "sum_shuffle", "sum_mean"]
CLASS = {
    "Arith. forced wide": ("gd0p0_gw1p0_mask_NUM+SYM", "gd0p0_gw1p0_mask_RANDOM", "gd0p0_gw1p0_mask_NUM+RANDOMMATCHED+SYM"),
    "Func. words forced deep": ("gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+VERB", "gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOM+VERB",
                                "gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOMMATCHED+VERB"),
}


def loss(model_dir: Path, name: str, source: str):
    f = model_dir / f"{name}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["sources"].get(source, {}).get("mean_loss")


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> None:
    models = sorted(d for d in RES.iterdir() if d.is_dir() and (d / "learned.json").exists())
    print(f"{len(models)} checkpoints with a learned baseline\n")
    print("Loss increase over learned gates (mean +- sd across checkpoints, n)")
    for name in MIX:
        cells = []
        for s in SOURCES:
            d = [loss(m, name, s) - loss(m, "learned", s) for m in models if loss(m, name, s) is not None]
            cells.append(f"{st.mean(d):+.3f} (+-{st.stdev(d):.3f}, n={len(d)})" if len(d) > 1 else "n/a")
        print(f"  {name:14}", " | ".join(cells))
    print("\nClass-specific excess over random control (paired Wilcoxon across checkpoints)")
    for label, (cls, rnd, matched) in CLASS.items():
        for ctrl_name, ctrl in (("count-matched", rnd), ("perturbation-matched", matched)):
            cells = []
            for s in SOURCES:
                d = [loss(m, cls, s) - loss(m, ctrl, s) for m in models
                     if loss(m, cls, s) is not None and loss(m, ctrl, s) is not None]
                if len(d) > 1:
                    p = wilcoxon(d).pvalue
                    cells.append(f"{st.mean(d):+.3f}{stars(p)} (p={p:.3g}, n={len(d)})")
                else:
                    cells.append("n/a")
            print(f"  {label:24} vs {ctrl_name:21}", " | ".join(cells))
    # perturbation match quality
    for label, (_, _, matched) in CLASS.items():
        t, c = [], []
        for m in models:
            f = m / f"{matched}.json"
            if f.exists():
                for s in SOURCES:
                    pm = json.loads(f.read_text())["sources"].get(s, {}).get("perturbation_match")
                    if pm:
                        t.append(pm["target_mean_delta"]); c.append(pm["control_mean_delta"])
        if t:
            print(f"\n  {label}: mean |delta g| target {st.mean(t):.3f} vs matched control {st.mean(c):.3f}")


if __name__ == "__main__":
    main()
