#!/usr/bin/env python
"""Figure 2 (parameter-quality plane) rebuilt from the paper's own Tables 5/6.

Uses the hierarchical aggregate of Sec. 4.2: mean of C4, WikiText-103, commonsense BPB (mean of six
tasks) and math BPB (mean of seven OLMo3 base-easy tasks). The original figure used an unweighted mean
over all 17 per-task BPBs instead (reviewer YXfZ). Numbers are parsed from main.tex, so the figure
cannot drift from the tables. Single seed, as in Tables 5/6 (the K=2,3 configurations have one seed).

Palette (alpha = share of FFN FLOPs on the deep path): blue = capacity-leaning a25, purple = a50,
terracotta = compute-leaning a75, matching the paper's wide/deep colours. Validated for CVD
separation (protan/deutan dE >= 15 on every pair, normal-vision dE >= 19, OKLab x100).

Usage: make_fig2_scatter.py <main.tex> <out_stem>
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = ["PureWide", "PureLoop K2", "PureLoop K3", "PureLoop K4",
        "a25 K2", "a50 K2", "a75 K2", "a25 K3", "a50 K3", "a75 K3", "a25 K4", "a50 K4", "a75 K4"]
ALPHA_COLOR = {"a25": "#3F8AC2", "a50": "#684681", "a75": "#C7734C"}
K_MARKER = {"K2": "o", "K3": "P", "K4": "X"}
CONTROL = "#6B6B6B"
INK, MUTED = "#1F2A44", "#56607A"


def numbers(line: str) -> list[float]:
    cells = line.split("&")[1:]
    return [float(re.sub(r"[^\d.]", "", c.replace("\\\\", ""))) for c in cells]


def parse(tex: str, label: str) -> dict:
    block = tex[tex.index(f"\\label{{{label}}}"):]
    block = block[:block.index("\\end{tabular}")]
    lines = block.splitlines()
    find = lambda pat: next(l for l in lines if re.match(pat, l.strip()))
    params = [float(x.rstrip("M")) for x in re.findall(r"\\textit\{([\d.]+M)\}", find(r"\\textit\{Params\}"))]
    rows = {
        "cs": numbers(find(r"\\quad BPB")),               # first BPB row = commonsense average
        "math": numbers(find(r"\\quad Easy math avg BPB")),
        "c4": numbers(find(r"\\quad C4 ")),
        "wiki": numbers(find(r"\\quad WikiText-103")),
    }
    out = {}
    for i, name in enumerate(COLS):
        agg = (rows["c4"][i] + rows["wiki"][i] + rows["cs"][i] + rows["math"][i]) / 4
        out[name] = (params[i], agg)
    return out


def panel(ax, data: dict, title: str) -> None:
    loop = [data[n] for n in ("PureLoop K4", "PureLoop K3", "PureLoop K2", "PureWide")]
    ax.plot([p for p, _ in loop], [a for _, a in loop], ls="--", lw=1.0, color=CONTROL, zorder=1)
    for n in ("PureLoop K4", "PureLoop K3", "PureLoop K2"):
        ax.plot(*data[n], marker="^", ms=8, color=CONTROL, mec="white", mew=1.2, ls="none", zorder=3)
    ax.plot(*data["PureWide"], marker="s", ms=8, color=CONTROL, mec="white", mew=1.2, ls="none", zorder=3)
    for a, c in ALPHA_COLOR.items():
        pts = [data[f"{a} {k}"] for k in ("K2", "K3", "K4")]
        ax.plot([p for p, _ in pts], [v for _, v in pts], lw=1.5, color=c, zorder=2)
        for k, m in K_MARKER.items():
            ax.plot(*data[f"{a} {k}"], marker=m, ms=9, color=c, mec="white", mew=1.2, ls="none", zorder=4)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Parameters (M)", fontsize=9, color=MUTED)
    ax.grid(True, color="#E6E6E3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#B8B8B4")
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    tex, out = open(sys.argv[1]).read(), sys.argv[2]
    f80, f160 = parse(tex, "tab:appendix-f80m"), parse(tex, "tab:appendix-f160m")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    panel(axes[0], f80, r"$F_M=80$M ($\sim$1.28G weight-matmul FLOPs/token)")
    panel(axes[1], f160, r"$F_M=160$M ($\sim$2.56G weight-matmul FLOPs/token)")
    axes[0].set_ylabel("Aggregate BPB (lower is better)", fontsize=9, color=MUTED)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", ls="none", color=CONTROL, ms=7, label="PureWide"),
               Line2D([], [], marker="^", ls="--", color=CONTROL, ms=7, label="PureLoop ($K$=4,3,2)")]
    handles += [Line2D([], [], color=c, lw=1.5, label=f"Dual $\\alpha$={a[1:]}") for a, c in ALPHA_COLOR.items()]
    handles += [Line2D([], [], marker=m, ls="none", color=MUTED, ms=7, label=f"$K$={k[1]}") for k, m in K_MARKER.items()]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    for name in ("PureWide", "a50 K4", "a25 K4"):
        print(name, "F80M", f80[name], "F160M", f160[name])


if __name__ == "__main__":
    main()
