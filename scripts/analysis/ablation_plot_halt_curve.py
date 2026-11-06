#!/usr/bin/env python
"""
Plot quality vs compute for adaptive-halting ablations.

Reads ablation_eval.py output JSONs from a directory, builds two compute axes
(per-token theoretical E[steps], and batch-bounded max_active_steps), and
plots mean_loss against each, per Paloma source.

Usage:
    python plot_halt_curve.py --in-dir /path/to/paloma_diagnostics/ablations \\
        --out plot.png

What you get
------------
Two side-by-side panels:
  Left:  x = mean E[steps],   y = mean_loss  (theoretical compute)
  Right: x = max-active steps, y = mean_loss  (actual wall-clock compute)
One curve per Paloma source, plus a marker for the learned baseline (its
"compute" = trained max_loops, since with no threshold every loop runs).
Force-K points are shown too if present, as dotted lines for reference.

Why two panels
--------------
E[steps] is the per-token compute the router *would* spend. With depth-
batching (group tokens by halt-step) you can actually realize this number.
But with vanilla batched inference, the slowest token in each batch decides
how long the loop runs — that's max-active-steps. Reviewers will ask for
both.
"""
import argparse
import json
import re
from pathlib import Path


def parse_halt_threshold(name):
    """'halt0p1' -> 0.1, 'halt0p001' -> 0.001, else None."""
    m = re.match(r"^halt([0-9p]+)$", name)
    if not m:
        return None
    try:
        return float(m.group(1).replace("p", "."))
    except ValueError:
        return None


def parse_force_k(name):
    """'K4' -> 4, else None."""
    m = re.match(r"^K(\d+)$", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Directory of ablation JSONs (output of ablation_eval.py)")
    ap.add_argument("--out", default="halt_curve.png",
                    help="Output plot file (.png, .pdf, .svg ...)")
    ap.add_argument("--mode", choices=["compute", "threshold", "aggregate"],
                    default="compute",
                    help="Plot mode. 'compute' = loss vs E[steps] and max_active "
                         "(default, two panels, K-sweep overlaid). 'threshold' = "
                         "loss vs threshold, one subplot per source, with E[steps] "
                         "and max_active on a secondary y-axis. 'aggregate' = "
                         "single panel, average across sources, loss + E[steps] "
                         "vs threshold.")
    ap.add_argument("--baseline", default="learned",
                    help="Ablation name to use as the baseline reference point")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Subset of sources to plot (default: all)")
    ap.add_argument("--log-x", action="store_true",
                    help="Log-scale x axis (useful for log-spaced thresholds, "
                         "but E[steps] is roughly linear so usually skip)")
    ap.add_argument("--annotate-thresholds", action="store_true",
                    help="Label each point with its threshold value")
    ap.add_argument("--max-k-shown", type=int, default=None,
                    help="Cap K-sweep curve at this K (e.g. trained max_loops). "
                         "OOD K values (above trained) confuse the comparison.")
    ap.add_argument("--highlight-crossing", action="store_true",
                    help="At each K, shade the loss gap between K-sweep and the "
                         "halt curve at matched compute. Positive gap = halt wins "
                         "= per-token routing pays off.")
    ap.add_argument("--show-delta", action="store_true",
                    help="Add a third panel showing (K_loss - halt_loss) at each "
                         "K, interpolating the halt curve to matched compute. "
                         "Positive = halt is better than fixed K. This is the "
                         "paper figure for 'per-token routing pays off'.")
    ap.add_argument("--delta-axis", choices=["etheory", "actual"], default="actual",
                    help="Which compute axis to use when matching K to halt in "
                         "the delta panel. 'actual' (max-active) is the honest "
                         "comparison (K=2 means 2 batch steps; match halt's "
                         "max_active=2). 'etheory' uses per-token E[steps] but "
                         "this only spans a narrow range so few K points match.")
    args = ap.parse_args()

    # Lazy imports so --help doesn't require matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {in_dir}")

    runs = [json.loads(p.read_text()) for p in files]

    # Group runs.
    baseline = next((r for r in runs if r["ablation"] == args.baseline), None)
    halt_runs = []
    k_runs = []
    for r in runs:
        t = parse_halt_threshold(r["ablation"])
        if t is not None:
            halt_runs.append((t, r))
            continue
        k = parse_force_k(r["ablation"])
        if k is not None:
            k_runs.append((k, r))
    halt_runs.sort(key=lambda tr: tr[0])
    k_runs.sort(key=lambda kr: kr[0])

    if not halt_runs:
        raise SystemExit("no halt_threshold runs found in --in-dir")

    # Sources to plot. Use intersection across all halt runs (defensive).
    available_sources = set(halt_runs[0][1]["sources"].keys())
    for _, r in halt_runs[1:]:
        available_sources &= set(r["sources"].keys())
    if args.sources:
        available_sources &= set(args.sources)
    sources = sorted(available_sources)
    if not sources:
        raise SystemExit("no common sources to plot")

    # Build the data: per source, lists of (compute_etheory, compute_actual, loss, threshold).
    series = {s: {"etheory": [], "actual": [], "loss": [], "thresh": []}
              for s in sources}
    for t, r in halt_runs:
        stats = r.get("halt_stats") or {}
        e_t = stats.get("mean_expected_steps_global")
        e_a = stats.get("mean_max_active_steps")
        if e_t is None:
            print(f"  WARN: {r['ablation']} has no halt_stats, skipping")
            continue
        for s in sources:
            src = r["sources"].get(s)
            if src is None:
                continue
            series[s]["etheory"].append(e_t)
            series[s]["actual"].append(e_a if e_a is not None else float("nan"))
            series[s]["loss"].append(src["mean_loss"])
            series[s]["thresh"].append(t)

    # K-sweep (reference): one point per K, where compute = K (per-token AND batch).
    k_series = {s: {"k": [], "loss": []} for s in sources}
    for k, r in k_runs:
        if args.max_k_shown is not None and k > args.max_k_shown:
            continue
        for s in sources:
            src = r["sources"].get(s)
            if src is None:
                continue
            k_series[s]["k"].append(k)
            k_series[s]["loss"].append(src["mean_loss"])

    # Baseline marker. learned has no halt_stats; its compute is trained max_loops,
    # but we don't know that from the JSON. Best we can do: use max K from the
    # K-sweep where the loss matches (or where the user says K=trained_max).
    # Cleaner: plot the baseline as a horizontal dashed line per source.
    base_losses = {}
    if baseline is not None:
        for s in sources:
            src = baseline["sources"].get(s)
            if src is not None:
                base_losses[s] = src["mean_loss"]

    # Plot.
    if args.mode == "compute":
        _plot_compute_mode(args, sources, series, k_series, base_losses, plt)
    elif args.mode == "threshold":
        _plot_threshold_mode(args, sources, series, k_series, base_losses, plt)
    else:
        _plot_aggregate_mode(args, sources, series, k_series, base_losses, plt)

    # Also dump a CSV next to it for downstream plotting.
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("source,threshold,e_steps_theoretical,max_active_steps,mean_loss\n")
        for s in sources:
            for t, et, ea, ls in zip(series[s]["thresh"], series[s]["etheory"],
                                     series[s]["actual"], series[s]["loss"]):
                f.write(f"{s},{t},{et},{ea},{ls}\n")
    print(f"Wrote {csv_path}")


def _plot_compute_mode(args, sources, series, k_series, base_losses, plt):
    """Original two- (or three-) panel: loss vs compute axis."""
    n_panels = 3 if args.show_delta else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6),
                             sharey=False)
    if n_panels == 2:
        # Old behavior: shared y across the two compute panels
        axes[1].sharey(axes[0])
    else:
        # Three panels: first two share y, third is its own delta axis
        axes[1].sharey(axes[0])
    cmap = plt.get_cmap("tab10")

    for panel_idx, (ax, x_key, x_label) in enumerate([
        (axes[0], "etheory", "mean E[steps] per token (theoretical compute)"),
        (axes[1], "actual",  "mean max-active steps (batch-bounded compute)"),
    ]):
        for i, s in enumerate(sources):
            color = cmap(i % 10)
            xs = series[s][x_key]
            ys = series[s]["loss"]
            if not xs:
                continue
            # Sort by x for a sane line.
            order = sorted(range(len(xs)), key=lambda j: xs[j])
            xs_s = [xs[j] for j in order]
            ys_s = [ys[j] for j in order]
            ts_s = [series[s]["thresh"][j] for j in order]
            ax.plot(xs_s, ys_s, "-o", color=color,
                    label=f"{s} (halt)", alpha=0.9, markersize=5)

            if args.annotate_thresholds:
                for x, y, t in zip(xs_s, ys_s, ts_s):
                    ax.annotate(f"{t:.3g}", (x, y), fontsize=6,
                                xytext=(3, 3), textcoords="offset points",
                                color=color, alpha=0.7)

            # K-sweep: a real comparison curve, same color, dashed (not dotted).
            # Compute = K for both axes (K=2 means 2 steps, period).
            ks = k_series[s]["k"]
            kls = k_series[s]["loss"]
            if ks:
                k_order = sorted(range(len(ks)), key=lambda j: ks[j])
                ks_s = [ks[j] for j in k_order]
                kls_s = [kls[j] for j in k_order]
                ax.plot(ks_s, kls_s, "--s", color=color,
                        label=f"{s} (force K)" if panel_idx == 0 else None,
                        alpha=0.7, markersize=5, markerfacecolor="none")

                # Optional: shade the gap at each K.
                if args.highlight_crossing and xs_s:
                    import numpy as _np
                    halt_at_k = _np.interp(ks_s, xs_s, ys_s,
                                           left=_np.nan, right=_np.nan)
                    for kx, k_loss, h_loss in zip(ks_s, kls_s, halt_at_k):
                        if _np.isnan(h_loss):
                            continue
                        # vertical line from K-curve point to halt-curve point
                        ax.vlines(kx, min(k_loss, h_loss), max(k_loss, h_loss),
                                  color=color, alpha=0.25, linewidth=4)

            # Baseline horizontal line.
            if s in base_losses:
                ax.axhline(base_losses[s], color=color, linestyle=":",
                           linewidth=0.6, alpha=0.5)

        if args.log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)

    # Optional delta panel: (K_loss - halt_loss) at matched compute.
    if args.show_delta:
        import numpy as _np
        ax_d = axes[2]
        # Use the theoretical-compute axis (etheory) for matching, since
        # K is exact compute and E[steps] is the per-token equivalent.
        for i, s in enumerate(sources):
            color = cmap(i % 10)
            ks = k_series[s]["k"]
            kls = k_series[s]["loss"]
            xs = series[s][args.delta_axis]
            ys = series[s]["loss"]
            if not ks or not xs:
                continue
            # Sort halt curve by x for interp
            order = sorted(range(len(xs)), key=lambda j: xs[j])
            xs_s = [xs[j] for j in order]
            ys_s = [ys[j] for j in order]
            # At each K, interpolate halt-loss
            k_order = sorted(range(len(ks)), key=lambda j: ks[j])
            ks_s = [ks[j] for j in k_order]
            kls_s = [kls[j] for j in k_order]
            halt_at_k = _np.interp(ks_s, xs_s, ys_s,
                                   left=_np.nan, right=_np.nan)
            deltas = [kl - hl for kl, hl in zip(kls_s, halt_at_k)
                      if not _np.isnan(hl)]
            ks_kept = [k for k, hl in zip(ks_s, halt_at_k) if not _np.isnan(hl)]
            if not ks_kept:
                continue
            ax_d.plot(ks_kept, deltas, "-o", color=color, label=s,
                      markersize=6, linewidth=2)
        ax_d.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax_d.set_xlabel(f"compute (K loops, matched to halt on {args.delta_axis})")
        ax_d.set_ylabel("loss(force_K) − loss(halt) at matched compute")
        ax_d.set_title("Per-token routing benefit\n(positive = halt beats fixed K)",
                       fontsize=10)
        ax_d.legend(fontsize=8, loc="best")
        ax_d.grid(True, alpha=0.3)

    axes[0].set_ylabel("mean cross-entropy loss")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Adaptive halting vs forced-K compute\n"
                 "(circles = halt_threshold sweep, squares = force_K, "
                 "dotted = learned baseline; gap shaded if --highlight-crossing)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


def _plot_threshold_mode(args, sources, series, k_series, base_losses, plt):
    """One subplot per source. x = threshold, primary y = loss, secondary y = compute.

    K reference: horizontal lines on the loss axis at K=1, K=2, K=trained_max.
    We can't put K on the threshold x-axis (K isn't a threshold), so K lives
    as horizontal annotations: "at this loss level, you'd need K=2."
    """
    import numpy as _np
    n = len(sources)
    # Up to 3 columns, wrap rows.
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Pick which K values to render as horizontal references on the loss axis.
    # Common K values across all sources, sorted, deduplicated, capped.
    all_ks = set()
    for s in sources:
        for k in k_series[s]["k"]:
            if args.max_k_shown is not None and k > args.max_k_shown:
                continue
            all_ks.add(k)
    ks_to_show = sorted(all_ks)

    for ax_idx, s in enumerate(sources):
        ax = axes_flat[ax_idx]

        # Sort halt data by threshold
        thr = series[s]["thresh"]
        loss = series[s]["loss"]
        etheory = series[s]["etheory"]
        actual = series[s]["actual"]
        if not thr:
            ax.text(0.5, 0.5, f"no halt data for {s}",
                    transform=ax.transAxes, ha="center")
            continue
        order = sorted(range(len(thr)), key=lambda j: thr[j])
        thr_s = [thr[j] for j in order]
        loss_s = [loss[j] for j in order]
        etheory_s = [etheory[j] for j in order]
        actual_s = [actual[j] for j in order]

        # Primary axis: loss (left)
        l1, = ax.plot(thr_s, loss_s, "-o", color="C0",
                      label="mean loss", markersize=5, linewidth=2)
        ax.set_xlabel("halt threshold T  (token halts when prob_remain ≤ T)")
        ax.set_ylabel("mean cross-entropy loss", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.grid(True, alpha=0.3)

        # K reference: horizontal lines on the loss axis, labeled with K=N.
        # Drop the K that equals the learned baseline (it visually doubles
        # the dashed "learned" line and clutters the label area).
        k_lookup = dict(zip(k_series[s]["k"], k_series[s]["loss"]))
        learned_loss = base_losses.get(s)
        x_label_pos = thr_s[-1] - 0.005 * (thr_s[-1] - thr_s[0])
        for k in ks_to_show:
            if k not in k_lookup:
                continue
            k_loss = k_lookup[k]
            # Skip if this K is effectively the learned baseline
            if learned_loss is not None and abs(k_loss - learned_loss) < 1e-3:
                continue
            ax.axhline(k_loss, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(x_label_pos, k_loss, f"K={k}", fontsize=8,
                    color="gray", verticalalignment="bottom",
                    horizontalalignment="right", alpha=0.8)

        # Learned baseline as a labeled horizontal line, distinct style.
        if s in base_losses:
            ax.axhline(base_losses[s], color="black", linestyle="--",
                       linewidth=1.0, alpha=0.5)
            ax.text(x_label_pos, base_losses[s], "learned", fontsize=8,
                    color="black", verticalalignment="bottom",
                    horizontalalignment="right", alpha=0.7)

        # Secondary axis: compute (right)
        ax2 = ax.twinx()
        l2, = ax2.plot(thr_s, etheory_s, "-s", color="C3",
                       label="E[steps] (per-token)", markersize=4,
                       linewidth=1.5, alpha=0.85)
        l3, = ax2.plot(thr_s, actual_s, "-^", color="C2",
                       label="max-active (batch)", markersize=4,
                       linewidth=1.5, alpha=0.85)
        ax2.set_ylabel("loop steps", color="dimgray")
        ax2.tick_params(axis="y", labelcolor="dimgray")
        # Tie y-range to a sensible bound: 0.8 to slightly above max_active max
        max_compute = max(max(actual_s), max(etheory_s))
        ax2.set_ylim(0.8, max_compute * 1.05)

        ax.set_title(f"{s}", fontsize=11, fontweight="bold")
        # One combined legend per subplot
        ax.legend(handles=[l1, l2, l3], fontsize=8, loc="best")

    # Hide unused axes if sources don't fill the grid
    for i in range(len(sources), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Adaptive halting: loss & compute vs halt threshold\n"
                 "(gray dotted = force_K reference levels, black dashed = learned baseline)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


def _plot_aggregate_mode(args, sources, series, k_series, base_losses, plt):
    """Single panel, mean loss and mean E[steps] averaged across sources, vs threshold.

    Loss and compute pull in opposite directions as threshold rises, so we
    encode them with contrasting hues: a muted indigo for loss (the cost we
    pay) and a warm amber for compute (the resource we save). Both are
    print-safe and friendly to red-green color blindness.

    Loss is shown with a ±1σ band across sources to give a sense of how
    consistently the trend holds across Paloma sources. E[steps] is averaged
    similarly but without a band (compute is router-driven, not source-
    driven, so cross-source variance is small and noisy).
    """
    import numpy as _np

    # Aesthetic choices
    LOSS_COLOR    = "#3D348B"   # deep indigo
    LOSS_BAND     = "#3D348B"
    COMPUTE_COLOR = "#E6A817"   # warm amber
    REF_COLOR     = "#999999"   # muted gray for K reference lines
    BASELINE_COLOR = "#222222"  # near-black for learned baseline

    # Collect (threshold, loss_delta_per_source, etheory_per_source).
    # We aggregate Δloss = loss(halt) - loss(learned) per source, NOT raw
    # loss. Raw loss differs hugely across sources (gsm8k ~2.1, triviaqa ~4.1)
    # so a std band across raw losses just measures source difficulty.
    # The delta isolates the halt effect, which is what we want to show.
    by_thr = {}
    for s in sources:
        if s not in base_losses:
            continue
        b = base_losses[s]
        for thr, ls, et in zip(series[s]["thresh"], series[s]["loss"],
                               series[s]["etheory"]):
            by_thr.setdefault(thr, {})[s] = (ls - b, et)
    if not by_thr:
        raise SystemExit("no halt data to plot")

    thresholds = sorted(by_thr.keys())
    dloss_mean = []
    dloss_std  = []
    es_mean    = []
    for t in thresholds:
        dls = [by_thr[t][s][0] for s in sources if s in by_thr[t]]
        es  = [by_thr[t][s][1] for s in sources if s in by_thr[t]]
        dloss_mean.append(_np.mean(dls))
        dloss_std.append(_np.std(dls))
        es_mean.append(_np.mean(es))

    dloss_mean = _np.array(dloss_mean)
    dloss_std  = _np.array(dloss_std)
    es_mean    = _np.array(es_mean)

    # K-sweep deltas (vs learned) averaged across sources for reference lines.
    k_avg = {}
    for s in sources:
        if s not in base_losses:
            continue
        b = base_losses[s]
        for k, ls in zip(k_series[s]["k"], k_series[s]["loss"]):
            if args.max_k_shown is not None and k > args.max_k_shown:
                continue
            k_avg.setdefault(k, []).append(ls - b)
    k_avg = {k: float(_np.mean(v)) for k, v in k_avg.items()}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Primary axis: Δ mean loss (indigo) with ±1σ band
    ax.fill_between(thresholds, dloss_mean - dloss_std, dloss_mean + dloss_std,
                    color=LOSS_BAND, alpha=0.15, linewidth=0)
    line_loss, = ax.plot(thresholds, dloss_mean, "-o", color=LOSS_COLOR,
                         markersize=6, linewidth=2.2,
                         label="Δ mean loss vs learned",
                         markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("halt threshold T   (token halts when prob_remain ≤ T)",
                  fontsize=11)
    ax.set_ylabel("Δ loss vs learned baseline\n(averaged across sources, ±1σ)",
                  color=LOSS_COLOR, fontsize=11)
    ax.tick_params(axis="y", labelcolor=LOSS_COLOR)

    # Reference horizontal lines: K-deltas + a zero line for the baseline.
    x_label_pos = thresholds[-1] - 0.005 * (thresholds[-1] - thresholds[0])
    # Zero line = learned baseline
    ax.axhline(0.0, color=BASELINE_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.55)
    ax.text(x_label_pos, 0.0, "  learned  ", fontsize=8.5,
            color=BASELINE_COLOR, alpha=0.9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
    for k in sorted(k_avg.keys()):
        kd = k_avg[k]
        # Skip K lines too close to learned (would overlap the "learned"
        # label and not represent a meaningfully different reference point).
        if abs(kd) < 0.1:
            continue
        ax.axhline(kd, color=REF_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.55)
        ax.text(x_label_pos, kd, f"  K={k}  ", fontsize=8.5,
                color=REF_COLOR, alpha=0.9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1.0))

    # Secondary axis: mean E[steps] (amber)
    ax2 = ax.twinx()
    line_steps, = ax2.plot(thresholds, es_mean, "-s", color=COMPUTE_COLOR,
                            markersize=5.5, linewidth=2.0,
                            label="mean E[steps] per token",
                            markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_ylabel("mean expected loop steps per token",
                   color=COMPUTE_COLOR, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=COMPUTE_COLOR)
    # Pad the compute axis a bit so the line doesn't kiss the top/bottom
    es_range = max(es_mean.max() - es_mean.min(), 0.5)
    ax2.set_ylim(es_mean.min() - 0.1 * es_range,
                 es_mean.max() + 0.15 * es_range)

    # Subtle grid only on the primary axis to keep things calm
    ax.grid(True, axis="y", color=LOSS_COLOR, alpha=0.08, linewidth=0.7)
    ax.grid(True, axis="x", color="black", alpha=0.06, linewidth=0.7)
    ax.set_axisbelow(True)

    # De-clutter spines
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LOSS_COLOR)
    ax.spines["left"].set_alpha(0.4)
    ax2.spines["right"].set_color(COMPUTE_COLOR)
    ax2.spines["right"].set_alpha(0.4)

    # Combined legend
    ax.legend(handles=[line_loss, line_steps], fontsize=10,
              loc="upper left", frameon=True, framealpha=0.9,
              edgecolor="#cccccc")

    ax.set_title("Adaptive halting: quality vs compute as halt threshold rises\n"
                 f"averaged across {len(sources)} Paloma sources",
                 fontsize=12, pad=12)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()