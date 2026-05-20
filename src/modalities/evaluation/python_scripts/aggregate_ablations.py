#!/usr/bin/env python
"""
Aggregate ablation_eval.py outputs into a single comparison table.

Usage:
    python aggregate_ablations.py --in-dir /path/to/paloma_diagnostics/ablations
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--baseline", default="learned",
                    help="Name of the ablation to use as the reference for delta columns")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {in_dir}")

    rows = [json.loads(p.read_text()) for p in files]
    sources = sorted({s for r in rows for s in r["sources"]})

    # Header
    print(f"\n{'ablation':<32s} " + "  ".join(f"{s:>14s}" for s in sources))
    print("-" * (33 + 16 * len(sources)))

    # Group rows: baseline, K-sweep sorted by K, then other ablations.
    baseline_row = next((r for r in rows if r["ablation"] == args.baseline), None)
    rest = [r for r in rows if r["ablation"] != args.baseline]

    def k_value(r):
        """Extract K from ablation name like 'K2', or None if not a K-sweep."""
        n = r["ablation"]
        if n.startswith("K") and n[1:].isdigit():
            return int(n[1:])
        return None

    k_rows = sorted([r for r in rest if k_value(r) is not None], key=k_value)
    other_rows = [r for r in rest if k_value(r) is None]
    # Sort other_rows alphabetically for stability.
    other_rows.sort(key=lambda r: r["ablation"])

    def fmt_row(name, vals, deltas=None):
        cells = []
        for v, d in zip(vals, deltas or [None] * len(vals)):
            if v is None:
                cells.append(f"{'—':>14s}")
            elif d is None:
                cells.append(f"{v:>9.4f}      ")
            else:
                cells.append(f"{v:>7.4f} ({d:+.3f})")
        return f"{name:<32s} " + "  ".join(cells)

    def vals_for(r):
        return [r["sources"].get(s, {}).get("mean_loss") for s in sources]

    base_vals = vals_for(baseline_row) if baseline_row else None

    def deltas_for(r):
        if base_vals is None:
            return None
        v = vals_for(r)
        return [(a - b) if (a is not None and b is not None) else None
                for a, b in zip(v, base_vals)]

    if baseline_row:
        print(fmt_row(baseline_row["ablation"] + " (baseline)", base_vals))
    if k_rows:
        print()
        print("  --- compute (force K loops) ---")
        for r in k_rows:
            print(fmt_row(r["ablation"], vals_for(r), deltas_for(r)))
    if other_rows:
        print()
        print("  --- gate overrides ---")
        for r in other_rows:
            print(fmt_row(r["ablation"], vals_for(r), deltas_for(r)))

    print()
    print("Lower mean_loss is better. Delta = ablation − baseline.")
    print("Positive delta = ablation is WORSE than the learned router.")
    print("Near-zero delta on 'shuffle' = router didn't learn meaningful per-token routing.")


if __name__ == "__main__":
    main()