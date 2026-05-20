#!/usr/bin/env python
"""
Standalone ablation eval for the dual-path model.

Reuses the input_ids cached in paloma_diagnostics parquets (so no
re-tokenization, no dataset download, no OLMES). Reports mean per-token
cross-entropy loss + std per source, optionally split by POS using the
sidecar POS parquets.

Why this script exists
----------------------
For the EMNLP paper, the load-bearing ablations are:

  1. "loop-only at inference"   : g_deep=1, g_wide=0
  2. "wide-only at inference"   : g_deep=0, g_wide=1
  3. "uniform router"           : g_deep=0.5, g_wide=0.5
  4. "shuffled router"          : per-sequence permutation of learned gates
  5. "force K loops"            : exact loop budget, sweep K = 1, 2, 3, ...

If (1)-(4) all hurt vs the learned router, the per-token routing is real.
If (4) ≈ learned, the router is decorative. This script lets you generate
the whole ablation table from one model checkpoint with no retraining.

Usage
-----
  # Baseline (no override)
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation learned

  # Gate overrides
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation 'g_deep=1,g_wide=0'
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation 'g_deep=0.5,g_wide=0.5'
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation shuffle

  # Loop control (sweep K)
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation force_loops=2
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation force_loops=4

  # Combinations (comma-separated)
  python ablation_eval.py --ckpt CKPT --paloma-dir DIAG --ablation 'g_deep=1,g_wide=0,force_loops=2'

  # Limit work
  python ablation_eval.py ... --sources gsm8k --max-chunks 50

The output JSON ends up at <out>/<ablation_name>.json. Run several
ablations, then aggregate the JSONs into a paper table with a small helper.

How the monkey-patches work
---------------------------
- Gate override: wrap each DualPathGate*.forward to intercept the computed
  gate tensor right after the sigmoid, before it gets used.
- Force-K loops: set max_loops=K on every AdaptiveRecursiveBlock AND patch
  its router so halt_prob always returns 1.0 at step K-1. That makes
  prob_remain hit 0 exactly at the last step, so every token sees exactly
  K loop iterations, no leakage.

The patches are applied to the loaded model in-process. The checkpoint
on disk is untouched.
"""

import argparse
import json
import re
from pathlib import Path
from types import MethodType

import torch
import pyarrow.parquet as pq


# ----------------------------------------------------------------------------
# Ablation spec parsing
# ----------------------------------------------------------------------------

def parse_ablation(spec: str):
    """Parse a comma-separated ablation spec.

    Examples
    --------
      'learned'                          -> {}
      'g_deep=1,g_wide=0'                -> {'g_deep': 1.0, 'g_wide': 0.0}
      'shuffle'                          -> {'shuffle': True}
      'force_loops=2'                    -> {'force_loops': 2}
      'g_deep=1,g_wide=0,force_loops=2'  -> combined

    Returns a dict with optional keys: g_deep, g_wide, shuffle, force_loops.
    """
    out = {}
    if spec.strip().lower() == "learned":
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "shuffle":
            out["shuffle"] = True
            continue
        m = re.match(r"^(g_deep|g_wide|force_loops)\s*=\s*([\d.]+)$", part)
        if not m:
            raise ValueError(f"can't parse ablation part: {part!r}")
        key, val = m.group(1), m.group(2)
        if key == "force_loops":
            out["force_loops"] = int(val)
        else:
            out[key] = float(val)
    if "shuffle" in out and ("g_deep" in out or "g_wide" in out):
        raise ValueError("shuffle is incompatible with constant gate overrides")
    return out


def ablation_name(spec_dict):
    """Build a short, filesystem-safe name from a parsed ablation dict."""
    if not spec_dict:
        return "learned"
    parts = []
    if "g_deep" in spec_dict: parts.append(f"gd{spec_dict['g_deep']}")
    if "g_wide" in spec_dict: parts.append(f"gw{spec_dict['g_wide']}")
    if spec_dict.get("shuffle"): parts.append("shuffle")
    if "force_loops" in spec_dict: parts.append(f"K{spec_dict['force_loops']}")
    return "_".join(parts).replace(".", "p")


# ----------------------------------------------------------------------------
# Monkey-patches
# ----------------------------------------------------------------------------

def patch_gates(model, spec):
    """Install a forward-wrapper on every dual gate to apply the override.

    Returns the count of patched gates so the caller can sanity-check.
    """
    if not any(k in spec for k in ("g_deep", "g_wide", "shuffle")):
        return 0   # no gate patch needed

    g_deep = spec.get("g_deep", None)
    g_wide = spec.get("g_wide", None)
    do_shuffle = bool(spec.get("shuffle", False))

    n_patched = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "DualPathGateConvex":
            _wrap_convex_gate(module, g_deep, g_wide, do_shuffle)
            n_patched += 1
        elif cls_name == "DualPathGateTwoGates":
            _wrap_two_gates(module, g_deep, g_wide, do_shuffle)
            n_patched += 1
    return n_patched


def _wrap_convex_gate(gate_module, g_deep, g_wide, do_shuffle):
    """Patch DualPathGateConvex.forward. In convex mode g_wide = 1 - gate,
    so we override the single gate value. If g_deep given, use it; else if
    g_wide given, use 1 - g_wide; else if shuffle, permute along T."""
    original_forward = gate_module.forward.__func__   # unbound

    def patched_forward(self, x, h_deep, h_wide, record=False):
        # Recompute gate exactly as the original does. We can't just call
        # original_forward and rewrite the output because the gate value
        # feeds into the convex combination internally — we have to inject
        # before that combination. Easiest: replicate the math here.
        import torch as _t
        import torch.nn.functional as _F
        gate = _t.sigmoid(self.gate_proj(x))   # (B, T, 1)

        # ---- override ----
        if g_deep is not None:
            gate = _t.full_like(gate, float(g_deep))
        elif g_wide is not None:
            gate = _t.full_like(gate, 1.0 - float(g_wide))
        elif do_shuffle:
            gate = _shuffle_T(gate)
        # ------------------

        cross_w2d_norm = cross_d2w_norm = None
        if self.use_cross:
            s_d = _F.softplus(self.cross_scale_deep)
            s_w = _F.softplus(self.cross_scale_wide)
            c_w2d = s_d * self.proj_w2d(h_wide)
            c_d2w = s_w * self.proj_d2w(h_deep)
            h_deep_eff = h_deep + c_w2d
            h_wide_eff = h_wide + c_d2w
            if record:
                cross_w2d_norm = c_w2d.norm(dim=-1)
                cross_d2w_norm = c_d2w.norm(dim=-1)
        else:
            h_deep_eff, h_wide_eff = h_deep, h_wide
        out = gate * h_deep_eff + (1.0 - gate) * h_wide_eff
        return out, gate.squeeze(-1), cross_w2d_norm, cross_d2w_norm

    gate_module.forward = MethodType(patched_forward, gate_module)


def _wrap_two_gates(gate_module, g_deep, g_wide, do_shuffle):
    """Patch DualPathGateTwoGates.forward."""
    def patched_forward(self, x, h_deep, h_wide, record=False):
        import torch as _t
        import torch.nn.functional as _F
        gates = _t.sigmoid(self.gate_proj(x))   # (B, T, 2)

        # ---- override ----
        if do_shuffle:
            # Shuffle each channel independently — otherwise the learned
            # gd/gw correlation across tokens survives and the ablation is
            # less aggressive than it should be.
            gd_s = _shuffle_T(gates[..., 0:1])
            gw_s = _shuffle_T(gates[..., 1:2])
            gates = _t.cat([gd_s, gw_s], dim=-1)
        else:
            out_gates = gates.clone()
            if g_deep is not None:
                out_gates[..., 0] = float(g_deep)
            if g_wide is not None:
                out_gates[..., 1] = float(g_wide)
            gates = out_gates
        # ------------------

        gd = gates[..., 0:1]
        gw = gates[..., 1:2]
        cross_w2d_norm = cross_d2w_norm = None
        if self.use_cross:
            s_d = _F.softplus(self.cross_scale_deep)
            s_w = _F.softplus(self.cross_scale_wide)
            c_w2d = s_d * self.proj_w2d(h_wide)
            c_d2w = s_w * self.proj_d2w(h_deep)
            contam_w2d = gd * c_w2d
            contam_d2w = gw * c_d2w
            h_deep_branch = gd * h_deep + contam_w2d
            h_wide_branch = gw * h_wide + contam_d2w
            if record:
                cross_w2d_norm = contam_w2d.norm(dim=-1)
                cross_d2w_norm = contam_d2w.norm(dim=-1)
        else:
            h_deep_branch = gd * h_deep
            h_wide_branch = gw * h_wide
        out = h_deep_branch + h_wide_branch
        return out, gd.squeeze(-1), gw.squeeze(-1), cross_w2d_norm, cross_d2w_norm

    gate_module.forward = MethodType(patched_forward, gate_module)


def _shuffle_T(x):
    """Permute x (shape (B, T, C)) along T independently per batch row."""
    B, T = x.shape[:2]
    out = torch.empty_like(x)
    for b in range(B):
        perm = torch.randperm(T, device=x.device)
        out[b] = x[b, perm]
    return out


def patch_force_loops(model, K):
    """Force every adaptive block to run K loop iterations.

    For K <= trained max_loops:
      Just set max_loops=K. The block's original forward truncates the
      loop at K steps; any unhalted prob_remain is pooled into the final
      h_loop (matches training semantics).

    For K > trained max_loops:
      We're past training's loop budget, which is OOD. We make two
      conservative choices:
        (a) loop_scales[step] for step >= trained_max_loops repeats
            loop_scales[trained_max_loops - 1]. So extra steps use the
            same scale as the last trained step.
        (b) step_normalized passed to the router is clamped at 1.0
            (i.e., the value the router saw on the last trained step).
            Otherwise step_normalized = step / (K-1) would put steps
            0..3 at fractions like 0/9..3/9 instead of 0/3..3/3, totally
            shifting the router's input distribution.

    With these, the model is asked "what if there were more loop
    iterations at the same regime as the final trained step?" — a
    well-defined extrapolation, even if reviewers may still ask
    questions.
    """
    if K is None:
        return 0
    if K < 1:
        raise ValueError(f"force_loops must be >= 1, got {K}")

    n_patched = 0
    for module in model.modules():
        if type(module).__name__ != "AdaptiveRecursiveBlock":
            continue
        if not getattr(module, "has_loop_path", False):
            continue

        # Stash originals once. Re-applying the patch later (e.g. different K)
        # then reads back from these to start fresh.
        if not hasattr(module, "_orig_max_loops"):
            module._orig_max_loops = module.max_loops
        if not hasattr(module, "_orig_loop_scales"):
            module._orig_loop_scales = module.loop_scales.detach().clone()

        trained_max = int(module._orig_max_loops)
        trained_step_denom = max(1, trained_max - 1)

        # Set the new K.
        module.max_loops = K

        # ---- loop_scales handling ----
        if K <= trained_max:
            # Truncate (or keep). Slicing the original gives the first K entries,
            # which is exactly what the forward indexes into. We rewrap as
            # nn.Parameter so the type stays consistent, though it's not needed
            # for eval-only.
            new_scales = module._orig_loop_scales[:K].clone()
        else:
            # Extend by repeating the last trained scale.
            last_scale = module._orig_loop_scales[trained_max - 1]
            pad = last_scale.expand(K - trained_max).clone()
            new_scales = torch.cat([module._orig_loop_scales, pad], dim=0)

        # Replace the parameter. We use nn.Parameter to preserve type.
        # Since we're in eval mode this doesn't affect gradient flow.
        module.loop_scales = torch.nn.Parameter(
            new_scales.to(module.loop_scales.device).to(module.loop_scales.dtype),
            requires_grad=False,
        )

        # ---- router patch for step_normalized remapping ----
        # When K > trained_max, the block computes step_normalized using
        # the wrong denominator. We need to remap so the router sees the
        # value it would have at the equivalent step at train time:
        #     trained_normalized[step] = min(step, trained_max - 1) / trained_step_denom
        # So for K=10 with trained_max=4:
        #   step 0 -> 0/3 = 0.0   (= training step 0)
        #   step 1 -> 1/3 = 0.33  (= training step 1)
        #   step 2 -> 2/3 = 0.67  (= training step 2)
        #   step 3 -> 3/3 = 1.0   (= training step 3, the last trained)
        #   step 4..9 -> 1.0      (clamped at the last trained step's value)
        #
        # We need to know the current step number, but the block passes only
        # step_normalized (a float). We use a per-router counter, reset at
        # the start of each block forward via a thin wrapper.
        router = module.router
        if K > trained_max:
            if not hasattr(router, "_orig_forward"):
                router._orig_forward = router.forward

            def make_remapped(orig_forward, trained_denom):
                def patched(self, h, step_normalized, x=None):
                    # Pull and bump the counter.
                    s = getattr(self, "_force_step_counter", 0)
                    self._force_step_counter = s + 1
                    # Map to trained-time normalized step, clamped at 1.0.
                    sn = min(s, trained_denom) / trained_denom
                    return orig_forward.__func__(self, h, step_normalized=sn, x=x)
                return patched

            router.forward = MethodType(make_remapped(router._orig_forward, trained_step_denom), router)

            # Wrap the block forward to reset the router counter on entry.
            if not hasattr(module, "_orig_forward"):
                module._orig_forward = module.forward

            def make_counter_reset(orig_block_forward):
                def patched(self, x):
                    self.router._force_step_counter = 0
                    return orig_block_forward.__func__(self, x)
                return patched

            module.forward = MethodType(make_counter_reset(module._orig_forward), module)
        else:
            # Restore router and block forward if previously patched for K > trained_max.
            if hasattr(router, "_orig_forward"):
                router.forward = router._orig_forward
                del router._orig_forward
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
                del module._orig_forward

        n_patched += 1
    return n_patched


# ----------------------------------------------------------------------------
# Eval loop
# ----------------------------------------------------------------------------

def eval_one_source(model, parquet_path, max_chunks, device, pos_map=None):
    """Iterate over a source's parquet, run model with current patches,
    return per-token loss stats plus optional POS-grouped stats."""
    import numpy as np

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if max_chunks > 0:
        rows = rows[:max_chunks]

    total_loss_sum = 0.0
    total_loss_count = 0
    per_pos = {}   # tag -> [sum, count]

    for r in rows:
        ids = torch.tensor([r["tokens"]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        # We reuse the model's own labels=ids path, but it averages — we
        # want per-token to grade variance. Easier: recompute per-token.
        logits = out.logits if hasattr(out, "logits") else out[1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(1, -1).float().cpu().numpy()[0]   # (T-1,)

        total_loss_sum += ce.sum()
        total_loss_count += ce.size

        # Per-POS aggregation. POS tags are length T, loss is length T-1
        # (no next-token loss for the last position). Align by slicing.
        if pos_map is not None:
            key = (int(r["doc"]), int(r["chunk"]))
            tags = pos_map.get(key)
            if tags is not None and len(tags) == r["n_tokens"]:
                tags_aligned = tags[:-1]   # drop last (no loss)
                for tag, l in zip(tags_aligned, ce):
                    if tag in ("UNKNOWN", "SPECIAL"):
                        continue
                    if tag not in per_pos:
                        per_pos[tag] = [0.0, 0]
                    per_pos[tag][0] += float(l)
                    per_pos[tag][1] += 1

    mean = total_loss_sum / max(total_loss_count, 1)
    pos_summary = {
        tag: {"mean_loss": s / c, "n_tokens": c}
        for tag, (s, c) in per_pos.items() if c > 0
    }
    return {
        "n_chunks": len(rows),
        "n_tokens_with_loss": int(total_loss_count),
        "mean_loss": float(mean),
        "ppl": float(np.exp(mean)),
        "by_pos": pos_summary,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir")
    ap.add_argument("--paloma-dir", required=True,
                    help="Directory with paloma_<source>.parquet files "
                         "(typically <ckpt>/paloma_diagnostics)")
    ap.add_argument("--ablation", required=True,
                    help="Ablation spec, e.g. 'learned', 'g_deep=1,g_wide=0', "
                         "'shuffle', 'force_loops=2'")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Sources to eval (default: all parquets in --paloma-dir)")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Cap chunks per source (0 = use all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the JSON (default: --paloma-dir/ablations)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Affects shuffle ablation only")
    args = ap.parse_args()

    spec = parse_ablation(args.ablation)
    name = ablation_name(spec)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.paloma_dir) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    print(f"Ablation: {name}  ({spec})")
    print(f"Will write: {out_path}")

    torch.manual_seed(args.seed)

    print(f"Loading model from {args.ckpt}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True
    ).to(args.device).eval()

    # Apply patches BEFORE the eval loop. Order matters: gates first, then
    # loops — gate patches replace forward methods on gate modules, loop
    # patches replace forward on the enclosing AdaptiveRecursiveBlock and
    # the router, so they don't collide.
    n_gates = patch_gates(model, spec)
    n_loops = patch_force_loops(model, spec.get("force_loops", None))
    print(f"  patched {n_gates} dual gates, {n_loops} loop blocks")

    # Sources
    paloma_dir = Path(args.paloma_dir)
    parquets = sorted(paloma_dir.glob("paloma_*.parquet"))
    parquets = [p for p in parquets if not p.stem.endswith("_pos")]
    if args.sources:
        wanted = set(args.sources)
        parquets = [p for p in parquets if p.stem.replace("paloma_", "") in wanted]
    if not parquets:
        raise SystemExit(f"no parquets found in {paloma_dir}")

    results = {"ablation": name, "spec": spec, "sources": {}}
    for pq_path in parquets:
        source = pq_path.stem.replace("paloma_", "")
        # POS sidecar if available
        pos_path = pq_path.with_name(pq_path.stem + "_pos.parquet")
        pos_map = None
        if pos_path.exists():
            pos_map = {}
            for r in pq.read_table(pos_path).to_pylist():
                pos_map[(int(r["doc"]), int(r["chunk"]))] = r["pos"]

        print(f"\n[{source}]")
        r = eval_one_source(model, pq_path, args.max_chunks, args.device, pos_map)
        print(f"  mean_loss = {r['mean_loss']:.4f}  "
              f"ppl = {r['ppl']:.2f}  "
              f"({r['n_tokens_with_loss']:,} tokens over {r['n_chunks']} chunks)")
        if r["by_pos"]:
            top = sorted(r["by_pos"].items(), key=lambda kv: -kv[1]["n_tokens"])[:6]
            for tag, stats in top:
                print(f"    {tag:6s}  mean_loss={stats['mean_loss']:.3f}  n={stats['n_tokens']}")
        results["sources"][source] = r

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()