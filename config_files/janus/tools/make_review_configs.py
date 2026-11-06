#!/usr/bin/env python3
"""Generate the review-round configs into config_files/janus/review/.

Three groups, three seeds each (42, 43, 44: the sampler seeds behind the main table):

attnmatched: PureWide and PureLoop K=4 with their FFN widened until total forward FLOPs,
    including the causal attention-score products (QK^T, AV) that Eq. 9 of the paper leaves
    out, reach the most expensive K=4 dual config at the same budget. Widths are solved in
    the review notes; they are the smallest multiple of 64 that reaches the target.
gate: controls for the token-wise gate, built from the F80M Dual a50 K=4 main-table config:
    gatefixed     static 0.5/0.5 mixture (gate_mode: fixed)
    gateperlayer  learned per-layer gates shared by all tokens (gate_token_independent)
    gateungated   token-wise gates on the path updates only, plain residual (gate_residual)

For every run: keep only the newest checkpoint (enough to resume), and no inline downstream
eval. Training would otherwise wait on the OLMES job with all 16 nodes idle. The final
checkpoint is evaluated by scripts/evaluation/queue_eval_after_training.sh. A _WARMSTART twin
is written next to each config for runs that need a second 24h link.

Usage: make_review_configs.py
"""

import sys
from pathlib import Path

JANUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_warmstart_config import transform as to_warmstart  # noqa: E402

SEEDS = (42, 43, 44)
SWEEP = JANUS / "l16_sweep"
OUT = JANUS / "review"

ATTNMATCHED = [
    # (base config, key, old width, new width, output stem)
    ("8e7_loops2__dm768_L16_loop1_F80M_ffnW24576_purewide.yaml", "wide_ffn_hidden", 24576, 32704,
     "8e7_loops2__dm768_L16_loop1_F80M_ffnW32704_purewide_attnmatched"),
    ("8e7_loops4__dm768_L16_loop4_F80M_ffnL4864_pureloop.yaml", "ffn_hidden", 4864, 5504,
     "8e7_loops4__dm768_L16_loop4_F80M_ffnL5504_pureloop_attnmatched"),
    ("16e7_loops2__dm768_L16_loop1_F160M_ffnW50624_purewide.yaml", "wide_ffn_hidden", 50624, 58432,
     "16e7_loops2__dm768_L16_loop1_F160M_ffnW58432_purewide_attnmatched"),
    ("16e7_loops4__dm768_L16_loop4_F160M_ffnL11392_pureloop.yaml", "ffn_hidden", 11392, 11968,
     "16e7_loops4__dm768_L16_loop4_F160M_ffnL11968_pureloop_attnmatched"),
]

GATE_BASE = "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross.yaml"
GATE_CONTROLS = {
    "gatefixed": ('      gate_mode: "two_gates"\n', '      gate_mode: "fixed"\n'),
    "gateperlayer": ('      gate_mode: "two_gates"\n', '      gate_mode: "two_gates"\n      gate_token_independent: true\n'),
    "gateungated": ('      gate_mode: "two_gates"\n', '      gate_mode: "two_gates"\n      gate_residual: "ungated"\n'),
}


# capped: the K=4 Dual configs of the main table with the gate soft cap (gate_sum_cap: 1.0),
# all three alphas at both budgets. The seed-43 a50 runs already exist (softcap/), so they
# are not regenerated.
CAPPED_BASES = [
    "8e7_loops4__dm768_L16_loop4_F80M_ffnD64_ffnW17920_dual_a25_nocross.yaml",
    "8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross.yaml",
    "8e7_loops4__dm768_L16_loop4_F80M_ffnD3264_ffnW4864_dual_a75_nocross.yaml",
    "16e7_loops4__dm768_L16_loop4_F160M_ffnD1600_ffnW37440_dual_a25_nocross.yaml",
    "16e7_loops4__dm768_L16_loop4_F160M_ffnD4864_ffnW24448_dual_a50_nocross.yaml",
    "16e7_loops4__dm768_L16_loop4_F160M_ffnD8128_ffnW11392_dual_a75_nocross.yaml",
]
CAPPED_EXISTING = {("8e7_loops4__dm768_L16_loop4_F80M_ffnD1600_ffnW11392_dual_a50_nocross.yaml", 43),
                   ("16e7_loops4__dm768_L16_loop4_F160M_ffnD4864_ffnW24448_dual_a50_nocross.yaml", 43)}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}, found {text.count(old)}")
    return text.replace(old, new)


def common(text: str, seed: int, wandb_suffix: str) -> str:
    text = replace_once(text, "            seed: 43\n", f"            seed: {seed}\n")
    text = replace_once(text, "        k: -1\n", "        k: 1  # keep only the newest checkpoint\n")
    text = replace_once(
        text,
        "    eval_interval: ${settings.intervals.evaluation_interval_in_steps}\n",
        "    eval_interval: 999999  # final checkpoint is evaluated by queue_eval_after_training.sh\n",
    )
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith("    experiment_id: dm768_")]
    if len(hits) != 1:
        raise ValueError("expected one wandb experiment_id line")
    lines[hits[0]] = lines[hits[0]].rstrip("\n") + f"{wandb_suffix}\n"
    return "".join(lines)


def write(stem: str, text: str) -> list[str]:
    out = OUT / f"{stem}.yaml"
    out.write_text(text)
    (OUT / f"{stem}_WARMSTART.yaml").write_text(to_warmstart(text))
    return [str(out.relative_to(JANUS.parents[1]))]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lists: dict[str, list[str]] = {"attnmatched": [], "gate": [], "capped": []}
    for base, key, old, new, stem in ATTNMATCHED:
        src = (SWEEP / base).read_text()
        src = replace_once(src, f"{key}: {old}\n", f"{key}: {new}  # attention-score-matched, was {old}\n")
        # The W&B run name encodes the width too.
        tag = "ffnW" if key == "wide_ffn_hidden" else "ffnL"
        src = replace_once(src, f"_{tag}{old}_", f"_{tag}{new}_")
        for seed in SEEDS:
            lists["attnmatched"] += write(f"{stem}_s{seed}", common(src, seed, f"_attnmatched_s{seed}"))
    base_text = (SWEEP / GATE_BASE).read_text()
    for name, (old, new) in GATE_CONTROLS.items():
        src = replace_once(base_text, old, new)
        for seed in SEEDS:
            stem = f"{GATE_BASE.removesuffix('.yaml')}_{name}_s{seed}"
            lists["gate"] += write(stem, common(src, seed, f"_{name}_s{seed}"))
    for base in CAPPED_BASES:
        src = replace_once((SWEEP / base).read_text(), "      use_cross: false\n",
                           "      use_cross: false\n      gate_sum_cap: 1.0\n")
        for seed in SEEDS:
            if (base, seed) in CAPPED_EXISTING:
                continue
            stem = f"{base.removesuffix('.yaml')}_softcap1_s{seed}"
            lists["capped"] += write(stem, common(src, seed, f"_softcap1_s{seed}"))
    for group, paths in lists.items():
        (OUT / f"{group}.txt").write_text("\n".join(paths) + "\n")
        print(f"{group}: {len(paths)} configs -> review/{group}.txt")


if __name__ == "__main__":
    main()
