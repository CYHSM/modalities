#!/usr/bin/env python
"""Generates the loop-POSITION sweep: one layer looped, at each place that operator can sit.

WHAT THIS WAVE IS FOR
---------------------
Wave 2 ranked the loop arms A1 (Mamba) < A4 < A2 (MoE) < A5 < A6 < A3 (attention) and read that as
"depth is worth most on the recurrent operator". That reading is confounded. The base pattern is
``MEM*EMEMEM*E``, so Mamba first occurs at built index 0, MoE at 1 and attention at 3 -- **the
operator you loop determines where the loop can start**, and the Wave 2 loss ranking is almost
perfectly monotonic in the executed index of each arm's first loop group (Spearman +0.971, against
-0.086 for the update-diversity hypothesis it was originally attributed to). Operator class and
stack position are inseparable in Wave 2 at any sample size; see
docs/loopotron/representation_diagnostics.md section 2.

Neither diagnostic run on the trained checkpoints settles it. The per-group inference-time ablation
points the other way (late groups look more important) but carries the opposite confound -- ablating
late leaves less downstream depth to repair the damage. **A trained position sweep is the only clean
test**, and this is it.

THE DESIGN
----------
Within a family: hold the operator fixed, hold the loop count fixed (K=6), hold the executed depth
fixed, and vary only *where* the looped layer sits. One family per operator, named by the built
indices that operator occupies in ``MEM*EMEMEM*E``::

    P (mamba,     5 arms)  built indices 0, 2, 5, 7, 9   e.g. P1_loop_mamba_at_2      ME[M]^6*EMEMEM*E
    Q (moe,       5 arms)  built indices 1, 4, 6, 8, 11  e.g. Q0_loop_moe_at_1        M[E]^6M*EMEMEM*E
    R (attention, 2 arms)  built indices 3, 10           e.g. R0_loop_attention_at_3  MEM[*]^6EMEMEM*E

Every arm in every family is 12 built / 17 executed layers, so within a family active parameters and
modelled FLOPs per token are identical **by construction**, not by matching. A ``main()`` assertion
re-derives built, executed and per-type counts from each pattern and fails the generation if any arm
in a family differs, because an arm that quietly differs in executed depth would reintroduce exactly
the confound this wave exists to remove.

Across families the per-type execution counts differ by design (P adds 5 Mamba executions, Q adds 5
MoE, R adds 5 attention), and because an extra MoE execution costs ~2x an extra Mamba one and
attention ~0.59x, the three families do NOT cost the same. That is deliberate: the comparison across
families is of curve SHAPE -- where the optimum sits, whether loss degrades with depth -- not of
magnitude. Holding K rather than FLOPs fixed keeps "same loop, different operator, different place"
as the contrast and maximises signal in the two weaker operators.

Reading the result. P (run 2026-08-17, n=1) gave a 0.0498-nat spread, 23.7x the 0.0021 seed s.d. and
larger than Wave 2's entire 0.0361-nat operator spread, with the optimum at built index 2 and index 9
actually WORSE than not looping at all. So position is established as a large causal variable for
Mamba. What Q and R answer is whether that curve is a property of the operator or of the stack: if
all three families peak in the same region, position is a stack property and Wave 2's operator
ranking is largely a positional artifact; if the optima differ by operator, the two interact and
neither Wave 2's ranking nor a single position curve tells the whole story.

R is a single contrast rather than a curve -- the base pattern holds only two attention layers. That
is a property of the stack, not a choice; widening it would need a different base pattern, which
would break comparability with A0 and every Wave 2 arm.

WHY K=6
-------
Wave 3 measured the depth optimum for this architecture's Mamba loop at K=6 (2.4815, against 2.4916
at K=3 and 2.5005 at K=12), with gradient norms still flat there -- maximum 5.1 and zero logged
steps above 100, where K=12 peaks at 85,782. K=6 therefore buys the largest per-arm loop effect that
is still in the well-behaved regime, which matters because the informative outcome of this wave may
well be a null: a null at K=3 (+2 executions) would be indistinguishable from "too little loop to
see anything", while a null at K=6 (+5 executions) is a real answer.

WHAT IS DELIBERATELY NOT VARIED
-------------------------------
The loop is plain -- no FiLM iteration conditioning, no per-iteration norms, no input injection --
because Wave 2's A0 and A1, the two arms this sweep is read against, are plain. FiLM is worth
-0.0079 nats at K=6 and would be constant across all five arms, so it would not bias the comparison,
but it would break comparability with the arms already on disk.

Run from the repository root::

    python config_files/nemotron/loop_ablation_position_sweep/generate_arm_configs.py
    python config_files/nemotron/loop_ablation_position_sweep/generate_warmstart_configs.py

``--operators`` restricts which families are (re)generated; the default is all three. Regenerating a
family whose runs already exist is safe -- the output is deterministic -- but relaunching those arms
is not, since a pinned experiment id would resume from the finished run's checkpoint.
"""

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "config_files/nemotron/loop_ablation"))

from generate_arm_configs import Arm, _read_n_embd, _render_arm  # noqa: E402
import generate_arm_configs as _original_module  # noqa: E402

from modalities.models.nemotron.layer_pattern import (  # noqa: E402
    LayerSymbol,
    count_layers_by_type,
    get_num_built_layers,
    get_num_layers,
)

# Patch the module-level paths the imported helpers close over, rather than duplicating them. Same
# arrangement as loop_ablation_5b_cluster/generate_arm_configs.py.
_original_module.BASE_CONFIG_PATH = (
    REPOSITORY_ROOT / "config_files/nemotron/config_research_nemotron_loops_5b_cluster.yaml"
)
_original_module.OUTPUT_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation_position_sweep"
# Same wandb project as Wave 2 on purpose: these arms are only interpretable next to A0 and A1, and
# a separate project would put the reference curves one project away from the thing they explain.
# Experiment ids are distinct, so nothing collides.
_original_module.WANDB_PROJECT = "modalities_nemotron_loops_5b_cluster"
BASE_CONFIG_PATH = _original_module.BASE_CONFIG_PATH
OUTPUT_DIRECTORY = _original_module.OUTPUT_DIRECTORY

BASE_PATTERN = "MEM*EMEMEM*E"
LOOP_COUNT = 6

# One family per operator. The prefix is part of every experiment id, so it is FROZEN: "P" arms are
# already trained and collated (docs/loopotron/position_sweep_stats.json), and renaming them would
# orphan those runs. Attention gets only two families members because the base pattern contains only
# two attention layers -- that sweep is a single contrast, not a curve, which is a property of the
# stack rather than a choice made here.
FAMILIES: dict[str, tuple[str, str]] = {
    "M": ("P", "mamba"),
    "E": ("Q", "moe"),
    "*": ("R", "attention"),
}


def _build_arms(symbol: str) -> list[Arm]:
    """
    Derives one arm per position ``symbol`` occupies in the base pattern.

    Deriving the arms rather than typing them means the sweep cannot silently disagree with the base
    pattern: if the base config's ``layer_pattern`` ever changes, this either follows it or fails the
    equality assertion in :func:`main`.

    Args:
        symbol (str): The layer symbol to loop, a key of :data:`FAMILIES`.

    Returns:
        list[Arm]: One arm per occurrence of ``symbol`` in :data:`BASE_PATTERN`.
    """
    prefix, operator_name = FAMILIES[symbol]
    built_indices = [index for index, character in enumerate(BASE_PATTERN) if character == symbol]
    arms = []
    for sweep_index, built_index in enumerate(built_indices):
        pattern = BASE_PATTERN[:built_index] + f"[{symbol}]^{LOOP_COUNT}" + BASE_PATTERN[built_index + 1 :]
        arms.append(
            Arm(
                f"{prefix}{sweep_index}_loop_{operator_name}_at_{built_index}",
                pattern,
                f"Position sweep {sweep_index} of {len(built_indices)}: the single {operator_name} "
                f"layer at built index {built_index} runs {LOOP_COUNT}x. Identical to every other "
                f"{prefix} arm in "
                f"built layers, executed layers, per-type executions, active parameters and FLOPs -- "
                f"the ONLY difference across this wave is where the looped layer sits. Deconfounds "
                f"loop position from looped operator, which Wave 2 cannot separate.",
            )
        )
    return arms




def main() -> None:
    """Writes one config per position per requested operator, after checking the arms are matched."""
    parser = argparse.ArgumentParser(description="Generate loop-position sweep configs.")
    parser.add_argument(
        "--operators",
        nargs="+",
        default=list(FAMILIES),
        choices=list(FAMILIES),
        help="Layer symbols to sweep (default: all three families).",
    )
    arguments = parser.parse_args()

    base_config_text = BASE_CONFIG_PATH.read_text()
    n_embd = _read_n_embd(base_config_text)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for symbol in arguments.operators:
        prefix, operator_name = FAMILIES[symbol]
        arms = _build_arms(symbol)

        # The whole value of a family is that position is the only thing varying WITHIN it. Assert
        # that from the patterns rather than trusting the construction above. Families are not
        # compared to each other on this signature: they share 12 built / 17 executed layers, but
        # differ in which operator's executions were added, which is the point.
        signatures = {
            arm.name: (
                get_num_built_layers(arm.layer_pattern),
                get_num_layers(arm.layer_pattern),
                tuple(
                    sorted(
                        (layer_symbol.name, count)
                        for layer_symbol, count in count_layers_by_type(arm.layer_pattern).items()
                    )
                ),
            )
            for arm in arms
        }
        if len(set(signatures.values())) != 1:
            raise RuntimeError(
                f"{operator_name} position-sweep arms are not matched -- built/executed/per-type "
                f"counts differ across arms, which would reintroduce the confound this wave "
                f"removes: {signatures}"
            )

        print(f"--- {operator_name} family ({prefix}*), {len(arms)} position(s) ---")
        for arm in arms:
            output_path = OUTPUT_DIRECTORY / f"config_{arm.name}.yaml"
            text = _render_arm(base_config_text, arm, n_embd)

            # The banner _render_arm writes points at the 557M wave's generator; this wave has its own.
            text, n = re.subn(
                r"^# config_files/nemotron/loop_ablation/generate_arm_configs\.py\.$",
                "# config_files/nemotron/loop_ablation_position_sweep/generate_arm_configs.py.",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n != 1:
                raise RuntimeError(f"expected exactly one generator pointer in the banner for {arm.name}, found {n}")

            # Pinned, not ${modalities_env:experiment_id}: that resolver hashes the config path plus
            # the current timestamp, so it differs on every launch, and `modalities warmstart` exposes
            # no --experiment_id flag to override it. Pinning here is the only way a requeued arm
            # finds its own checkpoint directory again. Same reasoning as loop_ablation_5b_cluster.
            text, n = re.subn(
                r"^  experiment_id: \$\{modalities_env:experiment_id\}$",
                f"  experiment_id: {arm.name}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n != 1:
                raise RuntimeError(f"expected exactly one experiment_id line for arm {arm.name}, found {n}")

            output_path.write_text(text)
            counts = count_layers_by_type(arm.layer_pattern)
            print(
                f"{arm.name:28s} {arm.layer_pattern:26s} "
                f"built={get_num_built_layers(arm.layer_pattern):3d} "
                f"executed={get_num_layers(arm.layer_pattern):3d} "
                f"M/E/*={counts[LayerSymbol.MAMBA]}/{counts[LayerSymbol.MOE]}/{counts[LayerSymbol.ATTENTION]} "
                f"-> {output_path.relative_to(REPOSITORY_ROOT)}"
            )

        # The Mamba family's list keeps its original name: run_position_sweep.sh already points at it
        # and those five arms are trained, collated and must not be relaunched.
        list_name = "arm_list.txt" if symbol == "M" else f"arm_list_{operator_name}.txt"
        (OUTPUT_DIRECTORY / list_name).write_text("".join(f"{arm.name}\n" for arm in arms))
        print(f"{list_name} written with {len(arms)} arms (one per Slurm array task).\n")


if __name__ == "__main__":
    main()
