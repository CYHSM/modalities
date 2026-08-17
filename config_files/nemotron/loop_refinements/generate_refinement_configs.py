#!/usr/bin/env python
"""Generates the loop-refinement wave: A1's Mamba loop at K in {3, 6, 12}, with and without the
per-group refinements introduced in modalities.models.nemotron.nemotron_loop.

Why this wave exists
--------------------
Wave 1 ablated the two loop refinements that existed then (per-iteration norm, input injection) at
K=2..3 and found no measurable effect (docs/loopotron/loopotron.tex, Finding 4). The looped-
transformer literature ablates its refinements at loop counts of 8 and above, where the failure mode
they address -- unbounded growth of the residual stream across iterations, reported as residual
explosion and late-training loss spikes -- actually appears. At K=3 there is very little for a
stabilizer to stabilize, which is the most likely reason Wave 1 measured nothing.

So loop count is the independent variable here, not a fixed backdrop: the control and the full
refinement stack are both measured at K=3 (where Wave 2 says the plain loop is already fine) and at
K=6. The prediction being tested is an *interaction*: plain looping should degrade as K rises while
the stabilized recurrence holds up. A refinement that helps uniformly at every K, or nowhere, means
something different from a refinement whose benefit grows with depth. K=6 is a shallower probe of
that interaction than the literature's K=8+, so a null result here is weaker evidence against the
refinements than a positive result is for them.

The arms
--------
At every K, `simple` is the control -- the loop exactly as Wave 2 ran it. `parcae` adds the
stabilized recurrence (arXiv:2604.12946); `parcae_norm` adds the injection normalization the same
paper ablates separately; `film` isolates iteration conditioning, which is an orthogonal mechanism
(telling the shared weights which iteration they are on) rather than a stabilizer; `all` combines
them.

The full five-arm separation runs at K=6 and K=12. K=3 carries only the control and the full stack,
which is enough to say whether the gap opens with depth without paying for three more arms at a
depth where both Wave 1 and this wave's own K=3 pair found the refinements do nothing (a measured
0.0001 nats).

K=12 was added after the K=3/K=6 results came in: at K=6 the refinements are worth up to 0.0079 nats
where at K=3 they are worth nothing, so the interaction is real and the natural next question is
whether it keeps growing. It also probes the regime the stabilized recurrence was actually designed
for -- at K=6 the plain loop showed no instability at all (gradient norms 0.36-0.38, no loss
spikes), so the stabilizer had nothing to fix, and arXiv:2604.12946 reports its failure mode at
K=8 and above.

Budget
------
Wave 2's budget exactly: 76,250 steps = 4,997,120,000 tokens, at the same global batch of 65,536
tokens/step. This is deliberate and it is the point of the wave. `R_k3_simple` is then the *same
model on the same budget* as Wave 2's A1, so it doubles as a replication check, and every refined
arm can be read against A1's measured 2.4951 +/- 0.0013 rather than only against its own control.
An earlier revision of this generator cut the budget to 1.5B to make K=12 fit a 20h slot; that
saves wall clock at the cost of the only external reference point the wave has, which is a bad
trade.

Wall clock is set by loop count, from measured throughputs on 4 GPUs: K=3 6,154 steps/h -> 12.4h;
K=6 4,230 -> 18.0h; K=12 2,632 -> 29.0h. K=3 and K=6 fit a single 24h slot on ONE node.

K=12 does not, and cannot be made to on the `normal` QOS at any node count: the 24h ceiling belongs
to partition boost_usr_prod (MaxTime=1-00:00:00), not to the QOS. The single-node options are
therefore boost_qos_lprod (4-day wall, but an 8-node/32-GPU cap shared across the whole project
account) or scaling out to more nodes -- which changes world size, hence the resumable sampler's
order, hence the data stream, so multi-node K=12 arms would stay comparable to each other but not
to K=3 and K=6. See run_multinode_probe.sh, which measures whether scaling out is even fast enough
to matter.

Fitting one slot matters beyond convenience: a requeue resumes through `modalities warmstart`,
which is the path with the open checkpoint defect (loopotron.tex section 3.6). K=6 at 18.0h has
about 6h of headroom against the 24h limit; K=12 has none at all on `normal`, which is the whole
difficulty.

Run from the repository root::

    python config_files/nemotron/loop_refinements/generate_refinement_configs.py
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from modalities.models.nemotron.layer_pattern import get_num_built_layers, get_num_layers  # noqa: E402

BASE_CONFIG_PATH = REPOSITORY_ROOT / "config_files/nemotron/config_research_nemotron_loops_5b_cluster.yaml"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_refinements"
WANDB_PROJECT = "modalities_nemotron_loop_refinements"

# See the "Budget" note in the module docstring. Identical to Wave 2, so that R_k3_simple is a
# straight replication of A1. num_target_tokens must stay exactly num_target_steps * 65,536, which
# is the global batch every arm in every wave uses.
NUM_TARGET_STEPS = 76_250
TOKENS_PER_STEP = 65_536
NUM_TARGET_TOKENS = NUM_TARGET_STEPS * TOKENS_PER_STEP
WARMUP_STEPS = 1_500

# Nodes per loop count, and with it dp_degree (4 GPUs per node) and the per-GPU micro batch that
# keeps the global batch at 65,536 tokens/step. One node preserves Wave 2's exact data order, so
# every arm in this wave fits one, so every arm runs on one node and the whole wave shares Wave 2's
# data stream. Keep it that way: a different node count is a different world size, hence a different
# order out of the resumable sampler, which would confound the refinement comparison with data
# order -- the very comparison this wave exists to make.
# K=12 runs on TWO nodes. Measured 2026-08-15 (run_multinode_probe.sh), 1/2/4 nodes in one
# allocation, global batch held at 65,536 throughout:
#     1 node   24.6 samples/s   1.00x   100% eff   27.6h  -- exceeds the 24h ceiling
#     2 nodes  32.9 samples/s   1.34x    67% eff   20.6h  -- chosen
#     4 nodes  33.9 samples/s   1.38x    34% eff   20.0h  -- saturated, 3% over 2 nodes
# Two nodes is the only sensible point: one cannot run on `normal` at all (the 24h limit is
# partition boost_usr_prod's MaxTime, not the QOS's), and four buys 3% for twice the hardware
# and a further doubling of world size. The Wave 2 base config's warning that scaling out is a
# net loss holds in spirit -- efficiency is 67% at best -- but at K=12 the absolute throughput
# does improve, because gradient all-reduce volume is set by parameter count (identical at
# every K) while compute per step scales with executed depth. The cost is that world size 8 gives a different sampler order
# than the world size 4 used by K=3, K=6 and Wave 2: the five K=12 arms share a data stream
# with each other, so the refinement comparison AT K=12 is clean, but the K=6 -> K=12 depth
# step carries data-order variation on top of seed noise and must be reported as such.
NODES_PER_LOOP_COUNT = {3: 1, 6: 1, 12: 2}
GPUS_PER_NODE = 4
# micro_batch * dp_degree * sequence_length == 65,536, with sequence_length 2048.
GLOBAL_BATCH_IN_SAMPLES = 32

# A1's pattern with the loop count left open. A1 loops every Mamba layer, which Wave 2 found to be
# both the best loop of the six and the one with the most headroom against its fresh-weight anchor,
# so it is the arm where a better loop has the most to prove.
A1_PATTERN_TEMPLATE = "[M]^{k}E[M]^{k}*E[M]^{k}E[M]^{k}E[M]^{k}*E"


class Refinement(NamedTuple):
    """
    One refinement setting, applied identically at every loop count it is generated for.

    Attributes:
        suffix (str): Appended to the arm name.
        description (str): One line for the config banner.
        per_iteration_norm (bool): ``loop_config.per_iteration_norm``.
        input_injection (bool): ``loop_config.input_injection``.
        iteration_embedding (str): ``loop_config.iteration_embedding``.
        loop_counts (tuple[int, ...]): The loop counts this refinement is generated for.
    """

    suffix: str
    description: str
    per_iteration_norm: bool = False
    input_injection: bool = False
    iteration_embedding: str = "none"
    loop_counts: tuple[int, ...] = (3, 6, 12)


REFINEMENTS: list[Refinement] = [
    Refinement(
        "simple",
        "Control: the loop exactly as Wave 2 ran it. Every other arm at this K differs from it only "
        "in the refinement named.",
    ),
    Refinement(
        "film",
        "Iteration conditioning alone (FiLM scale and shift per iteration). Not a stabilizer: it "
        "tells the shared weights which iteration they are executing. Isolates that mechanism.",
        iteration_embedding="film",
        loop_counts=(6, 12),
    ),
]

# The wave ran three further arms -- `parcae` (stabilized recurrence), `parcae_norm` (recurrence
# plus injection norm) and `all` (both plus FiLM) -- whose results are Table 6 of loopotron.tex and
# whose configs are frozen under removed_refinements/. Both mechanisms have since been deleted from
# the model, so this generator can no longer emit those arms and LoopConfig rejects their keys. They
# are not listed here as disabled entries because a disabled entry invites someone to switch it back
# on; the finding was that the recurrence bought nothing (-0.0003 at K=12, 0.2 within-arm s.d.,
# against a real instability) and the injection norm actively hurt (+0.0475 at K=12, and it dragged
# every combination it appeared in below FiLM alone).


def _read_scalar(base_config_text: str, field: str) -> int:
    """
    Reads an integer field from the base config.

    Args:
        base_config_text (str): Contents of the base config.
        field (str): The field name, matched at any indentation.

    Raises:
        RuntimeError: If the field does not appear exactly once.

    Returns:
        int: The field's value.
    """
    matches = re.findall(rf"^\s*{field}: (\d+)$", base_config_text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{field}' line in {BASE_CONFIG_PATH.name}, found {len(matches)}.")
    return int(matches[0])


def _render_loop_config_block(refinement: Refinement, n_embd: int) -> str:
    """
    Renders the ``loop_config`` mapping for one refinement.

    Args:
        refinement (Refinement): The refinement being rendered.
        n_embd (int): Model dimension. Unused since the injection norm -- the only refinement that
            needed its own nested config -- was removed; kept so the call sites stay uniform.

    Returns:
        str: The YAML block, indented to sit under ``model_raw.config``.
    """
    del n_embd
    return "\n".join(
        [
            "    loop_config:",
            f"      per_iteration_norm: {str(refinement.per_iteration_norm).lower()}",
            f"      input_injection: {str(refinement.input_injection).lower()}",
            f"      iteration_embedding: {refinement.iteration_embedding}",
        ]
    )


def _substitute(text: str, description: str, pattern: str, replacement: str, expected: int = 1) -> str:
    """
    Replaces a value in the config text, failing loudly if the config changed shape.

    Args:
        text (str): The config text.
        description (str): What is being substituted, for the error message.
        pattern (str): The regex to replace, matched multiline.
        replacement (str): The replacement text.
        expected (int): How many substitutions must occur.

    Raises:
        RuntimeError: If the substitution count differs from ``expected``.

    Returns:
        str: The updated text.
    """
    text, count = re.subn(pattern, replacement.replace("\\", "\\\\"), text, flags=re.MULTILINE)
    if count != expected:
        raise RuntimeError(
            f"Expected {expected} {description} line(s) in {BASE_CONFIG_PATH.name}, found {count}. "
            f"The base config changed shape; update this generator."
        )
    return text


def _render_arm(base_config_text: str, name: str, refinement: Refinement, loop_count: int, n_embd: int) -> str:
    """
    Produces one arm's config text.

    Args:
        base_config_text (str): Contents of the base config.
        name (str): The arm name, used as the pinned experiment id.
        refinement (Refinement): The refinement setting.
        loop_count (int): K, the number of iterations each Mamba loop group performs.
        n_embd (int): Model dimension read from the base config.

    Returns:
        str: The arm's config text.
    """
    layer_pattern = A1_PATTERN_TEMPLATE.format(k=loop_count)
    n_layer = get_num_built_layers(layer_pattern)

    text = base_config_text
    text = _substitute(text, "layer_pattern", r'^    layer_pattern: ".*"$', f'    layer_pattern: "{layer_pattern}"')
    text = _substitute(text, "n_layer", r"^    n_layer: \d+$", f"    n_layer: {n_layer}")
    text = _substitute(text, "wandb project", r"^    project: .*$", f"    project: {WANDB_PROJECT}")
    text = _substitute(
        text, "num_target_tokens", r"^    num_target_tokens: \d+$", f"    num_target_tokens: {NUM_TARGET_TOKENS}"
    )
    text = _substitute(
        text, "num_target_steps", r"^    num_target_steps: \d+$", f"    num_target_steps: {NUM_TARGET_STEPS}"
    )
    text = _substitute(text, "warmup_steps", r"^    warmup_steps: \d+$", f"    warmup_steps: {WARMUP_STEPS}")
    # Node count -> dp_degree and per-GPU micro batch. The product is held constant so that the
    # global batch is 65,536 tokens/step at every node count; only its split across GPUs changes.
    # Getting these two out of step would silently change the batch size and make the arm
    # incomparable to everything else while still training perfectly well.
    nodes = NODES_PER_LOOP_COUNT[loop_count]
    dp_degree = nodes * GPUS_PER_NODE
    micro_batch, remainder = divmod(GLOBAL_BATCH_IN_SAMPLES, dp_degree)
    if remainder or micro_batch < 1:
        raise RuntimeError(
            f"{nodes} node(s) gives dp_degree {dp_degree}, which does not divide the "
            f"{GLOBAL_BATCH_IN_SAMPLES}-sample global batch into a whole per-GPU micro batch."
        )
    text = _substitute(
        text,
        "data_parallel_replicate_degree",
        r"^    data_parallel_replicate_degree: \d+$",
        f"    data_parallel_replicate_degree: {dp_degree}",
    )
    text = _substitute(
        text,
        "local_train_micro_batch_size",
        r"^    local_train_micro_batch_size: \d+$",
        f"    local_train_micro_batch_size: {micro_batch}",
    )
    # The per-group refinement parameters are per-channel scales and time constants, so they belong
    # with the norms and the SSM parameters rather than with the weight matrices. Adding "loop" is
    # safe even for the `simple` arms: the group is declared unconditionally and is simply empty
    # when no refinement is on, and an empty group contributes nothing to either optimizer group.
    text = _substitute(
        text,
        "weight_decay_groups_excluded",
        r"^        weight_decay_groups_excluded: \[.*\]$",
        "        weight_decay_groups_excluded: [embedding, layernorm, ssm, router, loop]",
    )
    # Replace the whole loop_config mapping: the base config's block predates the per-group
    # refinements and has neither the keys nor the nested injection norm.
    text = _substitute(
        text,
        "loop_config block",
        r"^    loop_config:\n(?:(?:      |    #).*\n)*?      input_injection: (?:true|false)$",
        _render_loop_config_block(refinement, n_embd),
    )
    text = _substitute(
        text,
        "experiment_id",
        r"^  experiment_id: \$\{modalities_env:experiment_id\}$",
        f"  experiment_id: {name}",
    )

    executed = get_num_layers(layer_pattern)
    banner = "\n".join(
        [
            "# " + "=" * 96,
            f"# LOOP-REFINEMENT ARM: {name}",
            f"# {refinement.description}",
            "#",
            f"#   loop count K  : {loop_count}",
            f"#   layer_pattern : {layer_pattern}",
            f"#   built layers  : {n_layer}   (sets of weights, identical across every K)",
            f"#   executed      : {executed}   layer applications per token",
            f"#   budget        : {NUM_TARGET_STEPS:,} steps = {NUM_TARGET_TOKENS:,} tokens"
            "   (Wave 2's budget exactly)",
            f"#   layout        : {nodes} node(s), dp_degree {dp_degree}, micro batch {micro_batch}"
            f"   (global batch {GLOBAL_BATCH_IN_SAMPLES * 2048:,} tokens/step, same at every K)",
            "#",
            "# Compare only against the arm of the SAME K: absolute nats move with K because the",
            "# executed depth does, so a cross-K comparison measures compute, not the refinement.",
            "# The K=3 arms additionally use Wave 2's node count, so R_k3_simple sees the same data",
            "# in the same order as A1_loop_mamba and is a straight replication of it.",
            "#",
            "# GENERATED FILE -- do not edit. Edit the base config and re-run",
            "# config_files/nemotron/loop_refinements/generate_refinement_configs.py.",
            "# " + "=" * 96,
            "",
        ]
    )
    return banner + text


def main() -> None:
    """Writes one config file per (refinement, loop count) pair, plus the launcher's arm list."""
    base_config_text = BASE_CONFIG_PATH.read_text()
    n_embd = _read_scalar(base_config_text, "n_embd")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    arm_names = []
    for refinement in REFINEMENTS:
        for loop_count in refinement.loop_counts:
            name = f"R_k{loop_count}_{refinement.suffix}"
            output_path = OUTPUT_DIRECTORY / f"config_{name}.yaml"
            output_path.write_text(_render_arm(base_config_text, name, refinement, loop_count, n_embd))
            arm_names.append(name)
            layer_pattern = A1_PATTERN_TEMPLATE.format(k=loop_count)
            print(
                f"{name:22s} K={loop_count:<3d} executed={get_num_layers(layer_pattern):3d} "
                f"iteration_embedding={refinement.iteration_embedding:5s} "
                f"-> {output_path.relative_to(REPOSITORY_ROOT)}"
            )

    arm_list_path = OUTPUT_DIRECTORY / "arm_list_refinements.txt"
    arm_list_path.write_text("\n".join(arm_names) + "\n")
    print(f"\n{len(arm_names)} arms -> {arm_list_path.relative_to(REPOSITORY_ROOT)}")

    # One list per node count. `#SBATCH --nodes` cannot vary across the tasks of a single array, so
    # the launcher is submitted once per list with a matching --nodes override; splitting here keeps
    # the mapping in the same file that decided it.
    for nodes in sorted(set(NODES_PER_LOOP_COUNT.values())):
        loop_counts = {k for k, v in NODES_PER_LOOP_COUNT.items() if v == nodes}
        names = [n for n in arm_names if int(n.split("_")[1][1:]) in loop_counts]
        path = OUTPUT_DIRECTORY / f"arm_list_{nodes}node.txt"
        path.write_text("\n".join(names) + "\n")
        print(f"  {nodes} node(s): {len(names)} arms -> {path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
