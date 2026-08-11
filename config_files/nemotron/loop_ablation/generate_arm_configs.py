#!/usr/bin/env python
"""Generates the Nemotron layer-loop ablation arm configs from the shared base config.

Every arm differs from the base in exactly two values, ``layer_pattern`` and ``n_layer``. Keeping
them generated rather than hand-maintained means a change to a shared setting (batch size, learning
rate, data path) is made once in the base config and propagates to all arms, so the arms cannot
silently drift apart mid-sweep -- which would invalidate the comparison.

Run from the repository root::

    python config_files/nemotron/loop_ablation/generate_arm_configs.py

The substitution is textual on purpose: the base config uses YAML anchors (``&nemotron_norm_config``
/ ``*nemotron_norm_config``), and a load/dump round trip through a YAML library would expand them
and drop every comment.
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from modalities.models.nemotron.layer_pattern import (  # noqa: E402
    LayerSymbol,
    count_layers_by_type,
    get_num_built_layers,
    get_num_layers,
    parse_layer_schedule,
)

BASE_CONFIG_PATH = REPOSITORY_ROOT / "config_files/nemotron/config_research_nemotron_loops_1gpu.yaml"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation"


class Arm(NamedTuple):
    """
    One ablation arm.

    Attributes:
        name (str): Arm name; becomes the config file name and the wandb run name prefix.
        layer_pattern (str): The arm's layer pattern.
        description (str): One line on what the arm tests, rendered into the config banner.
        per_iteration_norm (bool): Whether each loop iteration gets its own pre-normalization.
        input_injection (bool): Whether the group's input is re-injected before every iteration
            after the first.
    """

    name: str
    layer_pattern: str
    description: str
    per_iteration_norm: bool = False
    input_injection: bool = False


# The five loop arms are matched to ~1.24x the baseline's FLOPs per token, and each anchor is
# matched to its loop arm *exactly* -- same executed count of every layer type, same active
# parameters, same FLOPs -- differing only in whether the weights are shared or fresh. The loop
# is spread over every layer of the type rather than concentrated on one, so "which layer was
# looped" is not a confound.
ARMS: list[Arm] = [
    Arm("A0_baseline", "MEM*EMEMEM*E", "Reference. 12 layers, no loops."),
    Arm(
        "A1_loop_mamba",
        "[M]^3E[M]^3*E[M]^3E[M]^3E[M]^3*E",
        "Every Mamba layer runs 3x. Tests depth in the sequence mixer, which also carries all positional information.",
    ),
    Arm(
        "A2_loop_moe",
        "M[E]^2M*[E]^2M[E]^2M[E]^2M*[E]^2",
        "Every MoE layer runs 2x. The router re-routes on each visit, so iteration 2 is a "
        "different function, not a repeat of iteration 1.",
    ),
    Arm(
        "A3_loop_attention",
        "MEM[*]^4EMEMEM[*]^4E",
        "Every attention layer runs 4x. Cheapest in parameters, quadratic in sequence length.",
    ),
    Arm(
        "A4_loop_mamba_moe",
        "[ME]^2M*E[ME]^2[ME]^2M*E",
        "Three Mamba+MoE pairs run 2x. Universal-Transformer-shaped mix-then-transform block.",
    ),
    Arm(
        "A5_loop_mamba_attention",
        "ME[M*]^3EMEME[M*]^3E",
        "Both Mamba+attention pairs run 3x. Local mixing followed by global attention, repeated.",
    ),
    Arm(
        "A6_loop_attention_moe",
        "MEM[*E]^2MEMEM[*E]^2",
        "Both attention+MoE pairs run 2x. This is the classical transformer block (attention then "
        "feed-forward), so it is the closest thing in this hybrid to a Universal Transformer loop, "
        "and it uses the loop count reported as the best marginal return. At 1.179x baseline FLOPs "
        "it gets ~5% less compute than the other loop arms, which is the conservative direction.",
    ),
    Arm(
        "N4_anchor_attention_moe",
        "MEM*E*EMEMEM*E*E",
        "Iso-FLOP anchor for A6: the same 7 MoE and 4 attention executions with FRESH weights "
        "(16 built layers, 1.487B parameters). A6 and A4 had no anchors, which left the "
        "iso-parameter comparison (arm vs A0) as the only one available for them.",
    ),
    Arm(
        "N1_anchor_mamba",
        "MMMEMMM*EMMMEMMMEMMM*E",
        "Iso-FLOP anchor for A1: the same 15 Mamba executions with FRESH weights (22 built layers).",
    ),
    Arm(
        "N2_anchor_moe",
        "MEEM*EEMEEMEEM*EE",
        "Iso-FLOP anchor for A2: the same 10 MoE executions with FRESH weights (17 built layers).",
    ),
    Arm(
        "N3_anchor_attention",
        "MEM****EMEMEM****E",
        "Iso-FLOP anchor for A3: the same 8 attention executions with FRESH weights (18 built layers).",
    ),
    # The 2x2 on A6. A6 itself is the cell with both refinements off, so only three new configs
    # are needed. A6 is the Universal-Transformer-shaped arm -- a looped attention+feed-forward
    # block -- which is the shape per-iteration norms and input injection were reported for, and
    # it is the only loop arm with an exact iso-FLOP anchor (N4) to read the result against.
    Arm(
        "A6a_loop_attention_moe_per_iteration_norm",
        "MEM[*E]^2MEMEM[*E]^2",
        "A6 with a separate pre-norm per loop iteration. Isolates per-iteration conditioning from "
        "weight sharing: reported Universal Transformer gains mix the two, and A6 is the cell of "
        "the 2x2 that has neither.",
        per_iteration_norm=True,
    ),
    Arm(
        "A6b_loop_attention_moe_input_injection",
        "MEM[*E]^2MEMEM[*E]^2",
        "A6 with the loop group's input added back before iteration 2. Exactly iso-parameter with "
        "A6, so any difference is attributable to the conditioning alone.",
        input_injection=True,
    ),
    Arm(
        "A6c_loop_attention_moe_norm_and_injection",
        "MEM[*E]^2MEMEM[*E]^2",
        "A6 with both refinements, the fourth cell of the 2x2. Tells whether the two interact or simply add.",
        per_iteration_norm=True,
        input_injection=True,
    ),
]

WANDB_PROJECT = "nemo"


def _count_per_iteration_norm_parameters(layer_pattern: str, n_embd: int) -> int:
    """
    Counts the parameters a per-iteration-norm arm has that the same arm without them does not.

    Iteration 0 reuses the norm the layer would have had anyway, so the extra is ``(K - 1)``
    norms per looped layer. Small, but it means such an arm is not *exactly* iso-parameter with
    the baseline, which is worth stating rather than discovering later.

    Args:
        layer_pattern (str): The arm's layer pattern.
        n_embd (int): Model dimension; an RMS norm holds one weight per channel.

    Returns:
        int: The number of extra parameters.
    """
    _, schedule = parse_layer_schedule(layer_pattern)
    return sum((group.num_loops - 1) * len(group.layer_keys) * n_embd for group in schedule)


def _render_arm(base_config_text: str, arm: Arm, n_embd: int) -> str:
    """
    Produces one arm's config text from the base config.

    Args:
        base_config_text (str): Contents of the base config.
        arm (Arm): The arm to render.
        n_embd (int): Model dimension read from the base config, used for the banner's
            per-iteration-norm parameter count.

    Raises:
        RuntimeError: If a value that must be substituted was not found exactly once, which would
            mean the base config changed shape and this generator needs updating.

    Returns:
        str: The arm's config text.
    """
    layer_pattern = arm.layer_pattern
    n_layer = get_num_built_layers(layer_pattern)
    counts = count_layers_by_type(layer_pattern)

    text = base_config_text
    for description_of_field, pattern, replacement in (
        ("layer_pattern", r'^    layer_pattern: ".*"$', f'    layer_pattern: "{layer_pattern}"'),
        ("n_layer", r"^    n_layer: \d+$", f"    n_layer: {n_layer}"),
        ("wandb project", r"^    project: .*$", f"    project: {WANDB_PROJECT}"),
        (
            "per_iteration_norm",
            r"^      per_iteration_norm: (?:true|false)$",
            f"      per_iteration_norm: {str(arm.per_iteration_norm).lower()}",
        ),
        (
            "input_injection",
            r"^      input_injection: (?:true|false)$",
            f"      input_injection: {str(arm.input_injection).lower()}",
        ),
    ):
        text, substitutions = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if substitutions != 1:
            raise RuntimeError(
                f"Expected exactly one {description_of_field} line in {BASE_CONFIG_PATH.name}, "
                f"found {substitutions}. The base config changed shape; update this generator."
            )

    banner_lines = [
        "# " + "=" * 96,
        f"# ABLATION ARM: {arm.name}",
        f"# {arm.description}",
        "#",
        f"#   layer_pattern : {layer_pattern}",
        f"#   built layers  : {n_layer}   (sets of weights)",
        f"#   executed      : {get_num_layers(layer_pattern)}   layer applications per token",
        f"#   executions    : {counts[LayerSymbol.MAMBA]} Mamba / {counts[LayerSymbol.MOE]} MoE"
        f" / {counts[LayerSymbol.ATTENTION]} attention",
    ]
    if arm.per_iteration_norm or arm.input_injection:
        refinements = ", ".join(
            name
            for name, enabled in (
                ("per-iteration norm", arm.per_iteration_norm),
                ("input injection", arm.input_injection),
            )
            if enabled
        )
        banner_lines.append(f"#   loop refine.  : {refinements}")
        if arm.per_iteration_norm:
            extra = _count_per_iteration_norm_parameters(layer_pattern, n_embd)
            banner_lines.append(f"#   extra params  : {extra:,}   (per-iteration norms; NOT iso-parameter with A0)")
    banner_lines += [
        "#",
        "# GENERATED FILE -- do not edit. Edit the base config and re-run",
        "# config_files/nemotron/loop_ablation/generate_arm_configs.py.",
        "# " + "=" * 96,
        "",
    ]
    return "\n".join(banner_lines) + text


def _read_n_embd(base_config_text: str) -> int:
    """
    Reads the model dimension out of the base config.

    Args:
        base_config_text (str): Contents of the base config.

    Raises:
        RuntimeError: If the ``n_embd`` line is missing or ambiguous.

    Returns:
        int: The model dimension.
    """
    matches = re.findall(r"^    n_embd: (\d+)$", base_config_text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one n_embd line in {BASE_CONFIG_PATH.name}, found {len(matches)}. "
            f"The base config changed shape; update this generator."
        )
    return int(matches[0])


def main() -> None:
    """Writes one config file per ablation arm."""
    base_config_text = BASE_CONFIG_PATH.read_text()
    n_embd = _read_n_embd(base_config_text)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        output_path = OUTPUT_DIRECTORY / f"config_{arm.name}.yaml"
        output_path.write_text(_render_arm(base_config_text, arm, n_embd))
        refinements = "".join(
            flag for flag, enabled in (("N", arm.per_iteration_norm), ("I", arm.input_injection)) if enabled
        )
        print(
            f"{arm.name:32s} {arm.layer_pattern:34s} "
            f"built={get_num_built_layers(arm.layer_pattern):3d} "
            f"executed={get_num_layers(arm.layer_pattern):3d} {refinements:2s} "
            f"-> {output_path.relative_to(REPOSITORY_ROOT)}"
        )


if __name__ == "__main__":
    main()
