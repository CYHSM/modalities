#!/usr/bin/env python
"""Generates vanilla dense-transformer baseline configs from the shared Nemotron base config.

The layer-loop ablation compares hybrid arms against each other, which answers "where should depth
go *within* a Nemotron" but never "is the hybrid worth anything in the first place". These configs
supply the missing reference: an ordinary pre-norm dense transformer (RoPE, SwiGLU, MHA) trained on
the same tokens, in the same order, with the same batch size, the same LR schedule, the same number
of steps and the same held-out evaluations.

They are *generated from the Nemotron base config* rather than written by hand for the same reason
the arms are: every shared setting -- data path, step profile, learning rate, `num_target_steps`,
the six evaluation dataloaders, the metrics -- comes from one file. A hand-maintained baseline
would drift from the arms, and a drifted baseline is worse than no baseline, because the curves
still look comparable.

Six top-level blocks differ, all of them model-specific: `loss_fn` (no MoE auxiliary term),
`initialized_model` (gpt2 initializer, depth is `n_layer`), `fsdp_model` (block class),
`model_raw`, `optimizer` (no expert-bias update) and `mfu_calculator`.

Run from the repository root::

    python config_files/nemotron/loop_ablation/generate_dense_baseline_configs.py
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

BASE_CONFIG_PATH = REPOSITORY_ROOT / "config_files/nemotron/config_research_nemotron_loops_1gpu.yaml"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "config_files/nemotron/loop_ablation"
WANDB_PROJECT = "nemo"

VOCAB_SIZE = 128256


class DenseBaseline(NamedTuple):
    """
    One dense-transformer reference model.

    Attributes:
        name (str): Baseline name; becomes the config file name and the wandb run name prefix.
        n_embd (int): Model dimension.
        n_layer (int): Number of transformer blocks.
        ffn_hidden (int): SwiGLU hidden dimension.
        n_head (int): Number of attention heads (multi-head; no grouped-query sharing).
        parameters (int): Total parameters, **measured by instantiating the model**, not computed
            from a closed form. A hand-derived formula for this stack was wrong by 8-16% (SwiGLU's
            three projections and the tied head are easy to miscount), and a wrong number here
            would silently misstate what the baseline is matched on -- which is the only thing it
            exists to establish. Re-derive after changing a shape by building the config's
            ``model_raw`` and summing ``p.numel()``.
        matched_on (str): What this baseline holds equal to `A0_baseline`, for the banner.
        description (str): One line on how to read it.
        max_lr (float): Peak learning rate. The base config's 4.5e-4 was chosen for a 225M-active
            Nemotron; a 1.1B dense model at the same 65k tokens/step is a different regime, so the
            larger baseline is also run at half and double to bracket it. ``initial_lr`` and
            ``final_lr`` scale with it at the base config's 1/10 ratio, and the optimizer's ``lr``
            must equal it -- the scheduler factory rejects a parameter group whose lr differs from
            ``max_lr``.
    """

    name: str
    n_embd: int
    n_layer: int
    ffn_hidden: int
    n_head: int
    parameters: int
    matched_on: str
    description: str
    max_lr: float = 4.5e-4


# Two references, because "the same size" is ambiguous for a sparse model and the two readings give
# opposite answers. A0_baseline is 1.105B total but only 225.5M active, so:
#
#   D1 matches what A0 *costs* -- same active parameters, hence ~the same FLOPs per token and the
#      same training speed. This is the honest "is the hybrid buying anything per unit of compute"
#      comparison, and the one to lead with.
#   D2 matches what A0 *occupies* -- same total parameters, hence ~4.9x the FLOPs per token. At
#      matched steps it is not a controlled comparison at all; it answers the different question of
#      what the same memory budget buys if spent densely. Expect it to win, and say why.
#
# Depth was solved for the parameter target at a fixed, conventional SwiGLU aspect ratio
# (ffn_hidden ~ 8/3 * n_embd, rounded to a multiple of 128 for tensor-core alignment): vary depth,
# not width, so the two baselines differ from each other only in scale. Both land within 0.5% of
# their target.
BASELINES: list[DenseBaseline] = [
    DenseBaseline(
        name="D1_dense_flops_matched",
        n_embd=1024,
        n_layer=9,
        ffn_hidden=2816,
        n_head=16,
        parameters=225_700_000,
        matched_on="active parameters (225.7M vs A0's 225.5M active) -> same FLOPs/token",
        description="Vanilla dense transformer at A0's COMPUTE budget. The primary reference: if "
        "the hybrid is worth its complexity, it should beat this at matched steps.",
    ),
    DenseBaseline(
        name="D2_dense_param_matched",
        n_embd=2048,
        n_layer=21,
        ffn_hidden=5504,
        n_head=16,
        parameters=1_110_500_000,
        matched_on="total parameters (1.111B vs A0's 1.105B total) -> ~4.9x the FLOPs/token",
        description="Vanilla dense transformer at A0's MEMORY budget. Not iso-FLOP -- it spends "
        "~4.9x the compute per token, so a win here is expected and prices what sparsity saves.",
    ),
    # An LR bracket around D2, because 4.5e-4 is inherited from a model with 1/5 the active
    # parameters and there is no reason it should still be right. Bracketing rather than only
    # lowering: a half-LR win says "go lower", a double-LR win says the inherited value was
    # *under*-tuned, and a double-LR blow-up confirms 4.5e-4 already sits near the ceiling. Any of
    # the three is a usable answer; only searching downwards would leave the last one unaskable.
    #
    # D2 at 4.5e-4 showed grad norm 17.3 at step 5 (against ~1.8 for the small dense baseline),
    # which is what prompted this.
    #
    # Everything except the learning rate is identical to D2, so these are also a free check that
    # the run-to-run spread at fixed settings is small compared to the LR effect.
    DenseBaseline(
        name="D2a_dense_param_matched_lr_half",
        n_embd=2048,
        n_layer=21,
        ffn_hidden=5504,
        n_head=16,
        parameters=1_110_500_000,
        matched_on="total parameters, at HALF the base learning rate (2.25e-4)",
        description="D2 with max_lr halved. Large dense models at a small batch usually want a "
        "lower peak LR than the 225M-active config this number came from.",
        max_lr=2.25e-4,
    ),
    DenseBaseline(
        name="D2b_dense_param_matched_lr_double",
        n_embd=2048,
        n_layer=21,
        ffn_hidden=5504,
        n_head=16,
        parameters=1_110_500_000,
        matched_on="total parameters, at DOUBLE the base learning rate (9.0e-4)",
        description="D2 with max_lr doubled. Expected to be the least stable of the three; if it "
        "is not, the inherited LR was too conservative and the whole sweep is under-tuned.",
        max_lr=9.0e-4,
    ),
]


# Each entry replaces one top-level block of the base config, matched from its key up to (but not
# including) the next top-level key. Everything not listed here -- settings, data, dataloaders,
# evaluations, metrics, lr_scheduler, gradient_clipper, subscribers -- is inherited verbatim, which
# is the entire point of generating rather than copying.
def _model_specific_blocks(baseline: DenseBaseline) -> dict[str, str]:
    """
    Builds the replacement text for every block that differs between Nemotron and a dense model.

    Args:
        baseline (DenseBaseline): The baseline being rendered.

    Returns:
        dict[str, str]: Top-level config key to its full replacement text.
    """
    return {
        # No MoE, so no auxiliary load-balancing term. The Nemotron arms wrap the same
        # cross-entropy in a weighted_sum with the aux loss; here it stands alone, which makes the
        # logged training loss directly comparable to the arms' `answer_nll`-style reading.
        "loss_fn": """loss_fn:
  component_key: loss
  variant_key: clm_cross_entropy_loss
  config:
    target_key: ${settings.referencing_keys.target_key}
    prediction_key: ${settings.referencing_keys.prediction_key}
""",
        "initialized_model": """initialized_model:
  component_key: model
  variant_key: model_initialized
  config:
    model:
      instance_key: fsdp_model
      pass_type: BY_REFERENCE
    model_initializer:
      component_key: model_initialization
      variant_key: composed
      config:
        model_type: gpt2
        weight_init_type: scaled
        mean: 0.0
        std: auto
        hidden_dim: ${model_raw.config.n_embd}
        # A dense model executes each layer exactly once, so effective depth IS n_layer. The
        # Nemotron arms derive this from the layer pattern because a loop group executes its
        # layers more than once; there is no such distinction here.
        num_layers: ${model_raw.config.n_layer}
""",
        "fsdp_model": """fsdp_model:
  component_key: model
  variant_key: fsdp2_wrapped
  config:
    model:
      instance_key: activation_checkpointed_model
      pass_type: BY_REFERENCE
    device_mesh:
      instance_key: device_mesh
      pass_type: BY_REFERENCE
    mixed_precision_settings:
      param_dtype: BF_16
      reduce_dtype: FP_32
    layers_per_fsdp_unit: 1
    block_names: [GPT2Block]
""",
        "model_raw": f"""model_raw:
  component_key: model
  variant_key: gpt2
  config:
    use_meta_device: false
    # Tied, exactly as the Nemotron arms are. At a 128k vocabulary the embedding is a large share
    # of a model this size, so untying here would change the parameter accounting the match rests
    # on.
    use_weight_tying: true
    sample_key: ${{settings.referencing_keys.sample_key}}
    prediction_key: ${{settings.referencing_keys.prediction_key}}
    sequence_length: ${{settings.step_profile.sequence_length}}
    vocab_size: {VOCAB_SIZE}
    n_layer: {baseline.n_layer}
    n_head_q: {baseline.n_head}
    # Multi-head, not grouped-query: this is the plain reference architecture, and GQA is a
    # memory-bandwidth optimization that would only muddy a parameter-matched comparison.
    n_head_kv: {baseline.n_head}
    ffn_hidden: {baseline.ffn_hidden}
    n_embd: {baseline.n_embd}
    dropout: 0.0
    bias: false
    # RoPE, because a dense transformer has no other source of positional information. The
    # Nemotron arms use none at all -- their Mamba-2 mixers carry position -- so this is a real
    # architectural difference, not a settings mismatch, and it favours neither side.
    poe_type: NOPE
    attention_config:
      qkv_transforms:
        - type_hint: RotaryTransform
          config:
            n_embd: ${{model_raw.config.n_embd}}
            n_head: ${{model_raw.config.n_head_q}}
            seq_length_dim: -2
            base_freq: 10000
    attention_implementation: pytorch_flash
    activation_type: swiglu
    attention_norm_config: &dense_norm_config
      norm_type: rms_norm
      config:
        ndim: ${{model_raw.config.n_embd}}
        bias: false
        epsilon: 1e-5
    ffn_norm_config: *dense_norm_config
    lm_head_norm_config: *dense_norm_config
""",
        # The arms use the moe_load_balanced wrapper, whose only job is the auxiliary-loss-free
        # expert bias update. Nothing to balance in a dense model; the inner AdamW is identical.
        "optimizer": f"""optimizer:
  component_key: optimizer
  variant_key: adam_w
  config:
    # Must equal lr_scheduler.max_lr; the scheduler factory rejects any mismatch.
    lr: {baseline.max_lr:.6g}
    betas: [0.9, 0.95]
    eps: 1e-8
    weight_decay: 0.1
    weight_decay_groups_excluded: [embedding, layernorm]
    wrapped_model:
      instance_key: initialized_model
      pass_type: BY_REFERENCE
""",
        # NOTE: `wrapped_model`, not `model_parts` as the Nemotron calculator takes.
        # GPT2MFUCalculatorConfig carries @add_deprecated_alias("model_parts", "wrapped_model"),
        # and that decorator sets validation_alias=AliasChoices("wrapped_model") *without*
        # including the field's own name and without populate_by_name. The supposedly deprecated
        # spelling is therefore the only one pydantic accepts, and `model_parts` fails with
        # "Field required: wrapped_model" plus "Extra inputs are not permitted: model_parts"
        # minutes into a run. NemotronMFUCalculatorConfig is undecorated, which is why the arms
        # use the other spelling.
        "mfu_calculator": """mfu_calculator:
  component_key: mfu_calculator
  variant_key: gpt2
  config:
    n_layer: ${model_raw.config.n_layer}
    sequence_length: ${settings.step_profile.sequence_length}
    n_embd: ${model_raw.config.n_embd}
    world_size: ${settings.cuda_env.world_size}
    wrapped_model:
      instance_key: initialized_model
      pass_type: BY_REFERENCE
    device_mesh:
      instance_key: device_mesh
      pass_type: BY_REFERENCE
""",
    }


def _replace_block(text: str, key: str, replacement: str) -> str:
    """
    Replaces one top-level YAML block, from its key line to the next top-level key.

    Args:
        text (str): The config text.
        key (str): The top-level key whose block is replaced.
        replacement (str): The block's new text, including its key line.

    Raises:
        RuntimeError: If the block is not found exactly once, which means the base config changed
            shape and this generator needs updating.

    Returns:
        str: The config text with the block replaced.
    """
    pattern = rf"^{re.escape(key)}:\n(?:[ \t].*\n|\n)*"
    text, substitutions = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if substitutions != 1:
        raise RuntimeError(
            f"Expected exactly one top-level '{key}:' block in {BASE_CONFIG_PATH.name}, found "
            f"{substitutions}. The base config changed shape; update this generator."
        )
    return text


def _render(base_config_text: str, baseline: DenseBaseline) -> str:
    """
    Produces one dense baseline's config text.

    Args:
        base_config_text (str): Contents of the Nemotron base config.
        baseline (DenseBaseline): The baseline to render.

    Returns:
        str: The baseline's config text.
    """
    text = base_config_text
    for key, replacement in _model_specific_blocks(baseline).items():
        text = _replace_block(text, key, replacement)

    # The lr_scheduler block is inherited, so its three learning rates are rewritten in place.
    # initial_lr and final_lr keep the base config's 1/10 ratio to max_lr: an LR bracket has to
    # move the whole schedule, or a "half LR" arm would still warm up from, and decay to, the
    # original endpoints and would not be half of anything.
    for field, value in (
        ("max_lr", baseline.max_lr),
        ("initial_lr", baseline.max_lr / 10),
        ("final_lr", baseline.max_lr / 10),
        ("project", WANDB_PROJECT),
    ):
        replacement = f"    {field}: {value if isinstance(value, str) else format(value, '.6g')}"
        text, substitutions = re.subn(rf"^    {field}: .*$", replacement, text, flags=re.MULTILINE)
        if substitutions != 1:
            raise RuntimeError(
                f"Expected exactly one '{field}' line in {BASE_CONFIG_PATH.name}, found "
                f"{substitutions}. The base config changed shape; update this generator."
            )

    banner = "\n".join(
        [
            "# " + "=" * 96,
            f"# DENSE BASELINE: {baseline.name}",
            f"# {baseline.description}",
            "#",
            f"#   matched on    : {baseline.matched_on}",
            f"#   shape         : n_embd {baseline.n_embd} / n_layer {baseline.n_layer} / "
            f"ffn_hidden {baseline.ffn_hidden} / {baseline.n_head} heads",
            f"#   parameters    : {baseline.parameters / 1e6:.1f}M (tied embeddings; measured, not derived)",
            f"#   max_lr        : {baseline.max_lr:.6g}   (warmup from and decay to max_lr/10)",
            "#",
            "# Everything else -- data, step profile, learning rate schedule, num_target_steps and",
            "# all six evaluation dataloaders -- is inherited from the Nemotron base config, so the",
            "# curves are comparable to the ablation arms step for step.",
            "#",
            "# GENERATED FILE -- do not edit. Edit the base config and re-run",
            "# config_files/nemotron/loop_ablation/generate_dense_baseline_configs.py.",
            "# " + "=" * 96,
            "",
        ]
    )
    return banner + text


def main() -> None:
    """Writes one config file per dense baseline."""
    base_config_text = BASE_CONFIG_PATH.read_text()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for baseline in BASELINES:
        output_path = OUTPUT_DIRECTORY / f"config_{baseline.name}.yaml"
        output_path.write_text(_render(base_config_text, baseline))
        print(
            f"{baseline.name:26s} d={baseline.n_embd:5d} L={baseline.n_layer:3d} "
            f"ffn={baseline.ffn_hidden:5d}  {baseline.parameters / 1e6:8.1f}M "
            f"-> {output_path.relative_to(REPOSITORY_ROOT)}"
        )


if __name__ == "__main__":
    main()
