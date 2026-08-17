"""Tests for depth-wise weight-shared loops in the hybrid layer pattern.

A loop group ``[<symbols>]^<K>`` builds its layers once and executes them ``K`` times. The
properties that matter, and that these tests pin down, are:

* the loop is genuinely weight-shared (parameter count and state dict are those of the unlooped
  model that builds the same layers), and
* everything that is counted per *execution* rather than per *parameter* follows the loop:
  effective depth, the MoE auxiliary loss, and the MFU calculator's active-parameter count.
"""

import re
from typing import Optional

import pytest
import torch

from modalities.models.components.moe.experts import ExpertsBackend
from modalities.models.components.moe.moe import MoE
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.layer_pattern import (
    LayerSymbol,
    count_layers_by_type,
    get_num_built_layers,
    get_num_layers,
    parse_layer_schedule,
)
from modalities.models.nemotron.nemotron_layer_specs import (
    Mamba2LayerSpec,
    NemotronAttentionLayerSpec,
    NemotronMoELayerSpec,
)
from modalities.models.nemotron.nemotron_layers import PerIterationNorm
from modalities.models.nemotron.nemotron_loop import LoopIterationConditioning
from modalities.models.nemotron.nemotron_model import LoopConfig, NemotronLLM
from modalities.training.activation_checkpointing.activation_checkpointing import (
    ActivationCheckpointing,
    ActivationCheckpointingVariants,
)
from modalities.utils.nemotron_mfu import NemotronMFUCalculator

N_EMBD = 128
VOCAB_SIZE = 256
SEQUENCE_LENGTH = 32
NORM_CONFIG = {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": N_EMBD, "eps": 1e-5}}

# The patterns of the six ablation arms in config_research_nemotron_loops_1gpu.yaml.
ABLATION_ARMS = {
    "baseline": ("MEM*EMEMEM*E", 12, 12),
    "loop_mamba": ("[M]^3EM*EMEMEM*E", 12, 14),
    "loop_moe": ("M[E]^3M*EMEMEM*E", 12, 14),
    "loop_attention": ("MEM[*]^3EMEMEM*E", 12, 14),
    "loop_mamba_moe": ("[ME]^3M*EMEMEM*E", 12, 16),
    "loop_mamba_attention": ("MEM[M*]^3EMEMEM*E", 13, 17),
}


def _layer_specs(aux_loss_coeff: float = 0.0) -> dict:
    return {
        "M": Mamba2LayerSpec(
            n_embd=N_EMBD,
            mamba_n_heads=8,
            mamba_head_dim=16,
            mamba_state_dim=8,
            mamba_n_groups=2,
            chunk_size=8,
            norm_config=NORM_CONFIG,
        ),
        "E": NemotronMoELayerSpec(
            n_embd=N_EMBD,
            num_experts=8,
            moe_ffn_hidden=32,
            top_k=2,
            route_scale=2.5,
            num_shared_experts=1,
            aux_loss_coeff=aux_loss_coeff,
            experts_backend=ExpertsBackend.LOOPED,
            norm_config=NORM_CONFIG,
        ),
        "*": NemotronAttentionLayerSpec(
            n_embd=N_EMBD,
            n_head_q=4,
            n_head_kv=2,
            head_dim=32,
            norm_config=NORM_CONFIG,
        ),
    }


def _build_model(
    layer_pattern: str,
    aux_loss_coeff: float = 0.0,
    seed: int = 0,
    loop_config: Optional[LoopConfig] = None,
) -> NemotronLLM:
    torch.manual_seed(seed)
    return NemotronLLM(
        sample_key="input_ids",
        prediction_key="logits",
        aux_loss_key="moe_aux_loss",
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=get_num_built_layers(layer_pattern),
        layer_pattern=layer_pattern,
        layer_specs=_layer_specs(aux_loss_coeff=aux_loss_coeff),
        lm_head_norm_config=NormWrapperConfig.model_validate(NORM_CONFIG),
        loop_config=loop_config,
    )


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def test_bracket_free_pattern_yields_one_single_layer_group_per_symbol():
    layer_symbols, schedule = parse_layer_schedule("ME*-")
    assert layer_symbols == [LayerSymbol.MAMBA, LayerSymbol.MOE, LayerSymbol.ATTENTION, LayerSymbol.MLP]
    assert [(group.layer_keys, group.num_loops) for group in schedule] == [
        (("0",), 1),
        (("1",), 1),
        (("2",), 1),
        (("3",), 1),
    ]


def test_loop_group_builds_its_layers_once_and_repeats_them():
    layer_symbols, schedule = parse_layer_schedule("M[ME]^3E")
    # Four layers are built (M, M, E, E), not the six that are executed.
    assert layer_symbols == [LayerSymbol.MAMBA, LayerSymbol.MAMBA, LayerSymbol.MOE, LayerSymbol.MOE]
    assert [(group.layer_keys, group.num_loops) for group in schedule] == [
        (("0",), 1),
        (("1", "2"), 3),
        (("3",), 1),
    ]


def test_attention_symbol_inside_a_loop_group_is_not_the_repeat_operator():
    # "[M*]^2" must read as a group of Mamba+attention repeated twice, not as a repeat of "M".
    layer_symbols, schedule = parse_layer_schedule("[M*]^2E")
    assert layer_symbols == [LayerSymbol.MAMBA, LayerSymbol.ATTENTION, LayerSymbol.MOE]
    assert schedule[0].layer_keys == ("0", "1")
    assert schedule[0].num_loops == 2


@pytest.mark.parametrize(
    "pattern, built, executed",
    [(pattern, built, executed) for pattern, built, executed in ABLATION_ARMS.values()],
    ids=list(ABLATION_ARMS),
)
def test_ablation_arm_patterns_build_and_execute_the_expected_depths(pattern, built, executed):
    assert get_num_built_layers(pattern) == built
    assert get_num_layers(pattern) == executed


def test_count_layers_by_type_counts_executions_not_weight_sets():
    # The MoE layer is built once but visited three times.
    assert count_layers_by_type("M[E]^3*") == {
        LayerSymbol.MAMBA: 1,
        LayerSymbol.MOE: 3,
        LayerSymbol.ATTENTION: 1,
        LayerSymbol.MLP: 0,
    }


@pytest.mark.parametrize(
    "pattern, message",
    [
        ("M[ME]*3E", "Malformed loop group"),  # the '*' repeat operator is not accepted
        ("[ME]3", "Malformed loop group"),
        ("[[M]^2]^2", "Malformed loop group"),  # nesting
        ("[ME]", "Malformed loop group"),
        ("[]^3", "Empty loop group"),
        ("[M]^0", "at least 1"),
        ("M]", "Unmatched"),
        ("M^3", "Invalid layer symbol"),
        ("", "must not be empty"),
    ],
)
def test_malformed_patterns_are_rejected(pattern, message):
    with pytest.raises(ValueError, match=message):
        parse_layer_schedule(pattern)


@pytest.mark.parametrize("pattern", ["MEM*E", "ME*-", "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"])
def test_bracket_free_patterns_are_unchanged_by_the_loop_grammar(pattern):
    # Guards the shipped configs: without brackets, built depth, executed depth and the per-type
    # counts must all still be plain character counts.
    assert get_num_built_layers(pattern) == len(pattern)
    assert get_num_layers(pattern) == len(pattern)
    assert sum(count_layers_by_type(pattern).values()) == len(pattern)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------


def test_looped_model_shares_weights_with_the_unlooped_model_of_the_same_built_layers():
    # "M[ME]^3*E" builds exactly the layers that "MME*E" builds, so the two must be identical in
    # parameters and differ only in how often those parameters are applied.
    unlooped = _build_model("MME*E")
    looped = _build_model("M[ME]^3*E")

    assert sorted(looped.state_dict()) == sorted(unlooped.state_dict())
    assert sum(p.numel() for p in looped.parameters()) == sum(p.numel() for p in unlooped.parameters())
    assert unlooped.n_executed_layers == 5
    assert looped.n_executed_layers == 9


def test_execution_counts_report_how_often_each_layer_runs():
    looped = _build_model("M[ME]^3*E")
    assert looped.get_execution_counts() == {"0": 1, "1": 3, "2": 3, "3": 1, "4": 1}


def test_n_layer_must_count_built_layers_not_executed_layers():
    with pytest.raises(ValueError, match="builds 4 layers"):
        NemotronLLM(
            sample_key="input_ids",
            prediction_key="logits",
            sequence_length=SEQUENCE_LENGTH,
            vocab_size=VOCAB_SIZE,
            n_embd=N_EMBD,
            n_layer=6,  # the executed depth, which is the mistake this guards against
            layer_pattern="M[ME]^3E",
            layer_specs=_layer_specs(),
            lm_head_norm_config=NormWrapperConfig.model_validate(NORM_CONFIG),
        )


def test_forward_and_backward_pass_through_every_loop_iteration():
    looped = _build_model("M[ME]^3*E")
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    logits = looped({"input_ids": inputs})["logits"]
    assert logits.shape == (2, SEQUENCE_LENGTH, VOCAB_SIZE)

    logits.sum().backward()
    parameters_without_gradient = [name for name, p in looped.named_parameters() if p.grad is None]
    assert parameters_without_gradient == []


def test_unknown_injection_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown injection_mode"):
        NemotronLLM(
            sample_key="input_ids",
            prediction_key="logits",
            sequence_length=SEQUENCE_LENGTH,
            vocab_size=VOCAB_SIZE,
            n_embd=N_EMBD,
            n_layer=2,
            layer_pattern="ME",
            layer_specs=_layer_specs(),
            lm_head_norm_config=NormWrapperConfig.model_validate(NORM_CONFIG),
            loop_config=LoopConfig(injection_mode="concat_proj"),
        )


# --------------------------------------------------------------------------------------------
# Auxiliary loss and MFU: quantities that must follow executions, not parameters
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern, expected_moe_visits",
    [("ME", 1), ("M[ME]^1", 1), ("M[ME]^3", 3), ("M[E]^5", 5), ("[ME]^2E", 3)],
)
def test_aux_loss_accumulates_once_per_moe_visit(monkeypatch, pattern, expected_moe_visits):
    # Each visit contributes exactly 1.0, so the total is a pure count of visits. Overwriting
    # instead of accumulating (the pre-loop behaviour) would report 1.0 for every pattern and
    # would leave all but the last iteration's routing unpenalized.
    monkeypatch.setattr(MoE, "_compute_aux_loss", lambda self, scores, top_indices, batch_size: torch.tensor(1.0))

    model = _build_model(pattern, aux_loss_coeff=1e-4)
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    assert model({"input_ids": inputs})["moe_aux_loss"].item() == pytest.approx(expected_moe_visits)


def test_aux_loss_does_not_leak_across_forward_passes(monkeypatch):
    monkeypatch.setattr(MoE, "_compute_aux_loss", lambda self, scores, top_indices, batch_size: torch.tensor(1.0))

    model = _build_model("M[ME]^3", aux_loss_coeff=1e-4)
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    losses = [model({"input_ids": inputs})["moe_aux_loss"].item() for _ in range(3)]
    assert losses == pytest.approx([3.0, 3.0, 3.0])


def test_active_parameter_count_charges_a_looped_layer_once_per_iteration():
    unlooped = _build_model("MME*E")
    looped = _build_model("M[ME]^3*E")

    unlooped_active = NemotronMFUCalculator.count_active_parameters(unlooped)
    looped_active = NemotronMFUCalculator.count_active_parameters(looped)

    # The two models hold identical weights, so equal counts would mean the loop's extra work is
    # invisible to the MFU calculator and looped arms would report inflated MFU.
    assert looped_active > unlooped_active

    # Precisely: the twice-extra execution of layers "1" (Mamba) and "2" (MoE).
    layers = looped.transformer.h
    extra = 2 * sum(NemotronMFUCalculator._count_active_parameters_of_module(layers[key]) for key in ("1", "2"))
    assert looped_active == unlooped_active + extra


# --------------------------------------------------------------------------------------------
# Per-iteration norms and input injection
#
# Both are optional refinements of the loop, off by default so that the arms already run keep
# meaning what they meant. Each is measured against a hand-computed reference, because the failure
# mode of "the wrong norm was applied" or "the injection landed on the wrong iteration" is a model
# that trains perfectly well and answers a different question than the one being asked.
# --------------------------------------------------------------------------------------------


def _run_reference_mamba_group(model: NemotronLLM, inputs: torch.Tensor, num_loops: int, inject: bool) -> torch.Tensor:
    """Recomputes a single-layer Mamba loop group by hand, from the embedding to the logits."""
    layer = model.transformer.h["0"]
    h = model.transformer.wte(inputs)
    group_input = h
    for iteration in range(num_loops):
        if inject and iteration > 0:
            h = h + group_input
        norm = layer.norm.norms[iteration] if layer.has_per_iteration_norm else layer.norm
        h = h + layer.mixer(norm(h))
    return model.transformer.lm_head(model.transformer.lm_head_norm(h))


def test_both_refinements_are_off_by_default():
    loop_config = LoopConfig()
    assert (loop_config.per_iteration_norm, loop_config.input_injection, loop_config.injection_mode) == (
        False,
        False,
        "add",
    )


@pytest.mark.parametrize("pattern", ["MEM*E", "M[ME]^3*E"])
def test_defaults_leave_the_module_tree_and_the_output_untouched(pattern):
    # The flags may not be a silent rename: an arm run before they existed must still be
    # reproducible from a config that does not mention them.
    without = _build_model(pattern)
    with_explicit_defaults = _build_model(pattern, loop_config=LoopConfig(variant="simple"))

    assert sorted(without.state_dict()) == sorted(with_explicit_defaults.state_dict())
    assert all(not isinstance(module, PerIterationNorm) for module in with_explicit_defaults.modules())

    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    torch.testing.assert_close(
        without({"input_ids": inputs})["logits"], with_explicit_defaults({"input_ids": inputs})["logits"]
    )


def test_per_iteration_norm_builds_one_norm_per_iteration_only_inside_loop_groups():
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(per_iteration_norm=True))

    # Layers "1" and "2" are the loop group's; "0", "3" and "4" run once and keep a plain norm.
    for layer_key in ("1", "2"):
        norm = model.transformer.h[layer_key].norm
        assert isinstance(norm, PerIterationNorm)
        assert len(norm.norms) == 3
    for layer_key in ("0", "3", "4"):
        assert not isinstance(model.transformer.h[layer_key].norm, PerIterationNorm)

    assert "transformer.h.1.norm.norms.2.weight" in model.state_dict()


def test_per_iteration_norm_is_a_no_op_without_loops():
    # A0_baseline may carry the flag without becoming a different model, so the norm/no-norm
    # comparison is not confounded by the pattern.
    plain = _build_model("MEM*E")
    flagged = _build_model("MEM*E", loop_config=LoopConfig(per_iteration_norm=True))
    assert sorted(plain.state_dict()) == sorted(flagged.state_dict())
    assert flagged.num_per_iteration_norm_parameters == 0


def test_per_iteration_norm_applies_the_norm_belonging_to_the_current_iteration():
    model = _build_model("[M]^3", loop_config=LoopConfig(per_iteration_norm=True))
    # Distinct scales per iteration, so applying the wrong one cannot coincidentally agree.
    for iteration, scale in enumerate((0.5, 1.5, 2.5)):
        torch.nn.init.constant_(model.transformer.h["0"].norm.norms[iteration].weight, scale)

    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    torch.testing.assert_close(
        model({"input_ids": inputs})["logits"],
        _run_reference_mamba_group(model, inputs, num_loops=3, inject=False),
    )

    # The reference is only meaningful if the three norms actually differ; a model that ignored
    # the iteration index would match a reference built from norms.0 alone.
    only_first_norm = _run_reference_mamba_group(model, inputs, num_loops=1, inject=False)
    assert not torch.allclose(model({"input_ids": inputs})["logits"], only_first_norm)


def test_per_iteration_norms_add_exactly_one_norm_per_extra_iteration():
    # Section 7.3 of the research plan: a per-iteration-norm arm is no longer exactly
    # iso-parameter with the baseline, and the delta has to be reported rather than assumed zero.
    pattern = "M[ME]^3*E"  # one loop group of two layers, three iterations
    plain = _build_model(pattern)
    per_iteration = _build_model(pattern, loop_config=LoopConfig(per_iteration_norm=True))

    expected_extra = (3 - 1) * 2 * N_EMBD
    assert per_iteration.num_per_iteration_norm_parameters == expected_extra
    assert sum(p.numel() for p in per_iteration.parameters()) == sum(p.numel() for p in plain.parameters()) + (
        expected_extra
    )


def test_input_injection_adds_the_group_input_before_every_iteration_after_the_first():
    model = _build_model("[M]^3", loop_config=LoopConfig(input_injection=True))
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    torch.testing.assert_close(
        model({"input_ids": inputs})["logits"],
        _run_reference_mamba_group(model, inputs, num_loops=3, inject=True),
    )
    # Injecting is not the same as not injecting, so the reference above is a real constraint.
    assert not torch.allclose(
        model({"input_ids": inputs})["logits"],
        _run_reference_mamba_group(model, inputs, num_loops=3, inject=False),
    )


def test_input_injection_is_a_no_op_for_layers_that_are_not_looped():
    # Every non-looped layer is a group of one iteration. Injecting after each iteration (rather
    # than before each iteration but the first) would double the input of every plain layer in the
    # model, which would change A0_baseline the moment the flag was set.
    plain = _build_model("MEM*E")
    injected = _build_model("MEM*E", loop_config=LoopConfig(input_injection=True))
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    torch.testing.assert_close(plain({"input_ids": inputs})["logits"], injected({"input_ids": inputs})["logits"])


def test_both_refinements_combine():
    model = _build_model("[M]^3", loop_config=LoopConfig(per_iteration_norm=True, input_injection=True))
    for iteration, scale in enumerate((0.5, 1.5, 2.5)):
        torch.nn.init.constant_(model.transformer.h["0"].norm.norms[iteration].weight, scale)

    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    torch.testing.assert_close(
        model({"input_ids": inputs})["logits"],
        _run_reference_mamba_group(model, inputs, num_loops=3, inject=True),
    )


def test_every_per_iteration_norm_receives_a_gradient():
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(per_iteration_norm=True, input_injection=True))
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    model({"input_ids": inputs})["logits"].sum().backward()
    parameters_without_gradient = [name for name, p in model.named_parameters() if p.grad is None]
    assert parameters_without_gradient == []


def test_refinements_survive_activation_checkpointing():
    # The layers are called with an extra positional argument now, which has to make it through
    # the checkpoint wrapper. Full AC is what every ablation arm runs with.
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(per_iteration_norm=True, input_injection=True))
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    expected = model({"input_ids": inputs})["logits"]

    ActivationCheckpointing.apply_activation_checkpointing_(
        ac_variant=ActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING,
        layers_fqn="transformer.h",
        model=model,
        ac_fun_params=None,
    )
    checkpointed = model({"input_ids": inputs})["logits"]

    torch.testing.assert_close(checkpointed, expected)
    checkpointed.sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


def test_unknown_injection_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown injection_mode"):
        LoopConfig(input_injection=True, injection_mode="concat_proj")


@pytest.mark.parametrize("per_iteration_norm", [False, True])
def test_every_parameter_lands_in_exactly_one_weight_decay_group(per_iteration_norm):
    # A parameter matching no group is not merely undecayed: _build_optimizer_groups_via_weight_
    # decay_split builds the two optimizer groups from the group members alone, so an unmatched
    # parameter is never handed to the optimizer at all. A parameter matching two groups is added
    # twice and torch rejects the optimizer. Per-iteration norms are named
    # "transformer.h.<idx>.norm.norms.<iteration>.weight" precisely to stay inside "layernorm".
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(per_iteration_norm=per_iteration_norm))
    groups = model.weight_decay_groups

    for name, _ in model.named_parameters():
        matched = [
            group
            for group, expressions in groups.items()
            if any(re.search(expression, name) for expression in expressions)
        ]
        assert len(matched) == 1, f"{name} matched {matched}"
        if ".norm" in name or "lm_head_norm" in name:
            assert matched == ["layernorm"], name


# --------------------------------------------------------------------------------------------
# Per-group iteration conditioning
#
# This attaches to a loop *group* rather than to a layer, and lives in LoopIterationConditioning.
# It is checked against a hand-computed reference for the same reason the per-layer refinements
# are: conditioning read from the wrong iteration still trains and still converges -- it just
# answers a different question than the arm claims to.
#
# Two sibling refinements (a stabilized recurrence, `variant="parcae"`, and an injection norm,
# `injection_norm_config`) were removed after the K in {3, 6, 12} ablation. Their tests went with
# them; what remains is the assertion that a config still asking for them fails loudly.
# --------------------------------------------------------------------------------------------


def test_iteration_conditioning_is_off_by_default():
    loop_config = LoopConfig()
    assert loop_config.iteration_embedding == "none"
    assert not loop_config.needs_group_modulation


@pytest.mark.parametrize("pattern", ["MEM*E", "M[ME]^3*E"])
def test_defaults_build_no_group_modulation_at_all(pattern):
    # Conditioning may not be a silent rename either: a model with the defaults must have the same
    # module tree and the same state dict keys it had before LoopIterationConditioning existed.
    model = _build_model(pattern, loop_config=LoopConfig())
    assert "loop_mods" not in model.transformer
    assert all(not isinstance(module, LoopIterationConditioning) for module in model.modules())
    assert model.num_loop_refinement_parameters == 0


def test_input_injection_alone_builds_no_group_modulation():
    # input_injection needs no parameters of its own, so it must not drag an (empty) module into
    # the state dict -- an arm using it stays byte-for-byte comparable with the plain loop.
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(input_injection=True))
    assert "loop_mods" not in model.transformer
    assert model.num_loop_refinement_parameters == 0


@pytest.mark.parametrize("iteration_embedding", ["add", "film"])
def test_group_modulation_is_built_only_for_groups_that_loop(iteration_embedding):
    # "M[ME]^3*E" has groups (M), (M E)x3, (*), (E): exactly one loops, and only it needs a module.
    model = _build_model("M[ME]^3*E", loop_config=LoopConfig(iteration_embedding=iteration_embedding))
    assert sorted(model.transformer.loop_mods.keys()) == ["1"]


def test_group_modulation_is_absent_when_nothing_loops():
    model = _build_model("MEM*E", loop_config=LoopConfig(iteration_embedding="film"))
    assert len(model.transformer.loop_mods) == 0
    assert model.num_loop_refinement_parameters == 0


@pytest.mark.parametrize("iteration_embedding", ["add", "film"])
def test_iteration_conditioning_is_an_exact_no_op_at_initialization(iteration_embedding):
    # Zero-initialized tables mean enabling conditioning cannot move the loss at step 0, only the
    # trajectory -- so a conditioning arm and its control start from the same place.
    plain = _build_model("[ME]^3*E", seed=7)
    conditioned = _build_model("[ME]^3*E", seed=7, loop_config=LoopConfig(iteration_embedding=iteration_embedding))
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    torch.testing.assert_close(conditioned({"input_ids": inputs})["logits"], plain({"input_ids": inputs})["logits"])


@pytest.mark.parametrize("iteration_embedding", ["add", "film"])
def test_iteration_conditioning_applies_the_entry_of_the_current_iteration(iteration_embedding):
    model = _build_model("[M]^3", loop_config=LoopConfig(iteration_embedding=iteration_embedding))
    conditioning = model.transformer.loop_mods["0"]
    with torch.no_grad():
        conditioning.iter_shift.normal_(std=0.5)
        if iteration_embedding == "film":
            conditioning.iter_scale.normal_(std=0.5)
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    layer = model.transformer.h["0"]
    h = model.transformer.wte(inputs)
    for iteration in range(3):
        if iteration_embedding == "film":
            h = h * (1.0 + conditioning.iter_scale[iteration]) + conditioning.iter_shift[iteration]
        else:
            h = h + conditioning.iter_shift[iteration]
        h = h + layer.mixer(layer.norm(h))
    expected = model.transformer.lm_head(model.transformer.lm_head_norm(h))

    torch.testing.assert_close(model({"input_ids": inputs})["logits"], expected)


def test_add_conditioning_has_no_scale_table():
    # "add" is FiLM's shift-only degenerate case, so it must not silently allocate (and leave
    # untrained) a scale table that no code path reads.
    conditioning = _build_model("[M]^3", loop_config=LoopConfig(iteration_embedding="add")).transformer.loop_mods["0"]
    assert not hasattr(conditioning, "iter_scale")


def test_iteration_conditioning_table_is_sized_to_the_groups_loop_count():
    model = _build_model("[M]^5E[ME]^2*E", loop_config=LoopConfig(iteration_embedding="film"))
    assert model.transformer.loop_mods["0"].iter_shift.shape == (5, N_EMBD)
    assert model.transformer.loop_mods["2"].iter_scale.shape == (2, N_EMBD)


def test_unknown_iteration_embedding_is_rejected():
    with pytest.raises(ValueError, match="Unknown iteration_embedding"):
        LoopConfig(iteration_embedding="rotary")


@pytest.mark.parametrize(
    "removed_kwargs",
    [
        {"variant": "parcae"},
        {"injection_norm_config": NORM_CONFIG},
        {"variant": "parcae", "injection_norm_config": NORM_CONFIG},
    ],
)
def test_removed_refinements_are_rejected_rather_than_ignored(removed_kwargs):
    # LoopConfig sets extra="allow", so without an explicit check a refinement-wave config would
    # parse cleanly, train a plain loop and still report itself as a refined arm.
    with pytest.raises(ValueError, match="removed after"):
        LoopConfig(**removed_kwargs)


@pytest.mark.parametrize(
    "historical_block",
    [
        # Every loop config written before the refinements were removed carries `variant: simple`;
        # the refinement wave's controls also carry an explicit `injection_norm_config: null`.
        {"variant": "simple", "per_iteration_norm": False, "input_injection": False},
        {"variant": "simple", "per_iteration_norm": False, "input_injection": False, "injection_norm_config": None},
    ],
)
def test_historical_loop_config_blocks_survive_the_component_factorys_strict_validation(historical_block):
    # ComponentFactory validates every component config with `extra="forbid"`, which OVERRIDES this
    # model's `extra="allow"` and propagates into nested models. So a removed key that is merely
    # undeclared is a hard build failure for every historical arm config -- not the tolerated extra
    # that `extra="allow"` suggests. Deleting `variant` outright did exactly that, and the shipped-
    # config schema test did not catch it because it only inspects each component's *top-level* keys.
    assert LoopConfig.model_validate(historical_block, extra="forbid").needs_group_modulation is False


@pytest.mark.parametrize("inert_kwargs", [{"variant": "simple"}, {"injection_norm_config": None}, {"dt_min": 0.001}])
def test_configs_predating_the_refinements_still_load(inert_kwargs):
    # Every loop config shipped before the refinements carries `variant: simple`, and every
    # non-refined arm of the ablation carries an explicit `injection_norm_config: null`. Those
    # describe exactly the plain loop, so rejecting them would break the historical configs for
    # no gain.
    assert LoopConfig(**inert_kwargs).needs_group_modulation is False


def test_refinement_parameter_count_is_reported():
    model = _build_model("[M]^4E", loop_config=LoopConfig(iteration_embedding="film"))
    # film: (scale + shift) = 2 * 4 * n_embd, for the one looped group.
    assert model.num_loop_refinement_parameters == 2 * 4 * N_EMBD


@pytest.mark.parametrize(
    "loop_config",
    [
        LoopConfig(iteration_embedding="film"),
        LoopConfig(iteration_embedding="add", per_iteration_norm=True),
        LoopConfig(iteration_embedding="film", per_iteration_norm=True, input_injection=True),
    ],
)
def test_every_refinement_parameter_receives_a_gradient(loop_config):
    model = _build_model("M[ME]^3*E", loop_config=loop_config)
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))

    model({"input_ids": inputs})["logits"].sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize(
    "loop_config",
    [
        LoopConfig(iteration_embedding="film"),
        LoopConfig(iteration_embedding="film", per_iteration_norm=True, input_injection=True),
    ],
)
def test_every_refinement_parameter_lands_in_exactly_one_weight_decay_group(loop_config):
    # Same trap as the per-iteration norms: the conditioning tables are matched by the `loop`
    # group's explicit name list. Overlap makes the optimizer raise; a gap drops the parameter
    # from training silently.
    model = _build_model("M[ME]^3*E", loop_config=loop_config)
    groups = model.weight_decay_groups
    in_loop_group = []

    for name, _ in model.named_parameters():
        matched = [
            group
            for group, expressions in groups.items()
            if any(re.search(expression, name) for expression in expressions)
        ]
        assert len(matched) == 1, f"{name} matched {matched}"
        if "loop_mods" in name:
            assert matched == ["loop"], name
        if matched == ["loop"]:
            in_loop_group.append(name)

    # The complement of the "empty by default" assertion in test_nemotron_model.py: with
    # conditioning on, the group must actually claim parameters, or they are silently untrained.
    assert in_loop_group


def test_refinement_parameters_survive_activation_checkpointing():
    loop_config = LoopConfig(iteration_embedding="film", per_iteration_norm=True, input_injection=True)
    model = _build_model("M[ME]^3*E", loop_config=loop_config)
    with torch.no_grad():  # off the no-op initialization, so a dropped term would show
        model.transformer.loop_mods["1"].iter_shift.normal_(std=0.1)
    inputs = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
    expected = model({"input_ids": inputs})["logits"]

    ActivationCheckpointing.apply_activation_checkpointing_(
        ac_variant=ActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING,
        layers_fqn="transformer.h",
        model=model,
        ac_fun_params=None,
    )
    checkpointed = model({"input_ids": inputs})["logits"]

    torch.testing.assert_close(checkpointed, expected)
    checkpointed.sum().backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []
