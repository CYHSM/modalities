"""Tests for the loop-update recorder.

Two classes of property matter here, and they fail in opposite directions:

* The recorder must not change what the model computes. A measurement that perturbs its subject
  produces numbers that look fine and describe a different model.
* The recorded quantity must be the one claimed. ``delta = output - input`` is only the operator's
  contribution because every residual layer computes ``x + operator(norm(x))``; that identity is
  asserted here against the real layer classes rather than assumed from reading them.
"""

import re

import pytest
import torch

from modalities.analysis.loop_updates import LoopUpdateRecorder, _per_token_cosine, _per_token_norm_ratio
from modalities.models.components.moe.experts import ExpertsBackend
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.layer_pattern import get_num_built_layers
from modalities.models.nemotron.nemotron_layer_specs import (
    Mamba2LayerSpec,
    NemotronAttentionLayerSpec,
    NemotronMoELayerSpec,
)
from modalities.models.nemotron.nemotron_model import LoopConfig, NemotronLLM

N_EMBD = 128
VOCAB_SIZE = 256
SEQUENCE_LENGTH = 32
NORM_CONFIG = {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": N_EMBD, "eps": 1e-5}}


def _layer_specs() -> dict:
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
            aux_loss_coeff=0.0,
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


def _build_model(layer_pattern: str, loop_config: LoopConfig = None, seed: int = 0) -> NemotronLLM:
    torch.manual_seed(seed)
    return NemotronLLM(
        sample_key="input_ids",
        prediction_key="logits",
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=get_num_built_layers(layer_pattern),
        layer_pattern=layer_pattern,
        layer_specs=_layer_specs(),
        lm_head_norm_config=NormWrapperConfig.model_validate(NORM_CONFIG),
        loop_config=loop_config,
    )


def _inputs(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randint(0, VOCAB_SIZE, (batch, SEQUENCE_LENGTH))


# --------------------------------------------------------------------------------------------
# The recorded quantity is the operator's contribution
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["M", "E", "*"])
def test_layer_output_minus_input_is_exactly_the_operator_contribution(pattern):
    # The whole measurement rests on `_ResidualLayer.forward` being `x + operator(norm(x))`. If a
    # layer type ever gains a second residual path or a post-norm, delta stops being the operator's
    # contribution and every number this module reports silently changes meaning.
    model = _build_model(pattern)
    layer = model.transformer.h["0"]
    hidden = torch.randn(2, SEQUENCE_LENGTH, N_EMBD)

    delta = layer(hidden) - hidden
    expected = layer._operator(layer.norm(hidden))

    torch.testing.assert_close(delta, expected)


def test_recorder_does_not_change_the_forward_pass():
    model = _build_model("[M]^3E[ME]^2*E", loop_config=LoopConfig())
    model.eval()
    inputs = _inputs()

    with torch.no_grad():
        unhooked = model({"input_ids": inputs})["logits"]
        with LoopUpdateRecorder(model=model) as recorder:
            hooked = model({"input_ids": inputs})["logits"]

    torch.testing.assert_close(hooked, unhooked)
    assert recorder.calls, "recorder captured nothing"


def test_recorder_removes_its_hooks_on_exit():
    model = _build_model("[M]^3E")
    with LoopUpdateRecorder(model=model):
        pass
    layer = model.transformer.h["0"]
    assert not layer._forward_pre_hooks
    assert not layer._forward_hooks


# --------------------------------------------------------------------------------------------
# Structure: calls map onto the schedule
# --------------------------------------------------------------------------------------------


def test_each_layer_is_recorded_once_per_iteration():
    model = _build_model("[M]^3E[M*]^2E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    assert len(recorder.calls) == model.n_executed_layers
    assert [call.layer_type for call in recorder.calls] == list("MMMEM*M*E")


def test_group_report_covers_only_looped_groups():
    model = _build_model("[M]^3E[M*]^2E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})
    report = recorder.group_report()

    assert [entry["composition"] for entry in report] == ["M", "M*"]
    assert [entry["num_loops"] for entry in report] == [3, 2]
    # K iterations give K step norms and K-1 consecutive cosines.
    assert [len(entry["relative_step_norm"]) for entry in report] == [3, 2]
    assert [len(entry["update_cosine"]) for entry in report] == [2, 1]


def test_group_input_norm_is_reported_so_the_step_ratio_can_be_read():
    # relative_step_norm divides by the group's input norm, and that denominator varies by an order
    # of magnitude with depth -- the first group sits on the raw embeddings. Without the absolute
    # norm, a ratio above 1 cannot be told apart from an ordinary update on a short vector.
    model = _build_model("[M]^3E[M*]^2E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    for group in recorder.group_report():
        assert group["group_input_norm"]["median"] > 0
    assert recorder.stack_report()["reference_norm"]["median"] > 0


def test_member_decomposition_is_present_only_for_multi_member_groups():
    model = _build_model("[M]^3E[M*]^2E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})
    report = recorder.group_report()

    assert report[0]["members"] is None
    members = report[1]["members"]
    assert [entry["layer_type"] for entry in members["per_member_relative_norm"]] == ["M", "*"]
    assert members["between_member_cosine"][0]["layer_types"] == ["M", "*"]


def test_member_deltas_sum_to_the_group_delta_as_vectors():
    # Norms do NOT sum -- the triangle inequality makes that inequality, not equality -- so the
    # decomposition has to be checked on the vectors. Asserting on scalar norms would fail whenever
    # the two members are not collinear, which is the interesting case.
    model = _build_model("E[M*]^2E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    group_calls = [call for call in recorder.calls if call.layer_type in ("M", "*")]
    mamba, attention = group_calls[0], group_calls[1]
    layer_input = mamba.layer_input
    # The two members run in sequence, so the group's update is the sum of their contributions.
    group_delta = attention.layer_input + attention.delta - layer_input

    torch.testing.assert_close(group_delta, mamba.delta + attention.delta)


def test_stack_report_covers_every_executed_layer_including_unlooped_models():
    model = _build_model("MEM*E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    assert recorder.group_report() == []
    stack = recorder.stack_report()
    assert stack["n_executed_layers"] == 5
    assert stack["executed_types"] == list("MEM*E")
    assert len(stack["update_cosine"]) == 4


# --------------------------------------------------------------------------------------------
# The metrics themselves
# --------------------------------------------------------------------------------------------


def test_cosine_is_one_for_identical_and_minus_one_for_opposed_updates():
    update = torch.randn(2, SEQUENCE_LENGTH, N_EMBD)
    torch.testing.assert_close(_per_token_cosine(update, update), torch.ones(2, SEQUENCE_LENGTH))
    torch.testing.assert_close(_per_token_cosine(update, -update), -torch.ones(2, SEQUENCE_LENGTH))


def test_cosine_is_computed_per_token_not_over_the_flattened_tensor():
    # One token is given a huge, aligned update and the rest are opposed. A flattened cosine would be
    # dominated by the large token and come out positive; the per-token median must stay at -1.
    first = torch.randn(1, 4, N_EMBD)
    second = -first.clone()
    first[0, 0] *= 1000.0
    second[0, 0] = first[0, 0]

    cosines = _per_token_cosine(first, second)
    assert cosines[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert cosines[0, 1:].median().item() == pytest.approx(-1.0, abs=1e-5)


def test_relative_step_norm_matches_a_hand_computed_ratio():
    reference = torch.full((1, 2, N_EMBD), 2.0)
    update = torch.full((1, 2, N_EMBD), 1.0)
    # Both vectors are constant, so the ratio is 1/2 regardless of n_embd.
    torch.testing.assert_close(_per_token_norm_ratio(update, reference), torch.full((1, 2), 0.5))


def test_metrics_are_accumulated_in_float32_under_a_lower_precision_forward():
    model = _build_model("[M]^3E").to(torch.bfloat16)
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    assert all(call.delta.dtype == torch.float32 for call in recorder.calls)
    entry = recorder.group_report()[0]["update_cosine"][0]
    assert -1.0 <= entry["median"] <= 1.0


def test_summary_reports_spread_alongside_the_mean():
    model = _build_model("[M]^3E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs(batch=4)})

    entry = recorder.group_report()[0]["update_cosine"][0]
    assert {"mean", "std", "p25", "median", "p75", "n_tokens"} <= set(entry)
    assert entry["p25"] <= entry["median"] <= entry["p75"]
    assert entry["n_tokens"] == 4 * SEQUENCE_LENGTH


def test_recorder_can_be_reused_after_reset():
    model = _build_model("[M]^3E")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})
        first = len(recorder.calls)
        recorder.reset()
        model({"input_ids": _inputs()})

    assert len(recorder.calls) == first


def test_composition_string_names_the_layer_types_of_the_group():
    model = _build_model("M[*E]^2M")
    model.eval()
    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        model({"input_ids": _inputs()})

    assert re.fullmatch(r"\*E", recorder.group_report()[0]["composition"])
