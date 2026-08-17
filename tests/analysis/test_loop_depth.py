"""Tests for the loop-depth override and its paired loss comparison.

The measurement is a difference between two forward passes, so the properties that matter are that
the override changes exactly what it claims to, that it always restores the schedule, and that the
paired statistic is genuinely paired. An override that leaked into the next measurement, or a
"baseline" that quietly differed from the unmodified model, would produce deltas that look fine.
"""

import pytest
import torch

from modalities.analysis.loop_depth import (
    group_executed_positions,
    looped_group_indices,
    override_loop_counts,
    paired_delta,
    per_token_losses,
)
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
            n_embd=N_EMBD, n_head_q=4, n_head_kv=2, head_dim=32, norm_config=NORM_CONFIG
        ),
    }


def _build_model(layer_pattern: str, loop_config: LoopConfig = None) -> NemotronLLM:
    torch.manual_seed(0)
    model = NemotronLLM(
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
    return model.eval()


def _samples(n: int = 4) -> torch.Tensor:
    torch.manual_seed(99)
    return torch.randint(0, VOCAB_SIZE, (n, SEQUENCE_LENGTH + 1))


# --------------------------------------------------------------------------------------------
# The override changes exactly what it claims to
# --------------------------------------------------------------------------------------------


def test_overriding_to_the_native_count_reproduces_the_unmodified_forward():
    # The baseline setting is evaluated through the same override machinery as the ablations, so if
    # this identity failed every reported delta would carry a constant, invisible offset.
    model = _build_model("[M]^3E[M*]^2E")
    inputs = _samples()[:, :SEQUENCE_LENGTH]

    with torch.no_grad():
        unmodified = model({"input_ids": inputs})["logits"]
        with override_loop_counts(model, num_loops=None):
            passthrough = model({"input_ids": inputs})["logits"]

    torch.testing.assert_close(passthrough, unmodified)


def test_per_group_override_touches_only_the_named_group():
    model = _build_model("[M]^3E[M*]^2E")
    looped = looped_group_indices(model)
    original = [group.num_loops for group in model._schedule]

    with override_loop_counts(model, num_loops=1, group_index=looped[0]) as schedule:
        counts = [group.num_loops for group in schedule]

    assert counts[looped[0]] == 1
    assert counts[looped[1]] == original[looped[1]]
    assert [group.num_loops for group in model._schedule] == original


def test_global_override_leaves_unlooped_groups_alone():
    # Forcing a never-looped layer to repeat would evaluate an architecture that was never trained,
    # which is a different experiment from the one this module claims to run.
    model = _build_model("[M]^3E[M*]^2E")
    with override_loop_counts(model, num_loops=4) as schedule:
        counts = {index: group.num_loops for index, group in enumerate(schedule)}
    looped = set(looped_group_indices(model))

    assert all(counts[index] == 4 for index in looped)
    assert all(counts[index] == 1 for index in counts if index not in looped)


def test_override_changes_the_executed_depth():
    model = _build_model("[M]^4E")
    inputs = _samples()[:, :SEQUENCE_LENGTH]
    with torch.no_grad():
        deep = model({"input_ids": inputs})["logits"]
        with override_loop_counts(model, num_loops=1):
            shallow = model({"input_ids": inputs})["logits"]

    assert not torch.allclose(deep, shallow), "override did not change the computation"


def test_schedule_is_restored_after_an_exception():
    model = _build_model("[M]^3E")
    original = list(model._schedule)
    with pytest.raises(RuntimeError):
        with override_loop_counts(model, num_loops=1):
            raise RuntimeError("boom")

    assert model._schedule == original


@pytest.mark.parametrize("bad_kwargs", [{"num_loops": 0}, {"num_loops": -1}, {"num_loops": 2, "group_index": 99}])
def test_invalid_overrides_are_rejected(bad_kwargs):
    model = _build_model("[M]^3E")
    with pytest.raises(ValueError):
        with override_loop_counts(model, **bad_kwargs):
            pass


# --------------------------------------------------------------------------------------------
# Positions and bookkeeping
# --------------------------------------------------------------------------------------------


def test_executed_positions_account_for_earlier_loops():
    # "[M]^3 E [M*]^2 E": the second looped group starts at executed index 4, not at its schedule
    # index of 2 -- indexing the results by schedule index would mislabel every depth.
    model = _build_model("[M]^3E[M*]^2E")
    positions = group_executed_positions(model)

    assert positions == {0: 0, 1: 3, 2: 4, 3: 8}
    assert looped_group_indices(model) == [0, 2]


def test_looped_group_indices_is_empty_for_a_model_without_loops():
    model = _build_model("MEM*E")
    assert looped_group_indices(model) == []


# --------------------------------------------------------------------------------------------
# The paired statistic
# --------------------------------------------------------------------------------------------


def test_per_token_losses_have_one_entry_per_predicted_token():
    model = _build_model("[M]^3E")
    samples = _samples(n=6)
    losses = per_token_losses(model, samples, micro_batch_size=4)

    assert losses.shape == (6, SEQUENCE_LENGTH)
    assert losses.dtype == torch.float32
    assert torch.isfinite(losses).all()


def test_micro_batching_does_not_change_the_losses():
    model = _build_model("[M]^3E")
    samples = _samples(n=6)

    torch.testing.assert_close(
        per_token_losses(model, samples, micro_batch_size=2),
        per_token_losses(model, samples, micro_batch_size=6),
    )


def test_paired_delta_of_a_setting_against_itself_is_exactly_zero():
    model = _build_model("[M]^3E")
    losses = per_token_losses(model, _samples())
    result = paired_delta(losses, losses)

    assert result["delta"] == 0.0
    assert result["standard_error"] == 0.0


def test_paired_delta_reports_the_mean_increase_and_its_sequence_level_error():
    baseline = torch.zeros(4, SEQUENCE_LENGTH)
    ablated = torch.arange(4, dtype=torch.float32).view(4, 1).expand(4, SEQUENCE_LENGTH).contiguous()
    result = paired_delta(baseline, ablated)

    assert result["delta"] == pytest.approx(1.5)
    # Sequence means are 0,1,2,3: sample sd is sqrt(5/3), so the standard error is that over sqrt(4).
    assert result["standard_error"] == pytest.approx((5 / 3) ** 0.5 / 2, rel=1e-5)
    assert result["n_sequences"] == 4
    assert result["n_tokens"] == 4 * SEQUENCE_LENGTH


def test_paired_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_delta(torch.zeros(4, 8), torch.zeros(3, 8))


def test_ablating_a_group_increases_loss_on_a_trained_toy_model():
    # A randomly initialized model has no reason to prefer more depth, so the model is briefly
    # trained on a fixed batch first. This checks the sign of the whole pipeline end to end.
    model = _build_model("[M]^3E[ME]^2*E")
    samples = _samples(n=4)
    inputs, targets = samples[:, :SEQUENCE_LENGTH], samples[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        logits = model({"input_ids": inputs})["logits"]
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
        optimizer.step()
    model.eval()

    baseline = per_token_losses(model, samples)
    with override_loop_counts(model, num_loops=1):
        ablated = per_token_losses(model, samples)

    assert paired_delta(baseline, ablated)["delta"] > 0
