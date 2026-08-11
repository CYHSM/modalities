"""End-to-end check that the synthetic reasoning evaluation works against a real Nemotron model.

The unit tests cover the dataset and the metrics in isolation. What they cannot catch is a wiring
mistake between them -- targets shifted by one, the metric reading the wrong position, a looped
model's repeated MoE visits interfering with the evaluation forward pass. Those would all produce a
number, and a plausible one, so this test runs the whole path on a small model and pins the values
that a randomly initialized model must produce.
"""

import math

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

from modalities.dataloader.collate_fns.explicit_target_collator import ExplicitTargetCollateFn
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.synthetic_reasoning import SyntheticReasoningDataset, SyntheticReasoningTask
from modalities.evaluation_metrics import MaskedTokenAccuracy, MaskedTokenNLL
from modalities.evaluator import Evaluator
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.components.moe.experts import ExpertsBackend
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.layer_pattern import get_num_built_layers
from modalities.models.nemotron.nemotron_attention import NemotronAttentionImplementation
from modalities.models.nemotron.nemotron_layer_specs import (
    Mamba2LayerSpec,
    NemotronAttentionLayerSpec,
    NemotronMoELayerSpec,
)
from modalities.models.nemotron.nemotron_model import NemotronLLM

N_EMBD = 64
VOCAB_SIZE = 128
SEQUENCE_LENGTH = 64
SAMPLE_KEY = "input_ids"
TARGET_KEY = "target_ids"
PREDICTION_KEY = "logits"
NORM_CONFIG = {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": N_EMBD, "eps": 1e-5}}
# Token ids well inside the vocabulary, standing in for the ' A'..' Z' ids a real config uses.
ALPHABET = list(range(10, 26))
DELIMITERS = [5, 6]


def _layer_specs() -> dict:
    return {
        "M": Mamba2LayerSpec(
            n_embd=N_EMBD,
            mamba_n_heads=4,
            mamba_head_dim=16,
            mamba_state_dim=8,
            mamba_n_groups=2,
            chunk_size=8,
            norm_config=NORM_CONFIG,
        ),
        "E": NemotronMoELayerSpec(
            n_embd=N_EMBD,
            num_experts=4,
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
            head_dim=16,
            attention_implementation=NemotronAttentionImplementation.MANUAL,
            norm_config=NORM_CONFIG,
        ),
    }


def _build_model(layer_pattern: str) -> NemotronLLM:
    torch.manual_seed(0)
    return NemotronLLM(
        sample_key=SAMPLE_KEY,
        prediction_key=PREDICTION_KEY,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=get_num_built_layers(layer_pattern),
        layer_pattern=layer_pattern,
        layer_specs=_layer_specs(),
        lm_head_norm_config=NormWrapperConfig.model_validate(NORM_CONFIG),
    )


def _dataloader(dataset: SyntheticReasoningDataset, tag: str, batch_size: int = 8) -> LLMDataLoader:
    return LLMDataLoader(
        dataloader_tag=tag,
        dataset=dataset,
        batch_sampler=BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=True),
        collate_fn=ExplicitTargetCollateFn(sample_key=SAMPLE_KEY, target_key=TARGET_KEY),
    )


def _metrics(dataloader_tag: str) -> list:
    return [
        MaskedTokenAccuracy(
            target_key=TARGET_KEY,
            prediction_key=PREDICTION_KEY,
            tag="answer_accuracy",
            dataloader_tags=[dataloader_tag],
        ),
        MaskedTokenNLL(
            target_key=TARGET_KEY,
            prediction_key=PREDICTION_KEY,
            tag="answer_nll",
            dataloader_tags=[dataloader_tag],
        ),
    ]


@pytest.mark.parametrize(
    "layer_pattern",
    [
        "MEM*E",
        # A looped arm: the MoE layers are visited twice per forward pass, which is the case where
        # per-visit state (the auxiliary loss accumulator) could leak into an evaluation.
        "M[E]^2M*E",
    ],
)
def test_p_hop_evaluation_of_an_untrained_model_sits_at_chance(layer_pattern, progress_publisher_mock, set_env_cpu):
    dataset = SyntheticReasoningDataset(
        task=SyntheticReasoningTask.P_HOP_INDUCTION,
        num_samples=64,
        num_hops=2,
        symbol_token_ids=ALPHABET,
        sample_key=SAMPLE_KEY,
        target_key=TARGET_KEY,
        seed=7,
        prompt_length=SEQUENCE_LENGTH,
    )
    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        metrics=_metrics("p_hop_2"),
    )

    results = evaluator.evaluate(
        model=_build_model(layer_pattern),
        data_loaders=[_dataloader(dataset, tag="p_hop_2")],
        loss_fun=CLMCrossEntropyLoss(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY),
        num_train_steps_done=1,
    )
    metrics = results["p_hop_2"].metrics

    # An untrained model has no reason to prefer the answer, so accuracy must be near zero rather
    # than at the ~50% a metric reading the wrong axis or an unmasked position could report.
    assert 0.0 <= metrics["answer_accuracy"].value.item() <= 0.2
    # A random model spreads its mass over the whole vocabulary, not over the 16-symbol alphabet.
    assert metrics["answer_nll"].value.item() == pytest.approx(math.log(VOCAB_SIZE), rel=0.25)


def test_the_reported_loss_is_the_answer_nll_when_the_objective_is_plain_cross_entropy(
    progress_publisher_mock, set_env_cpu
):
    # Masking every non-answer position is what makes the ordinary CLM loss report the answer's
    # negative log-likelihood, so the two must agree. If they ever diverge, the masking is broken.
    dataset = SyntheticReasoningDataset(
        task=SyntheticReasoningTask.VARIABLE_BINDING,
        num_samples=32,
        num_hops=2,
        symbol_token_ids=ALPHABET,
        sample_key=SAMPLE_KEY,
        target_key=TARGET_KEY,
        seed=7,
        num_distractors=4,
        delimiter_token_ids=DELIMITERS,
    )
    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        metrics=_metrics("bind_2"),
    )

    results = evaluator.evaluate(
        model=_build_model("MEM*E"),
        data_loaders=[_dataloader(dataset, tag="bind_2")],
        loss_fun=CLMCrossEntropyLoss(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY),
        num_train_steps_done=1,
    )

    reported_loss = results["bind_2"].losses["CLMCrossEntropyLoss"].value.item()
    answer_nll = results["bind_2"].metrics["answer_nll"].value.item()
    assert reported_loss == pytest.approx(answer_nll, rel=1e-4)


def test_evaluation_does_not_disturb_training_state(progress_publisher_mock, set_env_cpu):
    # The evaluator flips the model into eval mode; a looped model additionally resets its MoE
    # auxiliary-loss accumulators on every forward. Neither may leave the model in a state that
    # changes the next training step's output.
    model = _build_model("M[E]^2M*E")
    inputs = {SAMPLE_KEY: torch.randint(0, VOCAB_SIZE, (2, 16))}
    before = model(inputs)[PREDICTION_KEY]

    dataset = SyntheticReasoningDataset(
        task=SyntheticReasoningTask.P_HOP_INDUCTION,
        num_samples=16,
        num_hops=1,
        symbol_token_ids=ALPHABET,
        sample_key=SAMPLE_KEY,
        target_key=TARGET_KEY,
        seed=7,
        prompt_length=SEQUENCE_LENGTH,
    )
    Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        metrics=_metrics("p_hop_1"),
    ).evaluate(
        model=model,
        data_loaders=[_dataloader(dataset, tag="p_hop_1")],
        loss_fun=CLMCrossEntropyLoss(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY),
        num_train_steps_done=1,
    )

    after = model(inputs)[PREDICTION_KEY]
    torch.testing.assert_close(before, after)
