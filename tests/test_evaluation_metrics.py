"""Tests for the held-out evaluation metrics and their integration into the Evaluator.

The properties worth pinning down are the ones that would quietly produce a plausible-looking but
wrong number: that masked positions are excluded, that the reduction is exact rather than an
average of per-batch averages (a short final batch must not be over-weighted), and that a metric
only runs on the dataloaders it was pointed at.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch

from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.constants import IGNORE_INDEX
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluation_metrics import MaskedTokenAccuracy, MaskedTokenNLL
from modalities.evaluator import Evaluator

TARGET_KEY = "target_ids"
PREDICTION_KEY = "logits"
SAMPLE_KEY = "input_ids"
VOCAB_SIZE = 8


def _one_hot_logits(predicted_tokens: list[list[int]], magnitude: float = 10.0) -> torch.Tensor:
    """Logits that argmax to the given tokens, with a controllable confidence."""
    logits = torch.zeros(len(predicted_tokens), len(predicted_tokens[0]), VOCAB_SIZE)
    for row, tokens in enumerate(predicted_tokens):
        for column, token in enumerate(tokens):
            logits[row, column, token] = magnitude
    return logits


def _result_batch(predicted_tokens: list[list[int]], targets: list[list[int]]) -> InferenceResultBatch:
    return InferenceResultBatch(
        targets={TARGET_KEY: torch.tensor(targets)},
        predictions={PREDICTION_KEY: _one_hot_logits(predicted_tokens)},
    )


def test_accuracy_ignores_masked_positions():
    metric = MaskedTokenAccuracy(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="accuracy")

    # The sole unmasked position is wrong. Counting the masked positions -- where the prediction
    # happens to match the ignore-index placeholder's neighbours -- would inflate this to 2/3.
    result = metric(_result_batch(predicted_tokens=[[3, 3, 3]], targets=[[IGNORE_INDEX, IGNORE_INDEX, 5]]))
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))

    # Only the unmasked position is right, and it is the only one that counts.
    result = metric(_result_batch(predicted_tokens=[[3, 3, 5]], targets=[[IGNORE_INDEX, IGNORE_INDEX, 5]]))
    torch.testing.assert_close(result, torch.tensor([1.0, 1.0]))

    # Nothing masked: every position is scored.
    result = metric(_result_batch(predicted_tokens=[[3, 3, 3]], targets=[[3, 3, 5]]))
    torch.testing.assert_close(result, torch.tensor([2.0, 3.0]))


def test_accuracy_counts_every_unmasked_position_across_the_batch():
    metric = MaskedTokenAccuracy(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="accuracy")
    result = metric(
        _result_batch(
            predicted_tokens=[[0, 1], [0, 2], [0, 7]],
            targets=[[IGNORE_INDEX, 1], [IGNORE_INDEX, 3], [IGNORE_INDEX, 7]],
        )
    )
    torch.testing.assert_close(result, torch.tensor([2.0, 3.0]))


def test_nll_matches_cross_entropy_on_the_unmasked_positions():
    metric = MaskedTokenNLL(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="nll")
    logits = torch.randn(4, 5, VOCAB_SIZE)
    targets = torch.full((4, 5), IGNORE_INDEX)
    targets[:, -1] = torch.tensor([1, 2, 3, 4])

    result = metric(InferenceResultBatch(targets={TARGET_KEY: targets}, predictions={PREDICTION_KEY: logits}))
    expected = torch.nn.functional.cross_entropy(logits[:, -1, :], targets[:, -1], reduction="sum")
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[1], torch.tensor(4.0))


def test_nll_of_a_uniform_predictor_is_the_log_of_the_vocabulary_size():
    metric = MaskedTokenNLL(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="nll")
    result = metric(
        InferenceResultBatch(
            targets={TARGET_KEY: torch.tensor([[IGNORE_INDEX, 2]])},
            predictions={PREDICTION_KEY: torch.zeros(1, 2, VOCAB_SIZE)},
        )
    )
    assert result[0].item() == pytest.approx(math.log(VOCAB_SIZE))


def test_metric_applies_only_to_the_dataloaders_it_names():
    metric = MaskedTokenAccuracy(
        target_key=TARGET_KEY,
        prediction_key=PREDICTION_KEY,
        tag="accuracy",
        dataloader_tags=["p_hop_2", "p_hop_3"],
    )
    assert metric.applies_to("p_hop_2")
    assert not metric.applies_to("test")

    # An unset filter means "every dataloader", which is what a metric on a single-dataloader
    # config wants.
    unfiltered = MaskedTokenAccuracy(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="accuracy")
    assert unfiltered.applies_to("anything")


def _dataset_batch(targets: list[list[int]]) -> DatasetBatch:
    target_tensor = torch.tensor(targets)
    return DatasetBatch(
        samples={SAMPLE_KEY: torch.zeros_like(target_tensor)},
        targets={TARGET_KEY: target_tensor},
    )


def test_evaluator_reduction_is_exact_across_unequal_batches(loss_mock, progress_publisher_mock, set_env_cpu):
    # Three answers in the first batch, one in the second. Two of the four are correct, so the
    # metric must be 0.5 -- averaging the per-batch accuracies (2/3 and 0/1) would give 0.333.
    batches = [
        _dataset_batch([[IGNORE_INDEX, 1], [IGNORE_INDEX, 2], [IGNORE_INDEX, 3]]),
        _dataset_batch([[IGNORE_INDEX, 4]]),
    ]
    predictions = iter(
        [
            _one_hot_logits([[0, 1], [0, 2], [0, 6]]),
            _one_hot_logits([[0, 5]]),
        ]
    )
    model_mock = MagicMock(side_effect=lambda _: {PREDICTION_KEY: next(predictions)})

    data_loader_mock = MagicMock(spec=LLMDataLoader)
    data_loader_mock.__iter__ = lambda _: iter(batches)
    data_loader_mock.dataloader_tag = "p_hop_2"

    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        metrics=[
            MaskedTokenAccuracy(
                target_key=TARGET_KEY,
                prediction_key=PREDICTION_KEY,
                tag="answer_accuracy",
                dataloader_tags=["p_hop_2"],
            )
        ],
    )
    results = evaluator.evaluate(
        model=model_mock, data_loaders=[data_loader_mock], loss_fun=loss_mock, num_train_steps_done=1
    )

    assert results["p_hop_2"].metrics["answer_accuracy"].value.item() == pytest.approx(0.5)


def test_evaluator_skips_metrics_on_dataloaders_they_do_not_name(loss_mock, progress_publisher_mock, set_env_cpu):
    batches = [_dataset_batch([[IGNORE_INDEX, 1]])]
    model_mock = MagicMock(return_value={PREDICTION_KEY: _one_hot_logits([[0, 1]])})

    data_loader_mock = MagicMock(spec=LLMDataLoader)
    data_loader_mock.__iter__ = lambda _: iter(batches)
    data_loader_mock.dataloader_tag = "test"

    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        metrics=[
            MaskedTokenAccuracy(
                target_key=TARGET_KEY,
                prediction_key=PREDICTION_KEY,
                tag="answer_accuracy",
                dataloader_tags=["p_hop_2"],
            )
        ],
    )
    results = evaluator.evaluate(
        model=model_mock, data_loaders=[data_loader_mock], loss_fun=loss_mock, num_train_steps_done=1
    )

    assert results["test"].metrics == {}


def test_evaluator_without_metrics_reports_none(loss_mock, progress_publisher_mock, set_env_cpu):
    # Every pre-existing config constructs the Evaluator without metrics; that path must be unchanged.
    batches = [_dataset_batch([[1, 2]])]
    model_mock = MagicMock(return_value={PREDICTION_KEY: _one_hot_logits([[1, 2]])})

    data_loader_mock = MagicMock(spec=LLMDataLoader)
    data_loader_mock.__iter__ = lambda _: iter(batches)
    data_loader_mock.dataloader_tag = "test"

    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )
    results = evaluator.evaluate(
        model=model_mock, data_loaders=[data_loader_mock], loss_fun=loss_mock, num_train_steps_done=1
    )

    assert results["test"].metrics == {}
    assert results["test"].losses != {}
