"""Tests for the pre-tokenized held-out evaluations (Minerva MATH, TriviaQA).

The prepared file is produced offline, so the failure this guards against is a *silently* wrong
file: shapes that disagree, a masking that scored nothing, or a path that does not exist yet. Each
of those would otherwise surface as a NaN or a plausible-looking number partway into a run.
"""

import json

import numpy as np
import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.constants import IGNORE_INDEX
from modalities.dataloader.collate_fns.explicit_target_collator import ExplicitTargetCollateFn
from modalities.dataloader.prepared_eval import PreparedEvalDataset
from modalities.evaluation_metrics import MaskedTokenNLL

SAMPLE_KEY = "input_ids"
TARGET_KEY = "target_ids"
PREDICTION_KEY = "logits"


def _write_prepared(path, inputs, targets, metadata=None):
    arrays = {"input_ids": np.asarray(inputs), "target_ids": np.asarray(targets)}
    if metadata is not None:
        arrays["metadata"] = json.dumps(metadata)
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture
def prepared_path(tmp_path):
    # Two problems: a two-token prompt followed by a two-token scored answer.
    inputs = [[5, 6, 7, 8], [5, 6, 9, 10]]
    targets = [[IGNORE_INDEX, 7, 8, IGNORE_INDEX], [IGNORE_INDEX, 9, 10, IGNORE_INDEX]]
    return _write_prepared(tmp_path / "prepared.npz", inputs, targets, {"benchmark": "unit_test"})


def test_loads_inputs_targets_and_metadata(prepared_path):
    dataset = PreparedEvalDataset(prepared_path=prepared_path, sample_key=SAMPLE_KEY, target_key=TARGET_KEY)

    assert len(dataset) == 2
    assert dataset.metadata["benchmark"] == "unit_test"
    assert dataset.num_scored_tokens == 4
    assert dataset.mean_scored_tokens_per_problem == pytest.approx(2.0)
    sample = dataset[0]
    assert sample[SAMPLE_KEY].tolist() == [5, 6, 7, 8]
    assert sample[TARGET_KEY].tolist() == [IGNORE_INDEX, 7, 8, IGNORE_INDEX]


def test_num_samples_truncates(prepared_path):
    dataset = PreparedEvalDataset(
        prepared_path=prepared_path, sample_key=SAMPLE_KEY, target_key=TARGET_KEY, num_samples=1
    )
    assert len(dataset) == 1
    assert dataset.num_scored_tokens == 2


def test_missing_file_names_the_script_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_text_evals.py"):
        PreparedEvalDataset(prepared_path=tmp_path / "absent.npz", sample_key=SAMPLE_KEY, target_key=TARGET_KEY)


def test_mismatched_shapes_are_rejected(tmp_path):
    path = _write_prepared(tmp_path / "bad.npz", [[1, 2, 3]], [[IGNORE_INDEX, 2]])
    with pytest.raises(ValueError, match="they must match"):
        PreparedEvalDataset(prepared_path=path, sample_key=SAMPLE_KEY, target_key=TARGET_KEY)


def test_a_fully_masked_file_is_rejected_rather_than_reporting_nan(tmp_path):
    # Every metric over this would divide by zero; failing at construction is the honest outcome.
    path = _write_prepared(tmp_path / "empty.npz", [[1, 2]], [[IGNORE_INDEX, IGNORE_INDEX]])
    with pytest.raises(ValueError, match="no scored position"):
        PreparedEvalDataset(prepared_path=path, sample_key=SAMPLE_KEY, target_key=TARGET_KEY)


def test_padding_positions_do_not_reach_the_metric(tmp_path):
    # Problems have different lengths, so the shorter one is right-padded. The pad positions must
    # contribute nothing: if they leaked in, the reported NLL would be diluted by whatever the
    # model happens to predict for padding.
    path = _write_prepared(
        tmp_path / "padded.npz",
        [[5, 6, 7, 8], [5, 9, 0, 0]],
        [[IGNORE_INDEX, 7, 8, IGNORE_INDEX], [9, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]],
    )
    dataset = PreparedEvalDataset(prepared_path=path, sample_key=SAMPLE_KEY, target_key=TARGET_KEY)
    collator = ExplicitTargetCollateFn(sample_key=SAMPLE_KEY, target_key=TARGET_KEY)
    batch = collator([dataset[0], dataset[1]])

    vocab_size = 16
    metric = MaskedTokenNLL(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, tag="answer_nll")
    result = metric(
        InferenceResultBatch(
            targets=batch.targets,
            predictions={PREDICTION_KEY: torch.zeros(2, 4, vocab_size)},
        )
    )
    # Three scored positions, not the eight the padded tensor contains.
    assert result[1].item() == 3.0
    assert result[0].item() == pytest.approx(3 * np.log(vocab_size))
