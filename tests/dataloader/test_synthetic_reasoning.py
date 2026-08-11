"""Tests for the synthetic reasoning evaluation datasets.

The point of these datasets is to be a *measuring instrument* for the layer-loop ablation, so the
properties that matter are the ones that would silently corrupt a comparison between arms rather
than raise: that every arm is asked the same questions (determinism), that the answer is genuinely
the p-hop answer (correctness against an independent reference), that only the answer position
contributes to the loss (masking), and that the chance floor is what the docs claim it is.
"""

import numpy as np
import pytest
import torch

from modalities.dataloader.collate_fns.explicit_target_collator import ExplicitTargetCollateFn
from modalities.dataloader.synthetic_reasoning import (
    IGNORE_INDEX,
    SyntheticReasoningDataset,
    SyntheticReasoningTask,
    resolve_p_hop,
)

SAMPLE_KEY = "input_ids"
TARGET_KEY = "target_ids"
# A 26-symbol alphabet, standing in for the capital letters a real config would use.
ALPHABET = list(range(1000, 1026))
DELIMITERS = [900, 901]


def _p_hop_dataset(**overrides) -> SyntheticReasoningDataset:
    kwargs = dict(
        task=SyntheticReasoningTask.P_HOP_INDUCTION,
        num_samples=16,
        num_hops=2,
        symbol_token_ids=ALPHABET,
        sample_key=SAMPLE_KEY,
        target_key=TARGET_KEY,
        seed=1234,
        prompt_length=256,
    )
    kwargs.update(overrides)
    return SyntheticReasoningDataset(**kwargs)


def _variable_binding_dataset(**overrides) -> SyntheticReasoningDataset:
    kwargs = dict(
        task=SyntheticReasoningTask.VARIABLE_BINDING,
        num_samples=16,
        num_hops=3,
        symbol_token_ids=ALPHABET,
        sample_key=SAMPLE_KEY,
        target_key=TARGET_KEY,
        seed=1234,
        num_distractors=6,
        delimiter_token_ids=DELIMITERS,
    )
    kwargs.update(overrides)
    return SyntheticReasoningDataset(**kwargs)


def _resolve_p_hop_by_brute_force(symbols: list[int], num_hops: int) -> int:
    """A deliberately naive reference implementation, written from the task description alone."""
    position = len(symbols) - 1
    query = symbols[position]
    for _ in range(num_hops):
        source = None
        for candidate in range(position - 1, -1, -1):
            if symbols[candidate] == query:
                source = candidate
                break
        assert source is not None, "the dataset must not emit a sample with an undefined hop"
        query = symbols[source + 1]
        position = source
    return query


@pytest.mark.parametrize("num_hops", [1, 2, 3, 4])
def test_p_hop_answers_match_an_independent_reference(num_hops: int):
    dataset = _p_hop_dataset(num_hops=num_hops, num_samples=32)
    symbol_of_token = {token_id: index for index, token_id in enumerate(ALPHABET)}

    for index in range(len(dataset)):
        sample = dataset[index]
        symbols = [symbol_of_token[token_id] for token_id in sample[SAMPLE_KEY].tolist()]
        expected_symbol = _resolve_p_hop_by_brute_force(symbols, num_hops)
        assert sample[TARGET_KEY][-1] == ALPHABET[expected_symbol]


def test_p_hop_never_answers_with_the_final_prompt_token():
    # A model that has only learned "repeat the last token" must score at chance, not above it.
    dataset = _p_hop_dataset(num_samples=64, num_hops=2)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample[TARGET_KEY][-1] != sample[SAMPLE_KEY][-1]


def test_p_hop_answers_are_close_to_uniform_over_the_alphabet():
    # Chance accuracy is documented as 1 / alphabet size, which only holds if answers are uniform.
    dataset = _p_hop_dataset(num_samples=2000, num_hops=2)
    answers = [dataset[index][TARGET_KEY][-1] for index in range(len(dataset))]
    counts = np.bincount(np.searchsorted(ALPHABET, answers), minlength=len(ALPHABET))
    expected = len(dataset) / len(ALPHABET)
    # Chi-square for 25 degrees of freedom stays below ~52 at p=0.001; the fixed seed makes this
    # test deterministic, so this is a guard against a skewed construction, not a flaky assertion.
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    assert chi_square < 52.0
    assert dataset.chance_accuracy == pytest.approx(1 / len(ALPHABET))


@pytest.mark.parametrize("dataset_builder", [_p_hop_dataset, _variable_binding_dataset])
def test_only_the_final_position_carries_a_target(dataset_builder):
    dataset = dataset_builder()
    for index in range(len(dataset)):
        sample = dataset[index]
        targets = sample[TARGET_KEY]
        assert len(targets) == len(sample[SAMPLE_KEY])
        assert (targets[:-1] == IGNORE_INDEX).all()
        assert targets[-1] != IGNORE_INDEX


@pytest.mark.parametrize("dataset_builder", [_p_hop_dataset, _variable_binding_dataset])
def test_generation_is_deterministic_in_the_seed(dataset_builder):
    # Arms are compared on their answers, so two arms must be asked literally the same questions.
    first = dataset_builder()
    second = dataset_builder()
    different_seed = dataset_builder(seed=4321)
    for index in range(len(first)):
        assert (first[index][SAMPLE_KEY] == second[index][SAMPLE_KEY]).all()
        assert (first[index][TARGET_KEY] == second[index][TARGET_KEY]).all()
    assert not (first[0][SAMPLE_KEY] == different_seed[0][SAMPLE_KEY]).all()


@pytest.mark.parametrize("num_hops", [1, 2, 3])
def test_variable_binding_answer_is_the_literal_the_chain_bottoms_out_in(num_hops: int):
    dataset = _variable_binding_dataset(num_hops=num_hops, num_samples=32, num_distractors=4)
    separator_token_id, assignment_token_id = DELIMITERS

    for index in range(len(dataset)):
        sample = dataset[index]
        token_ids = sample[SAMPLE_KEY].tolist()
        # The prompt is a sequence of (separator, variable, assignment, value) statements followed
        # by a (separator, variable, assignment) query.
        statements = {token_ids[offset + 1]: token_ids[offset + 3] for offset in range(0, len(token_ids) - 3, 4)}
        assert token_ids[-3] == separator_token_id and token_ids[-1] == assignment_token_id
        value = token_ids[-2]
        for _ in range(num_hops):
            value = statements[value]
        assert value not in statements, "the chain must bottom out in a literal, not another variable"
        assert sample[TARGET_KEY][-1] == value


def test_variable_binding_shortcut_floor_is_as_documented():
    # The answer is always a symbol that appears as a value but never as a variable, so a model
    # that has learned only the format can guess among num_distractors + 1 candidates. That floor
    # is documented and reported by `format_aware_chance_accuracy`; this pins the two together, so
    # a change to the generator that widens or narrows the shortcut cannot go unnoticed.
    num_distractors = 6
    dataset = _variable_binding_dataset(num_hops=3, num_samples=64, num_distractors=num_distractors)

    for index in range(len(dataset)):
        token_ids = dataset[index][SAMPLE_KEY].tolist()
        offsets = range(0, len(token_ids) - 3, 4)
        variables = {token_ids[offset + 1] for offset in offsets}
        values = {token_ids[offset + 3] for offset in offsets}
        assert len(values - variables) == num_distractors + 1
        assert dataset[index][TARGET_KEY][-1] in values - variables

    assert dataset.format_aware_chance_accuracy == pytest.approx(1 / (num_distractors + 1))
    # p-hop admits every symbol of the alphabet as an answer, so it has no such shortcut.
    assert _p_hop_dataset().format_aware_chance_accuracy == pytest.approx(1 / len(ALPHABET))


def test_variable_binding_statement_order_carries_no_signal():
    # If the chain were emitted in resolution order, a recency heuristic would solve the task.
    dataset = _variable_binding_dataset(num_hops=3, num_samples=64, num_distractors=6)
    query_variable_positions = []
    for index in range(len(dataset)):
        token_ids = dataset[index][SAMPLE_KEY].tolist()
        queried = token_ids[-2]
        variables = [token_ids[offset + 1] for offset in range(0, len(token_ids) - 3, 4)]
        query_variable_positions.append(variables.index(queried))
    assert len(set(query_variable_positions)) > 1


def test_infeasible_task_parameters_are_rejected_at_construction():
    # These must fail while the config is being built, not thousands of steps into a run.
    with pytest.raises(ValueError, match="too short"):
        _p_hop_dataset(num_hops=4, prompt_length=64)
    with pytest.raises(ValueError, match="distinct symbols"):
        _variable_binding_dataset(num_hops=3, num_distractors=20)
    # A repeated symbol would give some questions two correct answers.
    with pytest.raises(ValueError, match="must be distinct"):
        _p_hop_dataset(symbol_token_ids=ALPHABET[:-1] + [ALPHABET[0]])


def test_resolve_p_hop_returns_none_when_a_hop_has_nowhere_to_jump():
    assert resolve_p_hop(np.array([0, 1, 2, 3]), num_hops=1) is None
    # 'A B A' -> the query A occurs at index 0, so one hop answers B; a second hop finds no
    # earlier B and the sequence poses no well-formed two-hop question.
    assert resolve_p_hop(np.array([0, 1, 0]), num_hops=1) == 1
    assert resolve_p_hop(np.array([0, 1, 0]), num_hops=2) is None


def test_collator_stacks_samples_without_shifting():
    dataset = _p_hop_dataset(num_samples=4)
    collator = ExplicitTargetCollateFn(sample_key=SAMPLE_KEY, target_key=TARGET_KEY)
    batch = collator([dataset[index] for index in range(4)])

    samples = batch.samples[SAMPLE_KEY]
    targets = batch.targets[TARGET_KEY]
    assert samples.shape == targets.shape == (4, 256)
    for index in range(4):
        assert torch.equal(samples[index], torch.as_tensor(dataset[index][SAMPLE_KEY]))
        assert torch.equal(targets[index], torch.as_tensor(dataset[index][TARGET_KEY]))
