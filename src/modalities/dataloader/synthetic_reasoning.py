"""Deterministic synthetic reasoning tasks for held-out evaluation.

These datasets exist because perplexity is the wrong instrument for measuring what a looped model
buys. The published results are consistent: depth-wise weight sharing *loses* on validation
perplexity under iso-FLOP comparison and *wins* on reasoning, so a sweep that logs only training
loss systematically under-reports the effect it is trying to measure. See
``docs/components/nemotron_loops_research_plan.md`` section 6.

Two task families are provided, both parameterized by a **hop count**, which is the number of
serial lookups the answer requires. That parameterization is the whole point: ``num_hops: 1`` is a
single associative recall that needs no depth and acts as the control, while ``num_hops: 3`` cannot
be resolved in fewer than three serial steps. Comparing arms across the hop ladder separates "this
arm is better at everything" from "this arm is better at things that need depth".

``p_hop_induction``
    A random sequence of symbols. The last symbol is the query; find its most recent earlier
    occurrence and read off the symbol that follows it, then repeat with that symbol as the new
    query, searching only further to the left. One hop is exactly the induction-head operation.

``variable_binding``
    A shuffled list of ``<var> = <value>`` statements in which a chain of assignments dereferences
    to one literal, plus distractor statements, followed by a query for the end of the chain.
    Because the statements are shuffled, recency heuristics do not help; the chain has to be
    followed by content.

A sample is emitted pre-shifted and masked: ``target_key`` is
:data:`~modalities.constants.IGNORE_INDEX` everywhere
except the final position, which holds the answer token. ``CrossEntropyLoss`` ignores
:data:`IGNORE_INDEX`, so the loss a plain ``clm_cross_entropy_loss`` reports on one of these
dataloaders is the negative log-likelihood of the answer alone. Use
:mod:`modalities.evaluation_metrics` to additionally log answer accuracy, which is what the
literature reports and what is comparable to published numbers.

Symbols are given as **token ids**, not as strings, so that evaluation has no runtime tokenizer
dependency and the exact sequences are reproducible from the config alone. Pick ids that the
tokenizer emits as single tokens (for a Llama-3 tokenizer, the capital letters with a leading
space are a convenient alphabet) -- see ``docs/components/nemotron_loops.md`` for how to derive
them once.
"""

from enum import Enum
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from modalities.constants import IGNORE_INDEX
from modalities.dataloader.dataset import Dataset

# Random sequences are rejected and redrawn when a hop is undefined or when the answer would be
# reachable by copying. Both are rare, so this bound is generous; exceeding it means the task
# parameters are infeasible (e.g. a prompt too short to contain the requested number of hops)
# rather than that the draw was unlucky.
_MAX_REJECTION_ATTEMPTS = 1000


class SyntheticReasoningTask(str, Enum):
    """
    The synthetic reasoning tasks that :class:`SyntheticReasoningDataset` can generate.

    Attributes:
        P_HOP_INDUCTION (str): Chained induction over a random symbol sequence.
        VARIABLE_BINDING (str): Chained assignment resolution over shuffled statements.
    """

    P_HOP_INDUCTION = "p_hop_induction"
    VARIABLE_BINDING = "variable_binding"


class SyntheticReasoningDatasetConfig(BaseModel):
    """
    Configuration of a :class:`SyntheticReasoningDataset`.

    Attributes:
        task (SyntheticReasoningTask): Which task to generate.
        num_samples (int): Number of samples in the dataset.
        num_hops (int): Number of serial lookups the answer requires. 1 is the depth-free control.
        symbol_token_ids (list[int]): The alphabet, as token ids. Must be distinct. Chance accuracy
            is one over its length, so a longer alphabet gives a lower floor and more headroom.
        sample_key (str): Key under which the input token ids are emitted.
        target_key (str): Key under which the masked targets are emitted.
        seed (int): Seed for generation. Every arm must use the same seed, otherwise arms are
            compared on different questions.
        prompt_length (int): ``p_hop_induction`` only: number of symbols in the sequence.
        num_distractors (int): ``variable_binding`` only: number of assignment statements that are
            not part of the chain.
        delimiter_token_ids (list[int] | None): ``variable_binding`` only: exactly two token ids,
            used as the statement separator and the assignment operator, in that order.
    """

    task: SyntheticReasoningTask
    num_samples: Annotated[int, Field(strict=True, ge=1)]
    num_hops: Annotated[int, Field(strict=True, ge=1)]
    symbol_token_ids: Annotated[list[int], Field(min_length=2)]
    sample_key: str
    target_key: str
    seed: Annotated[int, Field(strict=True, ge=0)] = 42
    prompt_length: Annotated[int, Field(strict=True, ge=2)] = 256
    num_distractors: Annotated[int, Field(strict=True, ge=0)] = 8
    delimiter_token_ids: Optional[list[int]] = None

    @model_validator(mode="after")
    def _validate(self) -> "SyntheticReasoningDatasetConfig":
        if len(set(self.symbol_token_ids)) != len(self.symbol_token_ids):
            raise ValueError("symbol_token_ids must be distinct; a repeated symbol makes the answer ambiguous.")
        if self.task == SyntheticReasoningTask.VARIABLE_BINDING:
            if self.delimiter_token_ids is None or len(self.delimiter_token_ids) != 2:
                raise ValueError(
                    "The variable_binding task needs exactly two delimiter_token_ids: the statement "
                    "separator and the assignment operator."
                )
            if set(self.delimiter_token_ids) & set(self.symbol_token_ids):
                raise ValueError(
                    "delimiter_token_ids must not appear in symbol_token_ids; a delimiter that is "
                    "also a symbol makes the statement structure ambiguous."
                )
        return self


class SyntheticReasoningDataset(Dataset):
    """A deterministic, in-memory dataset of synthetic reasoning questions."""

    def __init__(
        self,
        task: SyntheticReasoningTask,
        num_samples: int,
        num_hops: int,
        symbol_token_ids: list[int],
        sample_key: str,
        target_key: str,
        seed: int = 42,
        prompt_length: int = 256,
        num_distractors: int = 8,
        delimiter_token_ids: Optional[list[int]] = None,
    ):
        """
        Initializes the dataset, generating every sample up front.

        Generation is eager rather than lazy so that infeasible task parameters fail at
        construction time instead of thousands of steps into a training run, and so that the
        dataset is a plain array by the time the dataloader workers fork it.

        Args:
            task (SyntheticReasoningTask): Which task to generate.
            num_samples (int): Number of samples to generate.
            num_hops (int): Number of serial lookups the answer requires.
            symbol_token_ids (list[int]): The alphabet, as distinct token ids.
            sample_key (str): Key under which the input token ids are emitted.
            target_key (str): Key under which the masked targets are emitted.
            seed (int): Seed for generation; the same seed yields the same questions.
            prompt_length (int): ``p_hop_induction`` only: number of symbols in the sequence.
            num_distractors (int): ``variable_binding`` only: number of off-chain statements.
            delimiter_token_ids (list[int] | None): ``variable_binding`` only: the statement
                separator and the assignment operator, in that order.

        Raises:
            ValueError: If the task parameters cannot be satisfied by the given alphabet.
        """
        super().__init__(raw_data_path=None, sample_key=sample_key)
        self.task = task
        self.target_key = target_key
        self.num_hops = num_hops
        # How many answers the task's *format* permits, which is the floor a model reaches without
        # doing the task. For p-hop every symbol is permissible; for variable binding only the
        # symbols that appear as a value but never as a variable are, and there are
        # num_distractors + 1 of those. Resolved here so that neither task carries a parameter that
        # is meaningless for it.
        self._num_format_permissible_answers = (
            len(symbol_token_ids) if task == SyntheticReasoningTask.P_HOP_INDUCTION else num_distractors + 1
        )
        self.symbol_token_ids = list(symbol_token_ids)
        # Checked here as well as in the config so that a directly constructed dataset cannot
        # produce questions with two correct answers.
        if len(set(self.symbol_token_ids)) != len(self.symbol_token_ids):
            raise ValueError("symbol_token_ids must be distinct; a repeated symbol makes the answer ambiguous.")

        rng = np.random.default_rng(seed)
        if task == SyntheticReasoningTask.P_HOP_INDUCTION:
            symbol_sequences, answers = self._generate_p_hop_induction(
                rng=rng, num_samples=num_samples, num_hops=num_hops, prompt_length=prompt_length
            )
        else:
            symbol_sequences, answers = self._generate_variable_binding(
                rng=rng,
                num_samples=num_samples,
                num_hops=num_hops,
                num_distractors=num_distractors,
                delimiter_token_ids=delimiter_token_ids,
            )

        # (num_samples, prompt_len) of token ids and (num_samples,) of answer token ids.
        self._inputs = symbol_sequences
        self._answers = answers

    @property
    def chance_accuracy(self) -> float:
        """
        The accuracy of a uniform random guess over the alphabet.

        Answers are marginally uniform over the alphabet by relabeling symmetry, so this is the
        floor an arm has to clear before any comparison between arms means anything.

        Returns:
            float: One over the alphabet size.
        """
        return 1.0 / len(self.symbol_token_ids)

    @property
    def format_aware_chance_accuracy(self) -> float:
        """
        The accuracy reachable by guessing uniformly among the answers the *format* allows.

        For ``variable_binding`` a model that has learned only the shape of the task can restrict
        its guess to the symbols that appear as a value but never as a variable, of which there are
        ``num_distractors + 1``. That is a much higher floor than
        :attr:`chance_accuracy`, and it is the number to read absolute accuracy against. For
        ``p_hop_induction`` every symbol of the alphabet is a permissible answer, so the two floors
        coincide.

        Returns:
            float: One over the number of format-permissible answers.
        """
        return 1.0 / self._num_format_permissible_answers

    def __len__(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self._inputs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Returns one pre-shifted, masked sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict[str, np.ndarray]: The input token ids under ``sample_key`` and the targets under
                ``target_key``, the latter masked with :data:`IGNORE_INDEX` everywhere except the
                final position.
        """
        inputs = self._inputs[idx]
        targets = np.full(len(inputs), IGNORE_INDEX, dtype=np.int64)
        targets[-1] = self._answers[idx]
        return {self.sample_key: inputs, self.target_key: targets}

    def _to_token_ids(self, symbol_indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.symbol_token_ids, dtype=np.int64)[symbol_indices]

    def _generate_p_hop_induction(
        self, rng: np.random.Generator, num_samples: int, num_hops: int, prompt_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        alphabet_size = len(self.symbol_token_ids)
        # Each hop consumes roughly one expected gap between occurrences of a symbol, so a prompt
        # shorter than this cannot reliably contain the chain and rejection sampling would spin.
        if prompt_length <= num_hops * alphabet_size:
            raise ValueError(
                f"prompt_length ({prompt_length}) is too short for {num_hops} hops over an alphabet "
                f"of {alphabet_size} symbols: a hop moves left by ~{alphabet_size} positions on "
                f"average, so the chain would rarely fit. Use prompt_length > "
                f"{num_hops * alphabet_size}."
            )

        sequences = np.empty((num_samples, prompt_length), dtype=np.int64)
        answers = np.empty(num_samples, dtype=np.int64)
        for sample_index in range(num_samples):
            for _ in range(_MAX_REJECTION_ATTEMPTS):
                symbols = rng.integers(0, alphabet_size, size=prompt_length)
                answer = resolve_p_hop(symbols=symbols, num_hops=num_hops)
                # Rejecting answer == final symbol removes the "repeat the last token" shortcut,
                # which a model could otherwise score above chance on without doing any hops.
                if answer is not None and answer != symbols[-1]:
                    break
            else:
                raise ValueError(
                    f"Could not generate a {num_hops}-hop sample within {_MAX_REJECTION_ATTEMPTS} "
                    f"attempts at prompt_length {prompt_length} and alphabet size {alphabet_size}. "
                    f"Increase prompt_length or reduce num_hops."
                )
            sequences[sample_index] = self._to_token_ids(symbols)
            answers[sample_index] = self.symbol_token_ids[answer]
        return sequences, answers

    def _generate_variable_binding(
        self,
        rng: np.random.Generator,
        num_samples: int,
        num_hops: int,
        num_distractors: int,
        delimiter_token_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        # The chain uses one variable per hop plus the literal it bottoms out in; every distractor
        # needs its own variable and value. All are drawn without replacement so that no symbol is
        # bound twice, which would make the answer ambiguous.
        #
        # A consequence worth stating: the answer is always a symbol that appears as a value but
        # never as a variable, and there are exactly num_distractors + 1 of those. A model that has
        # learned only the *format* can therefore reach 1/(num_distractors + 1) without following
        # the chain, well above the 1/alphabet_size floor of the p-hop tasks. That floor is the same
        # for every ablation arm, so it does not bias a comparison, but it does mean absolute
        # accuracy on this task should be read against 1/(num_distractors + 1). Raising
        # num_distractors lowers it. See test_variable_binding_shortcut_floor_is_as_documented.
        num_required_symbols = num_hops + 1 + 2 * num_distractors
        alphabet_size = len(self.symbol_token_ids)
        if num_required_symbols > alphabet_size:
            raise ValueError(
                f"variable_binding with num_hops {num_hops} and num_distractors {num_distractors} "
                f"needs {num_required_symbols} distinct symbols but the alphabet has "
                f"{alphabet_size}. Enlarge symbol_token_ids or reduce num_distractors."
            )
        if delimiter_token_ids is None or len(delimiter_token_ids) != 2:
            raise ValueError("variable_binding needs exactly two delimiter_token_ids.")
        separator_token_id, assignment_token_id = delimiter_token_ids

        num_statements = num_hops + num_distractors
        # Per statement: separator, variable, assignment operator, value. Then the query, which is
        # the same shape without the value.
        prompt_length = 4 * num_statements + 3
        sequences = np.empty((num_samples, prompt_length), dtype=np.int64)
        answers = np.empty(num_samples, dtype=np.int64)

        for sample_index in range(num_samples):
            drawn = rng.choice(alphabet_size, size=num_required_symbols, replace=False)
            chain_variables = drawn[:num_hops]
            literal = drawn[num_hops]
            distractor_symbols = drawn[num_hops + 1 :].reshape(num_distractors, 2)

            # chain_variables[0] = literal, chain_variables[k] = chain_variables[k - 1]. Querying
            # the last variable therefore takes num_hops dereferences to reach the literal.
            statements = [(chain_variables[0], literal)]
            statements += [(chain_variables[k], chain_variables[k - 1]) for k in range(1, num_hops)]
            statements += [(variable, value) for variable, value in distractor_symbols]
            # Shuffling is what makes this a matching task rather than a scanning one: the chain no
            # longer appears in resolution order, so recency carries no signal.
            rng.shuffle(statements)

            token_ids: list[int] = []
            for variable, value in statements:
                token_ids += [
                    separator_token_id,
                    self.symbol_token_ids[variable],
                    assignment_token_id,
                    self.symbol_token_ids[value],
                ]
            token_ids += [separator_token_id, self.symbol_token_ids[chain_variables[-1]], assignment_token_id]

            sequences[sample_index] = np.asarray(token_ids, dtype=np.int64)
            answers[sample_index] = self.symbol_token_ids[literal]
        return sequences, answers


def resolve_p_hop(symbols: np.ndarray, num_hops: int) -> Optional[int]:
    """
    Resolves the p-hop induction chain of a symbol sequence.

    Starting from the final symbol as the query, each hop finds the query's most recent earlier
    occurrence and takes the symbol that follows it as the next query. The search window shrinks to
    the left of that occurrence on every hop, which both guarantees termination and forces the hops
    to be serial: hop ``k + 1`` cannot start before hop ``k`` has produced its symbol *and* its
    position.

    Args:
        symbols (np.ndarray): The symbol sequence, as alphabet indices.
        num_hops (int): The number of hops to resolve.

    Returns:
        int | None: The resolved symbol, or None if some hop has no earlier occurrence to jump to,
            in which case the sequence does not pose a well-formed question.
    """
    query_position = len(symbols) - 1
    query = symbols[query_position]
    for _ in range(num_hops):
        earlier_occurrences = np.flatnonzero(symbols[:query_position] == query)
        if earlier_occurrences.size == 0:
            return None
        source_position = int(earlier_occurrences[-1])
        query = symbols[source_position + 1]
        query_position = source_position
    return int(query)
