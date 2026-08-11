"""Pre-tokenized held-out evaluations with masked targets.

The synthetic tasks in :mod:`modalities.dataloader.synthetic_reasoning` generate their own token
ids, so they need no tokenizer. Benchmarks built from real text -- Minerva MATH and TriviaQA --
cannot: they have to be tokenized, and doing that at training time would make every run depend on a
tokenizer being present and would let a tokenizer change silently alter what the arms are compared
on.

So the text is tokenized **once, offline**, by
``config_files/nemotron/loop_ablation/prepare_text_evals.py``, into a ``.npz`` holding the input ids
and the masked targets. This module just loads that file. The arrangement mirrors how the training
pbin files work, and it means every ablation arm is scored on byte-identical token sequences.

Targets carry :data:`~modalities.constants.IGNORE_INDEX` at every position that should not be
scored, so *which* tokens count is baked into the prepared file rather than decided here. The
prompt is masked and the reference answer is scored.

Sequences are right-padded to a common length. Padding at the end is safe for a causal model: no
scored position can attend to it, and the pad positions themselves are masked out of the targets.
"""

import json
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, Field

from modalities.constants import IGNORE_INDEX
from modalities.dataloader.dataset import Dataset


class PreparedEvalDatasetConfig(BaseModel):
    """
    Configuration of a :class:`PreparedEvalDataset`.

    Attributes:
        prepared_path (Path): The ``.npz`` written by the preparation script.
        sample_key (str): Key under which the input token ids are emitted.
        target_key (str): Key under which the masked targets are emitted.
        num_samples (int | None): Truncate to this many problems. None uses all of them. Must be
            the same across arms, since arms are compared on their answers to the same questions.
    """

    prepared_path: Path
    sample_key: str
    target_key: str
    num_samples: Optional[Annotated[int, Field(strict=True, ge=1)]] = None


class PreparedEvalDataset(Dataset):
    """A held-out evaluation loaded from a pre-tokenized file with masked targets."""

    def __init__(
        self,
        prepared_path: Path,
        sample_key: str,
        target_key: str,
        num_samples: Optional[int] = None,
    ):
        """
        Loads a prepared evaluation.

        Args:
            prepared_path (Path): The ``.npz`` written by the preparation script.
            sample_key (str): Key under which the input token ids are emitted.
            target_key (str): Key under which the masked targets are emitted.
            num_samples (int | None): Truncate to this many problems, or None for all.

        Raises:
            FileNotFoundError: If the prepared file does not exist, with the command that builds it.
            ValueError: If the file is malformed or contains no scored position.
        """
        super().__init__(raw_data_path=prepared_path, sample_key=sample_key)
        self.target_key = target_key

        if not Path(prepared_path).exists():
            raise FileNotFoundError(
                f"No prepared evaluation at {prepared_path}. Build it with\n"
                f"  python config_files/nemotron/loop_ablation/prepare_text_evals.py\n"
                f"which downloads Minerva MATH and TriviaQA, tokenizes them once, and writes the "
                f".npz files."
            )

        with np.load(prepared_path, allow_pickle=False) as archive:
            missing = {"input_ids", "target_ids"} - set(archive.files)
            if missing:
                raise ValueError(f"{prepared_path} is missing the arrays {sorted(missing)}.")
            self._inputs = archive["input_ids"].astype(np.int64)
            self._targets = archive["target_ids"].astype(np.int64)
            self.metadata = json.loads(str(archive["metadata"])) if "metadata" in archive.files else {}

        if self._inputs.shape != self._targets.shape:
            raise ValueError(
                f"{prepared_path} has input_ids of shape {self._inputs.shape} but target_ids of "
                f"shape {self._targets.shape}; they must match."
            )
        if num_samples is not None:
            self._inputs = self._inputs[:num_samples]
            self._targets = self._targets[:num_samples]

        num_scored = int((self._targets != IGNORE_INDEX).sum())
        if num_scored == 0:
            raise ValueError(
                f"{prepared_path} has no scored position: every target is the ignore index, so any "
                f"metric over it would be undefined. The preparation step masked everything."
            )
        self.num_scored_tokens = num_scored

    @property
    def mean_scored_tokens_per_problem(self) -> float:
        """
        Average number of scored positions per problem.

        Worth knowing when reading a benchmark's negative log-likelihood: a full-solution masking
        scores an order of magnitude more tokens than a final-answer one, and most of them are
        ordinary prose rather than reasoning.

        Returns:
            float: Scored tokens divided by problems.
        """
        return self.num_scored_tokens / len(self._inputs)

    def __len__(self) -> int:
        """
        Returns the number of problems.

        Returns:
            int: The number of problems.
        """
        return len(self._inputs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Returns one pre-shifted, masked problem.

        Args:
            idx (int): Index of the problem.

        Returns:
            dict[str, np.ndarray]: Input token ids under ``sample_key`` and masked targets under
                ``target_key``.
        """
        return {self.sample_key: self._inputs[idx], self.target_key: self._targets[idx]}
