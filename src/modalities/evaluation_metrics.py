"""Held-out evaluation metrics that are reported alongside the loss.

The loss is a poor instrument for a question-answering evaluation. It is averaged per batch rather
than per answer, it carries whatever auxiliary terms the training objective happens to include (for
a mixture-of-experts model, a load-balancing term whose size depends on how many times the MoE
layers are visited -- i.e. on the very thing a loop ablation varies), and it is the same function
for every dataloader. A metric here is instead attached to specific dataloaders by tag, sees only
the positions that carry an answer, and accumulates a numerator and a denominator so that the
reduction across micro-batches and ranks is exact rather than an average of averages.

Metrics are computed from an :class:`~modalities.batch.InferenceResultBatch`, so they require the
model to return logits. A model configured to skip its language-model head (as a chunked loss
does) does not expose them and cannot be scored this way.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from modalities.batch import InferenceResultBatch
from modalities.constants import IGNORE_INDEX


class EvaluationMetricIF(ABC):
    """Interface for a metric evaluated on held-out dataloaders."""

    def __init__(self, tag: str, dataloader_tags: Optional[list[str]] = None):
        """
        Initializes the metric.

        Args:
            tag (str): Name the metric is logged under, prefixed by the dataloader tag.
            dataloader_tags (list[str] | None): Dataloaders this metric applies to. None or an
                empty list means every dataloader, which is rarely what you want: a metric that
                assumes masked targets reports a meaningless number on a plain language-modelling
                dataloader, where no position is masked.
        """
        self._tag = tag
        self._dataloader_tags = tuple(dataloader_tags) if dataloader_tags else ()

    @property
    def tag(self) -> str:
        """
        The name this metric is logged under.

        Returns:
            str: The tag.
        """
        return self._tag

    @property
    def dataloader_tags(self) -> tuple[str, ...]:
        """
        The dataloaders this metric applies to; empty means all of them.

        Returns:
            tuple[str, ...]: The dataloader tags.
        """
        return self._dataloader_tags

    def applies_to(self, dataloader_tag: str) -> bool:
        """
        Whether this metric should be computed for the given dataloader.

        Args:
            dataloader_tag (str): The dataloader's tag.

        Returns:
            bool: True if the metric applies.
        """
        return len(self._dataloader_tags) == 0 or dataloader_tag in self._dataloader_tags

    @abstractmethod
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Computes this metric's contribution from one batch.

        Returning a numerator and a denominator rather than a value keeps the metric exact under
        arbitrary batching and sharding: the caller sums both across micro-batches and ranks and
        divides once, so a short final batch or an uneven split across ranks cannot skew it.

        Args:
            forward_batch (InferenceResultBatch): The batch's targets and predictions.

        Returns:
            torch.Tensor: A tensor of shape ``(2,)`` holding ``[numerator, denominator]``.
        """
        raise NotImplementedError


class _MaskedTokenMetric(EvaluationMetricIF):
    """Base for metrics scored only at positions whose target is not the ignore index."""

    def __init__(self, target_key: str, prediction_key: str, tag: str, dataloader_tags: Optional[list[str]] = None):
        super().__init__(tag=tag, dataloader_tags=dataloader_tags)
        self.target_key = target_key
        self.prediction_key = prediction_key

    def _scored_positions(self, forward_batch: InferenceResultBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the flattened logits and targets at the positions that carry an answer."""
        targets = forward_batch.get_targets(self.target_key)
        logits = forward_batch.get_predictions(self.prediction_key)
        targets = targets.to(logits.device).long().reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
        scored = targets != IGNORE_INDEX
        return logits[scored], targets[scored]


class MaskedTokenAccuracy(_MaskedTokenMetric):
    """Fraction of answer positions whose most likely token is the correct one.

    This is the quantity the looped-model literature reports (e.g. 34.3% vs 29.3% on math word
    problems), and unlike a loss it has an interpretable floor: on the synthetic tasks in
    :mod:`modalities.dataloader.synthetic_reasoning` chance is one over the alphabet size, so it is
    immediately visible whether an arm is doing anything at all.
    """

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Counts correct predictions and scored positions in one batch.

        Args:
            forward_batch (InferenceResultBatch): The batch's targets and predictions.

        Returns:
            torch.Tensor: ``[num_correct, num_scored]``.
        """
        logits, targets = self._scored_positions(forward_batch)
        num_correct = (logits.argmax(dim=-1) == targets).sum()
        return torch.stack([num_correct.float(), torch.tensor(targets.numel(), device=logits.device).float()])


class MaskedTokenNLL(_MaskedTokenMetric):
    """Mean negative log-likelihood of the correct token at answer positions.

    A graded companion to :class:`MaskedTokenAccuracy`: it moves while accuracy is still pinned at
    chance, which matters at the token budgets these ablations run at. It is computed in float32
    and carries none of the training objective's auxiliary terms, so it is comparable across arms
    in a way the reported evaluation loss is not.
    """

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Sums the negative log-likelihood over the scored positions of one batch.

        Args:
            forward_batch (InferenceResultBatch): The batch's targets and predictions.

        Returns:
            torch.Tensor: ``[summed_nll, num_scored]``.
        """
        logits, targets = self._scored_positions(forward_batch)
        summed_nll = F.cross_entropy(logits.float(), targets, reduction="sum")
        return torch.stack([summed_nll, torch.tensor(targets.numel(), device=logits.device).float()])


class MaskedTokenMetricConfig(BaseModel):
    """
    Configuration of :class:`MaskedTokenAccuracy` and :class:`MaskedTokenNLL`.

    Attributes:
        target_key (str): Key under which the batch's targets are found.
        prediction_key (str): Key under which the model's logits are found.
        tag (str): Name the metric is logged under, prefixed by the dataloader tag.
        dataloader_tags (list[str] | None): Dataloaders to score. Leave unset to score all, which
            only makes sense if every evaluation dataloader uses masked targets.
    """

    target_key: str
    prediction_key: str
    tag: Annotated[str, Field(min_length=1)]
    dataloader_tags: Optional[list[str]] = None
