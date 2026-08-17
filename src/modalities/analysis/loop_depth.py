"""Measures what a loop group's extra iterations are worth, by removing them at inference.

Wave 2's loss ranking is almost perfectly monotonic in *where* an arm's first loop group sits
(Spearman +0.971), not in which operator it loops -- and in that pattern the two cannot be separated,
because M first occurs at position 0, MoE at 1 and attention at 3, so the looped operator determines
where the loop can start. Training a position sweep would cost hundreds of node-hours. A trained model
already contains a cheaper version of the question: turn one group's iterations off and see how much
worse it gets.

Two settings are supported, both applied to a finished checkpoint:

* **Per-group ablation** -- one group's ``num_loops`` set to 1, the rest untouched. The loss increase
  is what that group's extra iterations are worth. Comparing groups *within one arm* holds the looped
  operator fixed by construction, which is exactly the comparison Wave 2 could not make.
* **Global depth override** -- every group set to a common ``K``, giving a loss-versus-depth curve
  either side of the trained value.

**This measures association, not causation.** A model trained at K=3 and run at K=1 is off its training
distribution, so the loss increase mixes "these iterations did useful work" with "the model was never
asked to run this shallow". The magnitude therefore overstates the causal value of the iterations; the
*ordering across positions within an arm* is the part worth reading, and even that is only suggestive
until a trained position sweep confirms it.

Overriding is done by rebuilding the schedule rather than mutating it: :class:`LoopGroup` is a frozen
dataclass, and the context manager restores the original list on exit, so nothing here can leak into a
later measurement.

Statistical note. The effects sought are ~0.01 nats, while an 8-sequence batch moved by 0.14 nats
across seeds in an earlier diagnostic -- 8 sequences is effectively 8 samples, not 16k tokens. Losses
are therefore compared **paired, on identical tokens**, via per-token differences whose sequence-level
variance largely cancels; :func:`paired_delta` never subtracts two independently estimated means.
"""

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def override_loop_counts(
    model: nn.Module,
    num_loops: Optional[int] = None,
    group_index: Optional[int] = None,
) -> Iterator[list]:
    """
    Temporarily changes how many times loop groups execute.

    Args:
        model (nn.Module): A model exposing ``_schedule``.
        num_loops (int | None): The iteration count to impose. None leaves the schedule unchanged,
            which is how the unmodified baseline is evaluated through the same code path.
        group_index (int | None): Restricts the change to one group of the schedule. None applies it
            to every *looped* group; groups that already execute once are never touched, since forcing
            them to loop would evaluate an architecture that was never trained.

    Raises:
        ValueError: If ``num_loops`` is below 1, or ``group_index`` is out of range.

    Yields:
        list: The schedule in force inside the block.
    """
    original = model._schedule
    try:
        if num_loops is not None:
            if num_loops < 1:
                raise ValueError(f"num_loops must be at least 1, got {num_loops}.")
            if group_index is not None and not 0 <= group_index < len(original):
                raise ValueError(f"group_index {group_index} is outside the schedule (length {len(original)}).")
            model._schedule = [
                replace(group, num_loops=num_loops)
                if (group.num_loops > 1 and (group_index is None or index == group_index))
                else group
                for index, group in enumerate(original)
            ]
        yield model._schedule
    finally:
        # Restored on the exception path too: a leaked schedule would silently corrupt every
        # subsequent measurement in the same process rather than failing.
        model._schedule = original


def looped_group_indices(model: nn.Module) -> list[int]:
    """
    Indices of the schedule's groups that execute more than once.

    Args:
        model (nn.Module): A model exposing ``_schedule``.

    Returns:
        list[int]: Schedule indices of the looped groups, in execution order.
    """
    return [index for index, group in enumerate(model._schedule) if group.num_loops > 1]


def group_executed_positions(model: nn.Module) -> dict[int, int]:
    """
    Maps each schedule index onto the executed-layer index at which that group starts.

    Position in *executed* terms is what the analysis is indexed by, since that is the depth of
    residual stream the group operates on. It differs from the schedule index whenever an earlier
    group loops.

    Args:
        model (nn.Module): A model exposing ``_schedule``.

    Returns:
        dict[int, int]: Schedule index -> first executed-layer index.
    """
    positions = {}
    executed = 0
    for index, group in enumerate(model._schedule):
        positions[index] = executed
        executed += group.num_executed_layers
    return positions


@torch.no_grad()
def per_token_losses(model: nn.Module, samples: torch.Tensor, micro_batch_size: int = 8) -> torch.Tensor:
    """
    Cross-entropy of every predicted token, in evaluation order.

    Returned per token rather than averaged so that two settings can be compared *paired*: the
    difference of two per-token vectors over identical tokens has far less variance than the
    difference of their means.

    Args:
        model (nn.Module): The model to evaluate.
        samples (torch.Tensor): Token ids of shape ``(N, sequence_length + 1)``.
        micro_batch_size (int): Sequences per forward pass.

    Returns:
        torch.Tensor: Per-token losses of shape ``(N, sequence_length)``, on the CPU in float32.
    """
    sequence_length = model.sequence_length
    losses = []
    for start in range(0, samples.shape[0], micro_batch_size):
        chunk = samples[start : start + micro_batch_size]
        inputs, targets = chunk[:, :sequence_length], chunk[:, 1 : sequence_length + 1]
        logits = model({model.sample_key: inputs})[model.prediction_key]
        token_losses = F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="none"
        ).view(targets.shape)
        losses.append(token_losses.to("cpu", torch.float32))
    return torch.cat(losses)


def paired_delta(baseline: torch.Tensor, ablated: torch.Tensor) -> dict[str, float]:
    """
    Summarizes the paired loss increase of one setting over another.

    The standard error is computed over *sequences*, not tokens: tokens within a sequence are strongly
    correlated, so a token-level standard error would understate the uncertainty by roughly the square
    root of the sequence length.

    Args:
        baseline (torch.Tensor): Per-token losses of the reference setting, ``(N, L)``.
        ablated (torch.Tensor): Per-token losses of the modified setting, same shape and same tokens.

    Raises:
        ValueError: If the two settings were not evaluated on the same tokens.

    Returns:
        dict[str, float]: Mean paired delta, its standard error, both mean losses, and the counts.
    """
    if baseline.shape != ablated.shape:
        raise ValueError(f"Paired comparison needs identical shapes, got {tuple(baseline.shape)} and "
                         f"{tuple(ablated.shape)}.")
    difference = ablated - baseline
    per_sequence = difference.mean(dim=-1)
    n = per_sequence.numel()
    standard_error = (per_sequence.std(unbiased=True) / (n**0.5)).item() if n > 1 else 0.0
    return {
        "delta": difference.mean().item(),
        "standard_error": standard_error,
        "baseline_loss": baseline.mean().item(),
        "ablated_loss": ablated.mean().item(),
        "n_sequences": int(n),
        "n_tokens": int(difference.numel()),
    }
