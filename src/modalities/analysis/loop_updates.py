"""Measures whether successive iterations of a loop group do *different* work.

Wave 2 ranks six loop arms by loss but says nothing about why looping a Mamba layer beats looping an
attention layer by 28 within-arm standard deviations. The hypothesis this module is built to test is
the cheapest one available: a layer benefits from depth recurrence only insofar as its repeated
applications are not redundant. If iteration ``k+1`` pushes the residual stream in the same direction
iteration ``k`` did, the extra pass bought a rescale rather than a computation.

One quantity is captured -- each layer's **additive contribution to the residual stream** -- and two
numbers are derived from it per loop group:

* **Update direction diversity**, ``cos(delta_k, delta_{k+1})``. Near 1 means successive iterations
  re-tread one direction; near 0 means each pass does genuinely different work; negative means later
  passes partly undo earlier ones.
* **Step decay**, ``||delta_k|| / ||h_0||``, i.e. how far each iteration actually moves the stream
  relative to what the group was handed.

Every :class:`~modalities.models.nemotron.nemotron_layers._ResidualLayer` computes
``x + self._operator(self.norm(x))``, so the contribution is recovered exactly as ``output - input``.
That identity is what makes this measurement unambiguous: no choice about pre- versus post-norm taps
arises, and it holds uniformly for Mamba, attention, MoE and MLP layers, whose operator submodules
have different attribute names. ``tests/analysis/test_loop_updates.py`` asserts the identity against
the real layer classes rather than trusting it.

Two design points that decide whether the numbers can be read at all:

* **Cosines are computed per token and then aggregated.** A cosine over the flattened tensor is a
  single number dominated by whichever tokens have the largest norm -- in a transformer, the attention
  sinks. Per-token values summarized by their median are robust to those few tokens without needing an
  arbitrary prefix mask.
* **Everything is accumulated in float32** regardless of the autocast dtype, because these are
  differences of similar-magnitude vectors.

The same machinery yields two views. ``groups`` covers the looped groups, which is the question.
``stack`` covers every layer application in execution order, which gives the *real depth* reference:
a baseline arm has no loop groups at all, and its layer-to-layer trajectory is what looped iteration
has to be compared against.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Maps a layer class name onto the layer_pattern symbol it was built from, so that a group's
# composition can be reported as "[M*]" rather than as a pair of Python class names.
_LAYER_SYMBOLS = {
    "Mamba2Layer": "M",
    "NemotronMoELayer": "E",
    "NemotronAttentionLayer": "*",
    "NemotronMLPLayer": "-",
}

# Guards the per-token normalizations against a zero-norm token. Packed continuous datasets contain
# no padding, so this should never bind; it exists so that a stray zero cannot produce a silent NaN
# that would then propagate into a median and look like a finite result.
_EPS = 1e-12


def _summarize(values: torch.Tensor) -> dict[str, float]:
    """
    Summarizes a per-token quantity by its centre and spread.

    Both are reported because they answer different questions: the mean is what a reader compares
    across arms, while the median and inter-quartile range say whether that mean describes the bulk
    of the tokens or is being carried by a heavy tail.

    Args:
        values (torch.Tensor): Per-token values of any shape; flattened before summarizing.

    Returns:
        dict[str, float]: Mean, standard deviation, median, 25th and 75th percentile, and count.
    """
    flat = values.flatten().float()
    quantiles = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75], dtype=flat.dtype, device=flat.device))
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "p25": quantiles[0].item(),
        "median": quantiles[1].item(),
        "p75": quantiles[2].item(),
        "n_tokens": int(flat.numel()),
    }


def _per_token_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """
    Cosine between two update fields, computed independently for every token.

    Args:
        first (torch.Tensor): Updates of shape ``(B, L, n_embd)``.
        second (torch.Tensor): Updates of the same shape.

    Returns:
        torch.Tensor: Cosines of shape ``(B, L)``, in ``[-1, 1]``.
    """
    return F.cosine_similarity(first.float(), second.float(), dim=-1, eps=_EPS)


def _per_token_norm_ratio(update: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Per-token ratio of an update's length to a reference state's length.

    Args:
        update (torch.Tensor): Updates of shape ``(B, L, n_embd)``.
        reference (torch.Tensor): The state to normalize by, of the same shape.

    Returns:
        torch.Tensor: Ratios of shape ``(B, L)``.
    """
    return update.float().norm(dim=-1) / reference.float().norm(dim=-1).clamp_min(_EPS)


@dataclass
class _LayerCall:
    """One application of one layer during a single forward pass."""

    order: int
    layer_key: str
    layer_type: str
    delta: torch.Tensor
    layer_input: torch.Tensor


@dataclass
class LoopUpdateRecorder:
    """
    Records every layer's additive contribution to the residual stream during one forward pass.

    Hooks are attached to the layer modules themselves rather than to their operator submodules, so
    that a single code path covers all four layer types. They are strictly observational: they read
    the input and output tensors and store detached float32 copies, and never return a value, so the
    forward pass is bit-identical with and without recording. ``tests/analysis/test_loop_updates.py``
    asserts that.

    Captured tensors are moved to ``storage_device`` (CPU by default). A 22-layer arm at 8x2048
    tokens holds roughly 1.5 GB in float32, which is worth keeping off the accelerator.

    Attributes:
        model (nn.Module): The model to instrument. Must expose ``transformer.h`` and ``_schedule``.
        storage_device (str): Where captured tensors are kept.
    """

    model: nn.Module
    storage_device: str = "cpu"
    _calls: list[_LayerCall] = field(default_factory=list, init=False)
    _handles: list[Any] = field(default_factory=list, init=False)
    _pending: dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    def __enter__(self) -> "LoopUpdateRecorder":
        for layer_key, layer in self.model.transformer.h.items():
            self._handles.append(layer.register_forward_pre_hook(self._make_pre_hook(layer_key)))
            self._handles.append(layer.register_forward_hook(self._make_post_hook(layer_key)))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()

    def _make_pre_hook(self, layer_key: str):
        def hook(module: nn.Module, args: tuple) -> None:
            # args[0] is the residual stream; a second positional argument carries the iteration
            # index when per_iteration_norm is enabled, and is irrelevant here.
            self._pending[layer_key] = args[0].detach().to(self.storage_device, torch.float32)

        return hook

    def _make_post_hook(self, layer_key: str):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            layer_input = self._pending.pop(layer_key)
            layer_output = output.detach().to(self.storage_device, torch.float32)
            self._calls.append(
                _LayerCall(
                    order=len(self._calls),
                    layer_key=layer_key,
                    layer_type=_LAYER_SYMBOLS.get(type(module).__name__, type(module).__name__),
                    # Exact because the layer computes `x + operator(norm(x))`.
                    delta=layer_output - layer_input,
                    layer_input=layer_input,
                )
            )

        return hook

    @property
    def calls(self) -> list[_LayerCall]:
        """The recorded layer applications, in execution order."""
        return self._calls

    def reset(self) -> None:
        """Discards recorded calls so the recorder can be reused for another batch."""
        self._calls.clear()
        self._pending.clear()

    def group_report(self) -> list[dict]:
        """
        Summarizes the update trajectory of every *looped* group.

        A group's update at iteration ``k`` is the sum of its member layers' contributions at that
        iteration, which is exactly the change the group makes to the residual stream in that pass.
        Groups that execute once are skipped: they have no iteration trajectory, and including them
        would pad the table with rows that cannot answer the question.

        Returns:
            list[dict]: One entry per looped group.
        """
        reports = []
        call_index = 0
        for group_index, group in enumerate(self.model._schedule):
            num_calls = len(group.layer_keys) * group.num_loops
            group_calls = self._calls[call_index : call_index + num_calls]
            call_index += num_calls
            if group.num_loops < 2:
                continue

            # Iteration k executed the group's layers in order, so the calls arrive in blocks of
            # len(layer_keys). Reshaping by that block size recovers the iteration structure.
            per_iteration = [
                group_calls[k * len(group.layer_keys) : (k + 1) * len(group.layer_keys)]
                for k in range(group.num_loops)
            ]
            group_deltas = [sum(call.delta for call in iteration) for iteration in per_iteration]
            group_input = per_iteration[0][0].layer_input

            reports.append(
                {
                    "group_index": group_index,
                    "num_loops": group.num_loops,
                    "layer_keys": list(group.layer_keys),
                    "composition": "".join(call.layer_type for call in per_iteration[0]),
                    "first_executed_index": group_calls[0].order,
                    # The absolute length of the group's input, which is the denominator of every
                    # relative step norm below. Without it a large ratio is ambiguous: the first
                    # group in the stack sits on the raw embeddings, whose norm is small, so it can
                    # report a ratio above 1 while making a perfectly ordinary-sized update. Ratios
                    # are comparable across the iterations of one group, but never across depths
                    # unless this is checked too.
                    "group_input_norm": _summarize(group_input.float().norm(dim=-1)),
                    "relative_step_norm": [
                        {"iteration": k, **_summarize(_per_token_norm_ratio(delta, group_input))}
                        for k, delta in enumerate(group_deltas)
                    ],
                    "update_cosine": [
                        {"from_iteration": k, "to_iteration": k + 1,
                         **_summarize(_per_token_cosine(group_deltas[k], group_deltas[k + 1]))}
                        for k in range(group.num_loops - 1)
                    ],
                    "members": self._member_report(per_iteration, group_input),
                }
            )
        return reports

    def _member_report(self, per_iteration: list[list[_LayerCall]], group_input: torch.Tensor) -> Optional[dict]:
        """
        Decomposes a multi-member group's update into its individual layers.

        A group such as ``[M*]`` mixes two operator classes, and the summed update cannot say whether
        one member is idle or whether the two are actively cancelling. Both the per-member step size
        and the angle *between* members are needed: a negative angle is destructive interference,
        while an angle near zero is complementary work.

        Args:
            per_iteration (list[list[_LayerCall]]): The group's calls, grouped by iteration.
            group_input (torch.Tensor): The group's input, used as the normalization reference.

        Returns:
            dict | None: The decomposition, or None for single-member groups where it is vacuous.
        """
        members = per_iteration[0]
        if len(members) < 2:
            return None

        per_member_norm = [
            {
                "member": position,
                "layer_key": members[position].layer_key,
                "layer_type": members[position].layer_type,
                "per_iteration": [
                    {"iteration": k, **_summarize(_per_token_norm_ratio(iteration[position].delta, group_input))}
                    for k, iteration in enumerate(per_iteration)
                ],
            }
            for position in range(len(members))
        ]
        pairwise = [
            {
                "members": [first, second],
                "layer_types": [members[first].layer_type, members[second].layer_type],
                "per_iteration": [
                    {
                        "iteration": k,
                        **_summarize(_per_token_cosine(iteration[first].delta, iteration[second].delta)),
                    }
                    for k, iteration in enumerate(per_iteration)
                ],
            }
            for first in range(len(members))
            for second in range(first + 1, len(members))
        ]
        return {"per_member_relative_norm": per_member_norm, "between_member_cosine": pairwise}

    def layer_profile(self) -> list[dict]:
        """
        Per-layer update magnitude measured against each layer's OWN input.

        Distinct from :meth:`stack_report`, which normalizes every layer's update by the *stack's*
        input (the embedding output). That quantity necessarily grows with depth because the residual
        stream itself grows -- on the unlooped baseline it runs from 4.7 at layer 0 to 111 at layer 11
        -- so it measures the stream's scale, not how hard a layer is working. Dividing by the layer's
        own input instead answers "how much does this layer change the state it was handed", which is
        the quantity that predicts whether re-running that layer buys anything.

        This is the same normalization the looped arms' ``member_step_norms`` use (a single-layer loop
        group's input *is* the layer's input), so a profile measured on the baseline is directly
        comparable to the per-member figures in :meth:`group_report`.

        Returns:
            list[dict]: One entry per layer application, in execution order.
        """
        return [
            {
                "step": call.order,
                "layer_key": call.layer_key,
                "layer_type": call.layer_type,
                "input_norm": _summarize(call.layer_input.float().norm(dim=-1)),
                "absolute_update_norm": _summarize(call.delta.float().norm(dim=-1)),
                "relative_to_own_input": _summarize(_per_token_norm_ratio(call.delta, call.layer_input)),
            }
            for call in self._calls
        ]

    def stack_report(self) -> dict:
        """
        Summarizes the same two quantities across the whole executed stack, layer by layer.

        This is the real-depth reference. A baseline arm has no loop groups, so ``group_report``
        returns nothing for it and this is the only view it has -- and it is precisely the comparison
        that matters: does weight-shared iteration resemble the trajectory of genuine distinct depth,
        or a degenerate version of it?

        Returns:
            dict: Per-step relative norms and consecutive-step cosines over the executed sequence.
        """
        deltas = [call.delta for call in self._calls]
        reference = self._calls[0].layer_input
        return {
            "n_executed_layers": len(self._calls),
            "executed_types": [call.layer_type for call in self._calls],
            "reference_norm": _summarize(reference.float().norm(dim=-1)),
            "relative_step_norm": [
                {"step": index, **_summarize(_per_token_norm_ratio(delta, reference))}
                for index, delta in enumerate(deltas)
            ],
            "update_cosine": [
                {"from_step": index, "to_step": index + 1,
                 **_summarize(_per_token_cosine(deltas[index], deltas[index + 1]))}
                for index in range(len(deltas) - 1)
            ],
        }
