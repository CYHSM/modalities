# The single-operator pre-norm residual layer structure is adapted from NVIDIA's Megatron-LM
# (megatron/core/ssm/mamba_layer.py::MambaLayer and
# megatron/core/models/hybrid/hybrid_block.py::HybridStack).
# Copyright (c) 2024-2026, NVIDIA CORPORATION. Copyright (c) 2024, Tri Dao, Albert Gu.
# Licensed under the Apache License, Version 2.0.

"""The four residual sublayer types of a hybrid Mamba-Transformer stack.

Unlike a classical transformer block, which bundles attention and a feed-forward network, every
layer here wraps exactly *one* operator in a pre-norm residual connection::

    x = x + operator(norm(x))

A model is then a sequence of such single-operator layers described by a layer pattern, e.g.
``"MEM*E"``. This is the structure of Nemotron-H and Nemotron-3 Nano, and it is what allows the
Mamba / attention / MoE ratio to be tuned independently.

A layer that sits inside a loop group is applied several times per forward pass. It is then passed
the index of the current iteration, which a :class:`PerIterationNorm` uses to select that
iteration's own pre-normalization.
"""

import torch
import torch.nn as nn

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer
from modalities.models.components.moe.moe import MoE
from modalities.models.nemotron.nemotron_attention import NemotronSelfAttention
from modalities.models.nemotron.nemotron_mlp import SquaredReLUMLP


class PerIterationNorm(nn.Module):
    """
    Holds one normalization per loop iteration and selects it by the iteration index.

    A looped layer applies the same operator weights several times to a residual stream whose scale
    and statistics change between iterations, which is why the literature treats per-iteration
    normalization as close to a prerequisite for loop counts above two or three. Only the norm
    parameters are per-iteration; the operator stays shared, which is the point of the loop.

    The attribute holding this module on the layer is still called ``norm``, so a parameter is named
    ``transformer.h.<idx>.norm.norms.<iteration>.weight``. That name still contains ``.norm.`` and
    therefore still matches the ``layernorm`` weight-decay group regex of
    :class:`~modalities.models.nemotron.nemotron_model.NemotronLLM`. Naming the attribute ``norms``
    instead would silently move these parameters out of every weight-decay group -- which drops them
    from the optimizer entirely, since a parameter matching no group ends up in neither of the two
    optimizer groups.
    """

    def __init__(self, norms: list[nn.Module]):
        """
        Initializes the per-iteration normalization.

        Args:
            norms (list[nn.Module]): One normalization module per loop iteration, in iteration
                order. Each owns its own parameters.
        """
        super().__init__()
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        """
        Applies the normalization belonging to the given loop iteration.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, n_embd)``.
            iteration (int): Index of the current loop iteration.

        Raises:
            IndexError: If ``iteration`` exceeds the number of iterations this module was built for.

        Returns:
            torch.Tensor: The normalized input, of the same shape.
        """
        if iteration >= len(self.norms):
            raise IndexError(
                f"Loop iteration {iteration} has no normalization; this layer was built with "
                f"{len(self.norms)} per-iteration norms."
            )
        return self.norms[iteration](x)


class _ResidualLayer(nn.Module):
    """Base class implementing the shared pre-norm residual structure."""

    def __init__(self, norm: nn.Module):
        """
        Initializes the residual layer.

        Args:
            norm (nn.Module): The pre-normalization module, owned exclusively by this layer. A
                :class:`PerIterationNorm` gives every loop iteration its own normalization.
        """
        super().__init__()
        self.norm = norm
        # Cached rather than checked per forward pass: the plain path must stay exactly what it was
        # before per-iteration norms existed, including the single-argument call into the norm.
        self.has_per_iteration_norm = isinstance(norm, PerIterationNorm)

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer's operator to the normalized input.

        Args:
            x (torch.Tensor): The normalized input of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: The operator output of shape ``(B, L, n_embd)``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        """
        Forward pass: a pre-norm residual application of the layer's operator.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, n_embd)``.
            iteration (int): Index of the loop iteration currently being executed. Ignored unless
                the layer was built with a :class:`PerIterationNorm`.

        Returns:
            torch.Tensor: Output of shape ``(B, L, n_embd)``.
        """
        normalized = self.norm(x, iteration) if self.has_per_iteration_norm else self.norm(x)
        return x + self._operator(normalized)


class Mamba2Layer(_ResidualLayer):
    """A residual layer whose operator is a Mamba-2 mixer."""

    def __init__(self, norm: nn.Module, mixer: Mamba2Mixer):
        """
        Initializes the Mamba2Layer.

        Args:
            norm (nn.Module): The pre-normalization module.
            mixer (Mamba2Mixer): The Mamba-2 mixer.
        """
        super().__init__(norm=norm)
        self.mixer = mixer

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class NemotronAttentionLayer(_ResidualLayer):
    """A residual layer whose operator is grouped-query causal self-attention."""

    def __init__(self, norm: nn.Module, attn: NemotronSelfAttention):
        """
        Initializes the NemotronAttentionLayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            attn (NemotronSelfAttention): The self-attention module.
        """
        super().__init__(norm=norm)
        self.attn = attn

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class NemotronMoELayer(_ResidualLayer):
    """A residual layer whose operator is a sparse mixture-of-experts feed-forward block."""

    def __init__(self, norm: nn.Module, moe: MoE):
        """
        Initializes the NemotronMoELayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            moe (MoE): The mixture-of-experts block.
        """
        super().__init__(norm=norm)
        self.moe = moe

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)


class NemotronMLPLayer(_ResidualLayer):
    """A residual layer whose operator is a dense squared-ReLU feed-forward network."""

    def __init__(self, norm: nn.Module, mlp: SquaredReLUMLP):
        """
        Initializes the NemotronMLPLayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            mlp (SquaredReLUMLP): The dense feed-forward network.
        """
        super().__init__(norm=norm)
        self.mlp = mlp

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
