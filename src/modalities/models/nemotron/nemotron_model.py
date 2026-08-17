# The hybrid model structure (embedding, pattern-driven layer stack, final norm, LM head) is
# adapted from NVIDIA's Megatron-LM (megatron/core/models/hybrid/hybrid_model.py::HybridModel and
# megatron/core/models/hybrid/hybrid_block.py::HybridStack).
# Copyright (c) 2024-2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.
#
# The module tree deliberately mirrors modalities/models/gpt2/gpt2_model.py::GPT2LLM so that the
# existing FSDP / activation-checkpointing / pipeline components apply unchanged.

"""Nemotron-style hybrid Mamba-Transformer language model.

The architecture (Nemotron-3 Nano 30B-A3B, arXiv:2512.20848) interleaves Mamba-2 mixers,
grouped-query attention and sparse mixture-of-experts feed-forward layers according to a layer
pattern string. Attention appears in only a handful of layers; the Mamba-2 layers carry both the
bulk of the sequence mixing and all positional information, which is why the model uses no
positional embeddings at all.

The module tree deliberately mirrors GPT2's (``transformer.wte`` / ``transformer.h`` /
``transformer.lm_head_norm`` / ``transformer.lm_head``) so that the existing Modalities components
for activation checkpointing, FSDP wrapping, pipeline splitting and the chunked loss apply
unchanged. ``transformer.h`` is an ``nn.ModuleDict`` because the activation checkpointing component
requires one.
"""

import logging
from typing import Annotated, Optional, overload

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modalities.config.pydantic_if_types import PydanticNemotronLayerSpecIFType
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.model import NNModel
from modalities.models.nemotron.layer_pattern import LayerSymbol, LoopGroup, parse_layer_schedule
from modalities.models.nemotron.nemotron_layer_specs import NemotronLayerSpecIF
from modalities.models.nemotron.nemotron_layers import PerIterationNorm
from modalities.models.nemotron.nemotron_loop import (
    INJECTION_MODES,
    ITERATION_EMBEDDINGS,
    LoopIterationConditioning,
)

logger = logging.getLogger(__name__)


class LoopConfig(BaseModel):
    """
    Configuration of how a loop group's iterations are combined.

    Every loop refinement defaults to off, so a config that does not mention them describes exactly
    the model it described before they existed -- same modules, same parameter names, same count.

    Two further refinements once lived here and were removed after the K in {3, 6, 12} ablation
    measured them as useless and harmful respectively: a stabilized recurrence (``variant="parcae"``)
    and an injection normalization (``injection_norm_config``). See
    :mod:`~modalities.models.nemotron.nemotron_loop` for the numbers. Their keys remain declared but
    accept only their inert value, so the many configs written before the removal keep building while
    a config actually *requesting* a removed refinement fails loudly instead of quietly training a
    plain loop under a refined arm's name.

    Attributes:
        per_iteration_norm (bool): Whether each iteration of a loop group gets its own
            pre-normalization instead of reusing the layer's single norm. Only the norm parameters
            are per-iteration; the operator stays shared. Adds ``n_embd`` parameters per extra
            iteration per looped layer, so an arm using it is no longer *exactly* iso-parameter
            with the baseline.
        input_injection (bool): Whether the group's input is added back to the hidden states at the
            start of every iteration after the first, so that no iteration is more than one
            residual step away from what the group was given.
        injection_mode (str): How the group's input is combined with the hidden states. Only
            ``"add"`` is implemented; see :meth:`NemotronLLM._run_loop_group`.
        iteration_embedding (str): How the current iteration index is made visible to the shared
            weights, one of
            :data:`~modalities.models.nemotron.nemotron_loop.ITERATION_EMBEDDINGS`. ``"add"`` learns
            a per-iteration shift, ``"film"`` a per-iteration scale and shift. Adds ``K * n_embd``
            (or twice that) parameters per looped group.
    """

    per_iteration_norm: bool = False
    input_injection: bool = False
    injection_mode: str = "add"
    iteration_embedding: str = "none"

    # Removed refinements, still DECLARED rather than deleted. Every loop config written before the
    # removal carries `variant: simple`, and the refinement wave's controls carry an explicit
    # `injection_norm_config: null`; both describe exactly the plain loop. They have to be declared
    # because ComponentFactory validates each component config with `extra="forbid"`, which overrides
    # this model's `extra="allow"` -- so an *undeclared* key here is a hard build failure, not an
    # ignored extra, and deleting these outright makes every historical arm config unloadable.
    # `_validate` below rejects any value other than the inert one.
    variant: str = "simple"
    injection_norm_config: Optional[dict] = None

    # Permit unknown keys so that a future refinement's hyperparameters can be configured without a
    # schema change; the code implementing it is responsible for reading them.
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate(self) -> "LoopConfig":
        if self.injection_mode not in _INJECTION_MODES:
            raise ValueError(f"Unknown injection_mode '{self.injection_mode}'. Available: {sorted(_INJECTION_MODES)}.")
        if self.iteration_embedding not in ITERATION_EMBEDDINGS:
            raise ValueError(
                f"Unknown iteration_embedding '{self.iteration_embedding}'. "
                f"Available: {sorted(ITERATION_EMBEDDINGS)}."
            )
        # Accepting the key but ignoring its value would let a refinement-wave config train a plain
        # loop while still reporting itself as a refined arm.
        requested = sorted(
            key for key, inert in _REMOVED_REFINEMENTS.items() if getattr(self, key, inert) != inert
        )
        if requested:
            raise ValueError(
                f"loop_config key(s) {requested} request loop refinements that were removed after "
                "the K in {3, 6, 12} ablation measured them as ineffective and harmful respectively "
                "(see the nemotron_loop module docstring). iteration_embedding='film' is the "
                "refinement that survived."
            )
        return self

    @property
    def needs_group_modulation(self) -> bool:
        """
        Whether a per-group :class:`LoopIterationConditioning` needs to be built.

        ``per_iteration_norm`` is deliberately absent: it is realized inside the layers themselves,
        not per group. So is ``input_injection``, which needs no parameters of its own.

        Returns:
            bool: True if iteration conditioning is enabled.
        """
        return self.iteration_embedding != "none"


# Matrix dimensions should be multiples of this for efficient tensor-core utilization, see
# https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/
_TENSOR_CORE_ALIGNMENT = 128

# Injection modes and iteration-conditioning modes are defined next to the module that implements
# them; re-exported here under the name this file already used.
#
# A "concat_proj" injection mode (concatenate and project back to n_embd) is deliberately still
# absent: it would add ~2 * n_embd^2 parameters per looped group, breaking the iso-parameter
# comparison that the whole ablation rests on. The per-group parameters that *do* exist
# (LoopIterationConditioning) are per-channel tables, which is why they do not.
_INJECTION_MODES = INJECTION_MODES

# Removed loop refinements, mapped to the only value of each key that is still a faithful
# description of what the model does. `variant: simple` and an absent injection norm are exactly the
# plain loop, so every config predating the refinements -- and every arm of the ablation that did not
# use them -- keeps loading unchanged; anything requesting the removed behaviour is rejected below
# rather than swallowed by `extra="allow"`. `dt_min`/`dt_max` only ever tuned the recurrence, so they
# are inert here and are accepted at any value.
_REMOVED_REFINEMENTS = {"variant": "simple", "injection_norm_config": None}


class NemotronLLMConfig(BaseModel):
    """
    Configuration of the :class:`NemotronLLM` model.

    Attributes:
        sample_key (str): Key under which the input token ids are found.
        prediction_key (str): Key under which the logits are stored.
        aux_loss_key (str | None): Key under which the summed MoE auxiliary loss is stored. When
            None, the auxiliary loss is not exposed in the model output.
        sequence_length (int): Maximum supported sequence length.
        vocab_size (int): Vocabulary size.
        n_embd (int): Model dimension.
        n_layer (int): Number of layers to build. Must equal the number of layers described by
            ``layer_pattern``, counting a loop group's layers once rather than once per iteration.
        layer_pattern (str): One symbol per layer, e.g. ``"MEM*E"``. May contain loop groups
            written ``[<symbols>]^<K>``, e.g. ``"M[ME]^3E"``.
        layer_specs (dict[str, NemotronLayerSpecIF]): Maps a layer pattern symbol to the builder
            that produces layers of that type. Must cover every symbol used in the pattern.
        loop_config (LoopConfig): How the iterations of a loop group are combined.
        lm_head_norm_config (NormWrapperConfig): Normalization applied before the language model
            head.
        use_weight_tying (bool): Whether to tie the input embedding and the output projection.
            Nemotron unties them.
        use_meta_device (bool): Whether to build the model on the meta device.
        enforce_tensor_core_alignment (bool): Whether to require that ``n_embd`` and ``vocab_size``
            are multiples of 128.
    """

    sample_key: str
    prediction_key: str
    aux_loss_key: Optional[str] = None
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    layer_pattern: str
    layer_specs: dict[str, PydanticNemotronLayerSpecIFType]
    loop_config: LoopConfig = LoopConfig()
    lm_head_norm_config: NormWrapperConfig
    use_weight_tying: bool = False
    use_meta_device: Optional[bool] = False
    enforce_tensor_core_alignment: bool = True

    # Avoid the pydantic warning about the protected 'model_' namespace.
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def _validate(self) -> "NemotronLLMConfig":
        layer_symbols, _ = parse_layer_schedule(self.layer_pattern)
        if len(layer_symbols) != self.n_layer:
            raise ValueError(
                f"n_layer ({self.n_layer}) does not match the number of layers built by "
                f"layer_pattern ('{self.layer_pattern}' builds {len(layer_symbols)} layers). Note "
                f"that a loop group's layers are built once, not once per iteration."
            )

        # Every symbol appearing in the pattern needs a spec, and every provided spec must declare
        # the symbol it was registered under. Both mistakes are easy to make in YAML.
        for symbol, spec in self.layer_specs.items():
            parsed_symbol = LayerSymbol(symbol)
            if spec.symbol != parsed_symbol:
                raise ValueError(
                    f"layer_specs['{symbol}'] is a spec for layer type '{spec.symbol.value}', "
                    f"not '{parsed_symbol.value}'."
                )
        missing = sorted({symbol.value for symbol in layer_symbols} - set(self.layer_specs))
        if missing:
            raise ValueError(
                f"layer_pattern uses the layer types {missing} but layer_specs only provides "
                f"{sorted(self.layer_specs)}."
            )
        unused = sorted(set(self.layer_specs) - {symbol.value for symbol in layer_symbols})
        if unused:
            logger.warning("layer_specs provides unused layer types %s for pattern '%s'.", unused, self.layer_pattern)

        if self.enforce_tensor_core_alignment:
            for name, value in (("n_embd", self.n_embd), ("vocab_size", self.vocab_size)):
                if value % _TENSOR_CORE_ALIGNMENT != 0:
                    raise ValueError(
                        f"{name} with value {value} should be divisible by {_TENSOR_CORE_ALIGNMENT} for "
                        f"efficient training. Set enforce_tensor_core_alignment=False to override."
                    )
        return self


class NemotronLLM(NNModel):
    """A hybrid Mamba-Transformer decoder-only language model."""

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        sequence_length: int,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        layer_pattern: str,
        layer_specs: dict[str, NemotronLayerSpecIF],
        lm_head_norm_config: NormWrapperConfig,
        loop_config: Optional[LoopConfig] = None,
        use_weight_tying: bool = False,
        aux_loss_key: Optional[str] = None,
    ):
        """
        Initializes the NemotronLLM.

        Args:
            sample_key (str): Key under which the input token ids are found.
            prediction_key (str): Key under which the logits are stored.
            sequence_length (int): Maximum supported sequence length.
            vocab_size (int): Vocabulary size.
            n_embd (int): Model dimension.
            n_layer (int): Number of layers to build.
            layer_pattern (str): One symbol per layer, optionally containing loop groups.
            layer_specs (dict[str, NemotronLayerSpecIF]): Builders keyed by layer pattern symbol.
            lm_head_norm_config (NormWrapperConfig): Normalization before the language model head.
            loop_config (LoopConfig | None): How loop group iterations are combined. Defaults to
                the simple (chained) strategy.
            use_weight_tying (bool): Whether to tie the embedding and the output projection.
            aux_loss_key (str | None): Key under which the summed MoE auxiliary loss is exposed.
        """
        weight_decay_groups = {
            "linear": [
                r"\.attn\.",
                r"\.mixer\.in_proj",
                r"\.mixer\.out_proj",
                r"\.mlp\.",
                r"\.shared_experts\.",
                r"\.lm_head\.weight",
            ],
            "experts": [r"\.experts\.w1", r"\.experts\.w2"],
            "router": [r"\.router\.gate"],
            "embedding": [r"\.wte\."],
            # A per-iteration norm is named "...norm.norms.<iteration>.weight", which the first
            # regex already matches. The second is defensive: a parameter matching no group at all
            # is not merely undecayed, it is dropped from the optimizer entirely.
            "layernorm": [r"\.norm\.", r"\.norms\.", r"\.lm_head_norm\."],
            "ssm": [r"\.A_log", r"\.D$", r"\.dt_bias", r"\.conv1d_weight", r"\.conv1d_bias"],
            # Per-loop-group iteration-conditioning tables (LoopIterationConditioning). Matched by
            # exact parameter name rather than by the `.loop_mods.` prefix so that adding a
            # differently-shaped parameter to that module later cannot land here by accident. A
            # parameter matching two groups is appended to both optimizer groups and makes the
            # optimizer raise; one matching no group is dropped from the optimizer silently.
            # Like `ssm` and `layernorm`, this group belongs in `weight_decay_groups_excluded`:
            # every parameter in it is a per-channel scale or shift, not a weight matrix.
            "loop": [r"\.loop_mods\.\d+\.(iter_scale|iter_shift)$"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups)

        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.aux_loss_key = aux_loss_key
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_pattern = layer_pattern
        self.loop_config = loop_config if loop_config is not None else LoopConfig()
        # When True, forward returns the post-norm hidden states instead of logits so that a
        # memory-efficient loss (e.g. ChunkedCLMCrossEntropyLoss) can apply the head in chunks.
        self._skip_lm_head = False

        layer_symbols, self._schedule = parse_layer_schedule(layer_pattern)
        if len(layer_symbols) != n_layer:
            raise ValueError(
                f"n_layer ({n_layer}) does not match the number of layers built by layer_pattern "
                f"('{layer_pattern}' builds {len(layer_symbols)} layers)."
            )
        specs_by_symbol = {LayerSymbol(symbol): spec for symbol, spec in layer_specs.items()}
        missing = sorted({symbol.value for symbol in layer_symbols} - {s.value for s in specs_by_symbol})
        if missing:
            raise ValueError(f"No layer spec provided for layer types {missing}.")

        # How often each built layer is executed, so that a layer inside a loop group can be given
        # one pre-norm per iteration. Every built layer belongs to exactly one group.
        loops_per_layer = {key: group.num_loops for group in self._schedule for key in group.layer_keys}

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                # A ModuleDict (rather than a ModuleList) is required by the Modalities activation
                # checkpointing component and matches the GPT2 layout.
                h=nn.ModuleDict(
                    {
                        str(layer_idx): specs_by_symbol[symbol].build(
                            layer_idx=layer_idx,
                            num_norm_iterations=(
                                loops_per_layer[str(layer_idx)] if self.loop_config.per_iteration_norm else 1
                            ),
                        )
                        for layer_idx, symbol in enumerate(layer_symbols)
                    }
                ),
                lm_head_norm=lm_head_norm_config.build(),
                lm_head=nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False),
            )
        )

        # Per-group conditioning tables, keyed by the group's index in the schedule. Built only for
        # groups that actually loop and only when conditioning is on, so a model with the defaults
        # has no `loop_mods` submodule at all and its state dict is byte-for-byte the shape it was
        # before this existed. Lives beside `h` rather than inside it because these parameters
        # belong to a *group* of layers, and `transformer.h` is keyed by single layer. The attribute
        # is still `loop_mods` (rather than something matching the class name) because it is part of
        # every conditioned checkpoint's parameter names, including the FiLM arms of the ablation.
        if self.loop_config.needs_group_modulation:
            self.transformer["loop_mods"] = nn.ModuleDict(
                {
                    str(group_idx): LoopIterationConditioning(
                        n_embd=n_embd,
                        num_loops=group.num_loops,
                        iteration_embedding=self.loop_config.iteration_embedding,
                    )
                    for group_idx, group in enumerate(self._schedule)
                    if group.num_loops > 1
                }
            )

        if use_weight_tying:
            self.transformer.wte.weight = self.transformer.lm_head.weight

    @property
    def n_executed_layers(self) -> int:
        """
        The number of layer applications per forward pass, counting each loop iteration.

        This, not ``n_layer``, is the model's effective depth: it is what the residual stream sees
        and therefore what depth-scaled weight initialization must be scaled by.

        Returns:
            int: The effective depth.
        """
        return sum(group.num_executed_layers for group in self._schedule)

    @property
    def num_per_iteration_norm_parameters(self) -> int:
        """
        The parameters that exist only because ``loop_config.per_iteration_norm`` is enabled.

        Iteration 0 reuses the norm the layer would have had anyway, so this counts the norms of
        iterations 1 and up: ``(K - 1) * n_embd`` per looped layer. It is what makes a
        per-iteration-norm arm not *exactly* iso-parameter with the baseline, and is worth
        reporting alongside such an arm's results.

        Returns:
            int: The number of extra parameters, zero when the flag is off or nothing is looped.
        """
        extra = 0
        for module in self.modules():
            if isinstance(module, PerIterationNorm):
                extra += sum(parameter.numel() for norm in module.norms[1:] for parameter in norm.parameters())
        return extra

    @property
    def num_loop_refinement_parameters(self) -> int:
        """
        The parameters that exist only because a per-group loop refinement is enabled.

        Counts the iteration-conditioning tables held by the :class:`LoopIterationConditioning`
        modules. They are per-channel, so this is kilobytes against a billion-parameter model, but
        an arm using them is not *exactly* iso-parameter with the baseline and this is the figure to
        report alongside its results (as :attr:`num_per_iteration_norm_parameters` is for the
        per-layer refinement).

        Returns:
            int: The number of extra parameters, zero when conditioning is off.
        """
        return sum(
            parameter.numel()
            for module in self.modules()
            if isinstance(module, LoopIterationConditioning)
            for parameter in module.parameters()
        )

    def get_execution_counts(self) -> dict[str, int]:
        """
        Returns how often each built layer is executed per forward pass.

        Used by the MFU calculator, for which a looped layer's parameters count once per iteration.

        Returns:
            dict[str, int]: Layer key (as used in ``transformer.h``) to execution count.
        """
        counts: dict[str, int] = {}
        for group in self._schedule:
            for layer_key in group.layer_keys:
                counts[layer_key] = counts.get(layer_key, 0) + group.num_loops
        return counts

    @property
    def lm_head(self) -> nn.Module:
        """The language-model head. Exposed so a memory-efficient loss can apply it chunk-by-chunk."""
        return self.transformer.lm_head

    def set_skip_lm_head(self, skip: bool) -> None:
        """
        Toggles whether forward returns post-norm hidden states instead of logits.

        Args:
            skip (bool): True to return hidden states, False to return logits.
        """
        self._skip_lm_head = skip

    @property
    def has_tied_word_embeddings(self) -> bool:
        """
        Whether the token embedding and the output projection currently share a weight tensor.

        Under pipeline parallelism a stage may contain neither submodule, in which case there is no
        tying to report.

        Returns:
            bool: True if the weights are tied.
        """
        if "wte" not in self.transformer or "lm_head" not in self.transformer:
            return False
        return self.transformer.wte.weight is self.transformer.lm_head.weight

    def get_moe_layers(self) -> list[nn.Module]:
        """
        Returns every mixture-of-experts block contained in this model (or pipeline stage).

        Returns:
            list[nn.Module]: The MoE blocks, in model order.
        """
        # Imported here to keep the module import graph acyclic at definition time.
        from modalities.models.components.moe.moe import MoE

        return [module for module in self.modules() if isinstance(module, MoE)]

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """
        Sums the auxiliary load-balancing losses recorded by the MoE layers in the last forward pass.

        Returns:
            torch.Tensor | None: The summed auxiliary loss, or None if no MoE layer produced one
                (either because no MoE layers exist on this stage or because the coefficient is 0).
        """
        losses = [moe.last_aux_loss for moe in self.get_moe_layers() if moe.last_aux_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).sum()

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): Either a dict containing the input
                token ids under ``sample_key``, or the token id tensor directly (used by pipeline
                parallelism, which passes raw tensors between stages).

        Returns:
            dict[str, torch.Tensor] | torch.Tensor: A dict with the logits under ``prediction_key``
                (and the auxiliary loss under ``aux_loss_key`` if configured), or the logits tensor.
        """
        if not isinstance(inputs, dict):
            return self.forward_impl(inputs)

        logits = self.forward_impl(inputs[self.sample_key])
        # Insertion order matters: InferenceResultBatch derives its length from the first entry,
        # so the token-shaped logits have to come first.
        predictions = {self.prediction_key: logits}
        if self.aux_loss_key is not None:
            aux_loss = self.get_aux_loss()
            if aux_loss is not None:
                predictions[self.aux_loss_key] = aux_loss
        return predictions

    def _run_loop_group(self, group: LoopGroup, group_idx: int, h: torch.Tensor) -> torch.Tensor:
        """
        Executes one group of the schedule, applying its layers ``group.num_loops`` times.

        This is the single place where loop semantics live. All refinements are off by default, and
        each is an exact no-op on a group that does not loop, so enabling any of them leaves a
        non-looped arm bit-identical.

        Per-layer refinement:

        * ``per_iteration_norm`` passes the iteration index into the layer, which then applies that
          iteration's own pre-normalization. The layers were built with the matching number of
          norms; nothing here needs to know how many.

        Per-group refinements:

        * ``iteration_embedding`` conditions the hidden states on the iteration index before each
          pass, so the shared weights can tell which iteration they are executing. Carried by this
          group's :class:`LoopIterationConditioning`.
        * ``input_injection`` adds the group's input back to the hidden states at the start of
          every iteration after the first, so every iteration operates one residual step away from
          what the group was handed rather than drifting further from it with each pass. Injecting
          *before* an iteration rather than after one is what makes this an exact no-op for
          non-looped groups.

        Under pipeline parallelism a stage holds only some layers, so keys absent from this stage's
        ``transformer.h`` are skipped.

        Args:
            group (LoopGroup): The group to execute.
            group_idx (int): Index of the group in the schedule, which keys its conditioning.
            h (torch.Tensor): Hidden states of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: Hidden states of shape ``(B, L, n_embd)``.
        """
        # A single path rather than a refined one beside a fast one: every refinement below is a
        # no-op when disabled, so the disabled case executes exactly the operations it did before
        # they existed, and there is no second copy of the loop to drift out of sync.
        conditioning = self._iteration_conditioning(group_idx)
        injected = h
        for iteration in range(group.num_loops):
            if conditioning is not None:
                h = conditioning.condition(h, iteration)
            if self.loop_config.input_injection and iteration > 0:
                h = h + injected
            for layer_key in group.layer_keys:
                if layer_key in self.transformer.h:
                    layer = self.transformer.h[layer_key]
                    h = layer(h, iteration) if self.loop_config.per_iteration_norm else layer(h)
        return h

    def _iteration_conditioning(self, group_idx: int) -> Optional[LoopIterationConditioning]:
        """
        Returns the per-group iteration-conditioning module for a group, if it has one.

        Absent for non-looped groups, for every group when conditioning is off, and on a pipeline
        stage that does not hold this group.

        Args:
            group_idx (int): Index of the group in the schedule.

        Returns:
            LoopIterationConditioning | None: The module, or None.
        """
        if "loop_mods" not in self.transformer:
            return None
        # nn.ModuleDict has no .get(); membership then index is the only safe access.
        key = str(group_idx)
        return self.transformer.loop_mods[key] if key in self.transformer.loop_mods else None

    def forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation operating on plain tensors.

        Every submodule access is guarded by ``hasattr`` so that a pipeline stage holding only a
        subset of the model still works.

        Args:
            inputs (torch.Tensor): Input token ids of shape ``(B, L)``, or hidden states of shape
                ``(B, L, n_embd)`` on a non-first pipeline stage.

        Returns:
            torch.Tensor: Logits of shape ``(B, L, vocab_size)``, or hidden states if the head is
                skipped or absent.
        """
        seq_len = inputs.size(1)
        if seq_len > self.sequence_length:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; the model's maximum input sequence "
                f"length is {self.sequence_length}."
            )

        # MoE layers accumulate their auxiliary loss across the visits of one forward pass, so the
        # accumulator has to start empty. Matters only for looped MoE layers, which are visited
        # more than once, but is unconditional to keep the two paths identical.
        for moe_layer in self.get_moe_layers():
            moe_layer.reset_aux_loss()

        h = self.transformer.wte(inputs) if hasattr(self.transformer, "wte") else inputs

        for group_idx, group in enumerate(self._schedule):
            h = self._run_loop_group(group, group_idx, h)

        h = self.transformer.lm_head_norm(h) if hasattr(self.transformer, "lm_head_norm") else h
        if self._skip_lm_head:
            return h
        return self.transformer.lm_head(h) if hasattr(self.transformer, "lm_head") else h
