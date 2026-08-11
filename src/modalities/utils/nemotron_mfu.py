"""Model FLOPs Utilization for the Nemotron hybrid Mamba-Transformer.

The GPT2 estimate ``6 * num_params + 12 * n_layer * seq_len * n_embd`` is wrong for this
architecture on two counts:

1. It counts *all* parameters. In a 128-expert MoE only the 6 routed experts a token actually visits
   contribute FLOPs, so using total parameters would overstate the work by roughly an order of
   magnitude and report an absurdly high MFU.
2. Its second term is the attention score/value cost, which assumes *every* layer attends. In
   Nemotron-3 Nano only 6 of 52 layers do; the other 46 are Mamba-2 or MoE layers whose cost is
   linear in the sequence length.

This calculator therefore takes the number of *active* parameters and adds the quadratic attention
term only for the attention layers.
"""

from typing import Annotated, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field

from modalities.config.pydantic_if_types import PydanticPytorchModuleOrListType
from modalities.models.nemotron.layer_pattern import LayerSymbol, count_layers_by_type
from modalities.utils.mfu import MFUCalculatorABC


class NemotronMFUCalculatorConfig(BaseModel):
    """
    Configuration of :class:`NemotronMFUCalculator`.

    Attributes:
        layer_pattern (str): The model's layer pattern, used to count attention layers.
        sequence_length (int): Training sequence length.
        n_embd (int): Model dimension.
        n_head_q (int): Number of query heads of the attention layers.
        head_dim (int): Head dimension of the attention layers.
        num_active_params (int | None): Number of parameters visited per token. Derived from
            ``model_parts`` when omitted, which is what you normally want: hardcoding it in a config
            silently goes stale as soon as the layer pattern or expert count changes.
        world_size (int): Number of ranks.
        model_parts (nn.Module | list[nn.Module]): The wrapped model (or pipeline stages).
        device_mesh (DeviceMesh | None): The device mesh, if any.
    """

    layer_pattern: str
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    head_dim: Annotated[int, Field(strict=True, ge=1)]
    num_active_params: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    world_size: Annotated[int, Field(strict=True, ge=1)]
    model_parts: PydanticPytorchModuleOrListType
    device_mesh: Optional[object] = None

    # avoid the pydantic warning about the protected 'model_' namespace
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)


class NemotronMFUCalculator(MFUCalculatorABC):
    """Computes the Model FLOPs Utilization of a hybrid Mamba-Transformer MoE model."""

    def __init__(
        self,
        layer_pattern: str,
        sequence_length: int,
        n_embd: int,
        n_head_q: int,
        head_dim: int,
        num_active_params: Optional[int] = None,
        world_size: int = 1,
        model_parts: torch.nn.Module | list[torch.nn.Module] = None,
        device_mesh: Optional[object] = None,
    ):
        """
        Initializes the NemotronMFUCalculator.

        Args:
            layer_pattern (str): The model's layer pattern.
            sequence_length (int): Training sequence length.
            n_embd (int): Model dimension.
            n_head_q (int): Number of query heads of the attention layers.
            head_dim (int): Head dimension of the attention layers.
            num_active_params (int | None): Number of parameters visited per token. Derived from
                ``model_parts`` when omitted.
            world_size (int): Number of ranks.
            model_parts (nn.Module | list[nn.Module]): The wrapped model or pipeline stages.
            device_mesh (DeviceMesh | None): The device mesh, if any.

        Raises:
            ValueError: If ``num_active_params`` is omitted and cannot be derived.
        """
        if num_active_params is None:
            num_active_params = NemotronMFUCalculator.count_active_parameters(model_parts)
        self._sequence_length = sequence_length
        self._num_attention_layers = count_layers_by_type(layer_pattern)[LayerSymbol.ATTENTION]
        self._theoretical_flops_per_token = NemotronMFUCalculator._get_theoretical_flops_per_token(
            num_active_params=num_active_params,
            num_attention_layers=self._num_attention_layers,
            sequence_length=sequence_length,
            n_head_q=n_head_q,
            head_dim=head_dim,
        )
        self._theoretical_gpu_peak_performance = MFUCalculatorABC._get_theoretical_gpu_peak_performance(
            model_parts, world_size
        )
        del n_embd, device_mesh  # part of the public config for symmetry with the GPT2 calculator

    @staticmethod
    def _get_theoretical_flops_per_token(
        num_active_params: int,
        num_attention_layers: int,
        sequence_length: int,
        n_head_q: int,
        head_dim: int,
    ) -> int:
        """
        Estimates the forward-plus-backward FLOPs per token.

        The first term is the usual ``6 * active_params`` (2 for the forward matmuls, 4 for the
        backward). The second term is the attention score and value matmuls, which are quadratic in
        the sequence length and only incurred by the attention layers:
        ``12 * num_attention_layers * seq_len * n_head_q * head_dim``.

        Args:
            num_active_params (int): Parameters visited per token.
            num_attention_layers (int): Number of attention layers.
            sequence_length (int): Sequence length.
            n_head_q (int): Number of query heads.
            head_dim (int): Attention head dimension.

        Returns:
            int: Estimated FLOPs per token.
        """
        dense_flops = 6 * num_active_params
        attention_flops = 12 * num_attention_layers * sequence_length * n_head_q * head_dim
        return dense_flops + attention_flops

    @staticmethod
    def count_active_parameters(model: torch.nn.Module) -> int:
        """
        Counts the parameters a single token actually visits.

        Every parameter counts once, except:

        * the routed experts of an MoE layer, of which only ``top_k`` of ``num_experts`` are
          evaluated per token, so their contribution is scaled accordingly (the router, the shared
          experts and all dense layers are always active); and
        * the parameters of a **looped** layer, which a token visits once per loop iteration and
          which therefore count once per iteration. Without this the FLOPs of a looped model would
          be understated and its MFU correspondingly overstated.

        With ``loop_config.per_iteration_norm`` a looped layer holds ``K`` norms of which each
        iteration visits one, while this counts all ``K`` on every iteration. The overcount is
        ``K * (K - 1) * n_embd`` per looped layer -- 2048 parameters of 225M for the ``K=2`` MoE
        arm -- i.e. far below the accuracy of the estimate itself, and not worth a special case
        that would have to be kept in sync with the layer internals.

        Works on plain and FSDP2-wrapped models: ``numel()`` on a ``DTensor`` reports the unsharded
        size, so the result is the global count regardless of sharding.

        Args:
            model (nn.Module | list[nn.Module]): The model. A list of pipeline stages is rejected,
                because summing per-stage counts would not give the whole model's active parameters.

        Raises:
            ValueError: If ``model`` is None or a list of pipeline stages.

        Returns:
            int: The number of active parameters.
        """
        if model is None:
            raise ValueError("num_active_params was omitted but no model was provided to derive it from.")
        if isinstance(model, list):
            raise ValueError(
                "Cannot derive num_active_params from a list of pipeline stages, since each stage holds "
                "only a subset of the layers. Pass num_active_params explicitly."
            )

        layers = NemotronMFUCalculator._get_layer_module_dict(model)
        if layers is None:
            # No recognizable layer stack (e.g. a bare module in a test): count everything once.
            return NemotronMFUCalculator._count_active_parameters_of_module(model)

        execution_counts = NemotronMFUCalculator._get_execution_counts(model)
        # Everything outside the layer stack (embeddings, final norm, LM head) is visited once.
        layer_parameter_ids = {id(p) for p in layers.parameters()}
        total = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in layer_parameter_ids)
        for layer_key, layer in layers.items():
            total += NemotronMFUCalculator._count_active_parameters_of_module(layer) * execution_counts.get(
                layer_key, 1
            )
        return total

    @staticmethod
    def _get_layer_module_dict(model: torch.nn.Module) -> Optional[torch.nn.ModuleDict]:
        """
        Returns the model's layer ``ModuleDict``, or None if the model does not expose one.

        Args:
            model (nn.Module): The (possibly wrapped) model.

        Returns:
            nn.ModuleDict | None: The layer stack.
        """
        transformer = getattr(model, "transformer", None)
        return getattr(transformer, "h", None) if transformer is not None else None

    @staticmethod
    def _get_execution_counts(model: torch.nn.Module) -> dict[str, int]:
        """
        Returns how often each layer of the stack is executed per forward pass.

        Args:
            model (nn.Module): The (possibly wrapped) model.

        Returns:
            dict[str, int]: Layer key to execution count; empty if the model does not report any,
                in which case every layer is assumed to run once.
        """
        get_counts = getattr(model, "get_execution_counts", None)
        return get_counts() if callable(get_counts) else {}

    @staticmethod
    def _count_active_parameters_of_module(module: torch.nn.Module) -> int:
        """
        Counts the parameters of one module that a single token visits, discounting routed experts.

        Args:
            module (nn.Module): The module, typically one layer.

        Returns:
            int: The number of active parameters.
        """
        from modalities.models.components.moe.moe import MoE

        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        for submodule in module.modules():
            if not isinstance(submodule, MoE):
                continue
            routed = sum(p.numel() for p in submodule.experts.parameters())
            active_fraction = submodule.router.top_k / submodule.router.num_experts
            total -= int(routed * (1.0 - active_fraction))
        return total

    def compute(self, num_samples_per_second: torch.Tensor) -> torch.Tensor:
        """
        Computes the MFU for a given throughput.

        Args:
            num_samples_per_second (torch.Tensor): Observed samples per second.

        Returns:
            torch.Tensor: The model FLOPs utilization.
        """
        return MFUCalculatorABC._compute_mfu_impl(
            num_samples_per_second=num_samples_per_second,
            sequence_length=self._sequence_length,
            theoretical_flops_per_token=self._theoretical_flops_per_token,
            theoretical_gpu_peak_performance=self._theoretical_gpu_peak_performance,
        )
