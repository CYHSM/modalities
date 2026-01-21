from abc import ABC, abstractmethod
from typing import overload

import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch


class Loss(ABC):
    def __init__(self, tag: str):
        self._tag = tag

    @property
    def tag(self) -> str:
        return self._tag

    @abstractmethod
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Calculates the loss
        :return: Loss tensor
        """
        raise NotImplementedError


class CLMCrossEntropyLoss(Loss):
    def __init__(self, target_key: str, prediction_key: str, tag: str = "CLMCrossEntropyLoss"):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        # Mean over the tokens in the local-batch (batch per rank)
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, lm_logits = self._parse_arguments(args, kwargs)

        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()
        # Flatten the tokens. We compute here, the loss per token.
        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def _parse_arguments(
        self,
        args: list[torch.Tensor] | list[InferenceResultBatch],
        kwargs: dict[str, torch.Tensor] | dict[str, InferenceResultBatch],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        return labels, lm_logits


class CLMCrossEntropyWithPonderLoss(Loss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        tag: str = "CLMCrossEntropyWithPonderLoss"
    ):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.ce_loss_fun = CrossEntropyLoss(reduction="mean")
        
        self._last_ce_loss = None
        self._last_ponder_loss = None
        self._last_ponder_cost_unweighted = None
        self._last_expected_steps = None
        self._last_normalized_steps = None
        self._last_per_layer_ponder_costs = None
        self._last_per_layer_cos_sims = None
        self._last_loop_scales = None
        self._last_halt_probs = None
        self._last_local_mem_scales = None
        self._last_global_mem_scales = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, outputs = self._parse_arguments(args, kwargs)
        
        if isinstance(outputs, dict):
            lm_logits = outputs["logits"]
            ponder_loss = outputs.get("ponder_loss", torch.tensor(0.0, device=lm_logits.device))
            ponder_cost_unweighted = outputs.get("ponder_cost_unweighted", torch.tensor(0.0, device=lm_logits.device))
            expected_steps = outputs.get("expected_steps", torch.tensor(0.0, device=lm_logits.device))
            normalized_steps = outputs.get("normalized_steps", torch.tensor(0.0, device=lm_logits.device))
            per_layer_ponder_costs = outputs.get("per_layer_ponder_costs", None)
            per_layer_cos_sims = outputs.get("per_layer_cos_sims", None)
            loop_scales = outputs.get("loop_scales", None)
            halt_probs = outputs.get("halt_probs", None)
            local_mem_scales = outputs.get("local_mem_scales", None)
            global_mem_scales = outputs.get("global_mem_scales", None)
        else:
            lm_logits = outputs
            ponder_loss = torch.tensor(0.0, device=lm_logits.device)
            ponder_cost_unweighted = torch.tensor(0.0, device=lm_logits.device)
            expected_steps = torch.tensor(0.0, device=lm_logits.device)
            normalized_steps = torch.tensor(0.0, device=lm_logits.device)
            per_layer_ponder_costs = None
            per_layer_cos_sims = None
            loop_scales = None
            halt_probs = None
            local_mem_scales = None
            global_mem_scales = None

        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()
        ce_loss = self.ce_loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        self._last_ce_loss = ce_loss.detach()
        self._last_ponder_loss = ponder_loss.detach() if isinstance(ponder_loss, torch.Tensor) else torch.tensor(0.0)
        self._last_ponder_cost_unweighted = ponder_cost_unweighted.detach() if isinstance(ponder_cost_unweighted, torch.Tensor) else torch.tensor(0.0)
        self._last_expected_steps = expected_steps.detach() if isinstance(expected_steps, torch.Tensor) else torch.tensor(0.0)
        self._last_normalized_steps = normalized_steps.detach() if isinstance(normalized_steps, torch.Tensor) else torch.tensor(0.0)
        self._last_per_layer_ponder_costs = per_layer_ponder_costs.detach() if per_layer_ponder_costs is not None else None
        self._last_per_layer_cos_sims = per_layer_cos_sims.detach() if per_layer_cos_sims is not None else None
        self._last_loop_scales = loop_scales if loop_scales is not None else None
        self._last_halt_probs = halt_probs.detach() if halt_probs is not None else None
        self._last_local_mem_scales = local_mem_scales.detach() if local_mem_scales is not None else None
        self._last_global_mem_scales = global_mem_scales.detach() if global_mem_scales is not None else None
        
        total_loss = ce_loss + ponder_loss
        return total_loss
    
    def get_loss_components(self) -> dict[str, torch.Tensor]:
        return {
            "ce_loss": self._last_ce_loss if self._last_ce_loss is not None else torch.tensor(0.0),
            "ponder_loss": self._last_ponder_loss if self._last_ponder_loss is not None else torch.tensor(0.0),
            "ponder_cost_unweighted": self._last_ponder_cost_unweighted if self._last_ponder_cost_unweighted is not None else torch.tensor(0.0),
            "expected_steps": self._last_expected_steps if self._last_expected_steps is not None else torch.tensor(0.0),
            "normalized_steps": self._last_normalized_steps if self._last_normalized_steps is not None else torch.tensor(0.0),
            "per_layer_ponder_costs": self._last_per_layer_ponder_costs if self._last_per_layer_ponder_costs is not None else torch.tensor([]),
            "per_layer_cos_sims": self._last_per_layer_cos_sims if self._last_per_layer_cos_sims is not None else torch.tensor([]),
            "loop_scales": self._last_loop_scales if self._last_loop_scales is not None else torch.tensor([]),
            "halt_probs": self._last_halt_probs if self._last_halt_probs is not None else torch.tensor([]),
            "local_mem_scales": self._last_local_mem_scales if self._last_local_mem_scales is not None else torch.tensor([]),
            "global_mem_scales": self._last_global_mem_scales if self._last_global_mem_scales is not None else torch.tensor([]),
        }

    def _parse_arguments(
        self,
        args: list[torch.Tensor | dict] | list[InferenceResultBatch],
        kwargs: dict[str, torch.Tensor | dict] | dict[str, InferenceResultBatch],
    ) -> tuple[torch.Tensor, torch.Tensor | dict]:
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            outputs = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            outputs = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2:
            outputs, labels = args
        elif "outputs" in kwargs and "targets" in kwargs:
            outputs = kwargs["outputs"]
            labels = kwargs["targets"]
        elif len(args) == 1 and "targets" in kwargs:
            outputs = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyWithPonderLoss.__call__")
        
        return labels, outputs
    

def nce_loss(
    embedding1: torch.Tensor, embedding2: torch.Tensor, device: torch.device, is_asymmetric: bool, temperature: float
) -> torch.Tensor:
    """
    This implementation calculates the noise contrastive estimation loss between embeddings of two different modalities
    Implementation slightly adapted from https://arxiv.org/pdf/1912.06430.pdf, https://github.com/antoine77340/MIL-NCE_HowTo100M
    changes include adding a temperature value and the choice of calculating asymmetric loss w.r.t. one modality
    This implementation is adapted to contrastive loss from CoCa model https://arxiv.org/pdf/2205.01917.pdf

    Args:
        embedding1 (torch.Tensor): embeddings from modality 1 of size batch_size x embed_dim.
        embedding2 (torch.Tensor): embeddings from modality 2 of size batch_size x embed_dim.
        device (torch.device): torch device for calculating loss.
        is_asymmetric (bool): boolean value to specify if the loss is calculated in one direction or both directions.
        temperature (float): temperature value for regulating loss.

    Returns:
            torch.Tensor: loss tensor.
    """
    # calculating the similarity matrix of size (batch_size x batch_size)
    sim_matrix = torch.matmul(embedding1, embedding2.t()) / temperature
    # numerator of loss: using similarity scores for all positive pairs (e.g., image and its caption)
    numerator = sim_matrix * torch.eye(sim_matrix.shape[0], device=device)
    numerator = numerator.sum(dim=0).view(sim_matrix.shape[0], -1)
    numerator = torch.logsumexp(numerator, dim=1)
    if is_asymmetric:
        # denominator of loss: using all similarity scores for all pairs (positive and negative)
        denominator = torch.logsumexp(sim_matrix, dim=1)
    else:
        # calculate bidirectional loss
        numerator *= 2
        denominator = torch.logsumexp(sim_matrix, dim=1) + torch.logsumexp(sim_matrix.t(), dim=1)
    return torch.mean(denominator - numerator)  # calculated in log space


class NCELoss(Loss):
    def __init__(
        self,
        prediction_key1: str,
        prediction_key2: str,
        is_asymmetric: bool = True,
        temperature: float = 1.0,
        tag: str = "NCELoss",
    ):
        """
        Noise Contrastive Estimation Loss

        Args:
            prediction_key1 (str): key to access embedding 1.
            prediction_key2 (str): key to access embedding 2.
            is_asymmetric (bool, optional): specifies symmetric or asymmetric calculation of NCEloss. Defaults to True.
            temperature (float, optional): temperature. Defaults to 1.0.
            tag (str, optional): Defaults to "NCELoss".
        """
        super().__init__(tag)
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.is_asymmetric = is_asymmetric
        self.temperature = temperature

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch.

        Returns:
            torch.Tensor: loss tensor.
        """
        embedding1 = forward_batch.get_predictions(self.prediction_key1)
        embedding2 = forward_batch.get_predictions(self.prediction_key2)

        contiguous_embedding1 = embedding1.contiguous()
        contiguous_embedding2 = embedding2.contiguous()

        loss = nce_loss(
            contiguous_embedding1, contiguous_embedding2, embedding1.device, self.is_asymmetric, self.temperature
        )
        return loss
