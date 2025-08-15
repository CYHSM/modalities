import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.optim import Optimizer

from modalities.optimizers.optimizer_factory import get_optimizer_groups


class MuonClip(Optimizer):
    """
    MuonClip optimizer combining Muon with QK-Clip for large-scale training stability.
    Based on: "Kimi K2: Scaling Up Quantized Training with MuonClip"

    Combines:
    1. Muon optimizer with Newton-Schulz orthogonalization
    2. Weight decay for stability
    3. RMS matching for Adam compatibility
    4. QK-Clip to prevent attention logits explosion
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        qk_clip_threshold: float = 100.0,
        ns_steps: int = 5,
        update_rms_scale: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            qk_clip_threshold=qk_clip_threshold,
            ns_steps=ns_steps,
            update_rms_scale=update_rms_scale,
        )
        super().__init__(params, defaults)
        self.model = None
        self.attention_modules = []

    def set_model(self, model: nn.Module):
        self.model = model
        self._find_attention_modules()

    def _find_attention_modules(self):
        if self.model is None:
            return
        self.attention_modules = []
        if isinstance(self.model, (FSDP1, FSDP2)):
            if hasattr(self.model, "module"):
                base_model = self.model.module  # FSDP1
            else:
                base_model = self.model  # FSDP2
        else:
            base_model = self.model

        # Find attention modules
        for name, module in base_model.named_modules():
            # Look for attention modules by name patterns
            if any(pattern in name.lower() for pattern in ["attention", "attn", "self_attn"]):
                # Store the module and its name for debugging
                self.attention_modules.append((name, module))

    def newton_schulz_step(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Perform Newton-Schulz orthogonalization.
        """
        # Tuned coefficients from the paper aka The Magic Values
        a, b, c = 3.4445, -4.7750, 2.0315

        # Initialize with normalized input
        X = G / (G.norm(dtype=torch.float32) + self.defaults["eps"])

        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        return X

    def compute_max_qk_score(self) -> Optional[float]:
        """
        Compute the maximum QK attention score across all attention heads.
        This is a simplified version - in production you'd track this during forward pass.
        """
        if not self.attention_modules:
            return None

        max_score = 0.0

        for name, module in self.attention_modules:
            # Look for Q and K projection weights
            q_weight = None
            k_weight = None

            # Common attribute names for attention projections
            for q_name in ["q_proj", "query", "q_lin", "q"]:
                if hasattr(module, q_name):
                    q_weight = getattr(module, q_name)
                    if hasattr(q_weight, "weight"):
                        q_weight = q_weight.weight
                    break

            for k_name in ["k_proj", "key", "k_lin", "k"]:
                if hasattr(module, k_name):
                    k_weight = getattr(module, k_name)
                    if hasattr(k_weight, "weight"):
                        k_weight = k_weight.weight
                    break

            if q_weight is not None and k_weight is not None:
                # Simplified score computation
                with torch.no_grad():
                    # Compute approximate max attention score
                    q_norm = q_weight.norm(dim=-1).max()
                    k_norm = k_weight.norm(dim=-1).max()
                    hidden_size = q_weight.size(-1)
                    score = (q_norm * k_norm / math.sqrt(hidden_size)).item()
                    max_score = max(max_score, score)

        return max_score if max_score > 0 else None

    def apply_qk_clip(self):
        """
        Apply QK-Clip to attention weights to prevent logits explosion.
        """
        if not self.attention_modules:
            return

        threshold = self.defaults["qk_clip_threshold"]
        max_score = self.compute_max_qk_score()

        if max_score is None or max_score <= threshold:
            return

        # Compute clipping ratio
        clip_ratio = math.sqrt(threshold / max_score)

        for name, module in self.attention_modules:
            # Apply clipping to Q and K projections
            for q_name in ["q_proj", "query", "q_lin", "q"]:
                if hasattr(module, q_name):
                    q_module = getattr(module, q_name)
                    if hasattr(q_module, "weight"):
                        with torch.no_grad():
                            q_module.weight.mul_(clip_ratio)

            for k_name in ["k_proj", "key", "k_lin", "k"]:
                if hasattr(module, k_name):
                    k_module = getattr(module, k_name)
                    if hasattr(k_module, "weight"):
                        with torch.no_grad():
                            k_module.weight.mul_(clip_ratio)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First, apply standard Muon updates
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            update_rms_scale = group["update_rms_scale"]
            ns_steps = group["ns_steps"]
            group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                grad = p.grad

                # State initialization
                if len(param_state) == 0:
                    param_state["step"] = 0
                    param_state["momentum_buffer"] = torch.zeros_like(p)

                momentum_buf = param_state["momentum_buffer"]
                param_state["step"] += 1

                # Apply weight decay directly to gradient
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update momentum buffer
                momentum_buf.mul_(momentum).add_(grad)

                # Apply Muon orthogonalization for 2D parameters
                if p.dim() >= 2:
                    # For matrices, apply Newton-Schulz orthogonalization
                    if p.dim() == 2:
                        update = self.newton_schulz_step(momentum_buf, ns_steps)
                    else:
                        # For higher-dimensional tensors, flatten to 2D
                        orig_shape = momentum_buf.shape
                        flat_momentum = momentum_buf.view(orig_shape[0], -1)
                        update = self.newton_schulz_step(flat_momentum, ns_steps)
                        update = update.view(orig_shape)

                    # Apply RMS scaling to match Adam updates
                    max_dim = max(update.shape[-2:]) if update.dim() >= 2 else 1
                    rms_scale = math.sqrt(max_dim) * update_rms_scale
                    update = update * rms_scale
                else:
                    # For 1D parameters (biases, etc.), use momentum directly
                    update = momentum_buf

                # Apply update
                p.add_(update, alpha=-lr)

        # After all parameter updates, apply QK-Clip
        self.apply_qk_clip()

        return loss


class MuonClipOptimizerFactory:
    """Factory class for creating MuonClip optimizers compatible with Modalities."""

    @staticmethod
    def get_muonclip(
        lr: float,
        momentum: float,
        weight_decay: float,
        qk_clip_threshold: float,
        weight_decay_groups_excluded: list[str],
        wrapped_model: nn.Module,
        ns_steps: int = 5,
        update_rms_scale: float = 0.2,
        eps: float = 1e-8,
    ) -> MuonClip:
        """
        Create MuonClip optimizer with FSDP-compatible parameter groups.

        Args:
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay factor
            qk_clip_threshold: QK clipping threshold
            weight_decay_groups_excluded: Parameter groups to exclude from weight decay
            wrapped_model: FSDP-wrapped model
            ns_steps: Number of Newton-Schulz iteration steps
            update_rms_scale: RMS scaling factor
            eps: Epsilon for numerical stability
        """
        # Get parameter groups with weight decay handling
        optimizer_groups = get_optimizer_groups(wrapped_model, weight_decay, weight_decay_groups_excluded)

        # Create MuonClip optimizer
        optimizer = MuonClip(
            params=optimizer_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            qk_clip_threshold=qk_clip_threshold,
            ns_steps=ns_steps,
            update_rms_scale=update_rms_scale,
            eps=eps,
        )

        # Set model reference for QK-Clip functionality
        optimizer.set_model(wrapped_model)

        return optimizer
