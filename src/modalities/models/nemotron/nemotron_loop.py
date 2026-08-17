"""Per-loop-group parameters for iteration conditioning.

A loop group applies one set of layer weights ``K`` times. Nothing in that arrangement tells the
shared weights *which* iteration they are executing -- iteration 0 and iteration K-1 see the same
parameters and differ only in what the residual stream happens to contain. Iteration conditioning
gives them that signal: a learned per-iteration shift (``"add"``), or a FiLM-style per-iteration
scale and shift (``"film"``, arXiv:2606.04678), applied to the hidden states at the start of each
pass. It is a strictly more expressive version of what ``per_iteration_norm`` already does through
the pre-norm gain, and it is the one loop refinement that survived the K in {3, 6, 12} ablation
(``docs/loopotron/loopotron.tex``, section 3.7): it helped at every depth, with the margin growing
in K, and cut the peak post-warmup gradient norm at K=12 by a factor of ~37.

Two other refinements were implemented, ablated and then removed, which is worth recording so they
are not re-derived from the same papers a second time:

* **Stabilized recurrence** (Parcae, arXiv:2604.12946) bounded the loop's transition eigenvalues in
  ``(0, 1)`` by construction. Against a *real* instability at K=12 it moved the loss by -0.0003
  (0.2 within-arm s.d.) and produced more large-gradient steps than plain looping, so it bought
  nothing at any depth we could reach.
* **Injection normalization** (Parcae's "prelude norm") was actively harmful: +0.0475 nats at K=12,
  and it dominated every combination it appeared in -- FiLM alone was -0.0109 while FiLM plus
  recurrence plus norm was +0.0186.

The conditioning tables are initialized to zero, so enabling conditioning is an exact no-op at step
0: it perturbs the trajectory of a known-good run rather than replacing it, and a conditioned arm
starts from the same loss as its control.

Parameter naming is load-bearing. ``NemotronLLM.weight_decay_groups`` matches these parameters by
name, and a parameter matching *no* group is dropped from the optimizer entirely, while one matching
*two* groups lands in both and makes the optimizer raise. ``iter_scale`` and ``iter_shift`` are
matched by the ``loop`` group and by nothing else.
"""

import torch
import torch.nn as nn

# How the group's input is driven back into the residual stream on each iteration.
INJECTION_MODES = frozenset({"add"})

# How the current iteration index is made visible to the shared weights.
ITERATION_EMBEDDINGS = frozenset({"none", "add", "film"})


class LoopIterationConditioning(nn.Module):
    """
    Holds the per-loop-group iteration-conditioning tables.

    One instance exists per *looped* group (``num_loops > 1``) when conditioning is enabled. Groups
    that are not looped never need one, and a model with conditioning off builds none at all, so its
    parameter names are exactly what they were before this module existed.

    The tables are ``(K, n_embd)``, so the total cost is a few kilobytes against a billion-parameter
    model -- which is what keeps a conditioned arm iso-parameter with its control for practical
    purposes. Report the exact figure with :attr:`NemotronLLM.num_loop_refinement_parameters`.
    """

    def __init__(self, n_embd: int, num_loops: int, iteration_embedding: str = "none"):
        """
        Initializes the per-group conditioning tables.

        Args:
            n_embd (int): Model dimension.
            num_loops (int): Iterations this group performs, which sizes the tables.
            iteration_embedding (str): One of :data:`ITERATION_EMBEDDINGS`.

        Raises:
            ValueError: If an unknown ``iteration_embedding`` is given.
        """
        super().__init__()
        if iteration_embedding not in ITERATION_EMBEDDINGS:
            raise ValueError(
                f"Unknown iteration_embedding '{iteration_embedding}'. Available: {sorted(ITERATION_EMBEDDINGS)}."
            )

        self.num_loops = num_loops
        self.iteration_embedding = iteration_embedding

        # Zero-initialized, which makes conditioning an exact no-op at step 0. Deliberately absent
        # from NAMED_PARAMETER_INIT_GROUPS: the model initializer's normal draw would destroy that
        # property, exactly as it would for the Mamba mixer's A_log and dt_bias.
        if iteration_embedding == "film":
            self.iter_scale = nn.Parameter(torch.zeros(num_loops, n_embd))
        if iteration_embedding in ("add", "film"):
            self.iter_shift = nn.Parameter(torch.zeros(num_loops, n_embd))

    def condition(self, h: torch.Tensor, iteration: int) -> torch.Tensor:
        """
        Applies this iteration's conditioning to the hidden states.

        Args:
            h (torch.Tensor): Hidden states of shape ``(B, L, n_embd)``.
            iteration (int): Index of the iteration about to be executed.

        Raises:
            IndexError: If ``iteration`` exceeds the number of iterations this group was built for.

        Returns:
            torch.Tensor: The conditioned hidden states, of the same shape.
        """
        if self.iteration_embedding == "none":
            return h
        if iteration >= self.num_loops:
            raise IndexError(
                f"Loop iteration {iteration} has no iteration embedding; this group was built for "
                f"{self.num_loops} iterations."
            )
        shift = self.iter_shift[iteration].to(h.dtype)
        if self.iteration_embedding == "add":
            return h + shift
        # FiLM: a per-channel rescale as well as a shift. Parameterized as (1 + scale) so that the
        # zero initialization is the identity rather than a multiplication by zero.
        scale = self.iter_scale[iteration].to(h.dtype)
        return h * (1.0 + scale) + shift
