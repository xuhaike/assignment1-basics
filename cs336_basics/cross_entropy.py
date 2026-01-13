"""
Cross-entropy loss implementation with numerical stability.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """
    Compute the cross-entropy loss with numerical stability.

    The cross-entropy loss is defined as:
        ℓ = -log(softmax(logits)[target])

    Using the log-sum-exp trick for numerical stability:
        ℓ = -logits[target] + log(Σ exp(logits - max(logits)))

    This simplifies the computation and cancels log and exp where possible.

    Args:
        logits: Unnormalized logits of shape (..., vocab_size)
        targets: Target indices of shape (...), each value in [0, vocab_size-1]

    Returns:
        Scalar tensor with the average cross-entropy loss across all examples
    """
    # Step 1: Subtract max for numerical stability
    # Shape: (..., vocab_size)
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits_shifted = logits - max_logits

    # Step 2: Compute log-sum-exp
    # log(Σ exp(logits - max)) = log(Σ exp(logits_shifted))
    # Shape: (..., 1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True))

    # Step 3: Compute log-softmax
    # log_softmax = (logits - max) - log(Σ exp(logits - max))
    # Shape: (..., vocab_size)
    log_softmax = logits_shifted - log_sum_exp

    # Step 4: Gather the log probabilities for the target indices
    # We need to index log_softmax[..., targets[...]]
    # Shape: (...)
    # Flatten batch dimensions for easier indexing
    original_shape = targets.shape
    batch_size = targets.numel()

    # Reshape to (batch_size, vocab_size) and (batch_size,)
    log_softmax_flat = log_softmax.view(batch_size, -1)
    targets_flat = targets.view(batch_size)

    # Gather the log probabilities at target indices
    # Shape: (batch_size,)
    log_probs = log_softmax_flat[torch.arange(batch_size, device=targets.device), targets_flat]

    # Step 5: Compute negative log-likelihood (cross-entropy)
    # Shape: (batch_size,)
    loss_per_example = -log_probs

    # Step 6: Return average across all examples
    # Shape: scalar
    return loss_per_example.mean()
