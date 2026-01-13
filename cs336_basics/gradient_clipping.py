"""
Gradient clipping implementation.
"""

import torch
from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip the combined gradients of parameters to have L2 norm at most max_l2_norm.

    Given the gradient g for all parameters, compute its L2-norm ||g||_2.
    - If ||g||_2 < max_l2_norm, leave g as is
    - Otherwise, scale g down by max_l2_norm / (||g||_2 + ε), where ε = 1e-6

    Args:
        parameters: Collection of trainable parameters with gradients
        max_l2_norm: Maximum L2-norm for the combined gradients

    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    # Collect all gradients
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)

    if not gradients:
        return

    # Compute the total L2 norm of all gradients
    # ||g||_2 = sqrt(sum of squares of all gradient elements)
    total_norm = torch.sqrt(
        sum(torch.sum(grad * grad) for grad in gradients)
    )

    # Small epsilon for numerical stability
    eps = 1e-6

    # Compute clipping factor
    # If total_norm <= max_l2_norm, factor will be >= 1, so no clipping
    # If total_norm > max_l2_norm, factor < 1, so gradients get scaled down
    clip_factor = max_l2_norm / (total_norm + eps)

    # Only clip if necessary (when total_norm > max_l2_norm)
    if clip_factor < 1.0:
        # Scale all gradients in-place
        for grad in gradients:
            grad.mul_(clip_factor)
