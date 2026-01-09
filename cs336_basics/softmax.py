"""
Softmax function with numerical stability.
"""

import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # For numerical stability, subtract the max value along the dimension
    # This prevents overflow when computing exp(large_value)
    # softmax(x) = softmax(x - c) for any constant c
    max_vals = torch.max(in_features, dim=dim, keepdim=True).values

    # Subtract max from all elements (numerical stability trick)
    shifted = in_features - max_vals

    # Apply exponential
    exp_vals = torch.exp(shifted)

    # Sum exp values along the dimension
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)

    # Normalize: divide each exp value by the sum
    return exp_vals / sum_exp
