"""
Root Mean Square Layer Normalization (RMSNorm).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes activations by dividing by the root mean square (RMS)
    and then scaling by learnable gain parameters.

    For a vector a ∈ R^d_model:
        RMSNorm(a_i) = (a_i / RMS(a)) * g_i
    where:
        RMS(a) = sqrt((1/d_model) * Σ(a_i^2) + ε)
        g_i are learnable gain parameters

    Args:
        d_model: Hidden dimension of the model
        eps: Epsilon value for numerical stability (default: 1e-5)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameters (one per dimension)
        # Initialize to ones
        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape (..., d_model)
               Typically (batch_size, sequence_length, d_model)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Save original dtype
        in_dtype = x.dtype

        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)

        # Compute RMS over the last dimension (d_model)
        # Square the input
        x_squared = x ** 2

        # Mean over the last dimension
        mean_squared = x_squared.mean(dim=-1, keepdim=True)

        # RMS = sqrt(mean_squared + eps)
        rms = torch.sqrt(mean_squared + self.eps)

        # Normalize by dividing by RMS
        x_normalized = x / rms

        # Scale by learnable gain parameters
        # self.weight has shape (d_model,)
        # x_normalized has shape (..., d_model)
        # Broadcasting will handle this correctly
        result = x_normalized * self.weight

        # Downcast back to original dtype
        return result.to(in_dtype)
