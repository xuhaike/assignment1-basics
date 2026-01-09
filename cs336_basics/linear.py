"""
Linear transformation module without bias.
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    A linear transformation module that performs y = xW^T.

    This is similar to torch.nn.Linear but without bias.
    The weight matrix W is stored with shape (out_features, in_features).

    Args:
        in_features: Size of each input sample (last dimension)
        out_features: Size of each output sample (last dimension)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix with shape (out_features, in_features)
        # Store as W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )

        # Initialize weights using truncated normal distribution
        # N(μ=0, σ²=2/(d_in+d_out)) truncated at [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Linear transformation: y = xW^T
        # x has shape (..., in_features)
        # self.weight has shape (out_features, in_features)
        # We need to compute x @ self.weight.T to get (..., out_features)
        return x @ self.weight.T
