"""
SwiGLU position-wise feed-forward network.
"""

import torch
import torch.nn as nn


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish.

    SiLU(x) = x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        Output tensor after applying SiLU activation
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    FFN(x) = W2(SiLU(W1x) ⊙ W3x)

    where ⊙ denotes element-wise multiplication (the GLU gate).

    Args:
        d_model: Model dimension (input and output dimension)
        d_ff: Inner feed-forward dimension (if None, computed as ~8/3 * d_model)
        device: Device to store parameters on
        dtype: Data type of parameters
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model

        # Compute d_ff as approximately 8/3 * d_model, rounded to nearest multiple of 64
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        # Import Linear from our implementation
        from cs336_basics.linear import Linear

        # W1: d_model -> d_ff (for SiLU path)
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        # W2: d_ff -> d_model (output projection)
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # W3: d_model -> d_ff (for gate path)
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1x: (..., d_model) -> (..., d_ff)
        w1_x = self.w1(x)

        # SiLU(W1x): (..., d_ff)
        silu_w1_x = silu(w1_x)

        # W3x: (..., d_model) -> (..., d_ff)
        w3_x = self.w3(x)

        # GLU: SiLU(W1x) ⊙ W3x (element-wise multiplication)
        gated = silu_w1_x * w3_x

        # W2(...): (..., d_ff) -> (..., d_model)
        output = self.w2(gated)

        return output
