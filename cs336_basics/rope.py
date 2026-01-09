"""
Rotary Positional Embedding (RoPE).
"""

import torch
import torch.nn as nn
import einx


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    RoPE encodes positional information by rotating pairs of dimensions in the
    query and key vectors based on their position in the sequence. This preserves
    relative positional information through the attention mechanism.

    The rotation is applied to consecutive pairs of dimensions:
    For position k and dimension pair i:
        x_{2i}' = x_{2i} * cos(k * θ_i) - x_{2i+1} * sin(k * θ_i)
        x_{2i+1}' = x_{2i} * sin(k * θ_i) + x_{2i+1} * cos(k * θ_i)

    where θ_i = Θ^(-2i/d_k) is the frequency for dimension pair i.

    Args:
        theta: Base value Θ for computing rotation frequencies
        d_k: Dimension of query/key vectors (must be even)
        max_seq_len: Maximum sequence length for precomputing sin/cos buffers
        device: Device to store buffers on
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute frequency for each dimension pair
        # θ_i = Θ^(-2i/d_k) for i = 0, 1, ..., d_k/2 - 1
        i = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        freqs = theta ** (-2.0 * i / d_k)

        # Precompute sin and cos for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Compute angles: outer product of positions and frequencies
        # Shape: (max_seq_len, d_k//2)
        angles = positions[:, None] * freqs[None, :]

        # Compute and cache cos and sin values
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # Register as non-persistent buffers (not saved in state_dict)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor using einops for clarity.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape as x
        """
        # Retrieve cos and sin values for the given token positions
        # cos_cached, sin_cached: (max_seq_len, d_k//2)
        # token_positions: (..., seq_len)
        # Result: (..., seq_len, d_k//2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Split d_k dimension into (pairs, two) where pairs=d_k//2, two=2
        # This separates consecutive pairs: [x0, x1, x2, x3, ...] -> [[x0, x1], [x2, x3], ...]
        # (..., d_k) -> (..., d_k//2, 2)
        x_pairs = einx.rearrange('... (pairs two) -> ... pairs two', x, two=2)

        # Extract even-indexed and odd-indexed elements from each pair
        x_even = x_pairs[..., 0]  # (..., d_k//2) - elements at positions 0, 2, 4, ...
        x_odd = x_pairs[..., 1]   # (..., d_k//2) - elements at positions 1, 3, 5, ...

        # Apply 2D rotation to each pair
        # Rotation matrix: [[cos, -sin], [sin, cos]]
        # x_even' = x_even * cos - x_odd * sin
        # x_odd' = x_even * sin + x_odd * cos
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # Stack the rotated even and odd elements back into pairs
        x_rotated_pairs = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)

        # Merge pairs back into the original d_k dimension
        # (..., d_k//2, 2) -> (..., d_k)
        x_rotated = einx.rearrange('... pairs two -> ... (pairs two)', x_rotated_pairs, two=2)

        return x_rotated
