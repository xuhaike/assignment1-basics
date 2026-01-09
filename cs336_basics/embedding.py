"""
Embedding lookup module.
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    A module that performs embedding lookup.

    This is similar to torch.nn.Embedding but with custom initialization.
    The embedding matrix is stored with shape (num_embeddings, embedding_dim).

    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors (d_model)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embedding matrix with shape (num_embeddings, embedding_dim)
        # Store with d_model (embedding_dim) as the final dimension
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )

        # Initialize weights using truncated normal distribution
        # N(μ=0, σ²=1) truncated at [-3σ, 3σ]
        std = 1
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Tensor of token IDs with shape (...)

        Returns:
            Embedding vectors with shape (..., embedding_dim)
        """
        # Perform embedding lookup
        # token_ids can have arbitrary shape, and we index into self.weight
        # self.weight has shape (num_embeddings, embedding_dim)
        # Result will have shape (..., embedding_dim)
        return self.weight[token_ids]
