"""
Transformer block implementation.
"""

import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.attention import multihead_self_attention
from cs336_basics.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Implements the following architecture:
        x = x + MultiHeadAttention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))

    Args:
        d_model: Dimensionality of the Transformer block inputs
        num_heads: Number of heads to use in multi-head self-attention
        d_ff: Dimensionality of the position-wise feed-forward inner layer
        max_seq_len: Maximum sequence length for RoPE
        theta: RoPE parameter
        device: Device to store parameters on
        dtype: Data type of parameters
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # First RMSNorm (before attention)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # Multi-head self-attention projections
        from cs336_basics.linear import Linear

        self.attn_q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn_k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn_v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn_output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        # Second RMSNorm (before feed-forward)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply Transformer block to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional token positions for RoPE

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm: Apply RMSNorm before attention
        x_norm = self.ln1(x)

        # Multi-head self-attention with RoPE
        attn_output = multihead_self_attention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            q_proj_weight=self.attn_q_proj.weight,
            k_proj_weight=self.attn_k_proj.weight,
            v_proj_weight=self.attn_v_proj.weight,
            o_proj_weight=self.attn_output_proj.weight,
            in_features=x_norm,
            rope=self.rope,
            token_positions=token_positions,
        )

        # Residual connection
        x = x + attn_output

        # Pre-norm: Apply RMSNorm before feed-forward
        x_norm = self.ln2(x)

        # Feed-forward network
        ffn_output = self.ffn(x_norm)

        # Residual connection
        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model.

    Full autoregressive language model with:
    - Token embeddings
    - Multiple Transformer blocks
    - Final RMSNorm
    - LM head for prediction

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta parameter
        device: Device to store parameters on
        dtype: Data type of parameters
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embeddings
        from cs336_basics.embedding import Embedding

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # LM head (output projection to vocabulary)
        from cs336_basics.linear import Linear

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer LM.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            token_positions: Optional token positions for RoPE

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens: (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embeddings(input_ids)

        # Pass through all Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Apply final RMSNorm
        x = self.ln_final(x)

        # Project to vocabulary: (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
