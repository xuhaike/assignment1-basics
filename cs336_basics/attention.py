
import torch
import einx
from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Get d_k from the last dimension of K (or Q)
    d_k = K.shape[-1]

    # Step 1: Compute QK^T (dot product between queries and keys)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # Result: (..., queries, keys)
    attention_scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K)

    # Step 2: Scale by 1/sqrt(d_k)
    attention_scores = attention_scores / (d_k ** 0.5)

    # Step 3: Apply mask if provided
    # Set masked positions to -inf so they become 0 after softmax
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

    # Step 4: Apply softmax along the keys dimension (last dimension)
    # This normalizes the attention weights
    from cs336_basics.softmax import softmax
    attention_weights = softmax(attention_scores, dim=-1)

    # Step 5: Multiply attention weights with values
    # attention_weights: (..., queries, keys), V: (..., values, d_v)
    # Note: keys dimension should equal values dimension
    # Result: (..., queries, d_v)
    output = einx.dot("... queries keys, ... keys d_v -> ... queries d_v", attention_weights, V)

    return output

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    rope: RotaryPositionalEmbedding | None,
    token_positions: torch.Tensor | None,
) -> torch.Tensor:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Get sequence length from input
    seq_len = in_features.shape[-2]

    # Step 1: Project input to Q, K, V using matrix multiplication
    # in_features: (..., seq_len, d_in)
    # q_proj_weight: (d_k, d_in) where d_k = num_heads * d_k_per_head
    # Result: (..., seq_len, d_k)
    Q = einx.dot("... seq d_in, d_k d_in -> ... seq d_k", in_features, q_proj_weight)
    K = einx.dot("... seq d_in, d_k d_in -> ... seq d_k", in_features, k_proj_weight)
    V = einx.dot("... seq d_in, d_v d_in -> ... seq d_v", in_features, v_proj_weight)

    # Step 2: Reshape to separate heads
    # (..., seq_len, d_k) -> (..., num_heads, seq_len, d_k_per_head)
    Q = einx.rearrange("... seq (h d) -> ... h seq d", Q, h=num_heads)
    K = einx.rearrange("... seq (h d) -> ... h seq d", K, h=num_heads)
    V = einx.rearrange("... seq (h d) -> ... h seq d", V, h=num_heads)

    if rope is not None:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=in_features.device)
        Q = rope(Q, token_positions)
        K = rope(K, token_positions)

    # Step 3: Create causal mask (lower triangular)
    # True where we want to attend (including diagonal)
    mask = ~torch.triu(torch.ones((seq_len, seq_len), device=in_features.device, dtype=torch.bool), diagonal=1)

    # Step 4: Apply scaled dot-product attention
    # Q, K, V: (..., num_heads, seq_len, d_per_head)
    attn_output = scaled_dot_product_attention(Q, K, V, mask)

    # Step 5: Concatenate heads
    # (..., num_heads, seq_len, d_v_per_head) -> (..., seq_len, d_v)
    attn_output = einx.rearrange("... h seq d -> ... seq (h d)", attn_output)

    # Step 6: Apply output projection
    # attn_output: (..., seq_len, d_v)
    # o_proj_weight: (d_model, d_v)
    # Result: (..., seq_len, d_model)
    output = einx.dot("... seq d_v, d_model d_v -> ... seq d_model", attn_output, o_proj_weight)

    return output