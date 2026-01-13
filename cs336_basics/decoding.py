"""
Decoding functions for generating text from a language model.

Supports:
- Temperature scaling
- Top-p (nucleus) sampling
- Prompt completion
- Maximum token generation
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional


def apply_temperature(logits: Float[Tensor, "vocab_size"], temperature: float) -> Float[Tensor, "vocab_size"]:
    """
    Apply temperature scaling to logits.

    Higher temperature (> 1.0) makes the distribution more uniform (more random).
    Lower temperature (< 1.0) makes the distribution more peaked (more deterministic).
    Temperature of 1.0 leaves the distribution unchanged.

    Args:
        logits: Unnormalized log probabilities
        temperature: Temperature value (must be > 0)

    Returns:
        Temperature-scaled logits
    """
    return logits / temperature


def top_p_sampling(
    logits: Float[Tensor, "vocab_size"],
    top_p: float,
    temperature: float = 1.0
) -> int:
    """
    Sample from logits using top-p (nucleus) sampling.

    Top-p sampling:
    1. Apply temperature scaling
    2. Sort tokens by probability (descending)
    3. Find the smallest set of tokens whose cumulative probability >= top_p
    4. Sample from this subset

    Args:
        logits: Unnormalized log probabilities for next token
        top_p: Cumulative probability threshold (0 < top_p <= 1.0)
        temperature: Temperature for scaling logits

    Returns:
        Sampled token ID
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative probability exceeds top_p
    # We want to include all tokens up to and including the one that pushes us over top_p
    cutoff_index = torch.where(cumulative_probs >= top_p)[0]

    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[0].item() + 1  # +1 to include the token that crossed the threshold
    else:
        cutoff_index = len(sorted_probs)  # Include all tokens if none exceed threshold

    # Keep only the top-p nucleus
    nucleus_probs = sorted_probs[:cutoff_index]
    nucleus_indices = sorted_indices[:cutoff_index]

    # Renormalize the nucleus probabilities
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    # Sample from the nucleus
    sampled_index = torch.multinomial(nucleus_probs, num_samples=1).item()
    sampled_token = nucleus_indices[sampled_index].item()

    return sampled_token


def generate(
    model: torch.nn.Module,
    prompt_tokens: Int[Tensor, "seq_len"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> Int[Tensor, "generated_seq_len"]:
    """
    Generate text completion from a language model.

    Args:
        model: Transformer language model
        prompt_tokens: Input token IDs (1D tensor)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        top_p: Top-p sampling threshold (1.0 = no filtering)
        eos_token_id: End-of-sequence token ID (generation stops if this is produced)
        device: Device to run generation on

    Returns:
        Complete sequence of token IDs (prompt + generated tokens)
    """
    model.eval()

    # Move prompt to device and ensure it's 2D (batch_size=1, seq_len)
    if prompt_tokens.dim() == 1:
        tokens = prompt_tokens.unsqueeze(0).to(device)  # (1, seq_len)
    else:
        tokens = prompt_tokens.to(device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for next token
            # Model returns (batch_size, seq_len, vocab_size)
            logits = model(tokens)

            # Get logits for the last position
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            # Sample next token using top-p sampling
            next_token = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)

            # Check for end-of-sequence
            if eos_token_id is not None and next_token == eos_token_id:
                generated_tokens.append(next_token)
                break

            # Append to generated tokens
            generated_tokens.append(next_token)

            # Append to input sequence for next iteration
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

    # Combine prompt and generated tokens
    result = torch.cat([
        prompt_tokens.to(device),
        torch.tensor(generated_tokens, dtype=torch.long, device=device)
    ])

    return result


def generate_batch(
    model: torch.nn.Module,
    prompt_tokens: Int[Tensor, "batch_size seq_len"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> list[Int[Tensor, "generated_seq_len"]]:
    """
    Generate text completions for a batch of prompts.

    Note: This processes prompts independently (no batched generation) to handle
    different stopping points per sequence.

    Args:
        model: Transformer language model
        prompt_tokens: Batch of input token IDs (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate per sequence
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        eos_token_id: End-of-sequence token ID
        device: Device to run generation on

    Returns:
        List of generated sequences (one per prompt)
    """
    batch_size = prompt_tokens.shape[0]
    results = []

    for i in range(batch_size):
        prompt = prompt_tokens[i]
        generated = generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            device=device,
        )
        results.append(generated)

    return results


def decode_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str = "cuda",
) -> str:
    """
    Generate text completion from a string prompt.

    Args:
        model: Transformer language model
        tokenizer: Tokenizer with encode() and decode() methods
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        eos_token: End-of-sequence token string
        device: Device to run generation on

    Returns:
        Generated text (including prompt)
    """
    # Encode prompt
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)

    # Get EOS token ID if available
    eos_token_id = None
    if hasattr(tokenizer, 'special_tokens') and tokenizer.special_tokens:
        # Try to find the EOS token ID
        for token_id, token_bytes in tokenizer.vocab.items():
            try:
                if token_bytes.decode('utf-8') == eos_token:
                    eos_token_id = token_id
                    break
            except:
                pass

    # Generate
    generated_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    # Decode
    generated_text = tokenizer.decode(generated_tokens.cpu().tolist())

    return generated_text
