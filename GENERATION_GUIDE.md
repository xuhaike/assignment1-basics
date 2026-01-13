# Text Generation Guide

This guide explains how to generate text from your trained Transformer language model.

## Overview

The generation system provides:

- **Temperature Scaling**: Control randomness (higher = more creative, lower = more focused)
- **Top-p (Nucleus) Sampling**: Sample from the most probable tokens whose cumulative probability exceeds p
- **Prompt Completion**: Generate text starting from a prompt
- **EOS Token Handling**: Stop generation when end-of-sequence token is produced
- **Max Token Control**: Limit the length of generated text

## Quick Start

### 1. Test the Generation Functions

First, verify the generation code works:

```bash
uv run python examples/test_generation.py
```

This tests temperature scaling, top-p sampling, and text generation.

### 2. Generate Text from a Checkpoint

```bash
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --vocab data/vocab.pkl \
    --merges data/merges.pkl \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --prompt "Once upon a time" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9
```

### 3. Interactive Mode

For multiple generations:

```bash
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --vocab data/vocab.pkl \
    --merges data/merges.pkl \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --interactive \
    --temperature 0.8 \
    --top_p 0.9
```

## Sampling Strategies

### Temperature Scaling

Temperature controls the "randomness" of generation:

- **Temperature = 0.1-0.5**: Very focused, deterministic, repetitive
- **Temperature = 0.7-0.9**: Balanced between creativity and coherence
- **Temperature = 1.0**: Use model's raw probabilities (default)
- **Temperature = 1.5-2.0**: Very creative, diverse, but potentially incoherent

**How it works:**
```python
scaled_logits = logits / temperature
probabilities = softmax(scaled_logits)
```

Lower temperature makes high-probability tokens even more likely (peaked distribution).
Higher temperature makes all tokens more equally likely (flatter distribution).

### Top-p (Nucleus) Sampling

Top-p sampling filters the vocabulary to only the most probable tokens:

- **top_p = 0.5**: Very conservative, only use tokens in top 50% cumulative probability
- **top_p = 0.9**: Balanced, excludes unlikely tokens (recommended)
- **top_p = 0.95**: Slightly more diverse
- **top_p = 1.0**: No filtering, sample from full vocabulary

**How it works:**
1. Sort tokens by probability (descending)
2. Find the smallest set whose cumulative probability ≥ top_p
3. Sample only from this "nucleus"
4. Renormalize probabilities

Example: If probabilities are [0.6, 0.2, 0.1, 0.05, 0.05], with top_p=0.9:
- Nucleus = first 3 tokens (cumulative = 0.9)
- Ignore last 2 tokens
- Sample from [0.6, 0.2, 0.1] after renormalizing to [0.67, 0.22, 0.11]

## API Reference

### `generate()`

Generate tokens autoregressively:

```python
from cs336_basics.decoding import generate

generated_tokens = generate(
    model=model,                    # Your TransformerLM
    prompt_tokens=torch.tensor([1, 2, 3]),  # Input token IDs
    max_new_tokens=100,             # Maximum tokens to generate
    temperature=0.8,                # Sampling temperature
    top_p=0.9,                      # Top-p threshold
    eos_token_id=50256,            # Stop if this token is generated
    device="cuda",                  # Device to run on
)
# Returns: tensor of token IDs (prompt + generated)
```

### `decode_text()`

High-level API for text-to-text generation:

```python
from cs336_basics.decoding import decode_text

generated_text = decode_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    eos_token="<|endoftext|>",
    device="cuda",
)
# Returns: string with generated text
```

### `top_p_sampling()`

Low-level sampling function:

```python
from cs336_basics.decoding import top_p_sampling

next_token_id = top_p_sampling(
    logits=next_token_logits,  # Unnormalized log probabilities
    top_p=0.9,                  # Nucleus threshold
    temperature=0.8,            # Temperature scaling
)
# Returns: integer token ID
```

## Examples

### Example 1: Deterministic Generation

For consistent, focused output:

```python
import torch
from cs336_basics.decoding import generate

prompt = torch.tensor([1, 2, 3, 4, 5])  # Your prompt tokens

output = generate(
    model=model,
    prompt_tokens=prompt,
    max_new_tokens=50,
    temperature=0.1,  # Low temperature for deterministic output
    top_p=1.0,        # No filtering
    device="cuda"
)
```

### Example 2: Creative Generation

For diverse, creative output:

```python
output = generate(
    model=model,
    prompt_tokens=prompt,
    max_new_tokens=100,
    temperature=1.2,  # Higher temperature for creativity
    top_p=0.9,        # Filter unlikely tokens
    device="cuda"
)
```

### Example 3: Balanced Generation (Recommended)

Good default for most use cases:

```python
output = generate(
    model=model,
    prompt_tokens=prompt,
    max_new_tokens=100,
    temperature=0.8,  # Balanced temperature
    top_p=0.9,        # Standard nucleus filtering
    eos_token_id=50256,  # Stop at <|endoftext|>
    device="cuda"
)
```

### Example 4: Programmatic Generation

Using the decoding module in your code:

```python
import torch
from cs336_basics.transformer import TransformerLM
from cs336_basics.decoding import generate
from cs336_basics.tokenizer import Tokenizer

# Load model
model = TransformerLM(...)
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = Tokenizer(vocab, merges, special_tokens)

# Encode prompt
prompt_text = "The quick brown fox"
prompt_tokens = torch.tensor(tokenizer.encode(prompt_text))

# Generate
generated_tokens = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    device="cuda"
)

# Decode
generated_text = tokenizer.decode(generated_tokens.tolist())
print(generated_text)
```

## Command-Line Interface

The `generate.py` script provides a CLI for text generation.

### Basic Usage

```bash
uv run python cs336_basics/generate.py \
    --checkpoint <path>           # Model checkpoint
    --vocab <path>                # Tokenizer vocabulary
    --merges <path>               # Tokenizer merges
    --vocab_size <int>            # Model vocab size
    --d_model <int>               # Model dimension
    --num_layers <int>            # Number of layers
    --num_heads <int>             # Number of heads
    --prompt "Your prompt here"   # Input text
    --max_tokens 100              # Max tokens to generate
    --temperature 0.8             # Sampling temperature
    --top_p 0.9                   # Top-p threshold
```

### Interactive Mode

```bash
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/checkpoint_final.pt \
    --vocab data/vocab.pkl \
    --merges data/merges.pkl \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --interactive
```

Then enter prompts interactively:

```
Prompt: The meaning of life is
Generating...
--------------------------------------------------------------------------------
The meaning of life is to find happiness and fulfillment through meaningful
connections with others and pursuing your passions...
--------------------------------------------------------------------------------

Prompt: In the year 2050,
Generating...
...
```

## Tips and Best Practices

### For Coherent Text

- Use lower temperature (0.5-0.8)
- Use top_p around 0.9
- Provide clear, complete prompts

### For Creative/Diverse Text

- Use higher temperature (1.0-1.5)
- Use top_p around 0.95
- Can use shorter, more open-ended prompts

### For Code Generation

- Use very low temperature (0.1-0.3)
- Use top_p = 0.95
- Provide detailed prompts with context

### Avoiding Repetition

- Use top_p < 1.0 (filters out unlikely repetitions)
- Avoid very low temperatures
- Consider implementing repetition penalty (not included yet)

### Controlling Length

- Set `max_new_tokens` appropriately
- Use EOS token to allow natural stopping
- For fixed-length: set `eos_token_id=None`

## Common Issues

### Generation Too Random

**Problem**: Output is incoherent or doesn't make sense

**Solutions**:
- Lower the temperature (try 0.5-0.7)
- Lower top_p (try 0.85-0.9)
- Check if model is well-trained

### Generation Too Repetitive

**Problem**: Model keeps repeating the same phrases

**Solutions**:
- Increase temperature (try 0.9-1.1)
- Increase top_p (try 0.95)
- Model may need more training

### Generation Stops Too Early

**Problem**: Output is very short

**Solutions**:
- Check `eos_token_id` is correct
- Set `eos_token_id=None` to disable early stopping
- Increase `max_new_tokens`

### Slow Generation

**Problem**: Generation is too slow

**Solutions**:
- Use CUDA if available: `--device cuda`
- Reduce model size for faster inference
- Consider implementing KV caching (not included yet)

## Advanced: Implementing in Your Code

### Custom Sampling Strategy

```python
import torch
from cs336_basics.decoding import apply_temperature

def custom_sampling(logits, temperature, min_p=0.01):
    """Custom sampling with minimum probability threshold."""
    # Apply temperature
    logits = apply_temperature(logits, temperature)
    probs = torch.softmax(logits, dim=-1)

    # Filter tokens below min_p
    mask = probs >= min_p
    filtered_probs = probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample
    token = torch.multinomial(filtered_probs, num_samples=1)
    return token.item()
```

### Batched Generation

```python
from cs336_basics.decoding import generate_batch

# Generate for multiple prompts
prompts = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
])

results = generate_batch(
    model=model,
    prompt_tokens=prompts,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    device="cuda"
)
# Returns list of generated sequences
```

## References

- **Temperature Sampling**: Standard technique in language model sampling
- **Top-p (Nucleus) Sampling**: Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration"
- **Decoding Strategies**: See Hugging Face blog on "How to generate text"
