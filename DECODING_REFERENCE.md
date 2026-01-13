# Decoding Quick Reference

## Core Functions

### `generate()` - Token-level Generation
```python
from cs336_basics.decoding import generate

tokens = generate(
    model,              # TransformerLM model
    prompt_tokens,      # torch.Tensor of token IDs
    max_new_tokens,     # int: max tokens to generate
    temperature=1.0,    # float: sampling randomness
    top_p=1.0,          # float: nucleus threshold
    eos_token_id=None,  # int: stop if generated
    device="cuda"       # str: device
)
```

### `decode_text()` - Text-to-Text Generation
```python
from cs336_basics.decoding import decode_text

text = decode_text(
    model,                      # TransformerLM model
    tokenizer,                  # Tokenizer instance
    prompt="Once upon a time",  # str: input text
    max_new_tokens=100,         # int: max tokens
    temperature=0.8,            # float: randomness
    top_p=0.9,                  # float: nucleus
    eos_token="<|endoftext|>", # str: stop token
    device="cuda"               # str: device
)
```

### `top_p_sampling()` - Single Token Sampling
```python
from cs336_basics.decoding import top_p_sampling

token_id = top_p_sampling(
    logits,          # torch.Tensor: unnormalized log probs
    top_p=0.9,       # float: nucleus threshold
    temperature=1.0  # float: temperature
)
```

## CLI Usage

### Single Prompt
```bash
uv run python cs336_basics/generate.py \
    --checkpoint model.pt \
    --vocab vocab.pkl \
    --merges merges.pkl \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --prompt "Your text here" \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_tokens 100
```

### Interactive Mode
```bash
uv run python cs336_basics/generate.py \
    --checkpoint model.pt \
    --vocab vocab.pkl \
    --merges merges.pkl \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --interactive
```

## Parameter Guide

### Temperature
| Value | Effect | Use Case |
|-------|--------|----------|
| 0.1-0.5 | Very deterministic | Code, factual text |
| 0.7-0.9 | Balanced | General text generation |
| 1.0 | Model's raw distribution | Default |
| 1.2-2.0 | Very creative | Creative writing |

### Top-p
| Value | Effect | Use Case |
|-------|--------|----------|
| 0.5-0.8 | Conservative | When you want safe outputs |
| 0.9 | Balanced (recommended) | General use |
| 0.95 | More diverse | Creative tasks |
| 1.0 | No filtering | Full vocabulary |

## Common Configurations

### Code Generation
```python
temperature=0.2, top_p=0.95
```

### Story Writing
```python
temperature=0.9, top_p=0.95
```

### General Text
```python
temperature=0.8, top_p=0.9
```

### Deterministic Output
```python
temperature=0.1, top_p=1.0
```

## Implementation Details

### Temperature Scaling
```python
scaled_logits = logits / temperature
# Lower temp → peaked distribution (deterministic)
# Higher temp → flat distribution (random)
```

### Top-p Algorithm
1. Compute probabilities: `probs = softmax(logits)`
2. Sort descending: `sorted_probs, indices = sort(probs)`
3. Cumulative sum: `cumsum = cumsum(sorted_probs)`
4. Find cutoff: `nucleus = where(cumsum >= top_p)[0]`
5. Sample from nucleus only

### Generation Loop
```
1. Start with prompt tokens
2. For each step:
   a. Run model forward pass
   b. Get logits for next position
   c. Apply temperature + top-p
   d. Sample next token
   e. Append to sequence
   f. Stop if EOS or max_tokens
3. Return full sequence
```

## Testing

```bash
# Test generation functions
uv run python examples/test_generation.py

# Test with tiny model (quick)
uv run python -c "
from cs336_basics.transformer import TransformerLM
from cs336_basics.decoding import generate
import torch

model = TransformerLM(100, 64, 128, 2, 4, 512, device='cpu')
tokens = generate(model, torch.tensor([1,2,3]), 10, device='cpu')
print('Generated:', tokens.tolist())
"
```

## Files

- `cs336_basics/decoding.py` - Core implementation
- `cs336_basics/generate.py` - CLI script
- `examples/test_generation.py` - Tests
- `examples/generate_example.sh` - Usage examples
- `GENERATION_GUIDE.md` - Detailed documentation
