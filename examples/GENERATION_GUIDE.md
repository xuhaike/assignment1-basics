# Text Generation Guide

This guide explains how to generate text from your trained language models.

## Quick Start

### 1. Simple Generation (Auto-detect latest experiment)

```bash
cd /home/ubuntu/assignment1-basics
./examples/run_generate.sh
```

This will automatically find the most recent experiment and generate text using the default prompt.

### 2. Custom Prompt

```bash
./examples/run_generate.sh --prompt "Once upon a time in a magical kingdom"
```

### 3. Interactive Mode

```bash
./examples/run_generate.sh --interactive
```

In interactive mode, you can enter multiple prompts and see the generated text in real-time. Press Ctrl+C to exit.

### 4. List Available Experiments

```bash
./examples/run_generate.sh --list
```

### 5. Use Specific Experiment

```bash
./examples/run_generate.sh --experiment experiments/baseline_17m_20260111_231919
```

### 6. Use Specific Checkpoint

```bash
./examples/run_generate.sh \
    --checkpoint checkpoints/baseline_17m/checkpoint_010000.pt \
    --config experiments/baseline_17m_20260111_231919/config.json \
    --prompt "The robot walked into"
```

## Advanced Options

### Temperature Control

Control randomness in generation (0.1 = very deterministic, 1.5 = very random):

```bash
./examples/run_generate.sh --temperature 0.7 --prompt "The cat sat"
```

### Top-p Sampling

Control diversity using nucleus sampling:

```bash
./examples/run_generate.sh --top-p 0.95 --prompt "In the forest"
```

### Maximum Tokens

Control how much text to generate:

```bash
./examples/run_generate.sh --max-tokens 500 --prompt "Chapter 1:"
```

### Device Selection

Run on CPU instead of GPU:

```bash
./examples/run_generate.sh --device cpu
```

## Python Script Usage

You can also use the Python script directly:

### Auto-detect Experiment

```bash
uv run python cs336_basics/simple_generate.py \
    --experiment experiments/baseline_17m_20260111_231919 \
    --prompt "Once upon a time"
```

### With Specific Checkpoint

```bash
uv run python cs336_basics/simple_generate.py \
    --checkpoint checkpoints/baseline_17m/checkpoint_010000.pt \
    --config experiments/baseline_17m_20260111_231919/config.json \
    --prompt "The adventure begins" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_p 0.9
```

### Interactive Mode

```bash
uv run python cs336_basics/simple_generate.py \
    --experiment experiments/baseline_17m_20260111_231919 \
    --interactive
```

## Understanding the Output

The script will show:

1. **Experiment Information**: Name, description, and configuration
2. **Model Details**: Number of parameters, device, vocab size, context length
3. **Generation Process**: Tokenization info and generation progress
4. **Generated Text**: The final output
5. **Generation Settings**: Temperature, top-p, and token count

## Important Notes

### Tokenizer Support

The `simple_generate.py` script now supports **automatic tokenizer loading**:

- **With tokenizer files**: If your experiment config contains `tokenizer_vocab_path` and `tokenizer_merges_path`, the script will automatically load and use the proper BPE tokenizer
- **Without tokenizer files**: Falls back to a simple demo character-level tokenizer (results will be gibberish)

### Setting Up Tokenizer for Generation

To enable proper text generation, ensure your training config includes tokenizer paths:

1. Save the tokenizer vocabulary and merges during data preparation
2. Add tokenizer paths to your experiment config JSON:

```json
{
  "experiment_name": "my_experiment",
  "vocab_size": 10000,
  ...
  "tokenizer_vocab_path": "path/to/vocab.pkl",
  "tokenizer_merges_path": "path/to/merges.pkl"
}
```

3. The generation script will automatically detect and use these files

### Using the Full Generate Script

Alternatively, you can use the full `generate.py` script with proper tokenizer files:

```bash
uv run python cs336_basics/generate.py \
    --checkpoint checkpoints/baseline_17m/checkpoint_010000.pt \
    --vocab path/to/vocab.pkl \
    --merges path/to/merges.pkl \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --prompt "Once upon a time"
```

## Batch Size Sweep Experiments

If you've run batch size sweep experiments, you can generate from any of them:

```bash
# Generate from batch size 16 experiment
./examples/run_generate.sh --experiment experiments/bz_sweep_16_lr_1e-3_*

# Generate from batch size 64 experiment
./examples/run_generate.sh --experiment experiments/bz_sweep_64_lr_1e-3_*

# Generate from batch size 128 experiment
./examples/run_generate.sh --experiment experiments/bz_sweep_128_lr_1e-3_*
```

## Tips for Better Generation

1. **Temperature**:
   - Lower (0.5-0.7): More focused, deterministic output
   - Medium (0.8-1.0): Balanced creativity and coherence
   - Higher (1.1-1.5): More creative but potentially less coherent

2. **Top-p**:
   - Lower (0.8-0.9): More conservative token selection
   - Higher (0.95-0.99): More diverse vocabulary

3. **Prompt Engineering**:
   - Start with complete sentence fragments
   - Use genre-specific prompts (e.g., "Once upon a time" for stories)
   - Be specific about the context you want

4. **Checkpoint Selection**:
   - Later checkpoints (higher iteration numbers) are typically better trained
   - Try comparing outputs from different checkpoints to see training progress

## Troubleshooting

### CUDA Out of Memory

```bash
./examples/run_generate.sh --device cpu
```

### No Experiments Found

```bash
# List available experiments
./examples/run_generate.sh --list

# Or specify an experiment explicitly
./examples/run_generate.sh --experiment experiments/YOUR_EXPERIMENT_DIR
```

### Model Not Generating Sensible Text

This is expected with the demo tokenizer. To get proper text generation:
1. Save your tokenizer during training/data preparation
2. Use the full `generate.py` script with proper tokenizer files
