# Training Guide

This guide explains how to train a Transformer language model using the provided training script.

## Overview

The training script (`cs336_basics/train.py`) provides a complete training loop with:

- **Configurable hyperparameters**: All model and training parameters can be set via command-line arguments
- **Memory-efficient data loading**: Uses `np.memmap` for handling large datasets
- **Checkpointing**: Save and resume training at any point
- **Logging**: Console logging and optional Weights & Biases integration
- **Validation**: Periodic evaluation on held-out data

## Quick Start

### 1. Prepare Your Data

First, tokenize your text data and save it in memmap format:

```bash
# Prepare training data
uv run python cs336_basics/prepare_data.py \
    --input /path/to/train.txt \
    --output data/train.bin \
    --vocab /path/to/vocab.pkl \
    --merges /path/to/merges.pkl

# Prepare validation data
uv run python cs336_basics/prepare_data.py \
    --input /path/to/val.txt \
    --output data/val.bin \
    --vocab /path/to/vocab.pkl \
    --merges /path/to/merges.pkl
```

### 2. Train a Model

Basic training command:

```bash
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --checkpoint_dir checkpoints/ \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 8 \
    --max_iters 100000
```

### 3. Resume Training

To resume from a checkpoint:

```bash
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --checkpoint_dir checkpoints/ \
    --resume_from checkpoints/checkpoint_050000.pt \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 8 \
    --max_iters 100000
```

## Configuration Options

### Model Architecture

- `--vocab_size`: Vocabulary size (default: 50257)
- `--context_length`: Maximum sequence length (default: 1024)
- `--d_model`: Model dimension (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--d_ff`: Feed-forward dimension (default: 4 * d_model)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

### Training Hyperparameters

- `--batch_size`: Batch size (default: 8)
- `--max_iters`: Maximum training iterations (default: 100000)
- `--seed`: Random seed (default: 42)
- `--device`: Device (cuda or cpu, default: cuda)

### Optimizer Settings

- `--max_lr`: Maximum learning rate (default: 6e-4)
- `--min_lr`: Minimum learning rate (default: 6e-5)
- `--beta1`: Adam beta1 (default: 0.9)
- `--beta2`: Adam beta2 (default: 0.999)
- `--weight_decay`: Weight decay (default: 0.1)
- `--grad_clip`: Gradient clipping max norm (default: 1.0)

### Learning Rate Schedule

- `--warmup_iters`: Warmup iterations (default: 2000)
- `--cosine_cycle_iters`: Cosine annealing iterations (default: 100000)

### Logging and Checkpointing

- `--log_interval`: Log every N iterations (default: 10)
- `--eval_interval`: Evaluate every N iterations (default: 500)
- `--checkpoint_interval`: Save checkpoint every N iterations (default: 5000)
- `--checkpoint_dir`: Directory for checkpoints
- `--resume_from`: Path to checkpoint to resume from

### Weights & Biases

- `--wandb_project`: W&B project name (None to disable)
- `--wandb_run_name`: W&B run name

## Example Configurations

### GPT-2 Small (124M parameters)

```bash
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --vocab_size 50257 \
    --context_length 1024 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --d_ff 3072 \
    --batch_size 8 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 2000 \
    --max_iters 100000 \
    --checkpoint_dir checkpoints/gpt2-small
```

### GPT-2 Medium (350M parameters)

```bash
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --vocab_size 50257 \
    --context_length 1024 \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --batch_size 4 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_iters 2000 \
    --max_iters 100000 \
    --checkpoint_dir checkpoints/gpt2-medium
```

### Small Test Model (for debugging)

```bash
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --vocab_size 50257 \
    --context_length 256 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --d_ff 1024 \
    --batch_size 16 \
    --max_iters 10000 \
    --log_interval 5 \
    --eval_interval 100 \
    --checkpoint_dir checkpoints/tiny
```

## Using Weights & Biases

To enable W&B logging:

```bash
# Login to W&B (first time only)
wandb login

# Train with W&B logging
uv run python cs336_basics/train.py \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --checkpoint_dir checkpoints/ \
    --wandb_project "my-lm-project" \
    --wandb_run_name "gpt2-small-run1" \
    --vocab_size 50257 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12
```

## Monitoring Training

The training script logs:

- **Training loss**: Average loss over recent batches
- **Learning rate**: Current learning rate from the schedule
- **Throughput**: Tokens processed per second
- **Validation loss**: Periodic evaluation on validation set

Example output:

```
================================================================================
Training Configuration
================================================================================
train_data_path.................... data/train.bin
val_data_path...................... data/val.bin
vocab_size......................... 50257
d_model............................ 768
num_layers......................... 12
...
================================================================================

Loading datasets...
Train dataset size: 10,000,000 tokens
Validation dataset size: 1,000,000 tokens

Initializing model...
Model parameters: 124,439,808

================================================================================
Starting training
================================================================================
iter      0 | loss 10.8234 | lr 0.000000 | 1234 tok/s
iter     10 | loss 9.2156 | lr 0.000003 | 1245 tok/s
iter     20 | loss 8.5432 | lr 0.000006 | 1238 tok/s
...
iter    500 | val_loss 7.1234
...
```

## Tips

1. **Start small**: Test your setup with a small model and dataset first
2. **Monitor validation loss**: Watch for overfitting
3. **Adjust batch size**: Based on your GPU memory
4. **Use gradient clipping**: Helps stabilize training (default: 1.0)
5. **Save checkpoints frequently**: Prevents data loss from crashes
6. **Learning rate**: Start with 6e-4 for small models, reduce for larger ones

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Reduce `--context_length`
- Use a smaller model (fewer layers or smaller `--d_model`)

### Training Unstable

- Increase `--warmup_iters`
- Reduce `--max_lr`
- Ensure `--grad_clip` is enabled (default: 1.0)

### Slow Training

- Increase `--batch_size` if you have GPU memory
- Ensure you're using CUDA: `--device cuda`
- Check data loading isn't the bottleneck
