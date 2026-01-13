#!/bin/bash
# Example training script for a small Transformer LM
#
# This script demonstrates how to train a small language model
# with the training script.

# Configuration
TRAIN_DATA="data/train.bin"
VAL_DATA="data/val.bin"
CHECKPOINT_DIR="checkpoints/example_run"
DEVICE="cuda"  # or "cpu"

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Train the model
uv run python cs336_basics/train.py \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    \
    --vocab_size 50257 \
    --context_length 512 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --d_ff 2048 \
    \
    --batch_size 8 \
    --max_iters 50000 \
    --device "$DEVICE" \
    \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --warmup_iters 2000 \
    --cosine_cycle_iters 50000 \
    \
    --log_interval 10 \
    --eval_interval 500 \
    --eval_batches 20 \
    --checkpoint_interval 5000 \
    \
    --seed 42

echo "Training complete! Checkpoints saved to $CHECKPOINT_DIR"
