#!/bin/bash
# Run baseline 17M parameter model experiment on TinyStories

set -e  # Exit on error

echo "Starting Baseline Experiment"
echo "============================"

# Configuration
EXPERIMENT_NAME="baseline_17m"
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"
CHECKPOINT_DIR="checkpoints/baseline_17m"
LOG_DIR="experiments"

# Check data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data not found at $VAL_DATA"
    exit 1
fi

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Run training
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --description "Baseline 17M parameter model on TinyStories dataset" \
    --tags "baseline,17m,tinystories" \
    --notes "Standard configuration with d_model=512, 4 layers, 16 heads" \
    \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000.0 \
    \
    --batch_size 64 \
    --max_iters 20000 \
    --seed 42 \
    --device cuda \
    \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    \
    --warmup_iters 2000 \
    --cosine_cycle_iters 15000 \
    \
    --log_interval 50 \
    --eval_interval 500 \
    --eval_batches 20 \
    --checkpoint_interval 500

echo ""
echo "Experiment complete!"
echo "Results saved to: $LOG_DIR"

# Generate analysis
echo ""
echo "Generating analysis plots..."
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs "$LOG_DIR/${EXPERIMENT_NAME}_"* \
    --plot_type loss_curves \
    --output "${CHECKPOINT_DIR}/loss_curves.png"

echo "Loss curves saved to: ${CHECKPOINT_DIR}/loss_curves.png"

# Print summary
echo ""
echo "Experiment Summary:"
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs "$LOG_DIR/${EXPERIMENT_NAME}_"* \
    --summary
