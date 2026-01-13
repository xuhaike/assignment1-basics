#!/bin/bash
# Run ablation study to test importance of architectural components

set -e

echo "Ablation Study"
echo "=============="

# Configuration
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"
LOG_DIR="experiments"
MAX_ITERS=20000  # Shorter runs for ablations

# Baseline configuration for reference
VOCAB_SIZE=50257
CONTEXT_LENGTH=512
D_MODEL=512
NUM_LAYERS=8
NUM_HEADS=8
D_FF=2048
BATCH_SIZE=8

echo "Running ablation experiments..."
echo ""

# ============================================
# Experiment 1: Baseline
# ============================================
echo "1. Running BASELINE experiment..."
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_baseline" \
    --description "Baseline configuration for comparison" \
    --tags "ablation,baseline" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "checkpoints/ablation/baseline" \
    --log_dir "$LOG_DIR" \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --seed 42 \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 10000

echo ""

# ============================================
# Experiment 2: No Layer Normalization
# ============================================
echo "2. Running NO LAYER NORM experiment..."
echo "(Note: Would require code modification to disable LayerNorm)"
echo "Skipping for now - implement in transformer.py first"
echo ""

# ============================================
# Experiment 3: Smaller FFN
# ============================================
echo "3. Running SMALLER FFN experiment..."
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_small_ffn" \
    --description "Reduce FFN size from 4x to 2x model dimension" \
    --tags "ablation,ffn" \
    --notes "Testing importance of FFN capacity" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "checkpoints/ablation/small_ffn" \
    --log_dir "$LOG_DIR" \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $((D_MODEL * 2)) \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --seed 42 \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 10000

echo ""

# ============================================
# Experiment 4: Fewer Layers
# ============================================
echo "4. Running FEWER LAYERS experiment..."
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_fewer_layers" \
    --description "Reduce layers from 8 to 4" \
    --tags "ablation,depth" \
    --notes "Testing depth vs width trade-off" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "checkpoints/ablation/fewer_layers" \
    --log_dir "$LOG_DIR" \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --num_layers 4 \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --seed 42 \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 10000

echo ""

# ============================================
# Experiment 5: Fewer Heads
# ============================================
echo "5. Running FEWER HEADS experiment..."
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_fewer_heads" \
    --description "Reduce attention heads from 8 to 4" \
    --tags "ablation,attention" \
    --notes "Testing multi-head attention importance" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "checkpoints/ablation/fewer_heads" \
    --log_dir "$LOG_DIR" \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads 4 \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --seed 42 \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 10000

echo ""

# ============================================
# Experiment 6: Shorter Context
# ============================================
echo "6. Running SHORTER CONTEXT experiment..."
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_short_context" \
    --description "Reduce context length from 512 to 256" \
    --tags "ablation,context" \
    --notes "Testing context length impact" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --checkpoint_dir "checkpoints/ablation/short_context" \
    --log_dir "$LOG_DIR" \
    --vocab_size $VOCAB_SIZE \
    --context_length 256 \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --seed 42 \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 10000

echo ""
echo "======================================"
echo "All ablation experiments complete!"
echo "======================================"

# Collect experiment directories
EXPERIMENT_DIRS=(
    $(ls -d ${LOG_DIR}/ablation_baseline_* 2>/dev/null | tail -1)
    $(ls -d ${LOG_DIR}/ablation_small_ffn_* 2>/dev/null | tail -1)
    $(ls -d ${LOG_DIR}/ablation_fewer_layers_* 2>/dev/null | tail -1)
    $(ls -d ${LOG_DIR}/ablation_fewer_heads_* 2>/dev/null | tail -1)
    $(ls -d ${LOG_DIR}/ablation_short_context_* 2>/dev/null | tail -1)
)

EXPERIMENT_NAMES=(
    "Baseline"
    "Small FFN (2x)"
    "Fewer Layers (4)"
    "Fewer Heads (4)"
    "Short Context (256)"
)

# Generate comparison
echo ""
echo "Generating comparison plot..."
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs "${EXPERIMENT_DIRS[@]}" \
    --names "${EXPERIMENT_NAMES[@]}" \
    --plot_type comparison \
    --output "experiments/ablation_comparison.png"

echo "Comparison saved to: experiments/ablation_comparison.png"

# Print summaries
echo ""
echo "======================================"
echo "Experiment Summaries"
echo "======================================"
for i in "${!EXPERIMENT_DIRS[@]}"; do
    echo ""
    echo "${EXPERIMENT_NAMES[$i]}:"
    echo "--------------------------------------"
    uv run python cs336_basics/analyze_experiments.py \
        --experiment_dirs "${EXPERIMENT_DIRS[$i]}" \
        --summary
done

echo ""
echo "Ablation study complete!"
echo "Review experiments/ablation_comparison.png for results"
