#!/bin/bash
# Run learning rate sweep experiment

set -e

echo "Learning Rate Sweep Experiment"
echo "=============================="

# Configuration
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"
LOG_DIR="experiments"

# Learning rates to test
LRS=("1e-4" "3e-4" "6e-4" "1e-3" "3e-3" "6e-3" "1e-2" "3e-2")
# LRS=("6e-3" "1e-2" "3e-2")

# echo "Testing learning rates: ${LRS[@]}"
# echo ""

# # Run experiment for each learning rate
# for lr in "${LRS[@]}"; do
#     echo "======================================"
#     echo "Running experiment with LR = $lr"
#     echo "======================================"

#     EXPERIMENT_NAME="lr_sweep_${lr}"
#     CHECKPOINT_DIR="checkpoints/lr_sweep/${lr}"
#     mkdir -p "$CHECKPOINT_DIR"

#     uv run python cs336_basics/train_with_logging.py \
#         --experiment_name "$EXPERIMENT_NAME" \
#         --description "Learning rate sweep: testing lr=$lr" \
#         --tags "sweep,lr,${lr}" \
#         --notes "Part of learning rate sweep to find optimal LR" \
#         \
#         --train_data_path "$TRAIN_DATA" \
#         --val_data_path "$VAL_DATA" \
#         --checkpoint_dir "$CHECKPOINT_DIR" \
#         --log_dir "$LOG_DIR" \
#         \
#         --vocab_size 10000 \
#         --context_length 256 \
#         --d_model 512 \
#         --num_layers 4 \
#         --num_heads 16 \
#         --d_ff 1344 \
#         --rope_theta 10000.0 \
#         \
#         --batch_size 64 \
#         --max_iters 10000 \
#         --seed 42 \
#         --device cuda \
#         \
#         --max_lr "$lr" \
#         --min_lr "$(python -c "print(float('$lr') / 10)")" \
#         --beta1 0.9 \
#         --beta2 0.999 \
#         --weight_decay 0.1 \
#         --grad_clip 1.0 \
#         \
#         --warmup_iters 1000 \
#         --cosine_cycle_iters 8000 \
#         \
#         --log_interval 50 \
#         --eval_interval 500 \
#         --eval_batches 20 \
#         --checkpoint_interval 500

#     echo ""
# done

echo ""
echo "======================================"
echo "All experiments complete!"
echo "======================================"

# Collect all experiment directories
EXPERIMENT_DIRS=()
EXPERIMENT_NAMES=()
for lr in "${LRS[@]}"; do
    dir=$(ls -d ${LOG_DIR}/lr_sweep_${lr}_* 2>/dev/null | tail -1)
    if [ -n "$dir" ]; then
        EXPERIMENT_DIRS+=("$dir")
        EXPERIMENT_NAMES+=("LR=$lr")
    fi
done

# Generate comparison plot
echo ""
echo "Generating comparison plot..."
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs "${EXPERIMENT_DIRS[@]}" \
    --names "${EXPERIMENT_NAMES[@]}" \
    --plot_type comparison \
    --output "experiments/lr_sweep_comparison.png"

echo "Comparison saved to: experiments/lr_sweep_comparison.png"

# Print summaries
echo ""
echo "======================================"
echo "Experiment Summaries"
echo "======================================"
for dir in "${EXPERIMENT_DIRS[@]}"; do
    uv run python cs336_basics/analyze_experiments.py \
        --experiment_dirs "$dir" \
        --summary
    echo ""
done

echo "Learning rate sweep complete!"
echo "Review experiments/lr_sweep_comparison.png for results"
