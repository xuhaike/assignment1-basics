#!/bin/bash
# Run batch size sweep experiment

set -e

echo "Batch Size Sweep Experiment"
echo "=============================="

# Configuration
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"
LOG_DIR="experiments"

# Training configuration
TOTAL_TOKENS=327680000
CONTEXT_LENGTH=256
WARMUP_RATIO=0.1
COSINE_RATIO=0.8

# Batch sizes to test
BATCH_SIZES=(16 64 128)
# BATCH_SIZES=(128)

# Learning rates to test for each batch size
LRS=("1e-2" "1e-3" "1e-4")

echo "Testing batch sizes: ${BATCH_SIZES[@]}"
echo "Testing learning rates: ${LRS[@]}"
echo "Total tokens: $TOTAL_TOKENS"
echo "Context length: $CONTEXT_LENGTH"
echo "Warmup ratio: ${WARMUP_RATIO} of max iters"
echo "Cosine cycle ratio: ${COSINE_RATIO} of max iters"
echo ""

# # Run experiment for each combination
# for bz in "${BATCH_SIZES[@]}"; do
#     # Calculate max_iters for this batch size
#     MAX_ITERS=$(python -c "print(int($TOTAL_TOKENS / $bz / $CONTEXT_LENGTH / 4))")
#     echo "======================================"
#     echo "Batch size: $bz -> Max iterations: $MAX_ITERS"
#     echo "======================================"

#     for lr in "${LRS[@]}"; do
#         echo "Running experiment with BZ = $bz, LR = $lr, MAX_ITERS = $MAX_ITERS"

#         EXPERIMENT_NAME="bz_sweep_${bz}_lr_${lr}"
#         CHECKPOINT_DIR="checkpoints/bz_sweep/bz_${bz}_lr_${lr}"
#         mkdir -p "$CHECKPOINT_DIR"

#         uv run python cs336_basics/train_with_logging.py \
#             --experiment_name "$EXPERIMENT_NAME" \
#             --description "Batch size sweep: testing bz=$bz, lr=$lr" \
#             --tags "sweep,batch_size,bz_${bz},lr_${lr}" \
#             --notes "Part of batch size sweep to find optimal batch size and learning rate combination" \
#             \
#             --train_data_path "$TRAIN_DATA" \
#             --val_data_path "$VAL_DATA" \
#             --checkpoint_dir "$CHECKPOINT_DIR" \
#             --log_dir "$LOG_DIR" \
#             \
#             --vocab_size 10000 \
#             --context_length "$CONTEXT_LENGTH" \
#             --d_model 512 \
#             --num_layers 4 \
#             --num_heads 16 \
#             --d_ff 1344 \
#             --rope_theta 10000.0 \
#             \
#             --batch_size "$bz" \
#             --max_iters "$MAX_ITERS" \
#             --seed 42 \
#             --device cuda \
#             \
#             --max_lr "$lr" \
#             --min_lr "$(python -c "print(float('$lr') / 10)")" \
#             --beta1 0.9 \
#             --beta2 0.999 \
#             --weight_decay 0.1 \
#             --grad_clip 1.0 \
#             \
#             --warmup_iters "$(python -c "print(int($MAX_ITERS * $WARMUP_RATIO))")" \
#             --cosine_cycle_iters "$(python -c "print(int($MAX_ITERS * $COSINE_RATIO))")" \
#             \
#             --log_interval 50 \
#             --eval_interval 500 \
#             --eval_batches 20 \
#             --checkpoint_interval 500

#         echo ""
#     done
# done

echo ""
echo "======================================"
echo "All experiments complete!"
echo "======================================"

# Collect all experiment directories for each batch size
for bz in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "======================================"
    echo "Results for Batch Size = $bz"
    echo "======================================"

    EXPERIMENT_DIRS=()
    EXPERIMENT_NAMES=()
    for lr in "${LRS[@]}"; do
        dir=$(ls -d ${LOG_DIR}/bz_sweep_${bz}_lr_${lr}_* 2>/dev/null | tail -1)
        if [ -n "$dir" ]; then
            EXPERIMENT_DIRS+=("$dir")
            EXPERIMENT_NAMES+=("BZ=${bz}, LR=${lr}")
        fi
    done

    # Generate comparison plot for this batch size
    if [ ${#EXPERIMENT_DIRS[@]} -gt 0 ]; then
        echo ""
        echo "Generating comparison plot for batch size $bz..."
        uv run python cs336_basics/analyze_experiments.py \
            --experiment_dirs "${EXPERIMENT_DIRS[@]}" \
            --names "${EXPERIMENT_NAMES[@]}" \
            --plot_type comparison \
            --output "experiments/bz_${bz}_comparison.png"

        echo "Comparison saved to: experiments/bz_${bz}_comparison.png"
    fi
done

# Generate overall comparison plot with all experiments
echo ""
echo "======================================"
echo "Generating Overall Comparison"
echo "======================================"

ALL_EXPERIMENT_DIRS=()
ALL_EXPERIMENT_NAMES=()
for bz in "${BATCH_SIZES[@]}"; do
    for lr in "${LRS[@]}"; do
        dir=$(ls -d ${LOG_DIR}/bz_sweep_${bz}_lr_${lr}_* 2>/dev/null | tail -1)
        if [ -n "$dir" ]; then
            ALL_EXPERIMENT_DIRS+=("$dir")
            ALL_EXPERIMENT_NAMES+=("BZ=${bz}, LR=${lr}")
        fi
    done
done

if [ ${#ALL_EXPERIMENT_DIRS[@]} -gt 0 ]; then
    echo "Generating overall comparison plot..."
    uv run python cs336_basics/analyze_experiments.py \
        --experiment_dirs "${ALL_EXPERIMENT_DIRS[@]}" \
        --names "${ALL_EXPERIMENT_NAMES[@]}" \
        --plot_type comparison \
        --output "experiments/bz_sweep_all_comparison.png"

    echo "Overall comparison saved to: experiments/bz_sweep_all_comparison.png"
fi

# Print summaries
echo ""
echo "======================================"
echo "Experiment Summaries"
echo "======================================"
for dir in "${ALL_EXPERIMENT_DIRS[@]}"; do
    uv run python cs336_basics/analyze_experiments.py \
        --experiment_dirs "$dir" \
        --summary
    echo ""
done

echo "======================================"
echo "Batch size sweep complete!"
echo "======================================"
echo "Review the following plots for results:"
for bz in "${BATCH_SIZES[@]}"; do
    echo "  - experiments/bz_${bz}_comparison.png (BZ=${bz} with different LRs)"
done
echo "  - experiments/bz_sweep_all_comparison.png (all combinations)"
