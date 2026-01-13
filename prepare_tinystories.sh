#!/bin/bash
# Prepare TinyStories data for training
# Tokenizes text files and saves as memory-mapped binary files

set -e  # Exit on error

echo "======================================"
echo "TinyStories Data Preparation"
echo "======================================"
echo ""

# Configuration
TRAIN_INPUT="/home/ubuntu/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
VAL_INPUT="/home/ubuntu/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
VOCAB="/home/ubuntu/assignment1-basics/output/tinystories_vocab.json"
MERGES="/home/ubuntu/assignment1-basics/output/tinystories_merges.txt"

# Output paths
OUTPUT_DIR="/home/ubuntu/assignment1-basics/data"
TRAIN_OUTPUT="${OUTPUT_DIR}/tinystories_train.bin"
VAL_OUTPUT="${OUTPUT_DIR}/tinystories_val.bin"

# Check input files exist
echo "Checking input files..."
if [ ! -f "$TRAIN_INPUT" ]; then
    echo "Error: Training file not found: $TRAIN_INPUT"
    exit 1
fi

if [ ! -f "$VAL_INPUT" ]; then
    echo "Error: Validation file not found: $VAL_INPUT"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "Error: Vocabulary file not found: $VOCAB"
    exit 1
fi

if [ ! -f "$MERGES" ]; then
    echo "Error: Merges file not found: $MERGES"
    exit 1
fi

echo "✓ All input files found"
echo ""

# Display file sizes
echo "Input files:"
echo "  Training:   $(du -h "$TRAIN_INPUT" | cut -f1)"
echo "  Validation: $(du -h "$VAL_INPUT" | cut -f1)"
echo "  Vocabulary: $(du -h "$VOCAB" | cut -f1)"
echo "  Merges:     $(du -h "$MERGES" | cut -f1)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process training data
echo "======================================"
echo "Processing Training Data"
echo "======================================"
echo ""

uv run python cs336_basics/prepare_data.py \
    --input "$TRAIN_INPUT" \
    --output "$TRAIN_OUTPUT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --dtype uint16

echo ""

# Process validation data
echo "======================================"
echo "Processing Validation Data"
echo "======================================"
echo ""

uv run python cs336_basics/prepare_data.py \
    --input "$VAL_INPUT" \
    --output "$VAL_OUTPUT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --dtype uint16

# Check output files were created
echo ""
echo "======================================"
echo "Verification"
echo "======================================"

if [ -f "$TRAIN_OUTPUT" ]; then
    TRAIN_SIZE=$(du -h "$TRAIN_OUTPUT" | cut -f1)
    echo "✓ Training data created: $TRAIN_OUTPUT ($TRAIN_SIZE)"
else
    echo "✗ Training data not created!"
    exit 1
fi

if [ -f "$VAL_OUTPUT" ]; then
    VAL_SIZE=$(du -h "$VAL_OUTPUT" | cut -f1)
    echo "✓ Validation data created: $VAL_OUTPUT ($VAL_SIZE)"
else
    echo "✗ Validation data not created!"
    exit 1
fi

echo ""
echo "======================================"
echo "Data preparation complete!"
echo "======================================"
echo ""
echo "You can now train your model with:"
echo ""
echo "uv run python cs336_basics/train_with_logging.py \\"
echo "    --experiment_name \"baseline_17m\" \\"
echo "    --description \"Baseline on TinyStories\" \\"
echo "    --train_data_path $TRAIN_OUTPUT \\"
echo "    --val_data_path $VAL_OUTPUT \\"
echo "    --vocab_size 32000 \\"
echo "    --d_model 512 \\"
echo "    --num_layers 8 \\"
echo "    --num_heads 8 \\"
echo "    --batch_size 8 \\"
echo "    --max_iters 50000"
echo ""
