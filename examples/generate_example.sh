#!/bin/bash
# Example script for text generation with a trained model

# Configuration
CHECKPOINT="checkpoints/checkpoint_final.pt"
VOCAB="data/vocab.pkl"
MERGES="data/merges.pkl"

# Model architecture (must match training config)
VOCAB_SIZE=50257
D_MODEL=768
NUM_LAYERS=12
NUM_HEADS=12
D_FF=3072

# Generation parameters
TEMPERATURE=0.8
TOP_P=0.9
MAX_TOKENS=100

# Example 1: Single prompt
echo "Example 1: Single prompt generation"
echo "====================================="
uv run python cs336_basics/generate.py \
    --checkpoint "$CHECKPOINT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --prompt "Once upon a time" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS

echo ""
echo ""

# Example 2: Interactive mode
echo "Example 2: Interactive mode"
echo "============================"
echo "Launching interactive generation (Ctrl+C to exit)..."
uv run python cs336_basics/generate.py \
    --checkpoint "$CHECKPOINT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --interactive \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS

# Example 3: Different temperature settings
echo ""
echo "Example 3: Temperature comparison"
echo "=================================="

echo "Low temperature (0.2) - more deterministic:"
uv run python cs336_basics/generate.py \
    --checkpoint "$CHECKPOINT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --prompt "The future of AI is" \
    --temperature 0.2 \
    --max_tokens 50

echo ""
echo "High temperature (1.5) - more creative:"
uv run python cs336_basics/generate.py \
    --checkpoint "$CHECKPOINT" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --prompt "The future of AI is" \
    --temperature 1.5 \
    --max_tokens 50
