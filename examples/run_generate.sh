#!/bin/bash
# Text generation script - easily generate text from trained models

set -e

echo "Text Generation from Trained Model"
echo "===================================="
echo ""

# Configuration
DEFAULT_EXPERIMENT="experiments/baseline_17m_20260111_231919"
DEFAULT_PROMPT="Once upon a time, there was a"

# Parse command line arguments
EXPERIMENT=""
CHECKPOINT=""
CONFIG=""
PROMPT=""
MAX_TOKENS=200
TEMPERATURE=0.8
TOP_P=0.9
DEVICE="cuda"
INTERACTIVE=false

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --experiment DIR      Use experiment directory (auto-loads config and latest checkpoint)"
    echo "  --checkpoint FILE     Use specific checkpoint file"
    echo "  --config FILE         Config file (required with --checkpoint)"
    echo "  --prompt TEXT         Prompt text for generation"
    echo "  --max-tokens N        Maximum tokens to generate (default: 200)"
    echo "  --temperature T       Sampling temperature (default: 0.8)"
    echo "  --top-p P             Top-p sampling (default: 0.9)"
    echo "  --device DEVICE       Device to use: cuda or cpu (default: cuda)"
    echo "  --interactive         Interactive mode"
    echo "  --list                List available experiments"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Generate from latest experiment (auto-detect)"
    echo "  $0"
    echo ""
    echo "  # Generate with custom prompt"
    echo "  $0 --prompt \"In a magical forest\""
    echo ""
    echo "  # Interactive mode"
    echo "  $0 --interactive"
    echo ""
    echo "  # Use specific experiment"
    echo "  $0 --experiment experiments/baseline_17m_20260111_231919"
    echo ""
    echo "  # Use specific checkpoint"
    echo "  $0 --checkpoint checkpoints/baseline_17m/checkpoint_010000.pt --config experiments/baseline_17m_20260111_231919/config.json"
    exit 0
}

# List available experiments
list_experiments() {
    echo "Available experiments:"
    echo ""
    for dir in experiments/*/; do
        if [ -f "$dir/config.json" ]; then
            exp_name=$(basename "$dir")
            echo "  📁 $dir"
            if [ -f "$dir/config.json" ]; then
                desc=$(python3 -c "import json; print(json.load(open('$dir/config.json')).get('description', 'N/A'))" 2>/dev/null || echo "N/A")
                echo "      Description: $desc"
            fi
            echo ""
        fi
    done
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --list)
            list_experiments
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Auto-detect latest experiment if none specified
if [ -z "$EXPERIMENT" ] && [ -z "$CHECKPOINT" ]; then
    echo "🔍 Auto-detecting latest experiment..."
    # Find the most recent experiment directory
    LATEST_EXP=$(ls -td experiments/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_EXP" ] && [ -f "$LATEST_EXP/config.json" ]; then
        EXPERIMENT="$LATEST_EXP"
        echo "✓ Found: $EXPERIMENT"
    else
        echo "❌ No experiments found. Please specify --experiment or --checkpoint"
        echo ""
        show_usage
    fi
fi

# Build command
CMD="uv run python cs336_basics/simple_generate.py"

if [ -n "$EXPERIMENT" ]; then
    CMD="$CMD --experiment \"$EXPERIMENT\""
elif [ -n "$CHECKPOINT" ]; then
    if [ -z "$CONFIG" ]; then
        echo "Error: --config is required when using --checkpoint"
        exit 1
    fi
    CMD="$CMD --checkpoint \"$CHECKPOINT\" --config \"$CONFIG\""
fi

if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
fi

CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --device $DEVICE"

if [ "$INTERACTIVE" = true ]; then
    CMD="$CMD --interactive"
fi

# Run generation
echo ""
eval $CMD
