##  Experiment Logging Guide

Complete guide for running and tracking experiments for CS336.

## Overview

The experiment logging infrastructure provides:

- **Comprehensive Metrics Tracking**: Log loss, learning rate, throughput with gradient steps and wallclock time
- **Experiment Metadata**: Save configurations, hyperparameters, and notes
- **Multiple Output Formats**: CSV for analysis, JSON for complete records
- **Weights & Biases Integration**: Optional cloud-based experiment tracking
- **Visualization Tools**: Plot loss curves and compare experiments
- **Experiment Registry**: Keep track of all experiments in one place

## Quick Start

### 1. Run Your First Experiment

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "baseline_17m" \
    --description "Baseline 17M model on TinyStories" \
    --tags "baseline,17m,tinystories" \
    --train_data_path data/tinystories_train.bin \
    --val_data_path data/tinystories_val.bin \
    --checkpoint_dir checkpoints/baseline_17m \
    --vocab_size 50257 \
    --context_length 512 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --batch_size 8 \
    --max_iters 50000 \
    --log_interval 10 \
    --eval_interval 500
```

This creates:
- `experiments/baseline_17m_TIMESTAMP/` directory
- `metrics.csv` - Training metrics
- `metrics.json` - Complete metrics log
- `config.json` - Experiment configuration
- Checkpoint information logs

### 2. Analyze Results

```bash
# Print experiment summary
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_17m_* \
    --summary

# Plot loss curves
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_17m_* \
    --plot_type loss_curves \
    --output baseline_losses.png
```

### 3. Compare Multiple Experiments

```bash
# Compare baseline vs ablation
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_* experiments/no_rope_* \
    --names "Baseline" "No RoPE" \
    --plot_type comparison \
    --output comparison.png
```

## Experiment Workflow

### Step 1: Plan Your Experiment

Before running, document in `EXPERIMENT_LOG_TEMPLATE.md`:
- **Motivation**: Why are you running this?
- **Hypothesis**: What do you expect?
- **Configuration**: What parameters are you using?

### Step 2: Run the Experiment

Use the enhanced training script with logging:

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "exp_descriptive_name" \
    --description "Brief description" \
    --tags "tag1,tag2,tag3" \
    --notes "Any additional notes" \
    [... other parameters ...]
```

### Step 3: Monitor Progress

During training, check:
- Console output for real-time metrics
- `experiments/exp_name/metrics.csv` for data
- Weights & Biases dashboard (if enabled)

### Step 4: Analyze Results

After training:
1. Generate visualizations
2. Compare with other experiments
3. Document findings in experiment log

### Step 5: Iterate

Based on results:
- Adjust hyperparameters
- Try ablations
- Test hypotheses

## File Structure

After running experiments, you'll have:

```
experiments/
├── registry.json                 # Registry of all experiments
├── baseline_17m_20240115_143022/
│   ├── config.json              # Experiment configuration
│   ├── metrics.csv              # Training metrics (step, time, loss, lr, etc.)
│   ├── metrics.json             # Complete metrics log
│   ├── checkpoints.json         # Checkpoint metadata
│   ├── experiment_config.txt    # Human-readable config
│   ├── log.txt                  # Text logs
│   └── completion.txt           # Training completion info
├── no_rope_20240115_150033/
│   └── ...
└── high_lr_20240115_153044/
    └── ...
```

## Logged Metrics

### Training Metrics (every `log_interval` steps)

- `step`: Current iteration
- `wallclock_time`: Time since training started (seconds)
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/tokens_per_sec`: Training throughput
- `train/iter_time`: Time per iteration

### Validation Metrics (every `eval_interval` steps)

- `val/loss`: Validation loss

### Final Metrics

- `val/final_loss`: Final validation loss

## Configuration Options

### Experiment Metadata

```bash
--experiment_name "exp_name"      # Required: unique name
--description "What this tests"   # Brief description
--tags "tag1,tag2"                # Comma-separated tags
--notes "Additional info"          # Detailed notes
--log_dir "experiments"           # Where to save logs
```

### Model Architecture

```bash
--vocab_size 50257
--context_length 512
--d_model 512                     # Model dimension
--num_layers 8                    # Number of Transformer layers
--num_heads 8                     # Number of attention heads
--d_ff 2048                       # FFN dimension (default: 4 * d_model)
--rope_theta 10000.0              # RoPE parameter
```

### Training Configuration

```bash
--batch_size 8
--max_iters 50000
--seed 42
--device cuda                     # or cpu
```

### Optimizer (AdamW)

```bash
--max_lr 6e-4                     # Maximum learning rate
--min_lr 6e-5                     # Minimum learning rate
--beta1 0.9                       # Adam beta1
--beta2 0.999                     # Adam beta2
--weight_decay 0.1                # Weight decay coefficient
--grad_clip 1.0                   # Gradient clipping (0 to disable)
```

### Learning Rate Schedule

```bash
--warmup_iters 2000               # Warmup steps
--cosine_cycle_iters 50000        # Cosine annealing cycle
```

### Logging

```bash
--log_interval 10                 # Log every N steps
--eval_interval 500               # Evaluate every N steps
--eval_batches 20                 # Batches for validation
--checkpoint_interval 5000        # Save checkpoint every N steps
```

### Weights & Biases

```bash
--wandb_project "cs336-assignment1"  # Enable W&B logging
```

## Example Experiments

### Baseline Experiment

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "baseline_17m" \
    --description "Baseline configuration for 17M model" \
    --tags "baseline,17m" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --checkpoint_dir checkpoints/baseline \
    --vocab_size 50257 \
    --d_model 512 --num_layers 8 --num_heads 8 \
    --batch_size 8 --max_iters 50000
```

### Learning Rate Sweep

```bash
for lr in 1e-4 3e-4 6e-4 1e-3; do
    uv run python cs336_basics/train_with_logging.py \
        --experiment_name "lr_sweep_${lr}" \
        --description "Testing learning rate ${lr}" \
        --tags "sweep,lr" \
        --train_data_path data/train.bin \
        --val_data_path data/val.bin \
        --max_lr $lr \
        --max_iters 10000
done
```

### Ablation Study (No RoPE)

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "ablation_no_rope" \
    --description "Remove positional encoding to test importance" \
    --tags "ablation,rope" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --rope_theta 0  # Disable RoPE
    --max_iters 50000
```

### Model Size Comparison

```bash
# Small (8M params)
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "size_8m" \
    --d_model 384 --num_layers 6 \
    --tags "size,8m" \
    [...]

# Medium (17M params)
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "size_17m" \
    --d_model 512 --num_layers 8 \
    --tags "size,17m" \
    [...]

# Large (35M params)
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "size_35m" \
    --d_model 640 --num_layers 12 \
    --tags "size,35m" \
    [...]
```

## Analysis Tools

### 1. Experiment Summary

```bash
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_* \
    --summary
```

Output:
```
================================================================================
Experiment: baseline_17m_20240115_143022
================================================================================

Configuration:
  experiment_name: baseline_17m
  vocab_size: 50257
  d_model: 512
  ...

Metrics logged: 5000 steps
First step: 0
Last step: 50000

Final metrics:
  train/loss: 2.3456
  val/loss: 2.4567

Best validation loss: 2.4123 at step 45000
================================================================================
```

### 2. Loss Curves

```bash
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/exp1 experiments/exp2 \
    --names "Experiment 1" "Experiment 2" \
    --plot_type loss_curves \
    --output loss_comparison.png
```

Generates plots of:
- Training loss vs steps
- Training loss vs wallclock time
- Validation loss vs steps (if available)

### 3. Comprehensive Comparison

```bash
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/* \
    --plot_type comparison \
    --output full_comparison.png
```

Generates 6 subplots:
1. Train loss vs steps
2. Train loss vs time
3. Validation loss vs steps
4. Learning rate schedule
5. Training throughput
6. Final metrics comparison

## Working with Metrics

### Loading Metrics Programmatically

```python
from cs336_basics.analyze_experiments import load_experiment_metrics

# Load metrics
metrics = load_experiment_metrics("experiments/baseline_17m_20240115_143022")

# Access data
for entry in metrics:
    step = entry['step']
    loss = entry['train/loss']
    time = entry['wallclock_time']
    print(f"Step {step}: loss={loss:.4f}, time={time:.1f}s")

# Find best validation loss
val_metrics = [m for m in metrics if 'val/loss' in m]
best = min(val_metrics, key=lambda m: m['val/loss'])
print(f"Best val loss: {best['val/loss']:.4f} at step {best['step']}")
```

### Custom Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load as DataFrame
df = pd.read_csv("experiments/baseline_17m_20240115_143022/metrics.csv")

# Plot custom analysis
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['train/loss'], label='Train Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('custom_plot.png')
```

## Tips and Best Practices

### 1. Naming Conventions

Use descriptive experiment names:
- ✅ `baseline_17m_tinystories`
- ✅ `ablation_no_rope`
- ✅ `lr_sweep_6e4`
- ❌ `exp1`
- ❌ `test`
- ❌ `final_final_v3`

### 2. Tagging

Use consistent tags for easy filtering:
- Experiment type: `baseline`, `ablation`, `sweep`
- Model size: `8m`, `17m`, `35m`
- Dataset: `tinystories`, `openwebtext`
- Architecture variant: `rope`, `no_rope`, `layernorm`

### 3. Documentation

Always document:
- **Before running**: Write hypothesis in experiment log
- **During running**: Monitor for issues
- **After running**: Document results and conclusions

### 4. Comparing Fairly

When comparing experiments:
- Use same random seed for reproducibility
- Use same data split
- Run to same number of steps OR wallclock time
- Average over multiple seeds for important comparisons

### 5. Resource Management

- Use smaller models for quick iterations
- Run full training only for final comparisons
- Use checkpoints to avoid re-running
- Monitor GPU memory usage

## Weights & Biases Integration

### Setup

```bash
# Install wandb
pip install wandb

# Login (one time)
wandb login
```

### Usage

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "my_experiment" \
    --wandb_project "cs336-assignment1" \
    [... other args ...]
```

W&B will track:
- All metrics (loss, learning rate, throughput)
- System metrics (GPU usage, memory)
- Experiment configuration
- Code version

View results at: https://wandb.ai/your-username/cs336-assignment1

## Troubleshooting

### Issue: Experiment directory already exists

**Solution**: Each run creates a timestamped directory. If you see this, it means the name + timestamp combination is duplicated (very rare).

### Issue: Metrics not logging

**Solution**: Check that `--log_interval` is reasonable (not too large).

### Issue: Visualization fails

**Solution**:
```bash
# Install matplotlib if missing
pip install matplotlib

# Use non-interactive backend
export MPLBACKEND=Agg
```

### Issue: W&B not working

**Solution**:
```bash
# Verify wandb is installed
pip install wandb

# Re-authenticate
wandb login

# Check project name is correct
```

## Complete Example Workflow

```bash
# 1. Run baseline
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "baseline" \
    --description "Standard configuration" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --max_iters 50000

# 2. Run ablation
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "no_rope" \
    --description "Test without RoPE" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --rope_theta 0 \
    --max_iters 50000

# 3. Compare results
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_* experiments/no_rope_* \
    --names "Baseline" "No RoPE" \
    --output comparison.png

# 4. View summaries
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/* \
    --summary

# 5. Document findings in EXPERIMENT_LOG_TEMPLATE.md
```

## Reference

- **Training script**: `cs336_basics/train_with_logging.py`
- **Logger**: `cs336_basics/experiment_logger.py`
- **Analysis tools**: `cs336_basics/analyze_experiments.py`
- **Log template**: `EXPERIMENT_LOG_TEMPLATE.md`
