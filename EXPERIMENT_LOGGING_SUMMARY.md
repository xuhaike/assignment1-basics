# Experiment Logging System - Quick Reference

## What's Implemented

### ✅ Core Infrastructure

1. **ExperimentLogger** (`cs336_basics/experiment_logger.py`)
   - Logs metrics with gradient steps and wallclock time
   - Saves to CSV (easy analysis) and JSON (complete records)
   - Tracks experiment configurations
   - Optional Weights & Biases integration

2. **Enhanced Training Script** (`cs336_basics/train_with_logging.py`)
   - All features from original train.py
   - Integrated experiment logging
   - Automatic metric tracking
   - Configuration saving

3. **Analysis Tools** (`cs336_basics/analyze_experiments.py`)
   - Load and visualize experiment results
   - Compare multiple experiments
   - Generate loss curves
   - Print experiment summaries

4. **Documentation**
   - `EXPERIMENT_LOG_TEMPLATE.md` - Template for documenting experiments
   - `EXPERIMENT_GUIDE.md` - Complete usage guide
   - This summary

### ✅ Example Scripts

- `examples/run_baseline_experiment.sh` - Run baseline 17M model
- `examples/run_lr_sweep.sh` - Learning rate sweep
- `examples/run_ablation_study.sh` - Architecture ablations

## Quick Start

### 1. Run an Experiment

```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "my_experiment" \
    --description "What I'm testing" \
    --tags "tag1,tag2" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    --vocab_size 50257 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --batch_size 8 \
    --max_iters 50000
```

### 2. Analyze Results

```bash
# View summary
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/my_experiment_* \
    --summary

# Plot loss curves
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/my_experiment_* \
    --plot_type loss_curves \
    --output results.png
```

### 3. Compare Experiments

```bash
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/exp1_* experiments/exp2_* \
    --names "Experiment 1" "Experiment 2" \
    --plot_type comparison \
    --output comparison.png
```

## What Gets Logged

### Metrics (Automatic)

| Metric | When | Description |
|--------|------|-------------|
| `train/loss` | Every log_interval | Training loss |
| `train/learning_rate` | Every log_interval | Current LR from schedule |
| `train/tokens_per_sec` | Every log_interval | Training throughput |
| `train/iter_time` | Every log_interval | Time per iteration |
| `val/loss` | Every eval_interval | Validation loss |
| `step` | Always | Gradient step / iteration |
| `wallclock_time` | Always | Seconds since training start |

### Files Created

```
experiments/
└── my_experiment_20240115_143022/
    ├── config.json              # All hyperparameters
    ├── metrics.csv              # Training metrics (step, time, loss, lr, etc.)
    ├── metrics.json             # Complete metrics log
    ├── checkpoints.json         # Checkpoint metadata
    ├── experiment_config.txt    # Human-readable config
    └── log.txt                  # Text logs
```

## Key Features

### ✅ Tracks Gradient Steps AND Wallclock Time

Every metric is logged with both:
- **step**: Iteration/gradient update number
- **wallclock_time**: Actual time elapsed

This lets you plot loss curves by either:
- Steps (useful for comparing convergence)
- Time (useful for comparing efficiency)

### ✅ Comprehensive Configuration Saving

Automatically saves:
- Model architecture (d_model, layers, heads, etc.)
- Training hyperparameters (batch size, iterations, etc.)
- Optimizer settings (learning rate, weight decay, etc.)
- Data paths
- Random seed
- Notes and tags

### ✅ Easy Comparison

Compare experiments with one command:
```bash
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs exp1/ exp2/ exp3/ \
    --names "Name1" "Name2" "Name3" \
    --output comparison.png
```

Generates 6 comparison plots:
1. Train loss vs steps
2. Train loss vs wallclock time
3. Validation loss vs steps
4. Learning rate schedule
5. Training throughput
6. Final metrics bar chart

### ✅ Weights & Biases Integration

Optional cloud-based tracking:
```bash
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "my_exp" \
    --wandb_project "cs336-assignment1" \
    [... other args ...]
```

## Usage Patterns

### Pattern 1: Baseline First

```bash
# 1. Run baseline
./examples/run_baseline_experiment.sh

# 2. Document in experiment log
# Edit EXPERIMENT_LOG_TEMPLATE.md

# 3. Analyze
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/baseline_* \
    --summary
```

### Pattern 2: Hyperparameter Sweep

```bash
# Run sweep
./examples/run_lr_sweep.sh

# Automatic comparison plot generated
# Review: experiments/lr_sweep_comparison.png
```

### Pattern 3: Ablation Study

```bash
# Run ablations
./examples/run_ablation_study.sh

# Automatic comparison plot generated
# Review: experiments/ablation_comparison.png
```

## Integration with Assignment

### For Problem (experiment_log)

You need to submit:

1. **Logging Infrastructure Code** ✅
   - `cs336_basics/experiment_logger.py`
   - `cs336_basics/train_with_logging.py`
   - `cs336_basics/analyze_experiments.py`

2. **Experiment Log Document** ✅
   - Use `EXPERIMENT_LOG_TEMPLATE.md`
   - Fill in your experiments as you run them
   - Include hypotheses, results, and conclusions

3. **Loss Curves** ✅
   - Automatically generated by analysis tools
   - Both gradient steps and wallclock time axes
   - Training and validation losses

### Required Features ✅

- ✅ Track experiments
- ✅ Loss curves with respect to gradient steps
- ✅ Loss curves with respect to wallclock time
- ✅ Logging infrastructure code
- ✅ Experiment log document

## Tips

### Naming Conventions

Good:
- `baseline_17m_tinystories`
- `ablation_no_rope`
- `lr_sweep_6e4`

Bad:
- `exp1`, `test`, `final_v3`

### Tagging

Use consistent tags:
- Type: `baseline`, `ablation`, `sweep`
- Size: `8m`, `17m`, `35m`
- Dataset: `tinystories`, `owt`

### Documentation

1. **Before**: Write hypothesis
2. **During**: Monitor progress
3. **After**: Document findings

## File Locations

| File | Purpose |
|------|---------|
| `cs336_basics/experiment_logger.py` | Core logging infrastructure |
| `cs336_basics/train_with_logging.py` | Enhanced training script |
| `cs336_basics/analyze_experiments.py` | Analysis and visualization |
| `EXPERIMENT_LOG_TEMPLATE.md` | Template for your log |
| `EXPERIMENT_GUIDE.md` | Detailed guide |
| `examples/run_baseline_experiment.sh` | Example: baseline |
| `examples/run_lr_sweep.sh` | Example: sweep |
| `examples/run_ablation_study.sh` | Example: ablations |

## Common Commands

```bash
# Run experiment
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "name" \
    --description "desc" \
    --train_data_path data/train.bin \
    --val_data_path data/val.bin \
    [model/training args...]

# View summary
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/name_* \
    --summary

# Plot losses
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/name_* \
    --plot_type loss_curves \
    --output losses.png

# Compare multiple
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/exp1_* experiments/exp2_* \
    --names "Exp1" "Exp2" \
    --plot_type comparison \
    --output compare.png
```

## Metrics Access

### From Python

```python
from cs336_basics.analyze_experiments import load_experiment_metrics

# Load metrics
metrics = load_experiment_metrics("experiments/my_exp_20240115_143022")

# Access
for m in metrics:
    print(f"Step {m['step']}: loss={m['train/loss']:.4f}, time={m['wallclock_time']:.1f}s")
```

### From CSV

```python
import pandas as pd

df = pd.read_csv("experiments/my_exp_20240115_143022/metrics.csv")
print(df[['step', 'train/loss', 'wallclock_time']].head())
```

## Support

- Detailed guide: `EXPERIMENT_GUIDE.md`
- Experiment log template: `EXPERIMENT_LOG_TEMPLATE.md`
- Example scripts: `examples/run_*.sh`
- Code documentation: Inline comments in source files

---

**Summary**: Complete experiment tracking infrastructure is ready to use. Run experiments with `train_with_logging.py`, analyze with `analyze_experiments.py`, and document in `EXPERIMENT_LOG_TEMPLATE.md`.
