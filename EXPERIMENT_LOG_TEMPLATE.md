# Experiment Log

**Student Name:** [Your Name]
**Date Started:** [Date]
**Assignment:** CS336 Assignment 1 - Basics

---

## Overview

This document tracks all experiments conducted for the CS336 assignment. Each experiment includes:
- Motivation and hypothesis
- Configuration details
- Results and observations
- Learning curves (loss vs steps and wallclock time)
- Conclusions and next steps

---

## Baseline Experiment

### Experiment 1: Baseline 17M Model on TinyStories

**Date:** YYYY-MM-DD
**Status:** ✅ Completed / 🔄 Running / ❌ Failed

**Motivation:**
Establish a baseline for the 17M parameter model on TinyStories dataset to compare against future ablations and variations.

**Hypothesis:**
With standard hyperparameters (learning rate 6e-4, batch size 8), the model should converge to a reasonable loss on the TinyStories dataset.

**Configuration:**
```yaml
Model Architecture:
  vocab_size: 50257
  context_length: 512
  d_model: 512
  num_layers: 8
  num_heads: 8
  d_ff: 2048
  parameters: ~17M

Training:
  batch_size: 8
  max_iters: 50000
  seed: 42

Optimizer:
  max_lr: 6e-4
  min_lr: 6e-5
  weight_decay: 0.1
  grad_clip: 1.0

Schedule:
  warmup_iters: 2000
  cosine_cycle_iters: 50000

Data:
  train: data/tinystories_train.bin
  val: data/tinystories_val.bin
```

**Results:**

| Metric | Value |
|--------|-------|
| Final Train Loss | X.XXXX |
| Final Val Loss | X.XXXX |
| Best Val Loss | X.XXXX (at step XXXXX) |
| Total Training Time | X.X hours |
| Tokens/sec | XXXX |

**Learning Curves:**
- See: `experiments/baseline_17m/loss_curves.png`
- Train loss decreased steadily from X.XX to X.XX
- Validation loss followed training loss closely
- No signs of overfitting/underfitting

**Observations:**
- [Note any interesting behaviors during training]
- [Convergence speed, stability, etc.]
- [Any unexpected results]

**Conclusions:**
- [What did we learn?]
- [Does this match expectations?]
- [What should we try next?]

**Next Steps:**
- [ ] Try different learning rates
- [ ] Ablate attention components
- [ ] Test different model sizes

---

## Ablation Studies

### Experiment 2: [Ablation Name]

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
[Why are we doing this experiment?]

**Hypothesis:**
[What do we expect to happen?]

**Changes from Baseline:**
```yaml
# List only what changed
Parameter: new_value  # was: old_value
```

**Results:**

| Metric | Baseline | This Experiment | Δ |
|--------|----------|-----------------|---|
| Final Train Loss | X.XXXX | Y.YYYY | ±Z.ZZ |
| Final Val Loss | X.XXXX | Y.YYYY | ±Z.ZZ |
| Training Time | X.X hrs | Y.Y hrs | ±Z% |

**Learning Curves:**
- See: `experiments/[exp_name]/comparison.png`

**Observations:**
- [Detailed observations]

**Conclusions:**
- [Key findings]
- [Unexpected results?]

**Impact:**
☐ Positive - Use this going forward
☐ Negative - Revert to baseline
☐ Neutral - No clear winner

---

### Experiment 3: No Positional Encoding

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Understand the importance of positional information (RoPE) in the Transformer.

**Hypothesis:**
Without positional encoding, the model will struggle to learn sequential patterns and perform worse.

**Changes from Baseline:**
```yaml
rope_theta: null  # Disabled RoPE
```

**Results:**
[Fill in after experiment]

---

### Experiment 4: [Learning Rate Variation]

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Find optimal learning rate for this model size and dataset.

**Hypothesis:**
[Your hypothesis]

**Changes from Baseline:**
```yaml
max_lr: [new_value]  # was: 6e-4
```

**Results:**
[Fill in]

---

## Hyperparameter Sweeps

### Experiment 5: Learning Rate Sweep

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Systematically explore learning rate space.

**Configurations:**
| Exp ID | Learning Rate | Max Steps | Notes |
|--------|---------------|-----------|-------|
| 5a | 1e-4 | 10000 | Very conservative |
| 5b | 3e-4 | 10000 | |
| 5c | 6e-4 | 10000 | Baseline |
| 5d | 1e-3 | 10000 | |
| 5e | 3e-3 | 10000 | Aggressive |

**Results Summary:**
| Exp ID | Final Train Loss | Final Val Loss | Stable? |
|--------|------------------|----------------|---------|
| 5a | X.XXXX | X.XXXX | ✅ |
| 5b | X.XXXX | X.XXXX | ✅ |
| 5c | X.XXXX | X.XXXX | ✅ |
| 5d | X.XXXX | X.XXXX | ⚠️ |
| 5e | X.XXXX | X.XXXX | ❌ |

**Visualization:**
- See: `experiments/lr_sweep/comparison.png`

**Conclusions:**
- [Which learning rate was best?]
- [At what point did training become unstable?]
- [Relationship between LR and convergence speed?]

---

## Architecture Variations

### Experiment 6: [Depth vs Width Trade-off]

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Compare deeper narrow models vs shallower wide models with same parameter count.

**Configurations:**
| Exp ID | Layers | d_model | Params | Notes |
|--------|--------|---------|--------|-------|
| 6a | 16 | 384 | ~17M | Deep & narrow |
| 6b | 8 | 512 | ~17M | Baseline |
| 6c | 4 | 720 | ~17M | Shallow & wide |

**Results:**
[Fill in comparison]

---

## Training Dynamics

### Experiment 7: [Batch Size Effects]

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Understand how batch size affects training dynamics and final performance.

**Configurations:**
| Exp ID | Batch Size | Effective LR | Gradient Updates |
|--------|------------|--------------|------------------|
| 7a | 4 | 6e-4 | 2x more frequent |
| 7b | 8 | 6e-4 | Baseline |
| 7c | 16 | 6e-4 | 2x less frequent |
| 7d | 16 | 1.2e-3 | Scaled LR |

**Results:**
[Fill in]

---

## Optimization Experiments

### Experiment 8: [Weight Decay Variations]

**Date:** YYYY-MM-DD
**Status:** ✅ / 🔄 / ❌

**Motivation:**
Study regularization effects of weight decay.

**Configurations:**
| Exp ID | Weight Decay | Train Loss | Val Loss | Generalization Gap |
|--------|--------------|------------|----------|-------------------|
| 8a | 0.0 | X.XXXX | Y.YYYY | Z.ZZ |
| 8b | 0.01 | X.XXXX | Y.YYYY | Z.ZZ |
| 8c | 0.1 | X.XXXX | Y.YYYY | Z.ZZ |
| 8d | 0.3 | X.XXXX | Y.YYYY | Z.ZZ |

**Results:**
[Fill in]

---

## Summary of Findings

### Key Insights

1. **Learning Rate:**
   - Optimal: [value]
   - Reasoning: [why]

2. **Architecture:**
   - Best configuration: [details]
   - Trade-offs observed: [what]

3. **Regularization:**
   - Most effective: [what]
   - Impact on generalization: [how]

4. **Training Dynamics:**
   - Convergence patterns: [observations]
   - Stability considerations: [what matters]

### Best Configuration

Based on all experiments, the best configuration is:

```yaml
Model:
  d_model: [value]
  num_layers: [value]
  num_heads: [value]
  d_ff: [value]

Training:
  batch_size: [value]
  max_lr: [value]
  weight_decay: [value]

Performance:
  Final Val Loss: X.XXXX
  Training Time: X.X hours
  Tokens/sec: XXXX
```

**Improvements over Baseline:**
- Validation loss: [X%] better
- Training time: [X%] faster
- [Other improvements]

### Lessons Learned

1. [Key lesson 1]
2. [Key lesson 2]
3. [Key lesson 3]
4. [Unexpected findings]

### Future Experiments

Promising directions to explore:
- [ ] [Idea 1]
- [ ] [Idea 2]
- [ ] [Idea 3]

---

## Appendix

### Experiment Commands

Quick reference for reproducing experiments:

```bash
# Baseline
uv run python cs336_basics/train.py \
    --train_data_path data/tinystories_train.bin \
    --val_data_path data/tinystories_val.bin \
    --checkpoint_dir checkpoints/baseline \
    --vocab_size 50257 --d_model 512 --num_layers 8 \
    --batch_size 8 --max_iters 50000

# Experiment 2: [Name]
uv run python cs336_basics/train.py \
    [modified parameters]
```

### Visualization Commands

```bash
# Compare experiments
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/exp1 experiments/exp2 \
    --names "Baseline" "Modified" \
    --output comparison.png

# Loss curves
uv run python cs336_basics/analyze_experiments.py \
    --experiment_dirs experiments/* \
    --plot_type loss_curves \
    --output all_losses.png
```

### Environment

- **Hardware:** [GPU model, RAM, etc.]
- **Software:** [CUDA version, PyTorch version]
- **Dataset:** TinyStories
- **Random Seeds:** 42 (unless noted otherwise)

---

## Notes and Observations

### General Training Tips

- [Any general insights about training]
- [Common pitfalls encountered]
- [Best practices discovered]

### Debugging Issues

| Issue | Experiment | Solution |
|-------|------------|----------|
| Loss exploded | Exp 5e | Reduce learning rate |
| [Other issues] | | |

---

**Last Updated:** [Date]
