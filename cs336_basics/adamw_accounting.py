"""
Resource accounting for training with AdamW optimizer.

This module provides functions to calculate memory usage and FLOPs for training
a Transformer language model with the AdamW optimizer.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TransformerConfig:
    """Configuration for a Transformer language model."""
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int = None  # Will be set to 4 * d_model if not provided

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


@dataclass
class MemoryBreakdown:
    """Memory breakdown in bytes for different components."""
    parameters: int
    activations: int
    gradients: int
    optimizer_state: int
    total: int

    def to_gb(self) -> Dict[str, float]:
        """Convert all memory values to GB."""
        gb_divisor = 1024**3
        return {
            'parameters_gb': self.parameters / gb_divisor,
            'activations_gb': self.activations / gb_divisor,
            'gradients_gb': self.gradients / gb_divisor,
            'optimizer_state_gb': self.optimizer_state / gb_divisor,
            'total_gb': self.total / gb_divisor,
        }


def count_parameters(config: TransformerConfig) -> int:
    """
    Count the total number of parameters in the Transformer LM.

    Args:
        config: Transformer configuration

    Returns:
        Total number of parameters
    """
    V = config.vocab_size
    d = config.d_model
    L = config.num_layers
    d_ff = config.d_ff

    # Token embeddings: V * d
    token_embeddings = V * d

    # Per layer:
    # - Attention: 4 * d * d (Q, K, V, O projections)
    # - 2 RMSNorms: 2 * d (gain parameters)
    # - FFN (SwiGLU): 3 * d * d_ff (W1, W2, W3)
    #   W1: d_model -> d_ff, W2: d_ff -> d_model, W3: d_model -> d_ff
    per_layer = 4 * d * d + 2 * d + 3 * d * d_ff

    # Final RMSNorm: d
    final_norm = d

    # LM head: V * d
    lm_head = V * d

    total = token_embeddings + L * per_layer + final_norm + lm_head

    return total


def count_activation_memory(config: TransformerConfig, batch_size: int) -> int:
    """
    Count the memory required for activations during forward pass.

    Activations need to be stored for the backward pass.

    Args:
        config: Transformer configuration
        batch_size: Batch size

    Returns:
        Total activation memory in elements (multiply by 4 for bytes in float32)
    """
    B = batch_size
    S = config.context_length
    d = config.d_model
    h = config.num_heads
    d_ff = config.d_ff
    L = config.num_layers
    V = config.vocab_size

    # Per Transformer block:
    per_layer_activations = 0

    # RMSNorm1 output: B * S * d
    per_layer_activations += B * S * d

    # QKV projections: 3 * B * S * d
    per_layer_activations += 3 * B * S * d

    # Q^T K (attention scores): B * h * S * S
    per_layer_activations += B * h * S * S

    # Softmax output: B * h * S * S
    per_layer_activations += B * h * S * S

    # Attention output (weighted sum): B * S * d
    per_layer_activations += B * S * d

    # Output projection: B * S * d
    per_layer_activations += B * S * d

    # RMSNorm2 output: B * S * d
    per_layer_activations += B * S * d

    # SwiGLU FFN:
    # W1 output: B * S * d_ff
    per_layer_activations += B * S * d_ff

    # SiLU output: B * S * d_ff
    per_layer_activations += B * S * d_ff

    # W3 output: B * S * d_ff (for the gating path)
    per_layer_activations += B * S * d_ff

    # W2 output: B * S * d
    per_layer_activations += B * S * d

    # Total for all layers
    total_activations = L * per_layer_activations

    # Final RMSNorm: B * S * d
    total_activations += B * S * d

    # Output embedding (logits): B * S * V
    total_activations += B * S * V

    # Cross-entropy (stores per-token losses): B * S
    total_activations += B * S

    return total_activations


def calculate_peak_memory(
    config: TransformerConfig,
    batch_size: int,
    bytes_per_element: int = 4
) -> MemoryBreakdown:
    """
    Calculate peak memory usage during training with AdamW.

    Args:
        config: Transformer configuration
        batch_size: Batch size
        bytes_per_element: Bytes per float (4 for float32)

    Returns:
        MemoryBreakdown with memory usage for each component
    """
    # Count parameters
    num_params = count_parameters(config)

    # Parameters memory
    params_memory = num_params * bytes_per_element

    # Activations memory
    num_activations = count_activation_memory(config, batch_size)
    activations_memory = num_activations * bytes_per_element

    # Gradients memory (same size as parameters)
    gradients_memory = num_params * bytes_per_element

    # Optimizer state for AdamW:
    # - First moment (m): same size as parameters
    # - Second moment (v): same size as parameters
    # Total: 2 * parameters
    optimizer_state_memory = 2 * num_params * bytes_per_element

    # Total memory
    total_memory = (
        params_memory +
        activations_memory +
        gradients_memory +
        optimizer_state_memory
    )

    return MemoryBreakdown(
        parameters=params_memory,
        activations=activations_memory,
        gradients=gradients_memory,
        optimizer_state=optimizer_state_memory,
        total=total_memory
    )


def find_max_batch_size(
    config: TransformerConfig,
    max_memory_gb: float = 80.0,
    bytes_per_element: int = 4
) -> int:
    """
    Find the maximum batch size that fits in the given memory.

    Args:
        config: Transformer configuration
        max_memory_gb: Maximum memory in GB
        bytes_per_element: Bytes per float (4 for float32)

    Returns:
        Maximum batch size
    """
    max_memory_bytes = max_memory_gb * (1024**3)

    # Memory = a * batch_size + b
    # We need to find the coefficients a and b

    # Fixed memory (independent of batch size)
    num_params = count_parameters(config)
    fixed_memory = num_params * bytes_per_element  # parameters
    fixed_memory += num_params * bytes_per_element  # gradients
    fixed_memory += 2 * num_params * bytes_per_element  # optimizer state (m, v)

    # Variable memory (depends on batch size)
    # Get activation memory for batch_size=1
    activations_per_batch = count_activation_memory(config, batch_size=1)
    variable_memory_per_batch = activations_per_batch * bytes_per_element

    # Solve: max_memory = fixed + variable_per_batch * batch_size
    max_batch_size = int((max_memory_bytes - fixed_memory) / variable_memory_per_batch)

    return max_batch_size


def count_adamw_flops(num_params: int) -> int:
    """
    Count FLOPs for one AdamW optimizer step.

    For each parameter:
    - Update m: 3 FLOPs (multiply by beta1, multiply grad by (1-beta1), add)
    - Update v: 4 FLOPs (multiply by beta2, square grad, multiply by (1-beta2), add)
    - Compute sqrt(v): 1 FLOP per element
    - Add eps: 1 FLOP per element
    - Divide m by (sqrt(v) + eps): 1 FLOP per element
    - Multiply by -alpha_t: 1 FLOP per element
    - Add to parameters: 1 FLOP per element
    - Weight decay (if non-zero): 2 FLOPs (multiply p by -lr*lambda, add to p)

    Total per parameter: ~14 FLOPs (including weight decay)

    Args:
        num_params: Number of parameters

    Returns:
        Total FLOPs for AdamW step
    """
    # Conservative estimate:
    # m update: 3 FLOPs
    # v update: 4 FLOPs
    # sqrt(v) + eps: 2 FLOPs
    # m / (sqrt(v) + eps): 1 FLOP
    # multiply by -alpha_t: 1 FLOP
    # add to p: 1 FLOP
    # weight decay: 2 FLOPs
    flops_per_param = 14

    return num_params * flops_per_param


def estimate_training_time(
    config: TransformerConfig,
    batch_size: int,
    num_steps: int,
    peak_flops_per_second: float = 19.5e12,  # 19.5 TFLOP/s for A100
    mfu: float = 0.5,  # Model FLOPs Utilization
    forward_backward_ratio: float = 3.0,  # backward is 2x forward, so total is 3x
) -> float:
    """
    Estimate training time in days.

    Args:
        config: Transformer configuration
        batch_size: Batch size
        num_steps: Number of training steps
        peak_flops_per_second: Peak FLOPs/s of hardware
        mfu: Model FLOPs Utilization (fraction of peak)
        forward_backward_ratio: Total FLOPs = forward * this ratio (default 3 for 1 fwd + 2 bwd)

    Returns:
        Training time in days
    """
    from cs336_basics.transformer_accounting import count_flops

    # FLOPs per forward pass
    forward_flops = count_flops(config, config.context_length).total

    # FLOPs per step (forward + backward + optimizer)
    flops_per_step = forward_flops * forward_backward_ratio

    # Add optimizer FLOPs
    num_params = count_parameters(config)
    optimizer_flops = count_adamw_flops(num_params)
    flops_per_step += optimizer_flops

    # Total FLOPs for training
    total_flops = flops_per_step * num_steps

    # Effective throughput
    effective_flops_per_second = peak_flops_per_second * mfu

    # Time in seconds
    time_seconds = total_flops / effective_flops_per_second

    # Convert to days
    time_days = time_seconds / (24 * 3600)

    return time_days


if __name__ == "__main__":
    # Problem (a): Memory breakdown for GPT-2 XL
    print("=" * 80)
    print("Problem (a): Memory breakdown for GPT-2 XL")
    print("=" * 80)

    gpt2_xl = TransformerConfig(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400
    )

    # Calculate for batch_size = 1 to show the breakdown
    memory_b1 = calculate_peak_memory(gpt2_xl, batch_size=1)
    print(f"\nMemory breakdown (batch_size=1):")
    for key, value in memory_b1.to_gb().items():
        print(f"  {key}: {value:.4f} GB")

    # Problem (b): Find max batch size for 80GB
    print("\n" + "=" * 80)
    print("Problem (b): Maximum batch size for 80GB memory")
    print("=" * 80)

    num_params = count_parameters(gpt2_xl)
    print(f"\nTotal parameters: {num_params:,}")

    # Get coefficients for: memory = a * batch_size + b
    memory_b0 = calculate_peak_memory(gpt2_xl, batch_size=0)
    memory_b1 = calculate_peak_memory(gpt2_xl, batch_size=1)

    # Fixed memory (b)
    b = (memory_b0.parameters + memory_b0.gradients + memory_b0.optimizer_state)
    b_gb = b / (1024**3)

    # Variable memory per batch (a)
    a = memory_b1.activations  # Activations scale with batch_size
    a_gb = a / (1024**3)

    print(f"\nMemory = {a_gb:.4f} * batch_size + {b_gb:.4f} GB")

    max_bs = find_max_batch_size(gpt2_xl, max_memory_gb=80.0)
    print(f"Maximum batch size: {max_bs}")

    # Verify
    memory_max = calculate_peak_memory(gpt2_xl, batch_size=max_bs)
    print(f"Memory at max batch size: {memory_max.to_gb()['total_gb']:.4f} GB")

    # Problem (c): FLOPs for AdamW step
    print("\n" + "=" * 80)
    print("Problem (c): FLOPs for one AdamW step")
    print("=" * 80)

    adamw_flops = count_adamw_flops(num_params)
    print(f"\nAdamW FLOPs per step: {adamw_flops:,}")
    print(f"AdamW FLOPs per step: {adamw_flops:.2e}")
    print("\nJustification:")
    print("  Per parameter, AdamW performs:")
    print("  - m update: ~3 FLOPs (beta1*m + (1-beta1)*g)")
    print("  - v update: ~4 FLOPs (beta2*v + (1-beta2)*g^2)")
    print("  - sqrt(v) + eps: ~2 FLOPs")
    print("  - m / (sqrt(v) + eps): ~1 FLOP")
    print("  - multiply by -alpha_t: ~1 FLOP")
    print("  - add to parameter: ~1 FLOP")
    print("  - weight decay: ~2 FLOPs (alpha*lambda*theta)")
    print("  Total: ~14 FLOPs per parameter")

    # Problem (d): Training time
    print("\n" + "=" * 80)
    print("Problem (d): Training time for GPT-2 XL")
    print("=" * 80)

    training_time = estimate_training_time(
        config=gpt2_xl,
        batch_size=1024,
        num_steps=400_000,
        peak_flops_per_second=19.5e12,
        mfu=0.5,
        forward_backward_ratio=3.0
    )

    print(f"\nTraining configuration:")
    print(f"  Model: GPT-2 XL ({num_params:,} parameters)")
    print(f"  Batch size: 1024")
    print(f"  Training steps: 400,000")
    print(f"  Hardware: A100 (19.5 TFLOP/s peak for float32)")
    print(f"  MFU: 50%")
    print(f"  Forward:Backward ratio: 1:2 (total 3x forward)")
    print(f"\nEstimated training time: {training_time:.2f} days")

    # Additional breakdown
    from cs336_basics.transformer_accounting import count_flops
    forward_flops = count_flops(gpt2_xl, gpt2_xl.context_length).total
    total_flops_per_step = forward_flops * 3 + adamw_flops

    print(f"\nFLOPs breakdown per step:")
    print(f"  Forward pass: {forward_flops:.2e}")
    print(f"  Backward pass (2x forward): {2*forward_flops:.2e}")
    print(f"  AdamW optimizer: {adamw_flops:.2e}")
    print(f"  Total per step: {total_flops_per_step:.2e}")
    print(f"  Total for 400K steps: {total_flops_per_step * 400_000:.2e}")
