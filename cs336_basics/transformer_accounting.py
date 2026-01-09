"""
Transformer LM Resource Accounting

Functions to analyze parameter count, memory usage, and FLOPs for Transformer models.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TransformerConfig:
    """Configuration for a Transformer Language Model."""
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0


@dataclass
class FLOPBreakdown:
    """Breakdown of FLOPs for different components."""
    token_embedding: int
    qkv_projections: int
    attention_scores: int
    attention_output: int
    output_projection: int
    ffn_w1: int
    ffn_w2: int
    ffn_w3: int
    lm_head: int
    total: int

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for easy access."""
        return {
            'token_embedding': self.token_embedding,
            'qkv_projections': self.qkv_projections,
            'attention_scores': self.attention_scores,
            'attention_output': self.attention_output,
            'output_projection': self.output_projection,
            'ffn_w1': self.ffn_w1,
            'ffn_w2': self.ffn_w2,
            'ffn_w3': self.ffn_w3,
            'lm_head': self.lm_head,
            'total': self.total,
        }

    def print_breakdown(self, show_percentages: bool = True):
        """Print a nice breakdown of FLOPs."""
        print("\n" + "="*70)
        print("FLOP Breakdown")
        print("="*70)

        components = [
            ("Token Embedding", self.token_embedding),
            ("Q/K/V Projections", self.qkv_projections),
            ("Attention Scores (QK^T)", self.attention_scores),
            ("Attention Output (scores @ V)", self.attention_output),
            ("Output Projection", self.output_projection),
            ("FFN W1", self.ffn_w1),
            ("FFN W2", self.ffn_w2),
            ("FFN W3", self.ffn_w3),
            ("LM Head", self.lm_head),
        ]

        for name, flops in components:
            if show_percentages and self.total > 0:
                pct = (flops / self.total) * 100
                print(f"{name:.<40} {flops:>15,} ({pct:>5.1f}%)")
            else:
                print(f"{name:.<40} {flops:>15,}")

        print("-" * 70)
        print(f"{'Total FLOPs':.<40} {self.total:>15,}")
        print("=" * 70)


def count_parameters(config: TransformerConfig) -> Dict[str, int]:
    """
    Count the number of trainable parameters in the Transformer LM.

    Args:
        config: Transformer configuration

    Returns:
        Dictionary with parameter counts for each component
    """
    params = {}

    # Token embeddings
    params['token_embeddings'] = config.vocab_size * config.d_model

    # Per-layer parameters
    # Attention: Q, K, V, O projections (each d_model x d_model)
    params['attention_per_layer'] = 4 * (config.d_model * config.d_model)

    # RMSNorm: 2 per layer (before attention and before FFN)
    params['rmsnorm_per_layer'] = 2 * config.d_model

    # FFN: w1 (d_model x d_ff), w2 (d_ff x d_model), w3 (d_model x d_ff)
    params['ffn_per_layer'] = (config.d_model * config.d_ff) + \
                               (config.d_ff * config.d_model) + \
                               (config.d_model * config.d_ff)

    # Total per layer
    params['per_layer'] = params['attention_per_layer'] + \
                           params['rmsnorm_per_layer'] + \
                           params['ffn_per_layer']

    # All layers
    params['all_layers'] = params['per_layer'] * config.num_layers

    # Final RMSNorm
    params['final_rmsnorm'] = config.d_model

    # LM head
    params['lm_head'] = config.vocab_size * config.d_model

    # Total
    params['total'] = params['token_embeddings'] + \
                       params['all_layers'] + \
                       params['final_rmsnorm'] + \
                       params['lm_head']

    return params


def calculate_memory_gb(num_parameters: int, bytes_per_param: int = 4) -> float:
    """
    Calculate memory required to store model parameters.

    Args:
        num_parameters: Number of parameters
        bytes_per_param: Bytes per parameter (4 for float32, 2 for float16)

    Returns:
        Memory in gigabytes
    """
    bytes_total = num_parameters * bytes_per_param
    gb = bytes_total / (1024**3)
    return gb


def count_flops(config: TransformerConfig, seq_len: int = None) -> FLOPBreakdown:
    """
    Count FLOPs for a forward pass through the Transformer LM.

    For matrix multiply (M, K) @ (K, N), we count 2*M*K*N FLOPs.

    Args:
        config: Transformer configuration
        seq_len: Sequence length (defaults to context_length if not provided)

    Returns:
        FLOPBreakdown with detailed FLOP counts
    """
    if seq_len is None:
        seq_len = config.context_length

    S = seq_len
    d = config.d_model
    h = config.num_heads
    d_head = d // h
    L = config.num_layers
    d_ff = config.d_ff
    V = config.vocab_size

    # Token embedding: just lookup, negligible FLOPs
    token_embedding_flops = 0

    # Per-layer FLOPs
    # Q, K, V projections: each is (S, d) @ (d, d) = 2*S*d*d FLOPs
    qkv_proj_flops = L * 3 * (2 * S * d * d)

    # Attention scores: QK^T for each head
    # Q: (S, d_head), K: (S, d_head) -> scores: (S, S)
    # For each head: (S, d_head) @ (d_head, S) = 2*S*d_head*S FLOPs
    # Total for h heads: h * 2*S*d_head*S = 2*S*S*d (since h*d_head = d)
    attn_scores_flops = L * (2 * h * S * d_head * S)
    # Simplifies to: L * 2 * S * S * d

    # Attention output: scores @ V for each head
    # scores: (S, S), V: (S, d_head) -> output: (S, d_head)
    # For each head: (S, S) @ (S, d_head) = 2*S*S*d_head FLOPs
    # Total for h heads: h * 2*S*S*d_head = 2*S*S*d
    attn_output_flops = L * (2 * h * S * S * d_head)

    # Output projection: (S, d) @ (d, d) = 2*S*d*d FLOPs
    output_proj_flops = L * (2 * S * d * d)

    # FFN projections
    # w1: (S, d) @ (d, d_ff) = 2*S*d*d_ff FLOPs
    ffn_w1_flops = L * (2 * S * d * d_ff)

    # w2: (S, d_ff) @ (d_ff, d) = 2*S*d_ff*d FLOPs
    ffn_w2_flops = L * (2 * S * d_ff * d)

    # w3: (S, d) @ (d, d_ff) = 2*S*d*d_ff FLOPs
    ffn_w3_flops = L * (2 * S * d * d_ff)

    # LM head: (S, d) @ (d, V) = 2*S*d*V FLOPs
    lm_head_flops = 2 * S * d * V

    # Total
    total_flops = (
        token_embedding_flops +
        qkv_proj_flops +
        attn_scores_flops +
        attn_output_flops +
        output_proj_flops +
        ffn_w1_flops +
        ffn_w2_flops +
        ffn_w3_flops +
        lm_head_flops
    )

    return FLOPBreakdown(
        token_embedding=token_embedding_flops,
        qkv_projections=qkv_proj_flops,
        attention_scores=attn_scores_flops,
        attention_output=attn_output_flops,
        output_projection=output_proj_flops,
        ffn_w1=ffn_w1_flops,
        ffn_w2=ffn_w2_flops,
        ffn_w3=ffn_w3_flops,
        lm_head=lm_head_flops,
        total=total_flops,
    )


def analyze_model(config: TransformerConfig, seq_len: int = None, verbose: bool = True):
    """
    Comprehensive analysis of a Transformer LM.

    Args:
        config: Transformer configuration
        seq_len: Sequence length for FLOP analysis (defaults to context_length)
        verbose: Whether to print detailed output

    Returns:
        Tuple of (parameters_dict, memory_gb, flop_breakdown)
    """
    # Count parameters
    params = count_parameters(config)

    # Calculate memory
    memory_gb = calculate_memory_gb(params['total'])

    # Count FLOPs
    flops = count_flops(config, seq_len)

    if verbose:
        print("\n" + "="*70)
        print(f"Transformer LM Analysis: {config.num_layers} layers, {config.d_model} d_model")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  vocab_size: {config.vocab_size:,}")
        print(f"  context_length: {config.context_length:,}")
        print(f"  num_layers: {config.num_layers}")
        print(f"  d_model: {config.d_model:,}")
        print(f"  num_heads: {config.num_heads}")
        print(f"  d_ff: {config.d_ff:,}")

        print(f"\n{'Parameter Count':.<40}")
        print(f"{'  Token Embeddings':.<40} {params['token_embeddings']:>15,}")
        print(f"{'  Per Layer':.<40} {params['per_layer']:>15,}")
        print(f"{'    - Attention':.<40} {params['attention_per_layer']:>15,}")
        print(f"{'    - FFN':.<40} {params['ffn_per_layer']:>15,}")
        print(f"{'    - RMSNorm':.<40} {params['rmsnorm_per_layer']:>15,}")
        print(f"{'  All Layers':.<40} {params['all_layers']:>15,}")
        print(f"{'  Final RMSNorm':.<40} {params['final_rmsnorm']:>15,}")
        print(f"{'  LM Head':.<40} {params['lm_head']:>15,}")
        print(f"{'-'*70}")
        print(f"{'  Total Parameters':.<40} {params['total']:>15,}")
        print(f"\n{'Memory (float32)':.<40} {memory_gb:>10.2f} GB")

        # Print FLOP breakdown
        actual_seq_len = seq_len if seq_len is not None else config.context_length
        print(f"\nFLOPs for forward pass (sequence length = {actual_seq_len:,}):")
        flops.print_breakdown(show_percentages=True)

        # Identify most expensive component
        flop_dict = flops.to_dict()
        del flop_dict['total']
        max_component = max(flop_dict.items(), key=lambda x: x[1])
        print(f"\nMost expensive component: {max_component[0]} ({(max_component[1]/flops.total)*100:.1f}% of total FLOPs)")

    return params, memory_gb, flops


def compare_models(configs: Dict[str, TransformerConfig], seq_len: int = None):
    """
    Compare multiple model configurations.

    Args:
        configs: Dictionary of {model_name: config}
        seq_len: Sequence length for FLOP analysis
    """
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)

    results = {}
    for name, config in configs.items():
        params, memory, flops = analyze_model(config, seq_len, verbose=False)
        results[name] = {
            'params': params['total'],
            'memory_gb': memory,
            'flops': flops,
        }

    # Print comparison table
    print(f"\n{'Model':<20} {'Parameters':>15} {'Memory (GB)':>12} {'Total FLOPs':>18}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{name:<20} {result['params']:>15,} {result['memory_gb']:>12.2f} {result['flops'].total:>18,}")

    # Print FLOP breakdown for each model
    for name, result in results.items():
        print(f"\n{name}:")
        result['flops'].print_breakdown(show_percentages=True)


if __name__ == "__main__":
    # GPT-2 XL configuration from the problem
    gpt2_xl = TransformerConfig(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
    )

    print("Problem (a): GPT-2 XL - Parameters and Memory")
    analyze_model(gpt2_xl, seq_len=1024)

    print("\n\nProblem (b) & (c): FLOP Analysis (context_length tokens)")
    # Already shown above

    print("\n\nProblem (d): Model Size Comparison")
    models = {
        "GPT-2 Small": TransformerConfig(
            vocab_size=50257,
            context_length=1024,
            num_layers=12,
            d_model=768,
            num_heads=12,
            d_ff=3072,
        ),
        "GPT-2 Medium": TransformerConfig(
            vocab_size=50257,
            context_length=1024,
            num_layers=24,
            d_model=1024,
            num_heads=16,
            d_ff=4096,
        ),
        "GPT-2 Large": TransformerConfig(
            vocab_size=50257,
            context_length=1024,
            num_layers=36,
            d_model=1280,
            num_heads=20,
            d_ff=5120,
        ),
        "GPT-2 XL": gpt2_xl,
    }
    compare_models(models, seq_len=1024)

    print("\n\nProblem (e): Context Length Impact")
    print(f"\nGPT-2 XL with context_length = 1024:")
    _, _, flops_1024 = analyze_model(gpt2_xl, seq_len=1024, verbose=False)

    print(f"\nGPT-2 XL with context_length = 16384:")
    _, _, flops_16384 = analyze_model(gpt2_xl, seq_len=16384, verbose=False)

    ratio = flops_16384.total / flops_1024.total
    print(f"\nFLOP ratio (16384 / 1024): {ratio:.2f}x")
    print(f"\nContext length ratio: {16384 / 1024:.2f}x")

    print("\nComponent-wise FLOP changes:")
    components = ['qkv_projections', 'attention_scores', 'attention_output',
                   'output_projection', 'ffn_w1', 'ffn_w2', 'ffn_w3', 'lm_head']
    for comp in components:
        flops_old = getattr(flops_1024, comp)
        flops_new = getattr(flops_16384, comp)
        if flops_old > 0:
            ratio = flops_new / flops_old
            print(f"  {comp:.<30} {ratio:>6.2f}x")
