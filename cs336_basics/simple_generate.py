"""
Simplified text generation script that automatically loads config from experiment directory.

This script makes it easier to generate text from a trained model by:
1. Automatically loading the model config from the experiment directory
2. Supporting both checkpoint path and automatic latest checkpoint selection
3. Providing simple command-line interface

Usage:
    python simple_generate.py --experiment experiments/baseline_17m_20260111_231919 --prompt "Once upon a time"
    python simple_generate.py --checkpoint checkpoints/baseline_17m/checkpoint_010000.pt --config experiments/baseline_17m_20260111_231919/config.json --prompt "Once upon a time"
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional


def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_tokenizer(config: dict):
    """
    Load tokenizer from config if available.

    Returns tokenizer object or None if tokenizer files not available.
    """
    import pickle
    from cs336_basics.tokenizer import Tokenizer

    vocab_path = config.get('tokenizer_vocab_path', '')
    merges_path = config.get('tokenizer_merges_path', '')

    if not vocab_path or not merges_path:
        return None

    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)

    if not vocab_path.exists() or not merges_path.exists():
        return None

    try:
        # Load tokenizer from pickle files
        if vocab_path.suffix == '.json' or merges_path.suffix == '.txt':
            tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path))
        else:
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            with open(merges_path, 'rb') as f:
                merges = pickle.load(f)
            tokenizer = Tokenizer(vocab, merges, [])

        return tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}")
        return None


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = sorted(checkpoint_path.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    return str(checkpoints[-1])


def load_model(checkpoint_path: str, config: dict, device: str = "cuda"):
    """Load model from checkpoint using config."""
    from cs336_basics.transformer import TransformerLM

    # Create model
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config.get('rope_theta', 10000.0),
        device=device,
        dtype=torch.float32,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint.get('iteration', 'unknown')

@torch.no_grad()
def generate_text(
    model,
    prompt_ids: list[int],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda",
) -> list[int]:
    """
    Generate text tokens from a prompt.

    Args:
        model: The language model
        prompt_ids: List of token IDs for the prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        List of generated token IDs
    """
    generated = list(prompt_ids)

    for _ in range(max_tokens):
        # Prepare input (only use last context_length tokens)
        context = generated[-model.context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        # Get model predictions
        logits = model(input_ids)  # Shape: (1, seq_len, vocab_size)

        # Get logits for the last position
        next_token_logits = logits[0, -1, :] / temperature

        # Apply top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[0] = False

        # Set logits to -inf for removed tokens
        next_token_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

        # Sample from the filtered distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        # Check for end of sequence (token 0 or specific EOS token)
        if next_token == 0:
            break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=str,
                      help="Path to experiment directory (will auto-load config and latest checkpoint)")
    group.add_argument("--checkpoint", type=str,
                      help="Path to specific checkpoint file")

    parser.add_argument("--config", type=str,
                       help="Path to config JSON file (required if using --checkpoint)")
    parser.add_argument("--checkpoint_dir", type=str,
                       help="Directory containing checkpoints (optional, for auto-selecting latest)")

    # Tokenizer options
    parser.add_argument("--tokenizer_vocab", type=str, default=None,
                       help="Path to tokenizer vocabulary file (overrides config)")
    parser.add_argument("--tokenizer_merges", type=str, default=None,
                       help="Path to tokenizer merges file (overrides config)")

    # Generation options
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda or cpu)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode (prompt repeatedly)")

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load config and checkpoint
    if args.experiment:
        experiment_dir = Path(args.experiment)
        config_path = experiment_dir / "config.json"

        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            return

        config = load_config(str(config_path))

        # Override tokenizer paths if provided via command line
        if args.tokenizer_vocab:
            config['tokenizer_vocab_path'] = args.tokenizer_vocab
        if args.tokenizer_merges:
            config['tokenizer_merges_path'] = args.tokenizer_merges

        # Find checkpoint directory from experiment name
        experiment_name = config.get('experiment_name', experiment_dir.name.split('_')[0])
        checkpoint_dir = Path("checkpoints") / experiment_name

        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
            return

        checkpoint_path = find_latest_checkpoint(str(checkpoint_dir))
        if not checkpoint_path:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            return

        print(f"Using checkpoint: {checkpoint_path}")
    else:
        if not args.config:
            print("Error: --config is required when using --checkpoint")
            return

        config = load_config(args.config)

        # Override tokenizer paths if provided via command line
        if args.tokenizer_vocab:
            config['tokenizer_vocab_path'] = args.tokenizer_vocab
        if args.tokenizer_merges:
            config['tokenizer_merges_path'] = args.tokenizer_merges

        checkpoint_path = args.checkpoint

    print("=" * 80)
    print(f"Experiment: {config.get('experiment_name', 'Unknown')}")
    print(f"Description: {config.get('description', 'N/A')}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, iteration = load_model(checkpoint_path, config, device)

    print(f"✓ Model loaded successfully!")
    print(f"  Checkpoint iteration: {iteration}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Context length: {config['context_length']}")
    print()

    # Try to load tokenizer from config
    tokenizer = load_tokenizer(config)
    use_real_tokenizer = tokenizer is not None

    if use_real_tokenizer:
        print("✓ Loaded tokenizer from config")
        print(f"  Vocab path: {config.get('tokenizer_vocab_path', 'N/A')}")
        print(f"  Merges path: {config.get('tokenizer_merges_path', 'N/A')}")
    else:
        print("⚠ WARNING: Using simple demo tokenizer.")
        print("   No tokenizer files found in config or files don't exist.")
        print("   For proper text generation, ensure tokenizer paths are in config.")
        print("   The generated text will be gibberish without the correct tokenizer.")
    print()

    # Generation loop
    if args.interactive:
        print("=" * 80)
        print("Interactive Mode (Ctrl+C to exit)")
        print("=" * 80)

        while True:
            try:
                prompt = input("\n🤖 Prompt: ")
                if not prompt:
                    continue

                # Tokenize prompt
                prompt_ids = tokenizer.encode(prompt)

                print(f"   (Tokenized to {len(prompt_ids)} tokens)")

                # Generate
                print("   Generating...", end='', flush=True)
                generated_ids = generate_text(
                    model=model,
                    prompt_ids=prompt_ids,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
                print(" Done!")

                # Detokenize
                if use_real_tokenizer:
                    generated_text = tokenizer.decode(generated_ids)
                else:
                    generated_text = simple_detokenize(generated_ids)

                print()
                print("-" * 80)
                print(generated_text)
                print("-" * 80)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
    else:
        # Single generation
        print(f"💭 Prompt: {args.prompt}")

        # Tokenize prompt
        prompt_ids = tokenizer.encode(args.prompt)
        print(f"   (Tokenized to {len(prompt_ids)} tokens)")

        # Generate
        print("   Generating...", end='', flush=True)
        generated_ids = generate_text(
            model=model,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(" Done!")

        # Detokenize
        generated_text = tokenizer.decode(generated_ids)

        print()
        print("=" * 80)
        print("📝 Generated Text:")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)
        print()
        print(f"Generation settings:")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Tokens generated: {len(generated_ids) - len(prompt_ids)}")
        if use_real_tokenizer:
            print(f"  Using: Real BPE tokenizer")
        else:
            print(f"  Using: Demo character tokenizer (results will be gibberish)")


if __name__ == "__main__":
    main()
