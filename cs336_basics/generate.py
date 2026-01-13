"""
Text generation CLI script.

Generate text completions from a trained language model.
"""

import argparse
import pickle
import torch
from pathlib import Path

from cs336_basics.transformer import TransformerLM
from cs336_basics.decoding import decode_text


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str] = None):
    """Load tokenizer from pickle files."""
    from cs336_basics.tokenizer import Tokenizer

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)

    return Tokenizer(vocab, merges, special_tokens or [])


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load a model from a checkpoint.

    Note: You need to provide model configuration separately or store it in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If checkpoint contains config, use it
    if 'config' in checkpoint:
        config = checkpoint['config']
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
    else:
        raise ValueError(
            "Checkpoint does not contain model config. "
            "Please provide model architecture parameters."
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def generate_cli(args):
    """Main generation function."""
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=args.special_tokens,
    )

    print("Loading model...")
    # Create model with provided config
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print()

    # Interactive or single prompt mode
    if args.interactive:
        print("=" * 80)
        print("Interactive mode (Ctrl+C to exit)")
        print("=" * 80)
        while True:
            try:
                prompt = input("\nPrompt: ")
                if not prompt:
                    continue

                print("Generating...")
                generated_text = decode_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token=args.eos_token,
                    device=device,
                )

                print("-" * 80)
                print(generated_text)
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Single prompt mode
        if not args.prompt:
            print("Error: Please provide --prompt or use --interactive mode")
            return

        print(f"Prompt: {args.prompt}")
        print("Generating...")
        generated_text = decode_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token=args.eos_token,
            device=device,
        )

        print()
        print("=" * 80)
        print(generated_text)
        print("=" * 80)

    # Show generation parameters
    if not args.interactive:
        print()
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Max tokens: {args.max_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained language model")

    # Model files
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to tokenizer vocabulary (pickle)")
    parser.add_argument("--merges", type=str, required=True,
                        help="Path to tokenizer merges (pickle)")

    # Model architecture (needed to reconstruct model)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="Feed-forward dimension (default: 4 * d_model)")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling threshold")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>",
                        help="End-of-sequence token")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=None,
                        help="Special tokens")

    # Mode
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode (prompt repeatedly)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")

    args = parser.parse_args()

    # Set default d_ff
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    generate_cli(args)


if __name__ == "__main__":
    main()
