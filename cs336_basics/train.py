"""
Training script for Transformer language model.

This script provides a complete training loop with:
- Configurable hyperparameters
- Memory-efficient data loading with np.memmap
- Checkpointing and resumption
- Training and validation logging
- Optional Weights & Biases integration
"""

import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path

from cs336_basics.transformer import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


def load_dataset_memmap(path: str, dtype=np.uint16):
    """
    Load a dataset using memory-mapped file for efficient memory usage.

    Args:
        path: Path to the dataset file
        dtype: Data type of the tokens

    Returns:
        Memory-mapped numpy array
    """
    return np.memmap(path, dtype=dtype, mode='r')


def estimate_loss(model, dataset, batch_size, context_length, device, num_batches=20):
    """
    Estimate the average loss over multiple batches from a dataset.

    Args:
        model: The language model
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation
        context_length: Context length
        device: Device to run on
        num_batches: Number of batches to average over

    Returns:
        Average loss
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)

            # Reshape for cross-entropy: (batch_size * context_length, vocab_size) and (batch_size * context_length,)
            batch_size_actual, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)

            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())

    model.train()
    return np.mean(losses)


def train(args):
    """
    Main training function.

    Args:
        args: Command-line arguments containing hyperparameters
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:.<30} {value}")
    print("=" * 80)

    # Load datasets using memmap for memory efficiency
    print(f"\nLoading datasets...")
    train_data = load_dataset_memmap(args.train_data_path, dtype=np.uint16)
    val_data = load_dataset_memmap(args.val_data_path, dtype=np.uint16) if args.val_data_path else None
    print(f"Train dataset size: {len(train_data):,} tokens")
    if val_data is not None:
        print(f"Validation dataset size: {len(val_data):,} tokens")

    # Initialize model
    print(f"\nInitializing model...")
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

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Initialize Weights & Biases if requested
    use_wandb = args.wandb_project is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            print(f"\nLogging to Weights & Biases: {args.wandb_project}")
        except ImportError:
            print("\nWarning: wandb not installed, disabling W&B logging")
            use_wandb = False

    # Create checkpoint directory
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    model.train()
    train_losses = []
    start_time = time.time()

    for iteration in range(start_iter, args.max_iters):
        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(inputs)

        # Compute loss
        # Reshape: (batch_size, context_length, vocab_size) -> (batch_size * context_length, vocab_size)
        batch_size_actual, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # Track loss
        train_losses.append(loss.item())

        # Logging
        if iteration % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:]) if train_losses else loss.item()
            elapsed = time.time() - start_time
            tokens_per_sec = (iteration - start_iter + 1) * args.batch_size * args.context_length / elapsed

            log_str = f"iter {iteration:6d} | loss {avg_loss:.4f} | lr {lr:.6f} | {tokens_per_sec:.0f} tok/s"
            print(log_str)

            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "iteration": iteration,
                })

        # Validation
        if val_data is not None and iteration % args.eval_interval == 0 and iteration > 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_batches)
            print(f"iter {iteration:6d} | val_loss {val_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "iteration": iteration,
                })

            model.train()

        # Save checkpoint
        if args.checkpoint_dir and iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    if args.checkpoint_dir:
        final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
        print(f"\nSaved final checkpoint to {final_checkpoint_path}")

    # Final validation
    if val_data is not None:
        final_val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_batches)
        print(f"\nFinal validation loss: {final_val_loss:.4f}")

        if use_wandb:
            wandb.log({
                "val/final_loss": final_val_loss,
            })

    if use_wandb:
        wandb.finish()

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data (memmap file)")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data (memmap file)")

    # Model arguments
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

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--max_iters", type=int, default=100000,
                        help="Maximum number of training iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")

    # Optimizer arguments
    parser.add_argument("--max_lr", type=float, default=6e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum learning rate")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay coefficient")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")

    # Learning rate schedule arguments
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000,
                        help="Number of cosine annealing iterations")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N iterations")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval_batches", type=int, default=20,
                        help="Number of batches for validation evaluation")

    # Checkpointing arguments
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Weights & Biases arguments
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name (None to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")

    args = parser.parse_args()

    # Set default d_ff if not provided
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    train(args)


if __name__ == "__main__":
    main()
