"""
Enhanced training script with experiment logging infrastructure.

This version integrates the ExperimentLogger for comprehensive tracking.
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
from cs336_basics.experiment_logger import ExperimentLogger, ExperimentConfig, format_experiment_summary


def load_dataset_memmap(path: str, dtype=np.uint16):
    """Load a dataset using memory-mapped file."""
    return np.memmap(path, dtype=dtype, mode='r')


def estimate_loss(model, dataset, batch_size, context_length, device, num_batches=20):
    """Estimate average loss over multiple batches."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)

            batch_size_actual, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)

            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())

    model.train()
    return np.mean(losses)


def train(args):
    """Main training function with experiment logging."""
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        description=args.description,
        tags=args.tags.split(',') if args.tags else [],
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        seed=args.seed,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path or "",
        device=args.device,
        notes=args.notes,
    )

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Initialize experiment logger
    with ExperimentLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        config=config,
        use_wandb=args.wandb_project is not None,
        wandb_project=args.wandb_project,
    ) as logger:

        # Print experiment summary
        print("\n" + format_experiment_summary(config))
        logger.log_text(format_experiment_summary(config), "experiment_config.txt")

        # Load datasets
        print(f"Loading datasets...")
        train_data = load_dataset_memmap(args.train_data_path, dtype=np.uint16)
        val_data = load_dataset_memmap(args.val_data_path, dtype=np.uint16) if args.val_data_path else None
        print(f"Train dataset size: {len(train_data):,} tokens")
        if val_data is not None:
            print(f"Validation dataset size: {len(val_data):,} tokens")

        # Initialize model
        print(f"Initializing model...")
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

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        logger.log_text(f"Model parameters: {num_params:,}", "model_info.txt")

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
            print(f"Resuming from checkpoint: {args.resume_from}")
            start_iter = load_checkpoint(args.resume_from, model, optimizer)
            print(f"Resumed from iteration {start_iter}")
            logger.log_text(f"Resumed from iteration {start_iter}", "resume_info.txt")

        # Create checkpoint directory
        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Training loop
        print("\n" + "=" * 80)
        print("Starting training")
        print("=" * 80 + "\n")

        model.train()
        start_time = time.time()

        for iteration in range(start_iter, args.max_iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            iter_start_time = time.time()

            # Get learning rate
            lr = get_lr_cosine_schedule(
                it=iteration,
                max_learning_rate=args.max_lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Get batch
            inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

            # Forward pass
            logits = model(inputs)

            # Compute loss
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

            # Calculate metrics
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            iter_time = time.time() - iter_start_time
            wallclock_time = time.time() - start_time
            tokens_per_sec = (args.batch_size * args.context_length) / iter_time

            # Logging
            if iteration % args.log_interval == 0:
                metrics = {
                    'train/loss': loss.item(),
                    'train/learning_rate': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/iter_time': iter_time,
                }

                logger.log_metrics(metrics, step=iteration, wallclock_time=wallclock_time)

                log_str = (f"iter {iteration:6d} | "
                          f"loss {loss.item():.4f} | "
                          f"lr {lr:.6f} | "
                          f"{tokens_per_sec:.0f} tok/s | "
                          f"time {wallclock_time:.1f}s")
                print(log_str)

            # Validation
            if val_data is not None and iteration % args.eval_interval == 0 and iteration > 0:
                val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_batches)

                metrics = {
                    'val/loss': val_loss,
                }
                logger.log_metrics(metrics, step=iteration, wallclock_time=wallclock_time)

                print(f"iter {iteration:6d} | val_loss {val_loss:.4f}")
                model.train()

            # Save checkpoint
            if args.checkpoint_dir and iteration % args.checkpoint_interval == 0 and iteration > 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration:06d}.pt")
                save_checkpoint(model, optimizer, iteration, checkpoint_path)

                # Log checkpoint info
                checkpoint_metrics = {
                    'train/loss': loss.item(),
                }
                if val_data is not None:
                    # Quick validation
                    val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, 10)
                    checkpoint_metrics['val/loss'] = val_loss
                    model.train()

                logger.save_checkpoint_info(checkpoint_path, iteration, checkpoint_metrics)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Final checkpoint
        if args.checkpoint_dir:
            final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
            save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
            print(f"\nSaved final checkpoint to {final_checkpoint_path}")

        # Final validation
        if val_data is not None:
            final_val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_batches)
            final_wallclock = time.time() - start_time

            logger.log_metrics({'val/final_loss': final_val_loss}, step=args.max_iters, wallclock_time=final_wallclock)
            print(f"\nFinal validation loss: {final_val_loss:.4f}")

        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        logger.log_text(f"Training completed in {total_time/3600:.2f} hours", "completion.txt")


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM with experiment logging")

    # Experiment metadata
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of the experiment")
    parser.add_argument("--description", type=str, default="",
                        help="Description of the experiment")
    parser.add_argument("--tags", type=str, default="",
                        help="Comma-separated tags (e.g., 'baseline,17m,tinystories')")
    parser.add_argument("--notes", type=str, default="",
                        help="Additional notes about this experiment")
    parser.add_argument("--log_dir", type=str, default="experiments",
                        help="Directory for experiment logs")

    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data")

    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Optimizer arguments
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Schedule arguments
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=50000)

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)

    # Checkpointing arguments
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--resume_from", type=str, default=None)

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default=None)

    args = parser.parse_args()

    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    train(args)


if __name__ == "__main__":
    main()
