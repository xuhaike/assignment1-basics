"""
Simple test to verify the training loop works end-to-end.

This creates a small synthetic dataset and trains a tiny model for a few iterations.
"""

import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from cs336_basics.transformer import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


def test_training_loop():
    """
    Test the training loop with a tiny model and synthetic data.
    """
    print("Creating synthetic dataset...")

    # Create synthetic dataset (random token IDs)
    vocab_size = 1000
    dataset_size = 10000
    dataset = np.random.randint(0, vocab_size, size=dataset_size, dtype=np.uint16)

    # Create temporary file for memmap
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
        tmp_path = tmp_file.name

    # Save as memmap
    memmap_data = np.memmap(tmp_path, dtype=np.uint16, mode='w+', shape=dataset.shape)
    memmap_data[:] = dataset[:]
    memmap_data.flush()

    print("Initializing model...")

    # Tiny model configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    print("Training for 10 iterations...")

    # Load data from memmap
    train_data = np.memmap(tmp_path, dtype=np.uint16, mode='r')

    model.train()
    batch_size = 4
    context_length = 64

    for iteration in range(10):
        # Learning rate schedule
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=5,
            cosine_cycle_iters=10,
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = get_batch(train_data, batch_size, context_length, device)

        # Forward pass
        logits = model(inputs)

        # Compute loss
        batch_size_actual, seq_len, vocab_size_actual = logits.shape
        logits_flat = logits.view(-1, vocab_size_actual)
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=1.0)

        # Optimizer step
        optimizer.step()

        print(f"Iteration {iteration}: loss = {loss.item():.4f}, lr = {lr:.6f}")

    print("\nTesting checkpointing...")

    # Test checkpoint save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = os.path.join(tmp_dir, "test_checkpoint.pt")

        # Save checkpoint
        save_checkpoint(model, optimizer, iteration=10, out=checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Get model state before loading
        old_param = next(model.parameters()).clone()

        # Create new model and optimizer
        new_model = TransformerLM(
            vocab_size=vocab_size,
            context_length=64,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            rope_theta=10000.0,
            device=device,
            dtype=torch.float32,
        )
        new_optimizer = AdamW(new_model.parameters(), lr=1e-3)

        # Load checkpoint
        loaded_iter = load_checkpoint(checkpoint_path, new_model, new_optimizer)
        print(f"Loaded checkpoint from iteration {loaded_iter}")

        # Verify parameters match
        new_param = next(new_model.parameters())
        assert torch.allclose(old_param, new_param), "Checkpoint loading failed!"
        print("Checkpoint parameters match!")

    # Cleanup
    os.unlink(tmp_path)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_training_loop()
