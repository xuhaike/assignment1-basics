"""
Model checkpointing utilities for saving and loading training state.
"""

import os
import torch
from typing import BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer, and iteration state to a checkpoint file.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: The current training iteration number
        out: Path or file-like object to save the checkpoint to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        src: Path or file-like object to load the checkpoint from
        model: The model to restore state into
        optimizer: The optimizer to restore state into

    Returns:
        The iteration number saved in the checkpoint
    """
    checkpoint = torch.load(src, weights_only=False)

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the iteration number
    return checkpoint['iteration']
