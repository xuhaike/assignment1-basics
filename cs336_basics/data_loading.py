"""
Data loading utilities for language modeling.
"""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from a dataset.

    For each example, we sample a random starting position and extract:
    - Input: tokens at positions [start : start + context_length]
    - Target: tokens at positions [start + 1 : start + context_length + 1]

    This creates the standard language modeling setup where each input token
    predicts the next token.

    Args:
        dataset: 1D numpy array of integer token IDs in the dataset
        batch_size: Number of sequences to sample
        context_length: Length of each sampled sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')

    Returns:
        Tuple of two torch.LongTensors, each of shape (batch_size, context_length):
        - inputs: Sampled input sequences
        - targets: Corresponding next-token prediction targets
    """
    # Randomly sample starting positions
    # We need context_length + 1 tokens total (context_length inputs + 1 target)
    # So valid starting positions are from 0 to len(dataset) - context_length - 1
    max_start_idx = len(dataset) - context_length - 1

    # Sample batch_size random starting positions
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Extract sequences
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        # Input: tokens at [start_idx : start_idx + context_length]
        inputs[i] = dataset[start_idx : start_idx + context_length]

        # Target: tokens at [start_idx + 1 : start_idx + context_length + 1]
        # This is shifted by 1, so each input token predicts the next token
        targets[i] = dataset[start_idx + 1 : start_idx + context_length + 1]

    # Convert to PyTorch tensors and move to the specified device
    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)

    return inputs_tensor, targets_tensor
