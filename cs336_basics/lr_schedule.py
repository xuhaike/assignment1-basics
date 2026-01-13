"""
Learning rate scheduling functions.
"""

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute the learning rate at iteration `it` using a cosine schedule with warmup.

    The schedule has three phases:
    1. Warmup (t < Tw): Linear warmup from 0 to max_learning_rate
    2. Cosine annealing (Tw ≤ t ≤ Tc): Cosine decay from max to min learning rate
    3. Post-annealing (t > Tc): Constant at min_learning_rate

    Args:
        it: Current iteration number (t)
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine annealing iterations (T_c)

    Returns:
        Learning rate at iteration `it`
    """
    # Phase 1: Warmup
    if it < warmup_iters:
        # Linear warmup: αt = (t / Tw) * αmax
        return (it / warmup_iters) * max_learning_rate

    # Phase 2: Cosine annealing
    elif it <= cosine_cycle_iters:
        # Cosine decay: αt = αmin + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * π)) * (αmax - αmin)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    # Phase 3: Post-annealing
    else:
        # Constant at minimum: αt = αmin
        return min_learning_rate
