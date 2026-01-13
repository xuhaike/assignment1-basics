"""
AdamW optimizer implementation.
"""

import torch
from typing import Iterable


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Implements the AdamW algorithm as described in Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize the AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (α in the algorithm)
            betas: Tuple of (β1, β2) coefficients for computing running averages
            eps: Term added to denominator for numerical stability (ϵ)
            weight_decay: Weight decay coefficient (λ)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional callable to recompute the loss

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get or initialize state for this parameter
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    # First moment estimate (m)
                    state["m"] = torch.zeros_like(p.data)
                    # Second moment estimate (v)
                    state["v"] = torch.zeros_like(p.data)

                # Get state variables
                m = state["m"]
                v = state["v"]
                t = state["t"]

                # Increment timestep (note: t starts at 1 in the algorithm)
                t += 1
                state["t"] = t

                # Update biased first moment estimate: m ← β1*m + (1 − β1)*g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate: v ← β2*v + (1 − β2)*g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                # αt ← α * sqrt(1−(β2)^t) / (1−(β1)^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters: θ ← θ − αt * m / (sqrt(v) + ϵ)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay: θ ← θ − α*λ*θ
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
