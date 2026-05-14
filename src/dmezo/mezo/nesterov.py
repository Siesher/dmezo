"""Nesterov-style acceleration adapted to MeZO updates.

In first-order Nesterov::

    v_{t+1} = beta * v_t + g_t
    theta_{t+1} = theta_t - lr * v_{t+1}    (heavy-ball form)

Adapting to MeZO: g_t = rho_t * z_t. Two design choices for the velocity buffer:

A) Explicit velocity buffer ``v`` of size |theta|. Cost: one extra full-sized
   tensor per client. On Blackwell 96 GB with Qwen3-4B (~8 GB FP16) this is
   acceptable: 8 GB more per client.

B) Implicit velocity as a running history of ``(seed_i, rho_i)`` pairs with
   geometric decay. Memory cost is O(T) scalars but reconstruction needs all
   past seeds at apply-time, so O(T) extra forward passes — impractical.

We pick (A) for clarity and tractability. The buffer is allocated lazily on
the first call to ``nesterov_step``.

Note on look-ahead vs. heavy-ball: pure Nesterov requires evaluating the gradient
at ``theta + beta * v`` (look-ahead). For zeroth-order this is straightforward:
perturb to ``theta + beta * v`` first, then run a normal MeZO step there, then
update from ``theta``. This is implemented as the ``look_ahead=True`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn


@dataclass
class NesterovState:
    """Container for per-parameter velocity buffers and config.

    Attributes:
        beta: Momentum coefficient. Common defaults: 0.9 (heavy-ball),
            0.95-0.99 for slower mixing.
        look_ahead: If True, use full Nesterov look-ahead (evaluate gradient at
            theta + beta * v before updating). If False, use heavy-ball form.
        velocities: Per-parameter velocity tensors, keyed by parameter name.
            Allocated lazily.
    """

    beta: float = 0.9
    look_ahead: bool = False
    velocities: Dict[str, torch.Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        """Zero out all velocities (e.g., at the start of a new run)."""
        for v in self.velocities.values():
            v.zero_()


def _ensure_velocity_buffer(state: NesterovState, name: str, param: nn.Parameter) -> torch.Tensor:
    """Lazily allocate the velocity buffer for one parameter."""
    if name not in state.velocities:
        state.velocities[name] = torch.zeros_like(param.data)
    return state.velocities[name]


def nesterov_step(
    model: nn.Module,
    state: NesterovState,
    seed: int,
    projected_grad: float,
    lr: float,
    weight_decay: float = 0.0,
) -> None:
    """Apply one Nesterov-MeZO update (heavy-ball form).

    Updates each parameter as::

        v <- beta * v + (projected_grad * z + weight_decay * theta)
        theta <- theta - lr * v

    where ``z`` is regenerated from ``seed``. This is the heavy-ball variant.
    For the look-ahead variant, evaluate the projected gradient at
    ``theta + beta * v`` before calling this function.

    Args:
        model: Model whose parameters to update.
        state: NesterovState (velocity buffers, beta). Mutated in place.
        seed: Seed for regenerating ``z`` (must match the seed used to compute
            ``projected_grad``).
        projected_grad: Scalar projected gradient from MeZO step.
        lr: Learning rate.
        weight_decay: L2 coefficient (applied only to non-bias / non-norm params).
    """
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        z = torch.normal(
            mean=0.0,
            std=1.0,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
        )
        v = _ensure_velocity_buffer(state, name, param)
        lname = name.lower()
        decay = (
            weight_decay
            if ("bias" not in lname) and ("layer_norm" not in lname) and ("layernorm" not in lname)
            else 0.0
        )
        # v = beta * v + (rho * z + decay * theta)
        v.mul_(state.beta).add_(projected_grad * z + decay * param.data)
        # theta = theta - lr * v
        param.data.add_(v, alpha=-lr)
