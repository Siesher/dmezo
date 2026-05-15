"""MeZO step: forward-pass-only gradient estimation and parameter update.

Implements the canonical MeZO step:

1. Sample seed s_t.
2. Perturb parameters by +eps * z_t (z_t generated from s_t).
3. Compute loss L+.
4. Perturb by -2*eps * z_t -> parameters now at theta - eps*z_t.
5. Compute loss L-.
6. Projected gradient: rho_t = (L+ - L-) / (2*eps).
7. Restore parameters by +eps * z_t.
8. Update: theta <- theta - lr * (rho_t * z_t + weight_decay * theta).

Reference: Malladi et al. 2023 (arXiv:2305.17333),
princeton-nlp/MeZO `large_models/trainer.py::zo_step` and `::zo_update`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dmezo.mezo.perturbation import perturb_parameters


@dataclass
class MeZOConfig:
    """Configuration for the MeZO optimizer.

    Attributes:
        lr: Learning rate. For MeZO this is typically 1e-6 to 1e-7 (much smaller
            than for first-order methods on the same model).
        eps: Perturbation magnitude epsilon. Default 1e-3 (Malladi 2023).
        weight_decay: L2 regularization coefficient. Applied only to non-bias
            and non-(layer-)norm parameters, matching Princeton reference.
        seed_rng: Optional seed for the seed-generator itself (controls the
            sequence of per-step seeds across the run).
        rho_clip: Optional symmetric clip threshold on the projected gradient
            ρ_t. If set, the returned ρ_t is clipped to ``[-rho_clip, +rho_clip]``
            before further processing (downstream update + Nesterov velocity).
            Motivation: variance-reduction rescue for Nesterov-MeZO — Day 6
            heavy-ball β=0.9 diverged after a ρ spike to +905; clipping bounds
            the worst-case noise contribution to the velocity buffer per round.
            Must be positive when set; ``None`` (default) disables clipping.
    """

    lr: float = 1e-6
    eps: float = 1e-3
    weight_decay: float = 0.0
    seed_rng: int | None = None
    rho_clip: float | None = None


def _collect_optim_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Collect (name, param) for all parameters with requires_grad=True."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _is_decay_param(name: str) -> bool:
    """Heuristic from Princeton MeZO: apply weight_decay only to non-bias, non-norm params."""
    lname = name.lower()
    return ("bias" not in lname) and ("layer_norm" not in lname) and ("layernorm" not in lname)


@torch.inference_mode()
def _forward_loss(
    model: nn.Module,
    inputs: dict,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
) -> float:
    """Compute scalar loss with autograd disabled, dropout off."""
    was_training = model.training
    model.eval()
    try:
        loss = loss_fn(model, inputs)
        return float(loss.detach().item())
    finally:
        if was_training:
            model.train()


def mezo_step(
    model: nn.Module,
    inputs: dict,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: MeZOConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[int, float, float]:
    """Run one MeZO gradient-estimation step (no parameter update yet).

    Args:
        model: HF model (or any nn.Module). Must have parameters with requires_grad.
        inputs: Input batch (dict of tensors) to pass to ``loss_fn``.
        loss_fn: Callable ``(model, inputs) -> torch.Tensor`` returning scalar loss.
        config: MeZO hyperparameters.
        rng: Optional numpy Generator for picking the per-step seed. If None,
            uses np.random.

    Returns:
        Tuple ``(seed, projected_grad, loss_plus)``:
            - seed: int seed used for z_t this step. PASS THIS TO ``mezo_update``.
            - projected_grad: estimated <grad, z_t>.
            - loss_plus: loss at theta + eps*z (for logging).

    Note:
        After this call, parameters are RESTORED to their original state.
        The caller is responsible for invoking ``mezo_update`` with the returned
        seed and projected_grad to actually move theta.
    """
    if config.rho_clip is not None and config.rho_clip <= 0:
        raise ValueError(
            f"rho_clip must be positive or None; got {config.rho_clip}. "
            "Use None to disable clipping."
        )
    if rng is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    else:
        seed = int(rng.integers(0, 2**31 - 1))

    named = _collect_optim_params(model)

    # +eps * z
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=config.eps)
    loss_plus = _forward_loss(model, inputs, loss_fn)

    # -2*eps * z  (net: parameters at theta - eps*z)
    perturb_parameters(named, seed=seed, scaling_factor=-2.0, eps=config.eps)
    loss_minus = _forward_loss(model, inputs, loss_fn)

    # +eps * z  (restore)
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=config.eps)

    projected_grad = (loss_plus - loss_minus) / (2.0 * config.eps)
    if config.rho_clip is not None:
        # Symmetric clipping. Preserves sign; caps |ρ| at the threshold.
        projected_grad = max(-config.rho_clip, min(config.rho_clip, projected_grad))
    return seed, projected_grad, loss_plus


def mezo_update(
    model: nn.Module,
    seed: int,
    projected_grad: float,
    config: MeZOConfig,
) -> None:
    """Apply the MeZO update using a previously computed (seed, projected_grad).

    Updates each parameter as::

        param <- param - lr * (projected_grad * z + weight_decay * param)

    where ``z`` is regenerated from ``seed``. Weight decay only applies to
    non-bias / non-(layer-)norm parameters.

    Args:
        model: Model whose parameters to update.
        seed: Seed returned by ``mezo_step`` for this step.
        projected_grad: Projected gradient returned by ``mezo_step``.
        config: Optimizer configuration (lr, weight_decay).
    """
    torch.manual_seed(seed)
    for name, param in _collect_optim_params(model):
        z = torch.normal(
            mean=0.0,
            std=1.0,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
        )
        if _is_decay_param(name):
            param.data.add_(projected_grad * z + config.weight_decay * param.data, alpha=-config.lr)
        else:
            param.data.add_(projected_grad * z, alpha=-config.lr)
