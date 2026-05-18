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
        k_directions: Number of independent SPSA directions averaged per step
            (Multi-Direction MeZO). Default 1 = standard MeZO. With ``K > 1``
            we compute ``K`` independent ρ_k via fresh seeds s_k and average:
            ``g_tilde = (1/K) Σ_k ρ_k z_k``. This reduces the variance of the
            gradient estimator by factor K at the cost of ``K``× more forward
            passes per step. Per-direction ρ_k still respects ``rho_clip`` if
            set. Communication scales as O(K) per neighbour in federated mode.
            Reference: Spall 1992 (multi-direction SPSA); Malladi 2023 §G
            (mentions K-direction as an open extension).
    """

    lr: float = 1e-6
    eps: float = 1e-3
    weight_decay: float = 0.0
    seed_rng: int | None = None
    rho_clip: float | None = None
    k_directions: int = 1


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


# ---------------------------------------------------------------------------
# Multi-Direction MeZO (MD-MeZO): K-direction SPSA averaging
# ---------------------------------------------------------------------------


def md_mezo_step(
    model: nn.Module,
    inputs: dict,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: MeZOConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], list[float], float]:
    """Multi-direction MeZO gradient estimation: K independent SPSA probes.

    Performs ``K = config.k_directions`` independent MeZO sub-steps with fresh
    seeds, returning all per-direction (seed, ρ) pairs. The averaged
    "estimated gradient" is implicit:

    .. math::

        \\tilde g_t = \\frac{1}{K} \\sum_{k=1}^{K} \\hat\\rho_k z_{s_k}

    Variance scales as :math:`O(1/K)` vs. single-direction MeZO at the cost
    of ``2K`` forward passes per step (vs. 2 for ``K=1``). Per-direction ρ
    still respects ``config.rho_clip`` if set.

    Args:
        model: HF model (or any nn.Module). Must have parameters with
            ``requires_grad`` set.
        inputs: Input batch (dict of tensors) for ``loss_fn``.
        loss_fn: Callable ``(model, inputs) -> torch.Tensor`` returning scalar loss.
        config: MeZO hyperparameters. Uses ``config.k_directions`` for K
            (default 1, which makes this function equivalent to ``mezo_step``).
        rng: Optional numpy Generator for picking seeds. If None, uses
            ``np.random``.

    Returns:
        Tuple ``(seeds, projected_grads, mean_loss_plus)``:
            - seeds: list of length K — per-direction seeds. Pass to
              ``md_mezo_update``.
            - projected_grads: list of length K — per-direction ρ_k values
              (clipped if ``config.rho_clip`` set).
            - mean_loss_plus: mean of per-direction loss_plus (for logging).

    Notes:
        - ``K=1`` produces a single-element list and is regression-identical to
          ``mezo_step`` (verified by tests).
        - After this call, model parameters are RESTORED to their original state
          (each sub-step does its own +εz / -εz / +εz cycle independently).
    """
    K = max(1, int(config.k_directions))
    if config.rho_clip is not None and config.rho_clip <= 0:
        raise ValueError(
            f"rho_clip must be positive or None; got {config.rho_clip}. "
            "Use None to disable clipping."
        )
    if rng is None:
        # Match mezo_step behaviour: use np.random directly for seed generation
        # so K=1 produces the same seed sequence as a single mezo_step call
        # given the same global numpy RNG state.
        seeds = [int(np.random.randint(0, 2**31 - 1)) for _ in range(K)]
    else:
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(K)]

    named = _collect_optim_params(model)
    projected_grads: list[float] = []
    loss_pluses: list[float] = []

    for seed in seeds:
        # +eps * z_k
        perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=config.eps)
        loss_plus = _forward_loss(model, inputs, loss_fn)
        # -2*eps * z_k  (net: parameters at theta - eps*z_k)
        perturb_parameters(named, seed=seed, scaling_factor=-2.0, eps=config.eps)
        loss_minus = _forward_loss(model, inputs, loss_fn)
        # +eps * z_k  (restore)
        perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=config.eps)

        rho_k = (loss_plus - loss_minus) / (2.0 * config.eps)
        if config.rho_clip is not None:
            rho_k = max(-config.rho_clip, min(config.rho_clip, rho_k))
        projected_grads.append(rho_k)
        loss_pluses.append(loss_plus)

    mean_loss_plus = sum(loss_pluses) / len(loss_pluses) if loss_pluses else 0.0
    return seeds, projected_grads, mean_loss_plus


def md_mezo_update(
    model: nn.Module,
    seeds: list[int],
    projected_grads: list[float],
    config: MeZOConfig,
) -> None:
    """Apply the multi-direction MeZO update (1/K-averaged contributions).

    Updates each parameter as::

        param <- param - lr * ((1/K) Σ_k ρ_k z_{s_k} + weight_decay * param)

    Implementation: iterate ``K`` times, each time regenerate ``z_k`` from
    ``seeds[k]`` and subtract ``(lr/K) * ρ_k * z_k`` from each parameter.
    Weight decay is applied **once** per call (not K times) to match the
    semantics of standard MeZO update (which applies decay once per step).

    Args:
        model: Model whose parameters to update.
        seeds: Length-K seed list returned by ``md_mezo_step``.
        projected_grads: Length-K ρ values returned by ``md_mezo_step``.
        config: Optimizer configuration (lr, weight_decay).

    Raises:
        ValueError: If ``len(seeds) != len(projected_grads)`` or list is empty.
    """
    if len(seeds) != len(projected_grads):
        raise ValueError(
            f"seeds and projected_grads must have same length; "
            f"got {len(seeds)} vs {len(projected_grads)}."
        )
    K = len(seeds)
    if K == 0:
        raise ValueError("md_mezo_update requires at least one direction.")

    inv_K = 1.0 / K
    named = _collect_optim_params(model)

    # First apply weight decay once (matches single-direction semantics).
    if config.weight_decay > 0.0:
        for name, param in named:
            if _is_decay_param(name):
                param.data.add_(param.data, alpha=-config.lr * config.weight_decay)

    # Then accumulate (1/K) * Σ_k ρ_k z_k contributions.
    for seed, rho_k in zip(seeds, projected_grads):
        torch.manual_seed(seed)
        for _, param in named:
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data.add_(z, alpha=-config.lr * rho_k * inv_K)
