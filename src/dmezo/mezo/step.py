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

import math
from collections.abc import Callable
from dataclasses import dataclass, field

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
        lr_schedule: Learning-rate schedule mode. Default ``"constant"`` (the
            Princeton MeZO convention — yields convergence to noise neighborhood
            of radius :math:`O(G^2/\\mu)` per Theorem 3, not to true optimum).
            Set to ``"harmonic"`` for the SPSA classical schedule
            :math:`\\eta_t = \\eta_0 \\cdot a / (t+1)^{\\alpha}` (Spall 1992,
            convergence to :math:`\\theta^*`) or ``"cosine"`` for cosine annealing
            :math:`\\eta_t = \\eta_0 \\cdot 0.5(1 + \\cos(\\pi t/T))`. The
            effective lr is computed by :func:`effective_lr`; callers pass
            ``round_idx`` and ``num_rounds`` at use time.
        lr_decay_a: Numerator parameter ``a`` for harmonic decay (Spall 1992
            §3.2 notation). Ignored for non-harmonic schedules.
        lr_decay_alpha: Exponent ``α`` for harmonic decay. Spall's classical
            choice is 0.602 (gives near-optimal rate under PL).
    """

    lr: float = 1e-6
    eps: float = 1e-3
    weight_decay: float = 0.0
    seed_rng: int | None = None
    rho_clip: float | None = None
    k_directions: int = 1
    lr_schedule: str = "constant"
    lr_decay_a: float = 1.0
    lr_decay_alpha: float = 0.602
    # B1: adaptive ρ-clipping (data-driven threshold from running quantile).
    # When ``adaptive_clip_window > 0`` AND ``adaptive_clip_quantile in (0, 1)``,
    # callers should maintain an ``AdaptiveClipState`` and use its current
    # threshold via ``rho_clip_override`` instead of ``rho_clip`` below.
    # The ``MeZOConfig`` fields just record the policy parameters; the running
    # state lives in :class:`AdaptiveClipState` (declared below).
    adaptive_clip_window: int = 0
    adaptive_clip_quantile: float = 0.95
    adaptive_clip_alpha: float = 1.3
    # D2: differential-privacy Gaussian noise on the projected gradient.
    # When ``dp_sigma > 0``, ``mezo_step`` adds N(0, σ²) noise to ρ AFTER
    # clipping. Combined with ρ-clip ``C``, this provides Gaussian-mechanism
    # (ε, δ)-DP with sensitivity Δ = C (since per-record contribution to the
    # output ρ is bounded by the clip). See Dwork-Roth 2014 §3.5.3.
    # ``None`` or 0 disables DP noise.
    dp_sigma: float | None = None


def effective_lr(config: MeZOConfig, round_idx: int, num_rounds: int) -> float:
    """Compute the effective learning rate at a given round under the schedule.

    Schedules:
        - ``"constant"``: returns ``config.lr`` unchanged.
        - ``"harmonic"``: ``config.lr * lr_decay_a / (round_idx + 1) ** lr_decay_alpha``
          (Spall 1992 §3.2; ``α = 0.602`` is the classical choice).
        - ``"cosine"``: ``config.lr * 0.5 * (1 + cos(π * progress))`` where
          ``progress = round_idx / max(num_rounds - 1, 1)``. Reaches 0 at the
          final round.

    Args:
        config: Optimizer configuration.
        round_idx: 0-indexed current round / step (clamped to ≥ 0).
        num_rounds: Total horizon (used by cosine schedule). Must be ≥ 1.

    Returns:
        Effective learning rate (always positive for ``round_idx ≥ 0``).

    Raises:
        ValueError: If ``lr_schedule`` is unknown.
    """
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be ≥ 1, got {num_rounds}")
    t = max(0, int(round_idx))
    if config.lr_schedule == "constant":
        return float(config.lr)
    if config.lr_schedule == "harmonic":
        # Spall 1992 convention. Denominator (t+1) so first step uses η_0 * a / 1.
        return float(config.lr * config.lr_decay_a / ((t + 1) ** config.lr_decay_alpha))
    if config.lr_schedule == "cosine":
        T = max(int(num_rounds) - 1, 1)
        progress = min(t / T, 1.0)
        return float(config.lr * 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(
        f"Unknown lr_schedule {config.lr_schedule!r}. "
        "Supported: 'constant', 'harmonic', 'cosine'."
    )


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
    rho_clip_override: float | None = None,
    dp_sigma_override: float | None = None,
) -> tuple[int, float, float]:
    """Run one MeZO gradient-estimation step (no parameter update yet).

    Args:
        model: HF model (or any nn.Module). Must have parameters with requires_grad.
        inputs: Input batch (dict of tensors) to pass to ``loss_fn``.
        loss_fn: Callable ``(model, inputs) -> torch.Tensor`` returning scalar loss.
        config: MeZO hyperparameters.
        rng: Optional numpy Generator for picking the per-step seed. If None,
            uses np.random.
        rho_clip_override: B1 adaptive-clip — when not None, supersedes
            ``config.rho_clip`` for this step. Used by callers maintaining an
            :class:`AdaptiveClipState` to inject the data-driven threshold.
            Must be positive when set.
        dp_sigma_override: D2 differential-privacy — when not None, supersedes
            ``config.dp_sigma``. Adds ``N(0, σ²)`` noise to ρ after clipping.
            Set to 0 to disable for this step even if config has σ > 0.

    Returns:
        Tuple ``(seed, projected_grad, loss_plus)``:
            - seed: int seed used for z_t this step. PASS THIS TO ``mezo_update``.
            - projected_grad: estimated <grad, z_t>, after optional clip + DP.
            - loss_plus: loss at theta + eps*z (for logging).

    Note:
        After this call, parameters are RESTORED to their original state.
        The caller is responsible for invoking ``mezo_update`` with the returned
        seed and projected_grad to actually move theta.

        Adaptive clip ordering: clip is applied BEFORE DP noise (matches the
        Gaussian-mechanism convention — noise is added to the clipped output).
    """
    if config.rho_clip is not None and config.rho_clip <= 0:
        raise ValueError(
            f"rho_clip must be positive or None; got {config.rho_clip}. "
            "Use None to disable clipping."
        )
    if rho_clip_override is not None and rho_clip_override <= 0:
        raise ValueError(
            f"rho_clip_override must be positive or None; got {rho_clip_override}."
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
    # Effective clip: override > config.
    effective_clip = (
        rho_clip_override if rho_clip_override is not None else config.rho_clip
    )
    if effective_clip is not None:
        # Symmetric clipping. Preserves sign; caps |ρ| at the threshold.
        projected_grad = max(-effective_clip, min(effective_clip, projected_grad))
    # DP noise (post-clip, per Gaussian-mechanism convention).
    effective_dp = (
        dp_sigma_override if dp_sigma_override is not None else config.dp_sigma
    )
    if effective_dp is not None and effective_dp > 0:
        # Use the same numpy RNG so DP noise is reproducible per (seed, run).
        if rng is None:
            noise = float(np.random.normal(0.0, float(effective_dp)))
        else:
            noise = float(rng.normal(0.0, float(effective_dp)))
        projected_grad = projected_grad + noise
    return seed, projected_grad, loss_plus


def mezo_step_richardson(
    model: nn.Module,
    inputs: dict,
    loss_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    config: MeZOConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[int, float, float]:
    """Richardson-extrapolation 4-point central-difference variant of MeZO.

    Standard 2-point central diff has bias O(eps^2) coming from the third
    Taylor term ``eps^2 * z^T grad^3 L * zz / 6``. Section 6.7 shows this
    bias is the dominant failure mode for large-eps schedules in fp16 — it
    pushes the gradient direction off-axis and stochastic averaging cannot
    rescue it.

    Richardson combines two central-diff probes at eps and 2*eps:

        rho_R = (4 * rho(eps) - rho(2*eps)) / 3
              = (8 (L+ - L-) - (L++ - L--)) / (12 * eps)

    where ``L+- := L(theta +- eps z)`` and ``L++--  := L(theta +- 2 eps z)``.
    Same ``z`` is used at both scales (regenerated from one ``seed``), so
    the third-order term cancels exactly to leading order. Residual bias is
    O(eps^4) and variance contribution from the wider 2eps probe is bounded.

    Cost: 4 forward passes per step (vs 2 for ``mezo_step``). For an equal
    compute budget Richardson takes half the steps; the trade-off is worth
    it when bias dominates variance, which the autotuner sweep (§6.7) and
    the ε(t) ablation (§6.7 follow-up) suggest is the case in fp16.

    Args:
        model: HF model (or any nn.Module). Must have parameters with
            ``requires_grad`` set.
        inputs: Input batch (dict of tensors) passed to ``loss_fn``.
        loss_fn: ``(model, inputs) -> scalar loss``.
        config: MeZO hyperparameters. Uses ``config.eps`` (the small probe);
            the large probe uses ``2 * config.eps`` implicitly.
        rng: Optional numpy Generator for the per-step seed.

    Returns:
        ``(seed, projected_grad_R, loss_plus_at_eps)``. ``mezo_update`` (or
        ``nesterov_step``) consumes ``(seed, projected_grad_R)`` unchanged
        — the direction ``z`` is regenerated from the same seed, only the
        scalar magnitude changes.

    Note:
        For ``mezo_update`` correctness the same ``z`` regenerated from
        ``seed`` MUST match the ``z`` used here. Both call sites use
        ``torch.manual_seed(seed)`` before sampling, which is deterministic.
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
    eps = config.eps

    # Inner probe at +/- eps.
    # +eps z
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=eps)
    loss_plus = _forward_loss(model, inputs, loss_fn)
    # -2eps z (net theta - eps z)
    perturb_parameters(named, seed=seed, scaling_factor=-2.0, eps=eps)
    loss_minus = _forward_loss(model, inputs, loss_fn)
    # +eps z (restore to theta)
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=eps)

    # Outer probe at +/- 2 eps. We re-use ``eps`` as the base step and
    # multiply by 2 in scaling_factor — keeps the SAME z, just doubles the
    # magnitude. This is the key requirement for Richardson extrapolation.
    # +2eps z
    perturb_parameters(named, seed=seed, scaling_factor=+2.0, eps=eps)
    loss_plus_plus = _forward_loss(model, inputs, loss_fn)
    # -4eps z (net theta - 2 eps z)
    perturb_parameters(named, seed=seed, scaling_factor=-4.0, eps=eps)
    loss_minus_minus = _forward_loss(model, inputs, loss_fn)
    # +2eps z (restore to theta)
    perturb_parameters(named, seed=seed, scaling_factor=+2.0, eps=eps)

    rho_inner = (loss_plus - loss_minus) / (2.0 * eps)
    rho_outer = (loss_plus_plus - loss_minus_minus) / (4.0 * eps)
    projected_grad_R = (4.0 * rho_inner - rho_outer) / 3.0

    if config.rho_clip is not None:
        projected_grad_R = max(-config.rho_clip, min(config.rho_clip, projected_grad_R))
    return seed, projected_grad_R, loss_plus


def mezo_step_6point(
    model: nn.Module,
    inputs: dict,
    loss_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    config: MeZOConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[int, float, float]:
    """6-point extended-Richardson: cancels both O(eps^2) and O(eps^4) bias.

    Combines three central-diff probes at eps, 2*eps, 4*eps using the
    recursive Romberg-Richardson schedule

        T_1(h)   = (4 * rho(h) - rho(2h)) / 3                  bias O(h^4)
        T_1(2h)  = (4 * rho(2h) - rho(4h)) / 3                 bias O(h^4)
        T_2(h)   = (16 * T_1(h) - T_1(2h)) / 15                bias O(h^6)

    Expanded:

        rho_6 = (64 * rho(eps) - 20 * rho(2*eps) + rho(4*eps)) / 45

    Verified by Taylor expansion: the eps^2 and eps^4 coefficients vanish
    identically (64 - 80 + 16 = 0 and 64 - 320 + 256 = 0). Residual bias
    is O(eps^6). Variance is amplified by ``(64^2 + 20^2 + 1) / 45^2 ~= 2.22x``
    relative to a single rho(eps) estimate.

    The Richardson 4-point ablation (`docs/figures/fig17_*`) found that the
    4-point variant fails outside Taylor-validity (eps >= 1e-2 on Qwen3-0.6B
    full-attn). For 6-point the outermost probe is at ``4 * eps``, twice as
    far as 4-point's ``2 * eps``, so it's MORE susceptible to leaving the
    Taylor regime. The hypothesis worth testing: on hybrid linear-attention
    models where Taylor validity extends further (§6.7 cliff at eps=3e-1
    instead of eps=3e-1 → eps=1.0), 6-point may have a usable window at
    eps ~ 1e-3 ... 3e-3 with bias suppression to O(eps^6) ~= 1e-18.

    Cost: 6 forward passes per step (vs 2 for ``mezo_step``, 4 for
    ``mezo_step_richardson``). For equal compute use N/3 steps.

    Args:
        model: HF model with ``requires_grad`` parameters.
        inputs: Input batch passed to ``loss_fn``.
        loss_fn: ``(model, inputs) -> scalar loss``.
        config: MeZO hyperparameters. ``config.eps`` is the inner probe;
            outer probes use 2*eps and 4*eps with the SAME ``z``.
        rng: Optional numpy Generator for the per-step seed.

    Returns:
        ``(seed, projected_grad_6, loss_plus_at_eps)``. Direction z is
        regenerated from the same seed by ``mezo_update``.
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
    eps = config.eps

    # Inner probe at +/- eps.
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=eps)
    L1p = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=-2.0, eps=eps)
    L1m = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=+1.0, eps=eps)  # restore

    # Middle probe at +/- 2*eps.
    perturb_parameters(named, seed=seed, scaling_factor=+2.0, eps=eps)
    L2p = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=-4.0, eps=eps)
    L2m = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=+2.0, eps=eps)  # restore

    # Outer probe at +/- 4*eps.
    perturb_parameters(named, seed=seed, scaling_factor=+4.0, eps=eps)
    L3p = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=-8.0, eps=eps)
    L3m = _forward_loss(model, inputs, loss_fn)
    perturb_parameters(named, seed=seed, scaling_factor=+4.0, eps=eps)  # restore

    rho_1 = (L1p - L1m) / (2.0 * eps)
    rho_2 = (L2p - L2m) / (4.0 * eps)
    rho_3 = (L3p - L3m) / (8.0 * eps)
    projected_grad_6 = (64.0 * rho_1 - 20.0 * rho_2 + rho_3) / 45.0

    if config.rho_clip is not None:
        projected_grad_6 = max(-config.rho_clip, min(config.rho_clip, projected_grad_6))
    return seed, projected_grad_6, L1p


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


# ---------------------------------------------------------------------------
# B1: Adaptive ρ-clipping state (running quantile of |ρ| history)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveClipState:
    """Running-window quantile estimator for the ρ-clipping threshold.

    Maintains a sliding window of recent ``|ρ|`` values and exposes
    :meth:`current_threshold` returning ``α · quantile(|ρ| history)``.
    Callers pass this to :func:`mezo_step` via the ``rho_clip_override`` kwarg.

    Motivation: a fixed ``rho_clip = C`` chosen at design time may be too
    permissive in low-magnitude regimes (where it does nothing) or too tight
    in high-magnitude regimes (where it cuts too much signal). The 0.95
    quantile multiplied by ``α = 1.3`` adapts to the current distribution of
    ρ during training while still cutting the tail.

    Attributes:
        window: Number of most-recent ``|ρ|`` samples to keep (e.g., 50).
        quantile: Quantile in (0, 1) — typically 0.95 (cuts ~5% tail).
        alpha: Multiplier on the quantile to get the final clip threshold.
            ``α = 1.3`` lets the top 5% of the distribution survive ~30% larger
            magnitude before clipping kicks in.
        min_samples: Minimum samples in history before adaptive threshold is
            computed; below this, :meth:`current_threshold` returns ``None``
            and callers should fall back to ``config.rho_clip`` (if set).
        history: Internal — most recent ``|ρ|`` values. Don't mutate directly.

    Notes:
        - Per-client adaptive clip — each client tracks its own ρ distribution.
          In IID setup these should be similar; in non-IID Dir(α) settings they
          may drift apart, which is a feature not a bug (each client adapts to
          its own data shard).
        - The threshold is **monotonically increasing** in α and quantile, so
          ``α = 1.3`` with ``q = 0.95`` is more permissive than fixed C = 50
          when the median ρ is small (early/late phases of training).
    """

    window: int = 50
    quantile: float = 0.95
    alpha: float = 1.3
    min_samples: int = 10
    history: list[float] = field(default_factory=list)

    def update(self, abs_rho: float) -> None:
        """Append the latest ``|ρ|`` to the running buffer."""
        self.history.append(float(abs_rho))
        if len(self.history) > self.window:
            # Slice to drop oldest. List ops are O(n) here but window is small (~50).
            self.history = self.history[-self.window:]

    def current_threshold(self) -> float | None:
        """Compute the current clip threshold from the running buffer.

        Returns:
            ``α * quantile(|ρ| history)`` if at least ``min_samples`` history
            available, else ``None`` (caller should use ``config.rho_clip``
            fallback).
        """
        if len(self.history) < int(self.min_samples):
            return None
        q = float(np.quantile(self.history, float(self.quantile)))
        thr = float(self.alpha) * q
        # Guard against degenerate zero quantile (all-zero history).
        return thr if thr > 0 else None


# ---------------------------------------------------------------------------
# DP utility (D2): privacy accounting for Gaussian-mechanism MeZO
# ---------------------------------------------------------------------------


def dp_epsilon_from_sigma(sigma: float, sensitivity: float, delta: float) -> float:
    """Compute the (ε, δ)-DP guarantee for the Gaussian mechanism on ρ.

    Per Dwork-Roth 2014 (Theorem A.1, Gaussian mechanism), adding ``N(0, σ²)``
    noise to a function with L2-sensitivity Δ yields ``(ε, δ)``-DP for any
    ``δ ∈ (0, 1)`` such that

        σ ≥ Δ · √(2 ln(1.25 / δ)) / ε,

    rearranged:

        ε = Δ · √(2 ln(1.25 / δ)) / σ.

    For our setup, ``Δ = C`` (the ρ-clip threshold bounds per-record contribution).

    Args:
        sigma: Standard deviation of Gaussian noise applied to ρ.
        sensitivity: Per-record sensitivity bound (= clip threshold C).
        delta: Failure probability δ for (ε, δ)-DP. Typically 1/N where N is
            dataset size (here, 500 / 1000 = 1e-3 ish).

    Returns:
        Privacy parameter ε. Smaller is stronger privacy. ``σ → ∞`` gives ``ε → 0``;
        ``σ → 0`` gives ``ε → ∞`` (no privacy).
    """
    if sigma <= 0:
        return float("inf")
    return float(sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma)
