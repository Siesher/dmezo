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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class NesterovState:
    """Container for per-parameter velocity buffers and config.

    Attributes:
        beta: Momentum coefficient at the current round. With a schedule
            configured (``beta_end`` and ``num_rounds_total`` both set), this
            field is mutated by :meth:`update_schedule` each round; otherwise
            it stays at the initial value. Common defaults: 0.9 (heavy-ball),
            0.95-0.99 for slower mixing.
        look_ahead: If True, use full Nesterov look-ahead (evaluate gradient at
            theta + beta * v before updating). If False, use heavy-ball form.
        beta_end: If set together with ``num_rounds_total``, enables a linear
            β-schedule between the initial ``beta`` value and ``beta_end`` over
            ``num_rounds_total`` rounds. Default ``None`` keeps β constant.
        num_rounds_total: Horizon for the β-schedule. Must be ≥ 2 when set.
            Out-of-range round indices are clamped to the [0, total-1] interval.
        velocities: Per-parameter velocity tensors, keyed by parameter name.
            Allocated lazily.
        drift_window: Window size (in rounds) for late-stage drift detection.
            Default 0 = drift detection DISABLED (original behavior). When > 0,
            :meth:`check_drift_and_reset` compares current loss vs loss
            ``drift_window`` rounds ago; if increase exceeds ``drift_threshold``,
            zeroes out the velocity buffer (B5 improvement, 2026-05-20).
        drift_threshold: Threshold for drift detection — current loss minus loss
            ``drift_window`` rounds ago must exceed this value to trigger reset.
            Default 0.0 = effectively disabled (any drift triggers). Typical
            values: 0.05 (sensitive), 0.1 (moderate), 0.2 (conservative).
        loss_history: Internal — running window of (round_idx, loss) for drift
            detection. Bounded length ``drift_window + 1``. Don't mutate directly.
        n_resets: Counter of how many drift-resets occurred. Read for diagnostics.
    """

    beta: float = 0.9
    look_ahead: bool = False
    beta_end: float | None = None
    num_rounds_total: int | None = None
    velocities: dict[str, torch.Tensor] = field(default_factory=dict)
    # B5: drift detection / reset.
    drift_window: int = 0
    drift_threshold: float = 0.0
    loss_history: list[float] = field(default_factory=list)
    n_resets: int = 0
    # Snapshotted at construction so update_schedule can interpolate from the
    # original start even if ``beta`` has been mutated by prior schedule calls.
    _beta_start: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._beta_start = float(self.beta)

    def reset(self) -> None:
        """Zero out all velocities (e.g., at the start of a new run)."""
        for v in self.velocities.values():
            v.zero_()

    def update_schedule(self, round_idx: int) -> None:
        """Mutate ``self.beta`` based on ``round_idx`` if a schedule is configured.

        Linear interpolation: ``beta(t) = beta_start + (t / (T-1)) * (beta_end - beta_start)``
        where ``T = num_rounds_total``. Clamped to ``[0, T-1]`` for ``round_idx``
        below 0 or at/above ``T``.

        No-op when either ``beta_end`` or ``num_rounds_total`` is ``None``.
        """
        if self.beta_end is None or self.num_rounds_total is None:
            return
        total = int(self.num_rounds_total)
        if total < 2:
            # Degenerate schedule — single-round horizon. Snap to start.
            self.beta = self._beta_start
            return
        # Clamp round_idx to [0, total-1].
        r = min(max(round_idx, 0), total - 1)
        progress = r / (total - 1)
        self.beta = self._beta_start + progress * (float(self.beta_end) - self._beta_start)

    def check_drift_and_reset(self, current_loss: float) -> bool:
        """Drift detection / velocity reset (B5 improvement).

        If late-stage drift is detected (``current_loss`` exceeds loss from
        ``drift_window`` rounds ago by more than ``drift_threshold``), zero out
        all velocity buffers. Then loss_history is cleared so we don't
        immediately re-trigger on the next step.

        Args:
            current_loss: Current step's training loss (or any monotone signal
                of trajectory health, e.g., loss_plus from mezo_step).

        Returns:
            True if a reset was triggered this round, False otherwise.

        Notes:
            - When ``drift_window <= 0`` OR ``drift_threshold <= 0``, this method
              is a no-op (returns False without updating state).
            - First ``drift_window`` calls always return False (insufficient
              history to detect drift).
            - Resetting velocities effectively restarts D-MeZO-N as plain
              vanilla D-MeZO for one round (β still applied next round per
              schedule, but ``v = 0`` so no momentum carry-over).
        """
        if self.drift_window <= 0 or self.drift_threshold <= 0:
            return False
        self.loss_history.append(float(current_loss))
        if len(self.loss_history) > self.drift_window + 1:
            self.loss_history.pop(0)
        if len(self.loss_history) < self.drift_window + 1:
            return False
        drift = float(current_loss) - float(self.loss_history[0])
        if drift > float(self.drift_threshold):
            self.reset()
            self.n_resets += 1
            # Clear history so we don't fire again immediately on next step.
            self.loss_history = [float(current_loss)]
            return True
        return False


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


def md_nesterov_step(
    model: nn.Module,
    state: NesterovState,
    seeds: list[int],
    projected_grads: list[float],
    lr: float,
    weight_decay: float = 0.0,
) -> None:
    """Apply one Multi-Direction Nesterov-MeZO update (K-direction heavy-ball).

    Computes the K-direction-averaged gradient::

        g_tilde = (1/K) Σ_k ρ_k z_k

    and feeds it through the Nesterov velocity buffer::

        v <- beta * v + (g_tilde + weight_decay * theta)
        theta <- theta - lr * v

    K=1 is regression-equivalent to ``nesterov_step``. Per-direction z_k are
    regenerated from ``seeds[k]``. Weight decay is applied once per call
    (NOT K times) to match single-direction semantics.

    Args:
        model: Model whose parameters to update.
        state: NesterovState (velocity buffers, beta). Mutated in place.
        seeds: Length-K list of per-direction seeds.
        projected_grads: Length-K list of per-direction ρ_k (clipped if config
            had rho_clip set).
        lr: Learning rate.
        weight_decay: L2 coefficient (applied only to non-bias / non-norm params).

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
        raise ValueError("md_nesterov_step requires at least one direction.")
    inv_K = 1.0 / K

    # Phase 1: v <- beta * v + decay * theta  (decay applied once, not K times).
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        v = _ensure_velocity_buffer(state, name, param)
        v.mul_(state.beta)
        if weight_decay > 0.0:
            lname = name.lower()
            decay = (
                weight_decay
                if ("bias" not in lname)
                and ("layer_norm" not in lname)
                and ("layernorm" not in lname)
                else 0.0
            )
            if decay > 0.0:
                v.add_(param.data, alpha=decay)

    # Phase 2: v += (1/K) Σ_k ρ_k z_k  (accumulate per-direction contributions).
    for seed, rho_k in zip(seeds, projected_grads):
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
            v = state.velocities[name]
            v.add_(z, alpha=rho_k * inv_K)

    # Phase 3: theta <- theta - lr * v.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        v = state.velocities[name]
        param.data.add_(v, alpha=-lr)


def _apply_lookahead_shift(model: nn.Module, state: NesterovState, sign: float) -> None:
    """Add ``sign * state.beta * v`` to every trainable parameter in place.

    Used to move parameters to the look-ahead position ``theta + beta * v`` before
    MeZO probing, then back to ``theta`` afterwards. No-op for parameters whose
    velocity buffer hasn't been allocated yet (first call).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        v = state.velocities.get(name)
        if v is None:
            continue
        param.data.add_(v, alpha=sign * state.beta)


def nesterov_lookahead_step(
    model: nn.Module,
    state: NesterovState,
    inputs: dict,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: MeZOConfig,  # noqa: F821 — forward-imported below
    *,
    rng: np.random.Generator | None = None,  # noqa: F821
) -> tuple[int, float, float]:
    """One look-ahead Nesterov-MeZO step.

    Differs from the heavy-ball variant (``nesterov_step``): the projected
    gradient is evaluated at the *look-ahead* position ``theta + beta * v``
    rather than at ``theta``. This is the "true" Nesterov form (Sutskever
    et al. 2013 momentum reformulation), and for first-order optimization
    typically gives better convergence under noisy gradients than heavy-ball.

    Algorithm:
        1. theta <- theta + beta * v          (look-ahead shift)
        2. (seed, rho, loss_plus) <- mezo_step(theta_lookahead)
        3. theta <- theta - beta * v          (rollback)
        4. v <- beta * v + (rho * z + decay * theta)
        5. theta <- theta - lr * v

    Args:
        model: HF model whose params will be perturbed.
        state: Nesterov velocity state. Mutated in place.
        inputs: Batch dict passed to ``loss_fn``.
        loss_fn: ``(model, inputs) -> scalar tensor``.
        config: MeZO hyperparameters.
        rng: Numpy Generator for the per-step seed (passed through to ``mezo_step``).

    Returns:
        Tuple ``(seed, projected_grad, loss_plus)`` matching ``mezo_step``'s
        return signature, where ``projected_grad`` is computed at the look-ahead
        position.
    """
    # Imports moved here to avoid a top-of-module circular import: nesterov.py
    # now depends on mezo.step, and step.py does NOT import nesterov.
    from dmezo.mezo.step import mezo_step

    _apply_lookahead_shift(model, state, sign=+1.0)
    try:
        seed, rho, loss_plus = mezo_step(model, inputs, loss_fn, config, rng=rng)
    finally:
        _apply_lookahead_shift(model, state, sign=-1.0)

    nesterov_step(
        model,
        state,
        seed=seed,
        projected_grad=rho,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return seed, rho, loss_plus
