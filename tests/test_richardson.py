"""Tests for ``mezo_step_richardson`` (4-point central-difference variant).

Section 6.7 of the paper attributes the failure mode of large-eps schedules
to the O(eps^2) bias in the standard 2-point central difference. Richardson
extrapolation cancels exactly this term, leaving residual bias O(eps^4).

Tests:
    1. Determinism: same seed + same inputs -> same rho_R.
    2. Bias-cancellation: on a smooth model with known third-derivative,
       Richardson rho_R at large eps is much closer to the true projected
       gradient than 2-point rho.
    3. Agreement at small eps: at very small eps both methods agree (the
       O(eps^2) bias is negligible there, so Richardson's correction adds
       nothing).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_step_richardson


def _smooth_model_with_cubic_loss():
    """Linear model whose loss has a controlled cubic nonlinearity.

    L(theta) = a^T theta + (1/2) theta^T H theta + (1/6) C(theta, theta, theta)

    Returns model + loss_fn. We construct via a 1-layer net such that:
        out = w * x  (w is learnable, x is fixed input)
        loss = a*w + 0.5 * h * w^2 + (1/6) * c * w^3
    For a scalar w this isolates the third-order term.
    """
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    # The default Linear weight is small Gaussian. We don't care about its
    # initial value — we'll just measure rho at this fixed theta.
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _cubic_loss(model, batch):
    """L(w) = w + 0.5 * w^2 + (1/6) * 50 * w^3 (large cubic coefficient).

    Picked so that the third Taylor term is non-negligible at eps ~ 1e-2.
    Returns a scalar tensor.
    """
    w = next(model.parameters()).reshape(())  # () shape scalar
    # We need a "loss" returning a tensor with .item() — use weight directly.
    return w + 0.5 * w * w + (50.0 / 6.0) * w * w * w


def _quadratic_loss(model, batch):
    """L(w) = w + 0.5 * w^2 — pure quadratic, no third-order term.

    Both 2-point and Richardson should give EXACTLY the analytical gradient
    here (modulo floating-point arithmetic).
    """
    w = next(model.parameters()).reshape(())
    return w + 0.5 * w * w


class TestRichardsonDeterminism:
    def test_same_seed_same_rho(self):
        """Two calls with the same rng seed must return the same (seed, rho)."""
        model = _smooth_model_with_cubic_loss()
        batch = {}
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        seed_a, rho_a, _ = mezo_step_richardson(
            model, batch, _cubic_loss, cfg, rng=np.random.default_rng(42)
        )
        model2 = _smooth_model_with_cubic_loss()
        seed_b, rho_b, _ = mezo_step_richardson(
            model2, batch, _cubic_loss, cfg, rng=np.random.default_rng(42)
        )
        assert seed_a == seed_b
        assert abs(rho_a - rho_b) < 1e-9

    def test_richardson_eq_2point_on_quadratic(self):
        """For a pure quadratic L = a w + 0.5 b w^2 the third Taylor term is
        zero, so Richardson and 2-point central diff coincide (up to roundoff).
        """
        model = _smooth_model_with_cubic_loss()
        batch = {}
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        rng_seed = 7
        _, rho_2pt, _ = mezo_step(
            model, batch, _quadratic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )
        model2 = _smooth_model_with_cubic_loss()
        _, rho_R, _ = mezo_step_richardson(
            model2, batch, _quadratic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )
        # Both should give a + b*w + O(eps^4) ~= a + b*w (residual is double-precision noise).
        assert abs(rho_2pt - rho_R) < 1e-3, (
            f"On pure quadratic, 2-point and Richardson should agree; got "
            f"rho_2pt={rho_2pt}, rho_R={rho_R}"
        )


class TestRichardsonBiasCancellation:
    """At large eps, Richardson rho should be much closer to the true projected
    gradient than 2-point rho. This is the core motivation of the variant.
    """

    def _ground_truth_projected_grad(self, model, z_scalar: float) -> float:
        """Analytical <grad L, z> at the current theta for _cubic_loss.

        L = w + 0.5 w^2 + (50/6) w^3   =>   L'(w) = 1 + w + 25 w^2.
        Projected onto z: rho_true = L'(w) * z.
        """
        w = float(next(model.parameters()).item())
        grad = 1.0 + w + 25.0 * w * w
        return grad * z_scalar

    def _extract_z_scalar(self, seed: int, dtype: torch.dtype) -> float:
        """Reproduce the z used by mezo_step* for this 1-param model."""
        torch.manual_seed(seed)
        z = torch.normal(mean=0.0, std=1.0, size=(1, 1), device="cpu", dtype=dtype)
        return float(z.item())

    def test_richardson_beats_2point_at_large_eps(self):
        """On the cubic loss, at eps where O(eps^2) bias dominates:
            error_R << error_2pt.
        """
        model = _smooth_model_with_cubic_loss()
        # Push w away from 0 so the cubic term contributes meaningfully.
        with torch.no_grad():
            next(model.parameters()).fill_(0.3)
        batch = {}
        eps_large = 1e-2  # the regime where 2-point bias hurts

        cfg = MeZOConfig(lr=1e-3, eps=eps_large)
        rng_seed = 13
        seed_2pt, rho_2pt, _ = mezo_step(
            model, batch, _cubic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )

        model2 = _smooth_model_with_cubic_loss()
        with torch.no_grad():
            next(model2.parameters()).fill_(0.3)
        seed_R, rho_R, _ = mezo_step_richardson(
            model2, batch, _cubic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )
        assert seed_2pt == seed_R, "RNG draw must match across methods"

        z_scalar = self._extract_z_scalar(seed_R, next(model.parameters()).dtype)
        rho_true = self._ground_truth_projected_grad(model, z_scalar)

        err_2pt = abs(rho_2pt - rho_true)
        err_R = abs(rho_R - rho_true)
        # Hypothesis: residual is O(eps^2) for 2-point and O(eps^4) for Richardson.
        # With eps = 1e-2 the gap should be ~1e4× — extremely loose threshold below.
        assert err_R < err_2pt * 0.1, (
            f"At eps={eps_large}, Richardson should be >=10× more accurate. "
            f"err_2pt={err_2pt:.3e}, err_R={err_R:.3e}, rho_true={rho_true:.4f}, "
            f"rho_2pt={rho_2pt:.4f}, rho_R={rho_R:.4f}"
        )

    def test_richardson_close_to_2point_at_tiny_eps(self):
        """At very small eps both estimators are essentially equal (bias of
        2-point is negligible relative to roundoff), so Richardson's
        correction is in the noise. Sanity that we did not break the small-eps
        regime.
        """
        model = _smooth_model_with_cubic_loss()
        with torch.no_grad():
            next(model.parameters()).fill_(0.3)
        batch = {}
        cfg = MeZOConfig(lr=1e-3, eps=1e-5)

        rng_seed = 13
        _, rho_2pt, _ = mezo_step(
            model, batch, _cubic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )
        model2 = _smooth_model_with_cubic_loss()
        with torch.no_grad():
            next(model2.parameters()).fill_(0.3)
        _, rho_R, _ = mezo_step_richardson(
            model2, batch, _cubic_loss, cfg, rng=np.random.default_rng(rng_seed)
        )
        # At eps=1e-5 the O(eps^2) bias is ~1e-10, far below typical signal scale.
        # So 2-point and Richardson should agree to within fp32 roundoff (~1e-4).
        assert abs(rho_2pt - rho_R) < 5e-4, (
            f"At tiny eps={cfg.eps}, methods should match. "
            f"rho_2pt={rho_2pt}, rho_R={rho_R}"
        )
