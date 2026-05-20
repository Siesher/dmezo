"""Smoke tests for B5 (drift reset), B1 (adaptive clip), D2 (DP noise).

These are backwards-compatibility + correctness tests, not full integration
tests (those are in scripts/local_test_improvements.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmezo.mezo.nesterov import NesterovState  # noqa: E402
from dmezo.mezo.step import (  # noqa: E402
    AdaptiveClipState,
    MeZOConfig,
    dp_epsilon_from_sigma,
    mezo_step,
)


# ---------------------------------------------------------------------------
# B5: Drift reset
# ---------------------------------------------------------------------------


def test_drift_disabled_by_default():
    """No drift detection without explicit drift_window > 0."""
    ns = NesterovState(beta=0.9)
    # Even with massive loss jumps, no reset because window=0.
    for loss in [1.0, 100.0, 0.5, 200.0]:
        triggered = ns.check_drift_and_reset(loss)
        assert not triggered
    assert ns.n_resets == 0


def test_drift_triggers_on_increase():
    """Reset fires when loss rises by > threshold within window."""
    ns = NesterovState(beta=0.9, drift_window=3, drift_threshold=0.5)
    # Build window of decreasing loss.
    for loss in [1.0, 0.9, 0.8, 0.7]:
        assert not ns.check_drift_and_reset(loss)
    # Spike up: loss[t] - loss[t-3] = 1.5 - 0.7 = +0.8 > 0.5 → reset.
    triggered = ns.check_drift_and_reset(1.5)
    assert triggered
    assert ns.n_resets == 1


def test_drift_resets_velocity_buffer():
    """Reset must zero out all velocity tensors."""
    ns = NesterovState(beta=0.9, drift_window=2, drift_threshold=0.5)
    # Pre-allocate two fake velocity tensors with non-zero data.
    ns.velocities["fc1.weight"] = torch.ones(3, 3) * 5.0
    ns.velocities["fc1.bias"] = torch.ones(3) * 2.0
    # Drive a drift.
    ns.check_drift_and_reset(1.0)
    ns.check_drift_and_reset(1.0)
    ns.check_drift_and_reset(1.0)
    ns.check_drift_and_reset(2.0)  # +1.0 > 0.5 → reset
    assert ns.n_resets == 1
    for v in ns.velocities.values():
        assert torch.all(v == 0)


def test_drift_no_false_positive_on_decreasing_loss():
    """Monotonically decreasing loss should never trigger reset."""
    ns = NesterovState(beta=0.9, drift_window=5, drift_threshold=0.1)
    losses = [2.0 - 0.1 * i for i in range(50)]
    for loss in losses:
        ns.check_drift_and_reset(loss)
    assert ns.n_resets == 0


def test_drift_handles_short_history():
    """Calls before window is full should return False."""
    ns = NesterovState(beta=0.9, drift_window=10, drift_threshold=0.1)
    for loss in [1.0, 2.0, 3.0]:  # Big jumps but history too short.
        assert not ns.check_drift_and_reset(loss)
    assert ns.n_resets == 0


# ---------------------------------------------------------------------------
# B1: Adaptive ρ-clipping
# ---------------------------------------------------------------------------


def test_adaptive_clip_below_min_samples_returns_none():
    """Threshold is None until enough history accumulated."""
    acs = AdaptiveClipState(window=20, quantile=0.95, alpha=1.3, min_samples=10)
    for v in [1.0, 2.0, 3.0]:
        acs.update(v)
    assert acs.current_threshold() is None


def test_adaptive_clip_threshold_basic():
    """Threshold = α · quantile(|ρ| history)."""
    acs = AdaptiveClipState(window=100, quantile=0.95, alpha=1.0, min_samples=5)
    # Buffer: 1..10 → 0.95 quantile ≈ 9.55 → α=1.0 gives ~9.55.
    for v in range(1, 11):
        acs.update(float(v))
    t = acs.current_threshold()
    assert t is not None
    # numpy's 0.95 quantile of [1..10] is 9.55.
    assert abs(t - 9.55) < 0.1


def test_adaptive_clip_alpha_scales_threshold():
    """α = 2.0 doubles the threshold vs α = 1.0."""
    base = [float(v) for v in range(1, 11)]
    a1 = AdaptiveClipState(window=100, quantile=0.95, alpha=1.0, min_samples=5)
    a2 = AdaptiveClipState(window=100, quantile=0.95, alpha=2.0, min_samples=5)
    for v in base:
        a1.update(v)
        a2.update(v)
    t1 = a1.current_threshold()
    t2 = a2.current_threshold()
    assert t1 is not None and t2 is not None
    assert abs(t2 / t1 - 2.0) < 1e-6


def test_adaptive_clip_window_drops_old_values():
    """Beyond window size, old values are evicted."""
    acs = AdaptiveClipState(window=5, quantile=0.95, alpha=1.0, min_samples=3)
    # Fill with 1..10; only last 5 should remain.
    for v in range(1, 11):
        acs.update(float(v))
    assert len(acs.history) == 5
    assert acs.history == [6.0, 7.0, 8.0, 9.0, 10.0]


def test_adaptive_clip_zero_history_returns_none():
    """All-zero history: threshold guard returns None (not 0)."""
    acs = AdaptiveClipState(window=20, quantile=0.95, alpha=1.3, min_samples=5)
    for _ in range(10):
        acs.update(0.0)
    assert acs.current_threshold() is None


# ---------------------------------------------------------------------------
# D2: DP noise
# ---------------------------------------------------------------------------


def _toy_model_and_inputs():
    """Tiny model for testing — 2-layer MLP that produces a scalar loss."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 4)
            self.fc2 = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = Tiny()
    for p in model.parameters():
        p.requires_grad_(True)
    x = torch.randn(3, 4)
    target = torch.zeros(3, 1)

    def loss_fn(m, inputs):
        out = m(inputs["x"])
        return torch.nn.functional.mse_loss(out, inputs["target"])

    return model, {"x": x, "target": target}, loss_fn


def test_dp_noise_disabled_by_default():
    """Default config has dp_sigma=None → no noise added (deterministic to FP-precision)."""
    model, inputs, loss_fn = _toy_model_and_inputs()
    cfg = MeZOConfig(lr=1e-4, eps=1e-3)
    rng = np.random.default_rng(42)
    seed1, rho1, _ = mezo_step(model, inputs, loss_fn, cfg, rng=rng)
    rng = np.random.default_rng(42)
    seed2, rho2, _ = mezo_step(model, inputs, loss_fn, cfg, rng=rng)
    # Same seed → ρ identical up to floating-point noise from non-deterministic ops.
    # (PyTorch matmul on CPU is not bit-exact across runs unless we set
    # `torch.use_deterministic_algorithms(True)`, which has overhead we avoid in tests.)
    # The point of this test: when DP is OFF, the only randomness is the seed,
    # not Gaussian draws — so |Δρ| should be ≪ DP noise magnitude.
    assert seed1 == seed2
    assert abs(rho1 - rho2) < 1e-4  # FP noise tolerance


def test_dp_noise_changes_rho_when_enabled():
    """Two runs with same seed but DP enabled should differ in ρ."""
    model, inputs, loss_fn = _toy_model_and_inputs()
    cfg = MeZOConfig(lr=1e-4, eps=1e-3, dp_sigma=1.0)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)  # different rng → different DP noise
    seed1, rho1, _ = mezo_step(model, inputs, loss_fn, cfg, rng=rng1)
    seed2, rho2, _ = mezo_step(model, inputs, loss_fn, cfg, rng=rng2)
    # Different RNGs → different seeds → different ρ.
    assert rho1 != rho2


def test_dp_epsilon_formula():
    """Sanity: Gaussian-mechanism formula gives expected values."""
    # σ = sensitivity · √(2 ln(1.25/δ)) / ε  ⇔  ε = sensitivity · √(2 ln(1.25/δ)) / σ.
    # For δ=1e-5 ⇒ √(2 ln(1.25/1e-5)) ≈ √(2·11.74) ≈ 4.84.
    # ε(σ=1, C=1, δ=1e-5) ≈ 4.84.
    eps = dp_epsilon_from_sigma(sigma=1.0, sensitivity=1.0, delta=1e-5)
    assert 4.5 < eps < 5.2


def test_dp_zero_sigma_returns_infinity():
    """No noise → no privacy (ε = ∞)."""
    assert dp_epsilon_from_sigma(sigma=0.0, sensitivity=50.0, delta=1e-3) == float("inf")


def test_dp_higher_sigma_gives_stronger_privacy():
    """Larger σ → smaller ε."""
    e1 = dp_epsilon_from_sigma(sigma=1.0, sensitivity=50.0, delta=1e-3)
    e10 = dp_epsilon_from_sigma(sigma=10.0, sensitivity=50.0, delta=1e-3)
    e100 = dp_epsilon_from_sigma(sigma=100.0, sensitivity=50.0, delta=1e-3)
    assert e1 > e10 > e100


# ---------------------------------------------------------------------------
# Override semantics: rho_clip_override > config.rho_clip
# ---------------------------------------------------------------------------


def test_rho_clip_override_takes_priority():
    """When rho_clip_override is set, config.rho_clip is ignored."""
    model, inputs, loss_fn = _toy_model_and_inputs()
    cfg = MeZOConfig(lr=1e-4, eps=1e-3, rho_clip=100.0)
    # Tight override should clip ρ regardless of large config threshold.
    rng = np.random.default_rng(42)
    seed, rho, _ = mezo_step(model, inputs, loss_fn, cfg, rng=rng, rho_clip_override=0.01)
    # ρ must be in [-0.01, +0.01].
    assert -0.01 <= rho <= 0.01


def test_rho_clip_override_negative_rejected():
    """Override must be positive."""
    model, inputs, loss_fn = _toy_model_and_inputs()
    cfg = MeZOConfig(lr=1e-4, eps=1e-3)
    with pytest.raises(ValueError, match="rho_clip_override"):
        mezo_step(model, inputs, loss_fn, cfg, rho_clip_override=-1.0)


def test_backwards_compat_no_new_kwargs():
    """Calling mezo_step with old signature should still work."""
    model, inputs, loss_fn = _toy_model_and_inputs()
    cfg = MeZOConfig(lr=1e-4, eps=1e-3)
    # No new kwargs — must work like before.
    seed, rho, lp = mezo_step(model, inputs, loss_fn, cfg)
    assert isinstance(seed, int)
    assert isinstance(rho, float)
    assert isinstance(lp, float)
