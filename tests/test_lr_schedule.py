"""Tests for the lr-schedule helper :func:`effective_lr`.

Three schedules supported:

- ``"constant"``: returns ``config.lr`` unchanged at every round.
- ``"harmonic"``: ``η_t = η_0 · a / (t+1)^α`` (SPSA classical, Spall 1992 §3.2).
- ``"cosine"``: ``η_t = η_0 · 0.5 (1 + cos(π·t/(T-1)))``, reaches 0 at the final
  round.

All three pass through the default ``config.lr`` at ``round_idx = 0`` (modulo
the harmonic ``a`` multiplier).
"""

from __future__ import annotations

import math

import pytest

from dmezo.mezo import MeZOConfig, effective_lr


class TestConstantSchedule:
    def test_constant_returns_lr_for_any_round(self):
        cfg = MeZOConfig(lr=3e-7, lr_schedule="constant")
        for t in [0, 100, 999]:
            assert effective_lr(cfg, t, num_rounds=1000) == pytest.approx(3e-7)

    def test_default_schedule_is_constant(self):
        cfg = MeZOConfig(lr=1e-6)  # no lr_schedule kwarg
        assert cfg.lr_schedule == "constant"
        assert effective_lr(cfg, round_idx=500, num_rounds=1000) == pytest.approx(1e-6)


class TestHarmonicSchedule:
    def test_first_step_uses_a_times_lr(self):
        """With a=1, α=0.602: at t=0 the denominator is (0+1)^0.602 = 1, so
        effective lr equals η_0 · a."""
        cfg = MeZOConfig(lr=1e-6, lr_schedule="harmonic", lr_decay_a=1.0, lr_decay_alpha=0.602)
        assert effective_lr(cfg, round_idx=0, num_rounds=1000) == pytest.approx(1e-6)

    def test_decay_follows_spall_formula(self):
        """η_t = η_0 · a / (t+1)^α — verify on a few points."""
        cfg = MeZOConfig(lr=1.0, lr_schedule="harmonic", lr_decay_a=1.0, lr_decay_alpha=0.602)
        for t in [0, 1, 10, 100, 999]:
            expected = 1.0 / ((t + 1) ** 0.602)
            assert effective_lr(cfg, t, num_rounds=1000) == pytest.approx(expected, rel=1e-9)

    def test_a_parameter_scales_lr(self):
        """a > 1 increases the initial lr; a < 1 decreases."""
        cfg = MeZOConfig(lr=1.0, lr_schedule="harmonic", lr_decay_a=2.5, lr_decay_alpha=0.602)
        assert effective_lr(cfg, round_idx=0, num_rounds=100) == pytest.approx(2.5)

    def test_alpha_affects_decay_speed(self):
        """Larger α → faster decay."""
        cfg_slow = MeZOConfig(lr=1.0, lr_schedule="harmonic", lr_decay_alpha=0.5)
        cfg_fast = MeZOConfig(lr=1.0, lr_schedule="harmonic", lr_decay_alpha=1.0)
        # At t=99, denominator (100)^0.5 vs (100)^1.0 — fast schedule should give
        # SMALLER lr than slow.
        slow = effective_lr(cfg_slow, round_idx=99, num_rounds=1000)
        fast = effective_lr(cfg_fast, round_idx=99, num_rounds=1000)
        assert fast < slow, f"α=1.0 should decay faster than α=0.5: {fast} vs {slow}"


class TestCosineSchedule:
    def test_first_step_returns_full_lr(self):
        cfg = MeZOConfig(lr=3e-7, lr_schedule="cosine")
        # progress(0) = 0; cos(0) = 1; factor = 0.5·(1+1) = 1.
        assert effective_lr(cfg, round_idx=0, num_rounds=1000) == pytest.approx(3e-7)

    def test_final_step_returns_zero(self):
        cfg = MeZOConfig(lr=3e-7, lr_schedule="cosine")
        # progress(T-1) = 1; cos(π) = -1; factor = 0.5·(1-1) = 0.
        assert effective_lr(cfg, round_idx=999, num_rounds=1000) == pytest.approx(0.0, abs=1e-12)

    def test_midpoint_returns_half(self):
        cfg = MeZOConfig(lr=1.0, lr_schedule="cosine")
        # progress = 0.5; cos(π/2) = 0; factor = 0.5·(1+0) = 0.5.
        # Use odd num_rounds so midpoint is exact integer.
        assert effective_lr(cfg, round_idx=500, num_rounds=1001) == pytest.approx(0.5)

    def test_monotonically_decreasing(self):
        cfg = MeZOConfig(lr=1.0, lr_schedule="cosine")
        lrs = [effective_lr(cfg, t, num_rounds=100) for t in range(0, 100)]
        assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))


class TestErrors:
    def test_unknown_schedule_raises(self):
        cfg = MeZOConfig(lr=1.0, lr_schedule="exponential")
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            effective_lr(cfg, round_idx=0, num_rounds=100)

    def test_invalid_num_rounds_raises(self):
        cfg = MeZOConfig(lr=1.0)
        with pytest.raises(ValueError, match="num_rounds"):
            effective_lr(cfg, round_idx=0, num_rounds=0)

    def test_negative_round_idx_clamped(self):
        """Defensive: round_idx < 0 should clamp to 0 (use initial lr)."""
        cfg = MeZOConfig(lr=1.0, lr_schedule="harmonic", lr_decay_alpha=0.602)
        assert effective_lr(cfg, round_idx=-5, num_rounds=100) == pytest.approx(1.0)


class TestBackwardCompat:
    def test_existing_configs_unaffected(self):
        """A MeZOConfig built without any lr_schedule field keeps lr behaviour
        identical to pre-feature code (constant lr)."""
        # No new fields set — should default to constant.
        cfg = MeZOConfig(lr=3e-7, eps=1e-3, rho_clip=50.0, k_directions=3)
        assert cfg.lr_schedule == "constant"
        # effective_lr returns config.lr regardless of round.
        for t in [0, 100, 500, 999]:
            assert effective_lr(cfg, t, num_rounds=1000) == cfg.lr
