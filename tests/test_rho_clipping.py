"""Tests for ρ-clipping in MeZOConfig / mezo_step.

Motivation: Day 6 Nesterov β=0.9 diverged at round 140 when ρ spiked to +905
(normal range was ±150). Clipping ρ to a fixed range should cap the
worst-case variance contribution to the Nesterov velocity buffer.

Spec: cfg.rho_clip = None -> behavior unchanged. cfg.rho_clip = C > 0 ->
returned projected_grad is in [-C, +C].
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dmezo.mezo.step import MeZOConfig, mezo_step


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    m = nn.Linear(4, 2)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def _huge_loss_fn(model, batch):
    """Loss scaled to produce |ρ| in the 10^6 range so clipping is visible."""
    out = model(batch["x"])
    return (out * out).sum() * 1e6  # massive loss


def _normal_loss_fn(model, batch):
    out = model(batch["x"])
    return (out * out).sum()


class TestRhoClipping:
    def test_default_is_no_clip(self):
        """cfg.rho_clip = None -> returned ρ should match unclipped behaviour."""
        model = _tiny_model()
        batch = {"x": torch.randn(2, 4)}
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        assert cfg.rho_clip is None, "default rho_clip should be None (no clip)"

        seed, rho, _ = mezo_step(model, batch, _normal_loss_fn, cfg, rng=np.random.default_rng(0))
        # rho can be any finite value here; just verify no clipping happened by
        # comparing against re-run with explicit no-clip.
        cfg_explicit = MeZOConfig(lr=1e-3, eps=1e-3, rho_clip=None)
        model2 = _tiny_model()
        _, rho2, _ = mezo_step(
            model2, batch, _normal_loss_fn, cfg_explicit, rng=np.random.default_rng(0)
        )
        assert abs(rho - rho2) < 1e-9

    def test_clipping_bounds_rho_below_threshold(self):
        """When |ρ_unclipped| > C, ρ should be clipped to ±C."""
        model = _tiny_model()
        batch = {"x": torch.randn(2, 4)}
        cfg_clip = MeZOConfig(lr=1e-3, eps=1e-3, rho_clip=10.0)
        cfg_noclip = MeZOConfig(lr=1e-3, eps=1e-3)

        # Use huge loss to force |ρ| >> 10.
        m1 = _tiny_model()
        m2 = _tiny_model()
        _, rho_clipped, _ = mezo_step(
            m1, batch, _huge_loss_fn, cfg_clip, rng=np.random.default_rng(0)
        )
        _, rho_unclipped, _ = mezo_step(
            m2, batch, _huge_loss_fn, cfg_noclip, rng=np.random.default_rng(0)
        )
        assert abs(rho_unclipped) > 10.0, (
            f"setup precondition failed: huge_loss_fn should yield |ρ|>10, got {rho_unclipped}"
        )
        assert abs(rho_clipped) <= 10.0 + 1e-9, f"clipping failed: got {rho_clipped}"
        # Sign must be preserved.
        assert (rho_clipped >= 0) == (rho_unclipped >= 0)

    def test_clipping_passes_through_small_rho(self):
        """When |ρ_unclipped| < C, clipping should leave ρ unchanged."""
        model = _tiny_model()
        batch = {"x": torch.randn(2, 4) * 0.001}  # tiny loss
        cfg_clip = MeZOConfig(lr=1e-3, eps=1e-3, rho_clip=1e10)
        cfg_noclip = MeZOConfig(lr=1e-3, eps=1e-3)

        m1 = _tiny_model()
        m2 = _tiny_model()
        _, rho_clipped, _ = mezo_step(
            m1, batch, _normal_loss_fn, cfg_clip, rng=np.random.default_rng(0)
        )
        _, rho_unclipped, _ = mezo_step(
            m2, batch, _normal_loss_fn, cfg_noclip, rng=np.random.default_rng(0)
        )
        assert abs(rho_clipped - rho_unclipped) < 1e-9, (
            f"large clip threshold should be no-op: clipped={rho_clipped}, raw={rho_unclipped}"
        )

    def test_clipping_threshold_must_be_positive(self):
        """rho_clip <= 0 should raise."""
        import pytest

        with pytest.raises(ValueError, match="rho_clip"):
            cfg = MeZOConfig(lr=1e-3, eps=1e-3, rho_clip=0.0)
            mezo_step(
                _tiny_model(),
                {"x": torch.randn(2, 4)},
                _normal_loss_fn,
                cfg,
                rng=np.random.default_rng(0),
            )

        with pytest.raises(ValueError, match="rho_clip"):
            cfg = MeZOConfig(lr=1e-3, eps=1e-3, rho_clip=-5.0)
            mezo_step(
                _tiny_model(),
                {"x": torch.randn(2, 4)},
                _normal_loss_fn,
                cfg,
                rng=np.random.default_rng(0),
            )
