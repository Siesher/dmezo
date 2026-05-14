"""Tests for in-place seed-based perturbation."""

from __future__ import annotations

import torch
from torch import nn

from dmezo.mezo.perturbation import perturb_parameters


def _make_tiny_model() -> nn.Module:
    """A tiny module with a few parameters of different shapes."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 4),
    )


def test_perturbation_is_deterministic():
    """Same seed must produce the same parameter delta."""
    m1 = _make_tiny_model()
    m2 = _make_tiny_model()
    # Match initial weights so we can compare deltas.
    m2.load_state_dict(m1.state_dict())

    seed = 1234
    eps = 1e-3
    perturb_parameters(m1.named_parameters(), seed=seed, scaling_factor=1.0, eps=eps)
    perturb_parameters(m2.named_parameters(), seed=seed, scaling_factor=1.0, eps=eps)

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.data, p2.data, atol=1e-10), f"Param {n1} differs"


def test_perturbation_is_reversible():
    """+1, -2, +1 sequence should restore parameters exactly (modulo fp noise)."""
    m = _make_tiny_model()
    snapshot = {n: p.data.clone() for n, p in m.named_parameters()}

    seed = 9999
    eps = 1e-3
    perturb_parameters(m.named_parameters(), seed=seed, scaling_factor=+1.0, eps=eps)
    perturb_parameters(m.named_parameters(), seed=seed, scaling_factor=-2.0, eps=eps)
    perturb_parameters(m.named_parameters(), seed=seed, scaling_factor=+1.0, eps=eps)

    for n, p in m.named_parameters():
        assert torch.allclose(p.data, snapshot[n], atol=1e-6), f"Param {n} not restored"


def test_perturbation_magnitude_scales_with_eps():
    """Delta magnitude should scale linearly with eps."""
    m1 = _make_tiny_model()
    m2 = _make_tiny_model()
    m2.load_state_dict(m1.state_dict())
    snap = {n: p.data.clone() for n, p in m1.named_parameters()}

    perturb_parameters(m1.named_parameters(), seed=42, scaling_factor=1.0, eps=1e-3)
    perturb_parameters(m2.named_parameters(), seed=42, scaling_factor=1.0, eps=2e-3)

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        d1 = p1.data - snap[n1]
        d2 = p2.data - snap[n2]
        # d2 should be ~2 * d1.
        assert torch.allclose(d2, 2.0 * d1, atol=1e-9), f"Param {n1} non-linear in eps"


def test_different_seeds_give_different_perturbations():
    m1 = _make_tiny_model()
    m2 = _make_tiny_model()
    m2.load_state_dict(m1.state_dict())

    perturb_parameters(m1.named_parameters(), seed=111, scaling_factor=1.0, eps=1e-3)
    perturb_parameters(m2.named_parameters(), seed=222, scaling_factor=1.0, eps=1e-3)

    same = []
    for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        same.append(torch.allclose(p1.data, p2.data))
    assert not all(same), "Different seeds produced identical perturbations"
