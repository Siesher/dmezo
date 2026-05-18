"""Tests for Multi-Direction MeZO (MD-MeZO) — K-direction SPSA averaging.

Critical properties verified:

1. **K=1 regression-identity.** md_mezo_step + md_mezo_update with K=1 produces
   bit-exact identical parameter updates to mezo_step + mezo_update given the
   same RNG state. This guarantees the new code path doesn't break anything.

2. **K=3 variance reduction.** Averaging K=3 independent ρ estimates yields a
   gradient estimate with lower variance than K=1. We verify this empirically
   on a tiny CPU model: variance of repeated K=3 estimates should be lower
   than variance of repeated K=1 estimates by approximately factor 3.

3. **API contract.** seeds and projected_grads are returned as parallel lists of
   length K; rho_clip applies per-direction.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from dmezo.mezo.step import (
    MeZOConfig,
    md_mezo_step,
    md_mezo_update,
    mezo_step,
    mezo_update,
)
from tests._fixtures import make_tiny_causal_lm, synthetic_token_loader


def _causal_lm_loss(model, batch):
    out = model(**batch)
    return out.loss


def _state_clone(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def _max_abs_diff(state_a, state_b):
    return max((state_a[k] - state_b[k]).abs().max().item() for k in state_a)


# ---------------------------------------------------------------------------
# 1. K=1 regression-identity
# ---------------------------------------------------------------------------


class TestKOneEquivalence:
    def test_md_mezo_step_k1_returns_same_seed_and_rho_as_mezo_step(self):
        """For K=1, md_mezo_step's first (and only) seed/ρ must match
        what mezo_step produces given the same global RNG state."""
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)

        loader = synthetic_token_loader(seed=42)
        batch = next(iter(loader))
        cfg_k1 = MeZOConfig(lr=1e-3, eps=1e-3, k_directions=1)
        cfg_single = MeZOConfig(lr=1e-3, eps=1e-3)  # default k_directions=1

        # Reset global numpy RNG so both calls produce the same seed.
        np.random.seed(123)
        seeds_md, rhos_md, _ = md_mezo_step(model, batch, _causal_lm_loss, cfg_k1)
        np.random.seed(123)
        seed_single, rho_single, _ = mezo_step(model, batch, _causal_lm_loss, cfg_single)

        assert len(seeds_md) == 1, f"K=1 should yield one seed, got {len(seeds_md)}"
        assert seeds_md[0] == seed_single, f"seed mismatch: {seeds_md[0]} vs {seed_single}"
        assert abs(rhos_md[0] - rho_single) < 1e-12, (
            f"ρ mismatch: {rhos_md[0]} vs {rho_single}"
        )

    def test_md_mezo_update_k1_matches_mezo_update_param_for_param(self):
        """After md_mezo_update with K=1, parameters must match mezo_update
        with the same (seed, ρ) and config, to within fp32 round-off."""
        # Two independent model copies starting from the same state.
        model_a = make_tiny_causal_lm(seed=0)
        model_b = make_tiny_causal_lm(seed=0)
        for p in model_a.parameters():
            p.requires_grad_(True)
        for p in model_b.parameters():
            p.requires_grad_(True)

        cfg = MeZOConfig(lr=1e-3, eps=1e-3, weight_decay=0.0, k_directions=1)
        seed, rho = 12345, 0.42

        # Apply via mezo_update on A.
        mezo_update(model_a, seed=seed, projected_grad=rho, config=cfg)
        # Apply via md_mezo_update on B (with K=1).
        md_mezo_update(model_b, seeds=[seed], projected_grads=[rho], config=cfg)

        diff = _max_abs_diff(_state_clone(model_a), _state_clone(model_b))
        assert diff < 1e-6, f"K=1 update should match mezo_update, max diff: {diff}"


# ---------------------------------------------------------------------------
# 2. K=3 variance reduction
# ---------------------------------------------------------------------------


class TestKDirectionVarianceReduction:
    def test_k3_estimate_has_lower_variance_than_k1(self):
        """Averaging K=3 independent ρ estimates of the same target gradient
        should give an estimator with ~3× lower variance than K=1.

        We verify by sampling many K=1 and K=3 estimates of ρ_avg = ⟨z̄, ∇L⟩
        and checking var(K=3) / var(K=1) ≈ 1/3 (within a tolerance to account
        for the small sample size and tiny model used in this test)."""
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)

        loader = synthetic_token_loader(seed=42)
        batch = next(iter(loader))
        # Larger eps reduces the bias term so we measure noise more cleanly.
        cfg_k1 = MeZOConfig(lr=1e-3, eps=1e-2, k_directions=1)
        cfg_k3 = MeZOConfig(lr=1e-3, eps=1e-2, k_directions=3)

        n_samples = 60
        k1_estimates = []
        k3_estimates = []

        rng_k1 = np.random.default_rng(0)
        for _ in range(n_samples):
            _, rhos, _ = md_mezo_step(model, batch, _causal_lm_loss, cfg_k1, rng=rng_k1)
            k1_estimates.append(rhos[0])

        rng_k3 = np.random.default_rng(0)
        for _ in range(n_samples):
            _, rhos, _ = md_mezo_step(model, batch, _causal_lm_loss, cfg_k3, rng=rng_k3)
            # The aggregated estimator is the MEAN of the K per-direction ρ's
            # (this is the SCALAR projection averaging; the vector gradient
            # estimator g̃ = (1/K) Σ ρ_k z_k has variance scaling the same way
            # by linearity).
            k3_estimates.append(float(np.mean(rhos)))

        var_k1 = float(np.var(k1_estimates))
        var_k3 = float(np.var(k3_estimates))
        ratio = var_k3 / var_k1 if var_k1 > 0 else float("inf")

        # Expect ratio ≈ 1/K = 1/3 ≈ 0.33. Allow generous tolerance because
        # n_samples=60 gives noisy variance estimates.
        assert ratio < 0.7, (
            f"K=3 variance ({var_k3:.6f}) should be ~3× lower than K=1 ({var_k1:.6f}), "
            f"got ratio {ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# 3. API contract
# ---------------------------------------------------------------------------


class TestMDMezoAPI:
    def test_returns_lists_of_length_k(self):
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)
        batch = next(iter(synthetic_token_loader(seed=42)))
        cfg = MeZOConfig(lr=1e-3, eps=1e-3, k_directions=5)
        seeds, rhos, _ = md_mezo_step(model, batch, _causal_lm_loss, cfg)
        assert len(seeds) == 5
        assert len(rhos) == 5
        assert all(isinstance(s, int) for s in seeds)
        assert all(isinstance(r, float) for r in rhos)

    def test_rho_clip_applies_per_direction(self):
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)
        batch = next(iter(synthetic_token_loader(seed=42)))
        # Tight clip — ρ_k must lie in [-0.5, +0.5] regardless of source magnitude.
        cfg = MeZOConfig(lr=1e-3, eps=1e-3, k_directions=4, rho_clip=0.5)
        _, rhos, _ = md_mezo_step(model, batch, _causal_lm_loss, cfg)
        for rho in rhos:
            assert -0.5 <= rho <= 0.5, f"ρ={rho} violated clip [-0.5, +0.5]"

    def test_update_mismatch_lengths_raises(self):
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)
        cfg = MeZOConfig(lr=1e-3, k_directions=3)
        import pytest

        with pytest.raises(ValueError, match="same length"):
            md_mezo_update(model, seeds=[1, 2], projected_grads=[0.1, 0.2, 0.3], config=cfg)

    def test_update_empty_lists_raises(self):
        model = make_tiny_causal_lm(seed=0)
        for p in model.parameters():
            p.requires_grad_(True)
        cfg = MeZOConfig(lr=1e-3, k_directions=1)
        import pytest

        with pytest.raises(ValueError, match="at least one direction"):
            md_mezo_update(model, seeds=[], projected_grads=[], config=cfg)
