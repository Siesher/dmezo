"""Positive tests for nesterov_step in src/dmezo/mezo/nesterov.py.

The existing test_simulator suite only covers the negative case
(NotImplementedError when combining Nesterov with update_share consensus).
These tests cover the actual update math.
"""

from __future__ import annotations

import torch
from torch import nn

from dmezo.mezo.nesterov import NesterovState, nesterov_step


def _tiny_model(seed: int = 0) -> nn.Module:
    """Two-layer MLP with deterministic init."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    for p in m.parameters():
        p.requires_grad_(True)
    return m


class TestNesterovStep:
    def test_beta_zero_matches_plain_mezo_update(self):
        """With beta=0, velocity = projected_grad * z (heavy-ball collapses to SGD).

        Then theta_new - theta_old = -lr * projected_grad * z. We can verify
        this exactly by comparing against a hand-computed delta for the first
        parameter.
        """
        model = _tiny_model()
        state = NesterovState(beta=0.0)
        # Stash original weights for comparison.
        orig = {n: p.data.clone() for n, p in model.named_parameters()}

        seed = 7
        rho = 0.5
        lr = 1e-3
        nesterov_step(model, state, seed=seed, projected_grad=rho, lr=lr, weight_decay=0.0)

        # Regenerate the same z deterministically and compute expected delta.
        torch.manual_seed(seed)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            z = torch.normal(0.0, 1.0, size=param.data.size())
            expected = orig[name] - lr * rho * z
            assert torch.allclose(param.data, expected, atol=1e-6), f"mismatch on {name}"

    def test_velocity_accumulates_across_calls(self):
        """With beta=0.9, calling twice with same rho/seed should give a larger
        update than a single call (velocity carries over).
        """
        m1 = _tiny_model()
        s1 = NesterovState(beta=0.9)
        nesterov_step(m1, s1, seed=42, projected_grad=1.0, lr=1e-3)

        m2 = _tiny_model()
        s2 = NesterovState(beta=0.9)
        nesterov_step(m2, s2, seed=42, projected_grad=1.0, lr=1e-3)
        nesterov_step(m2, s2, seed=42, projected_grad=1.0, lr=1e-3)

        # After 2 calls with the same seed: velocity = 0.9 * (rho*z) + (rho*z) = 1.9 * rho*z
        # vs 1 call: velocity = 1.0 * rho*z. So delta is 1.9x larger.
        # Use the first linear layer's weight to compare deltas.
        p1 = dict(m1.named_parameters())["0.weight"].data
        p2 = dict(m2.named_parameters())["0.weight"].data
        m_orig = _tiny_model()
        po = dict(m_orig.named_parameters())["0.weight"].data
        delta1 = (po - p1).abs().mean().item()
        delta2 = (po - p2).abs().mean().item()
        assert delta2 > 2.0 * delta1, (
            f"two calls should accumulate to ~2.9x single-call delta, got ratio {delta2 / delta1:.2f}"
        )

    def test_velocity_buffer_lazily_allocated(self):
        """Before any call, state.velocities is empty. After one call, contains one
        entry per trainable parameter."""
        model = _tiny_model()
        state = NesterovState(beta=0.9)
        assert len(state.velocities) == 0
        nesterov_step(model, state, seed=1, projected_grad=0.1, lr=1e-3)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert len(state.velocities) == n_trainable

    def test_reset_zeros_velocities(self):
        """state.reset() should zero out all velocity buffers in place."""
        model = _tiny_model()
        state = NesterovState(beta=0.9)
        nesterov_step(model, state, seed=1, projected_grad=0.5, lr=1e-3)
        # Velocities should be non-zero after a step with non-zero rho.
        assert any(v.abs().sum().item() > 0 for v in state.velocities.values())
        state.reset()
        for v in state.velocities.values():
            assert torch.all(v == 0), "reset() did not zero velocity"

    def test_determinism_same_seed_same_update(self):
        """Two independent runs with the same seed/rho/lr should give identical
        post-step parameters."""
        m1 = _tiny_model()
        s1 = NesterovState(beta=0.9)
        nesterov_step(m1, s1, seed=999, projected_grad=0.3, lr=2e-4)

        m2 = _tiny_model()
        s2 = NesterovState(beta=0.9)
        nesterov_step(m2, s2, seed=999, projected_grad=0.3, lr=2e-4)

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"non-deterministic update on {n1}"

    def test_weight_decay_skips_bias_and_norm(self):
        """weight_decay should only apply to non-bias, non-norm parameters.

        Hand-checks the 'bias' branch via a model with a clear bias parameter.
        """
        model = nn.Linear(3, 3)
        model.weight.data.fill_(1.0)
        model.bias.data.fill_(1.0)
        state = NesterovState(beta=0.0)
        # rho=0 + wd>0 means: weight should change (wd applied), bias should not.
        orig_w = model.weight.data.clone()
        orig_b = model.bias.data.clone()
        nesterov_step(model, state, seed=0, projected_grad=0.0, lr=1.0, weight_decay=0.1)
        # bias: rho=0, decay=0 -> no change.
        assert torch.equal(model.bias.data, orig_b), "bias should not be touched by weight_decay"
        # weight: rho=0, decay=0.1 -> theta -= lr * decay * theta = theta - 0.1 * theta = 0.9 * theta
        assert torch.allclose(model.weight.data, 0.9 * orig_w, atol=1e-6)
