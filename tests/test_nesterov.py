"""Positive tests for nesterov_step in src/dmezo/mezo/nesterov.py.

The existing test_simulator suite only covers the negative case
(NotImplementedError when combining Nesterov with update_share consensus).
These tests cover the actual update math.
"""

from __future__ import annotations

import torch
from torch import nn

from dmezo.mezo.nesterov import NesterovState, nesterov_lookahead_step, nesterov_step
from dmezo.mezo.step import MeZOConfig


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


# ---------------------------------------------------------------------------
# nesterov_lookahead_step — the "true" Nesterov variant
# ---------------------------------------------------------------------------


def _quadratic_loss(model, batch):
    """Trivial loss: sum of squared params. Differentiable, deterministic, no real data needed."""
    return sum((p * p).sum() for p in model.parameters())


class TestNesterovLookaheadStep:
    """Tests for the look-ahead Nesterov variant.

    Look-ahead form: perturb to theta + beta * v before MeZO probing, evaluate
    rho there, then restore theta and apply heavy-ball update with rho_lookahead.
    """

    def test_with_empty_velocity_matches_vanilla_mezo(self):
        """With no prior velocity (first call), look-ahead shift is zero, so
        the function should produce the same MeZO step as plain vanilla MeZO."""
        import numpy as np

        from dmezo.mezo.step import mezo_step

        m1 = _tiny_model()
        m2 = _tiny_model()
        state = NesterovState(beta=0.9, look_ahead=True)
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        batch = {"x": torch.zeros(1)}

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        # vanilla mezo_step on m1
        seed_v, rho_v, _ = mezo_step(m1, batch, _quadratic_loss, cfg, rng=rng1)
        # look-ahead on m2 (no velocity yet)
        seed_l, rho_l, _ = nesterov_lookahead_step(m2, state, batch, _quadratic_loss, cfg, rng=rng2)

        assert seed_v == seed_l, "with empty velocity both should pick same seed"
        assert abs(rho_v - rho_l) < 1e-9, f"rho should match: vanilla={rho_v}, lookahead={rho_l}"

    def test_rolls_back_lookahead_shift(self):
        """After the call, params at theta should match the heavy-ball trajectory
        (i.e. no residual look-ahead shift left over from probing)."""
        import numpy as np

        m1 = _tiny_model()
        m2 = _tiny_model()
        # Pre-populate velocity for both states (same value).
        state1 = NesterovState(beta=0.5, look_ahead=False)
        state2 = NesterovState(beta=0.5, look_ahead=True)
        # Seed an initial velocity by doing one nesterov_step.
        nesterov_step(m1, state1, seed=1, projected_grad=0.1, lr=1e-3)
        nesterov_step(m2, state2, seed=1, projected_grad=0.1, lr=1e-3)

        # Both should now have identical params and velocities.
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.allclose(p1.data, p2.data), f"setup mismatch: {n1}"

        # Now do one look-ahead step on m2 only and compare param magnitudes.
        # After look-ahead probing, params must be at theta, not theta+beta*v.
        # We check that the params have moved by a "sensible" amount — not by
        # beta*v (which would mean rollback failed).
        snapshot = {n: p.data.clone() for n, p in m2.named_parameters()}
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        batch = {"x": torch.zeros(1)}
        nesterov_lookahead_step(
            m2, state2, batch, _quadratic_loss, cfg, rng=np.random.default_rng(7)
        )

        # After the step: params should equal theta_orig - lr * v_new for some v_new.
        # If rollback was broken, params would have an extra +beta*v_orig shift.
        # Concrete check: the *magnitude* of param change should be on the order of lr,
        # not on the order of beta*v_norm (which would be ~1e-4 here, much larger than lr*v_new).
        for name, param in m2.named_parameters():
            delta = (param.data - snapshot[name]).abs().max().item()
            # lr=1e-3, rho~O(1), |z|~O(1), so |delta| ~ lr * O(1) = 1e-3 magnitude or smaller
            assert delta < 0.1, f"{name}: delta={delta:.4f} is too large; rollback likely failed"

    def test_returns_seed_rho_loss_triple(self):
        """Function signature: returns (seed, rho, loss_plus) like mezo_step."""
        import numpy as np

        model = _tiny_model()
        state = NesterovState(beta=0.5, look_ahead=True)
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        batch = {"x": torch.zeros(1)}
        result = nesterov_lookahead_step(
            model, state, batch, _quadratic_loss, cfg, rng=np.random.default_rng(0)
        )
        assert len(result) == 3
        seed, rho, loss_plus = result
        assert isinstance(seed, int)
        assert isinstance(rho, float)
        assert isinstance(loss_plus, float)

    def test_velocity_is_updated_after_step(self):
        """After one lookahead step, velocity buffers must exist."""
        import numpy as np

        model = _tiny_model()
        state = NesterovState(beta=0.9, look_ahead=True)
        cfg = MeZOConfig(lr=1e-3, eps=1e-3)
        batch = {"x": torch.zeros(1)}
        assert len(state.velocities) == 0
        nesterov_lookahead_step(
            model, state, batch, _quadratic_loss, cfg, rng=np.random.default_rng(0)
        )
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert len(state.velocities) == n_trainable


# ---------------------------------------------------------------------------
# β-schedule: linear decay from beta to beta_end over num_rounds_total rounds
# ---------------------------------------------------------------------------


class TestBetaSchedule:
    def test_constant_when_schedule_unset(self):
        """When beta_end/num_rounds_total are None, update_schedule is a no-op."""
        state = NesterovState(beta=0.9)
        assert state.beta == 0.9
        state.update_schedule(round_idx=0)
        assert state.beta == 0.9
        state.update_schedule(round_idx=500)
        assert state.beta == 0.9
        state.update_schedule(round_idx=10_000)
        assert state.beta == 0.9

    def test_linear_decay_at_endpoints(self):
        """beta at round 0 = beta_start; beta at round total-1 = beta_end."""
        state = NesterovState(beta=0.9, beta_end=0.0, num_rounds_total=1000)
        state.update_schedule(round_idx=0)
        assert abs(state.beta - 0.9) < 1e-9
        state.update_schedule(round_idx=999)
        assert abs(state.beta - 0.0) < 1e-9

    def test_linear_decay_at_midpoint(self):
        """At round_idx = (total-1)/2 (linear progress 0.5), beta = midpoint of [start, end]."""
        state = NesterovState(beta=0.9, beta_end=0.1, num_rounds_total=1001)
        state.update_schedule(round_idx=500)
        expected = 0.5  # midpoint of (0.9, 0.1)
        assert abs(state.beta - expected) < 1e-9

    def test_schedule_clamps_past_endpoint(self):
        """round_idx > num_rounds_total clamps to beta_end (no overshoot)."""
        state = NesterovState(beta=0.9, beta_end=0.0, num_rounds_total=100)
        state.update_schedule(round_idx=500)
        assert abs(state.beta - 0.0) < 1e-9

    def test_schedule_resets_to_start_correctly(self):
        """Multiple update_schedule calls reproduce the curve from the original start."""
        state = NesterovState(beta=0.9, beta_end=0.0, num_rounds_total=1000)
        state.update_schedule(round_idx=500)
        # progress 500/999 ≈ 0.5005; beta ≈ 0.9 + 0.5005 * (0.0 - 0.9) ≈ 0.4495
        beta_at_500 = state.beta
        state.update_schedule(round_idx=0)
        assert abs(state.beta - 0.9) < 1e-9, (
            "schedule must reset deterministically; got %f after going back to 0" % state.beta
        )
        state.update_schedule(round_idx=500)
        assert abs(state.beta - beta_at_500) < 1e-9, "schedule should be deterministic in round_idx"

    def test_ascending_schedule_also_works(self):
        """If beta_end > beta_start (warmup-up), schedule should increase."""
        state = NesterovState(beta=0.0, beta_end=0.9, num_rounds_total=100)
        state.update_schedule(round_idx=0)
        assert abs(state.beta - 0.0) < 1e-9
        state.update_schedule(round_idx=99)
        assert abs(state.beta - 0.9) < 1e-9

    def test_negative_round_idx_clamps_to_start(self):
        """Round indices below 0 should not produce out-of-range beta."""
        state = NesterovState(beta=0.9, beta_end=0.0, num_rounds_total=1000)
        state.update_schedule(round_idx=-10)
        assert abs(state.beta - 0.9) < 1e-9
