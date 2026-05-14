"""Tests for the federated simulator (full-round behaviour)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dmezo.federated.simulator import SimulatorConfig, run_simulation
from dmezo.federated.topology import MixingMatrix
from dmezo.mezo.nesterov import NesterovState
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update
from tests._fixtures import (
    make_tiny_causal_lm,
    make_tiny_clients,
    synthetic_token_loader,
)


def _causal_lm_loss(model, batch):
    """Inline loss fn that matches dmezo.data.superglue.causal_lm_loss interface."""
    out = model(**batch)
    return out.loss


def _self_loop_topology():
    """A 1-node 'topology' with W=[[1]] for n=1 tests."""
    return MixingMatrix(W=np.array([[1.0]]), name="self_loop")


def test_no_double_update_in_update_share():
    """One round with W=[[1]] under update_share must equal one mezo_update step.

    This is the regression test for B2 (double-update bug). Before the fix,
    local_round applies the update AND consensus_via_updates applies it again,
    so the final theta is theta - 2*lr*(rho*z + ...).

    The test strategy:
    - Build two models with identical initial weights (same seed).
    - Reference path: run mezo_step on a clone of client 0's model, then apply
      exactly ONE mezo_update with the exact (seed, rho) produced.
    - Simulator path: run 1 round of run_simulation in update_share mode on the
      client (which holds a model with the same initial weights).
    - Compare final parameters: they must match to atol=1e-6.
    """
    # Fixed seeds for full determinism.
    MODEL_SEED = 42
    RNG_SEED = 7
    DATA_SEED = 100

    # Build client with deterministic RNG.
    cfg = MeZOConfig(lr=1e-3, eps=1e-3, weight_decay=0.0)
    model_client = make_tiny_causal_lm(seed=MODEL_SEED)
    for p in model_client.parameters():
        p.requires_grad_(True)
    loader = synthetic_token_loader(seed=DATA_SEED)

    from dmezo.federated.client import ClientState

    client = ClientState(
        client_id=0,
        model=model_client,
        dataloader=loader,
        mezo_config=cfg,
        rng=np.random.default_rng(RNG_SEED),
    )

    # Reference model: identical initial weights.
    reference_model = make_tiny_causal_lm(seed=MODEL_SEED)
    for p in reference_model.parameters():
        p.requires_grad_(True)

    # Verify identical starting weights.
    for (nr, pr), (nc, pc) in zip(
        reference_model.named_parameters(), client.model.named_parameters()
    ):
        assert nr == nc
        assert torch.allclose(pr.data, pc.data, atol=1e-12), (
            f"Pre-condition failed: reference and client models differ at {nr!r}"
        )

    # Pre-compute what one mezo_step would produce on this client's first batch.
    # We need to use the same RNG state as the client will use during run_simulation.
    ref_rng = np.random.default_rng(RNG_SEED)
    first_batch = next(iter(synthetic_token_loader(seed=DATA_SEED)))
    seed_ref, rho_ref, _ = mezo_step(
        reference_model, first_batch, _causal_lm_loss, cfg, rng=ref_rng
    )
    # Apply exactly ONE update on the reference path.
    mezo_update(reference_model, seed=seed_ref, projected_grad=rho_ref, config=cfg)

    # Simulator path: one round in update_share mode.
    # Client's rng must be reset to the SAME state as ref_rng was before mezo_step.
    # It already is because we constructed client.rng with RNG_SEED before this block.
    sim_cfg = SimulatorConfig(
        num_rounds=1,
        consensus_mode="update_share",
        eval_every=0,
        log_every=0,
    )
    run_simulation(
        clients=[client],
        topology=_self_loop_topology(),
        loss_fn=_causal_lm_loss,
        config=sim_cfg,
    )

    # Compare params: simulator path must equal exactly one mezo_update.
    for (nr, pr), (nc, pc) in zip(
        reference_model.named_parameters(), client.model.named_parameters()
    ):
        assert nr == nc
        assert torch.allclose(pr.data, pc.data, atol=1e-6), (
            f"Param {nr!r}: simulator path diverged from single mezo_update. "
            f"Likely double-update bug (B2). "
            f"Max abs diff: {(pr.data - pc.data).abs().max().item():.6e}"
        )


def test_update_share_requires_local_steps_eq_1():
    """update_share with local_steps > 1 must fail early (otherwise silently drops grads)."""
    clients = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)
    clients[0].local_steps = 2  # multi-step local would silently lose the first (seed, rho)

    cfg = SimulatorConfig(
        num_rounds=1,
        consensus_mode="update_share",
        eval_every=0,
        log_every=0,
    )
    with pytest.raises(ValueError, match="local_steps"):
        run_simulation(
            clients=clients,
            topology=_self_loop_topology(),
            loss_fn=_causal_lm_loss,
            config=cfg,
        )


def test_nesterov_update_share_raises():
    """Nesterov + update_share must raise NotImplementedError (guard for D1)."""
    clients = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)
    # Attach a Nesterov state to the client.
    clients[0].nesterov_state = NesterovState(beta=0.9)

    cfg = SimulatorConfig(
        num_rounds=1,
        consensus_mode="update_share",
        eval_every=0,
        log_every=0,
    )
    with pytest.raises(NotImplementedError, match="Nesterov"):
        run_simulation(
            clients=clients,
            topology=_self_loop_topology(),
            loss_fn=_causal_lm_loss,
            config=cfg,
        )


def test_weight_avg_mode_unaffected():
    """weight_avg mode must still converge toward centroid after one round.

    Regression guard: F2 changes must not break the weight_avg code path.
    After one round on a complete graph (W = [[0.5, 0.5], [0.5, 0.5]]),
    both clients must share the same parameters (they should be averaged).
    """
    from dmezo.federated.topology import complete_graph

    n = 2
    clients = make_tiny_clients(n=n, mezo_lr=1e-3, mezo_eps=1e-3, seed_offset=0)

    cfg = SimulatorConfig(
        num_rounds=1,
        consensus_mode="weight_avg",
        eval_every=0,
        log_every=0,
    )
    run_simulation(
        clients=clients,
        topology=complete_graph(n),
        loss_fn=_causal_lm_loss,
        config=cfg,
    )

    # After complete-graph weight_avg, all clients must have identical parameters.
    for (n0, p0), (n1, p1) in zip(
        clients[0].model.named_parameters(), clients[1].model.named_parameters()
    ):
        assert n0 == n1
        assert torch.allclose(p0.data, p1.data, atol=1e-6), (
            f"weight_avg broken: clients differ on param {n0!r} after complete-graph consensus"
        )


def test_full_round_runs_on_tiny_model():
    """End-to-end smoke: 2 clients, ring(2), 5 rounds of update_share runs cleanly."""
    from dmezo.federated.topology import ring_graph

    n = 2
    clients = make_tiny_clients(n=n, mezo_lr=1e-3, same_init=True)

    initial_params = [
        {name: p.data.clone() for name, p in c.model.named_parameters()} for c in clients
    ]

    cfg = SimulatorConfig(num_rounds=5, consensus_mode="update_share", eval_every=0, log_every=0)
    logs = run_simulation(
        clients=clients,
        topology=ring_graph(n),
        loss_fn=_causal_lm_loss,
        config=cfg,
    )

    assert len(logs) == 5
    for entry in logs:
        assert "mean_local_loss" in entry
        assert "mean_projected_grad" in entry
        assert np.isfinite(entry["mean_local_loss"])
        assert np.isfinite(entry["mean_projected_grad"])

    # At least one parameter changed for each client (params were not frozen).
    for i, c in enumerate(clients):
        changed = False
        for name, p in c.model.named_parameters():
            if not torch.allclose(p.data, initial_params[i][name]):
                changed = True
                break
        assert changed, f"Client {i} params unchanged after 5 rounds"
