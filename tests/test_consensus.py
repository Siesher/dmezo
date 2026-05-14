"""Tests for federated consensus mixing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dmezo.federated.consensus import consensus_via_updates, consensus_via_weights
from dmezo.federated.topology import complete_graph, ring_graph
from dmezo.mezo.step import mezo_update
from tests._fixtures import make_tiny_clients


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.data.clone() for name, p in model.named_parameters()}


def test_via_weights_averages_params_on_complete_graph():
    """One step of consensus_via_weights with complete graph hits the centroid."""
    n = 4
    clients = make_tiny_clients(n=n)
    snapshots = [_snapshot(c.model) for c in clients]

    # Expected per-name centroid across clients.
    param_names = [name for name, _ in clients[0].model.named_parameters()]
    expected: dict[str, torch.Tensor] = {
        name: torch.stack([snap[name] for snap in snapshots]).mean(dim=0) for name in param_names
    }

    W = complete_graph(n).W
    consensus_via_weights(clients, W)

    for i, c in enumerate(clients):
        for name, p in c.model.named_parameters():
            assert torch.allclose(p.data, expected[name], atol=1e-6), (
                f"Client {i} param {name!r} did not match centroid"
            )


def test_via_updates_n1_equivalent_to_mezo_update():
    """For n=1, W=[[1]], consensus_via_updates should match a direct mezo_update."""
    clients_a = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)
    clients_b = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)

    # Sanity: identical initial params.
    for (na, pa), (nb, pb) in zip(
        clients_a[0].model.named_parameters(), clients_b[0].model.named_parameters()
    ):
        assert na == nb
        assert torch.allclose(pa.data, pb.data, atol=1e-12)

    seed = 12345
    rho = 0.7
    cfg = clients_a[0].mezo_config

    # Path A: direct mezo_update.
    mezo_update(clients_a[0].model, seed=seed, projected_grad=rho, config=cfg)

    # Path B: consensus_via_updates with W=[[1]], same (seed, rho).
    W = np.array([[1.0]])
    consensus_via_updates(clients_b, W, seeds=[seed], projected_grads=[rho], config=cfg)

    for (na, pa), (nb, pb) in zip(
        clients_a[0].model.named_parameters(), clients_b[0].model.named_parameters()
    ):
        assert torch.allclose(pa.data, pb.data, atol=1e-6), (
            f"Param {na!r}: mezo_update and consensus_via_updates diverged"
        )


def test_via_updates_deterministic_under_replay():
    """Same (seeds, rhos) on the same initial params should produce the same outputs."""
    clients_a = make_tiny_clients(n=2)
    clients_b = make_tiny_clients(n=2)

    seeds = [111, 222]
    rhos = [0.3, -0.5]
    W = ring_graph(2).W
    cfg = clients_a[0].mezo_config

    consensus_via_updates(clients_a, W, seeds=seeds, projected_grads=rhos, config=cfg)
    consensus_via_updates(clients_b, W, seeds=seeds, projected_grads=rhos, config=cfg)

    for ca, cb in zip(clients_a, clients_b):
        for (na, pa), (nb, pb) in zip(ca.model.named_parameters(), cb.model.named_parameters()):
            assert na == nb
            assert torch.allclose(pa.data, pb.data, atol=1e-9), f"Replay diverged on {na!r}"


def test_via_updates_complete_graph_consensus():
    """Same-init clients on a complete graph stay close after many update_share rounds.

    With identical initial params and a complete (rho=0) graph, all clients
    receive the same averaged update each round, so they should stay bit-identical.
    This validates that the F2 apply-flag plumbing is wired correctly via the
    public API: a single ``run_simulation`` call.
    """
    from dmezo.federated.simulator import SimulatorConfig, run_simulation

    n = 4
    clients = make_tiny_clients(n=n, mezo_lr=5e-4, same_init=True)
    cfg = SimulatorConfig(num_rounds=20, consensus_mode="update_share", eval_every=0, log_every=0)
    run_simulation(
        clients=clients,
        topology=complete_graph(n),
        loss_fn=_causal_lm_loss_local,
        config=cfg,
    )

    # All clients should have nearly identical parameters (rho=0 → one-step consensus).
    ref = {name: p.data for name, p in clients[0].model.named_parameters()}
    for i in range(1, n):
        for name, p in clients[i].model.named_parameters():
            diff = (p.data - ref[name]).abs().max().item()
            assert diff < 1e-4, (
                f"Client {i} param {name!r}: max abs diff {diff} > 1e-4 from client 0"
            )


def _causal_lm_loss_local(model, batch):
    """Local copy of the loss fn (avoids cross-test import coupling)."""
    out = model(**batch)
    return out.loss
