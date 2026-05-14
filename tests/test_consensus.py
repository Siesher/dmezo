"""Tests for federated consensus mixing."""

from __future__ import annotations

import torch
from torch import nn

from dmezo.federated.consensus import consensus_via_weights
from dmezo.federated.topology import complete_graph
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
