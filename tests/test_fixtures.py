"""Sanity tests for shared fixtures."""

from __future__ import annotations

import torch

from tests._fixtures import (
    make_tiny_causal_lm,
    make_tiny_clients,
    synthetic_token_loader,
)


def test_tiny_model_produces_finite_loss():
    model = make_tiny_causal_lm(seed=0)
    loader = synthetic_token_loader(num_examples=8, batch_size=4)
    batch = next(iter(loader))
    out = model(**batch)
    assert torch.isfinite(out.loss).item(), f"Expected finite loss, got {out.loss}"
    assert out.loss.ndim == 0, f"Loss must be a scalar tensor, got shape {out.loss.shape}"


def test_make_tiny_clients_returns_n_clients_with_distinct_params():
    clients = make_tiny_clients(n=3)
    assert len(clients) == 3
    p0 = list(clients[0].model.parameters())[0].data
    p1 = list(clients[1].model.parameters())[0].data
    assert not torch.allclose(p0, p1), "Clients with different seeds should have different params"


def test_make_tiny_clients_same_init_makes_identical_params():
    clients = make_tiny_clients(n=2, same_init=True)
    for (n0, p0), (n1, p1) in zip(
        clients[0].model.named_parameters(), clients[1].model.named_parameters()
    ):
        assert n0 == n1
        assert torch.allclose(p0.data, p1.data, atol=1e-12), f"Param {n0} differs"
