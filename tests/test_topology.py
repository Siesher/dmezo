"""Tests for mixing matrices."""

from __future__ import annotations

import numpy as np
import pytest

from dmezo.federated.topology import (
    complete_graph,
    random_regular,
    ring_graph,
    spectral_gap,
    star_graph,
)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ring_is_doubly_stochastic(n: int):
    M = ring_graph(n)
    W = M.W
    assert W.shape == (n, n)
    assert np.allclose(W.sum(axis=0), 1.0)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W, W.T)


@pytest.mark.parametrize("n", [4, 8])
def test_complete_has_zero_spectral_gap(n: int):
    M = complete_graph(n)
    rho = M.spectral_gap()
    assert rho < 1e-8, f"Complete graph should have rho ~ 0, got {rho}"


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ring_has_positive_gap(n: int):
    """Ring should have rho between 0 and 1 (strictly)."""
    M = ring_graph(n)
    rho = M.spectral_gap()
    assert 0 < rho < 1, f"Ring rho out of bounds: {rho}"


def test_random_regular_doubly_stochastic():
    M = random_regular(n=8, degree=3, seed=0)
    W = M.W
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W, W.T)


def test_star_topology():
    M = star_graph(n=5)
    W = M.W
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W, W.T)
    # Non-center nodes should only connect to center (0).
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                assert W[i, j] == 0.0


def test_spectral_gap_function():
    W = np.full((3, 3), 1.0 / 3)
    assert abs(spectral_gap(W)) < 1e-8
