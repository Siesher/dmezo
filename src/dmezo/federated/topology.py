"""Mixing matrices for decentralized consensus.

Each topology produces a doubly-stochastic, symmetric mixing matrix W with
spectral gap rho = ||W - 11^T/n||_2 < 1. Smaller rho => faster consensus.

References:
    - Koloskova et al. 2020, "A Unified Theory of Decentralized SGD", arXiv:2003.10422.
    - Nedic, Olshevsky, Shi 2017, "DIGing", SIAM J. Optim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixingMatrix:
    """Container for a mixing matrix and metadata.

    Attributes:
        W: Doubly-stochastic, symmetric n x n matrix.
        name: Human-readable topology name.
        n: Number of nodes.
    """

    W: np.ndarray
    name: str

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def spectral_gap(self) -> float:
        """Return rho = ||W - 11^T/n||_2."""
        return spectral_gap(self.W)

    def __repr__(self) -> str:
        return f"MixingMatrix(name={self.name!r}, n={self.n}, rho={self.spectral_gap():.4f})"


def spectral_gap(W: np.ndarray) -> float:
    """Compute the spectral gap rho = ||W - (1/n) * 11^T||_2 (largest non-trivial sv)."""
    n = W.shape[0]
    P = W - np.ones((n, n)) / n
    # For symmetric W: spectral norm = max(|eigvals|)
    eigvals = np.linalg.eigvalsh(P)
    return float(np.max(np.abs(eigvals)))


def _metropolis_weights(adjacency: np.ndarray) -> np.ndarray:
    """Standard Metropolis-Hastings weights for a symmetric adjacency matrix.

    For each edge (i, j): W[i, j] = 1 / (1 + max(deg(i), deg(j))).
    Diagonal: W[i, i] = 1 - sum_{j != i} W[i, j].

    This guarantees doubly-stochastic + symmetric.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1) - np.diag(adjacency)  # exclude self-loops in count
    W = np.zeros_like(adjacency, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def ring_graph(n: int) -> MixingMatrix:
    """n-node ring with Metropolis weights. rho ~ 1 - O(1/n^2) for large n."""
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[i, (i - 1) % n] = 1
    W = _metropolis_weights(adj)
    return MixingMatrix(W=W, name=f"ring(n={n})")


def complete_graph(n: int) -> MixingMatrix:
    """Complete graph: W = (1/n) * 11^T. rho = 0 (one-step consensus)."""
    W = np.full((n, n), 1.0 / n)
    return MixingMatrix(W=W, name=f"complete(n={n})")


def random_regular(n: int, degree: int, seed: int = 0) -> MixingMatrix:
    """Random d-regular graph with Metropolis weights.

    Useful for moderate rho between ring and complete. Requires n*degree even.
    """
    if (n * degree) % 2 != 0:
        raise ValueError("n * degree must be even for a regular graph to exist")
    rng = np.random.default_rng(seed)
    # Simple configuration-model construction with rejection of self-loops/multi-edges.
    for _ in range(200):
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        adj = np.zeros((n, n), dtype=int)
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or adj[a, b] == 1:
                ok = False
                break
            adj[a, b] = 1
            adj[b, a] = 1
        if ok:
            W = _metropolis_weights(adj)
            return MixingMatrix(W=W, name=f"random_regular(n={n},d={degree})")
    raise RuntimeError("Failed to sample a valid random regular graph; try a different seed")


def star_graph(n: int) -> MixingMatrix:
    """Star topology centered at node 0 (FedAvg-like sanity baseline).

    Note: with Metropolis weights, star has poor mixing for large n. Provided
    as a baseline to compare against, not as a serious topology.
    """
    adj = np.zeros((n, n), dtype=int)
    for j in range(1, n):
        adj[0, j] = 1
        adj[j, 0] = 1
    W = _metropolis_weights(adj)
    return MixingMatrix(W=W, name=f"star(n={n})")
