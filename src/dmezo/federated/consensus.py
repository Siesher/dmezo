"""Consensus mixing for decentralized D-MeZO.

After all clients perform local steps, they exchange parameters with neighbors
per the mixing matrix W:

    theta_i^{new} = sum_j W[i, j] * theta_j

For a doubly-stochastic, symmetric W, this drives all theta_i toward the
average while each client only communicates with its neighbors (W[i, j] != 0).

In a real distributed deployment, this would use gossip protocols. Here, since
all clients are in one process on one GPU, we perform the averaging directly.

For seed-only D-MeZO with NO parameter exchange (the most communication-
efficient variant): clients exchange only ``(seed, projected_grad)`` pairs and
each client locally applies a *weighted sum of neighbor updates*. This is
implemented in ``consensus_via_updates`` and is the main novel contribution
direction of this project.
"""

from __future__ import annotations

import numpy as np
import torch

from dmezo.federated.client import ClientState
from dmezo.mezo.step import MeZOConfig, _collect_optim_params, _is_decay_param


def consensus_via_weights(clients: list[ClientState], W: np.ndarray) -> None:
    """Apply consensus by directly averaging client parameters per W.

    For each parameter p:
        p_i^{new} = sum_j W[i, j] * p_j

    This is the standard D-SGD consensus step. Bandwidth-heavy in a real
    deployment, but useful as a baseline to compare against the update-based
    consensus below.

    Args:
        clients: List of ClientState objects (must all have the same parameter shapes).
        W: n x n doubly-stochastic mixing matrix.

    Note:
        All parameter tensors are duplicated in memory temporarily during the
        weighted sum. For tight memory budgets, do this parameter-by-parameter
        and free intermediates.
    """
    n = len(clients)
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}, got {W.shape}")

    # Build (name -> [tensor per client]) snapshots.
    param_names = [name for name, _ in _collect_optim_params(clients[0].model)]
    for name in param_names:
        # Gather copies.
        tensors = []
        for c in clients:
            for n_, p_ in c.model.named_parameters():
                if n_ == name:
                    tensors.append(p_.data.clone())
                    break
        # Weighted sum: new_tensors[i] = sum_j W[i, j] * tensors[j]
        new_tensors = [torch.zeros_like(tensors[0]) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    new_tensors[i].add_(tensors[j], alpha=float(W[i, j]))
        # Write back.
        for i, c in enumerate(clients):
            for n_, p_ in c.model.named_parameters():
                if n_ == name:
                    p_.data.copy_(new_tensors[i])
                    break


def consensus_via_updates(
    clients: list[ClientState],
    W: np.ndarray,
    seeds: list[int],
    projected_grads: list[float],
    config: MeZOConfig,
) -> None:
    """Apply consensus by exchanging (seed, rho) pairs and locally combining updates.

    For each client ``i``, computes::

        theta_i <- theta_i - lr * (sum_j W[i, j] * rho_j * z(seed_j) + decay * theta_i)

    where ``z(seed_j)`` is regenerated locally and ``decay`` is the
    weight_decay coefficient (applied only to non-bias / non-norm params).

    Complexity per round: O(n_clients * n_neighbors * p), where ``p`` is the
    number of trainable parameters. Each ``z`` is generated exactly once per
    (i, j, p_k) triple.

    Args:
        clients: All client states. Must share parameter order and shapes.
        W: ``n x n`` doubly-stochastic mixing matrix.
        seeds: Per-client seeds from the round's MeZO step (``len == n_clients``).
        projected_grads: Per-client projected gradients (``len == n_clients``).
        config: MeZO config (``lr``, ``weight_decay``).

    Note:
        Assumes ``local_steps == 1`` per round and that
        ``ClientState.local_round`` was called with ``apply=False`` so that no
        local update has been applied yet — this function is the single owner
        of parameter mutation in ``consensus_mode="update_share"``.

        Side effect: ``torch.manual_seed`` is called inside the inner loop, so
        after this function returns the global torch RNG state reflects the
        last neighbor's seed. Callers that draw from the global torch RNG must
        re-seed (``mezo_step`` already does so internally).
    """
    n = len(clients)
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}, got {W.shape}")
    if len(seeds) != n or len(projected_grads) != n:
        raise ValueError("seeds and projected_grads must have len == n_clients")

    for i, client in enumerate(clients):
        named = [(name, p) for name, p in client.model.named_parameters() if p.requires_grad]
        if not named:
            continue

        accum = {name: torch.zeros_like(p.data) for name, p in named}

        # Each neighbor contributes W[i, j] * rho_j * z(seed_j) in a single
        # deterministic pass over named parameters.
        for j in range(n):
            wij = float(W[i, j])
            if wij == 0.0:
                continue
            coef = wij * float(projected_grads[j])
            torch.manual_seed(int(seeds[j]))
            for name, p in named:
                z = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=p.data.size(),
                    device=p.data.device,
                    dtype=p.data.dtype,
                )
                accum[name].add_(z, alpha=coef)

        # Apply: theta -= lr * (accum + decay * theta).
        for name, p in named:
            decay = config.weight_decay if _is_decay_param(name) else 0.0
            if decay != 0.0:
                accum[name].add_(p.data, alpha=decay)
            p.data.add_(accum[name], alpha=-config.lr)
