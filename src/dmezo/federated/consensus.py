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

from typing import List

import numpy as np
import torch
from torch import nn

from dmezo.mezo.step import MeZOConfig, _collect_optim_params
from dmezo.federated.client import ClientState


def consensus_via_weights(clients: List[ClientState], W: np.ndarray) -> None:
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
    clients: List[ClientState],
    W: np.ndarray,
    seeds: List[int],
    projected_grads: List[float],
    config: MeZOConfig,
) -> None:
    """Apply consensus by exchanging (seed, rho) pairs and locally combining updates.

    This is the *communication-efficient* consensus: each client receives from
    its neighbors only ``(seed_j, rho_j)`` pairs. Then for each parameter p_i:

        p_i^{new} = p_i - lr * sum_j W[i, j] * (rho_j * z_{seed_j} + decay * p_i)

    Each client regenerates ``z_{seed_j}`` locally from the seed. Total bandwidth
    per round: O(n^2) scalars in the worst case (every pair exchanges), but with
    sparse W only O(|E|) scalars are actually transmitted.

    Args:
        clients: List of ClientState.
        W: Mixing matrix.
        seeds: Per-client seeds from this round's local step (assumes local_steps=1).
        projected_grads: Per-client projected gradients.
        config: MeZO config (for lr, weight_decay).

    Note:
        Currently supports ``local_steps == 1`` for the simplest analysis.
        For local_steps > 1, clients accumulate (seed, rho) sequences and apply
        them in order — left as a follow-up implementation.
    """
    n = len(clients)
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}")
    if len(seeds) != n or len(projected_grads) != n:
        raise ValueError("seeds and projected_grads must have len == n_clients")

    for i, client in enumerate(clients):
        for name, param in client.model.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            decay = (
                config.weight_decay
                if ("bias" not in lname) and ("layer_norm" not in lname) and ("layernorm" not in lname)
                else 0.0
            )
            # Accumulate weighted update.
            update = torch.zeros_like(param.data)
            for j in range(n):
                wij = float(W[i, j])
                if wij == 0:
                    continue
                # Regenerate z_j from seed_j.
                # NOTE: torch.manual_seed sets a global state, so we must replay
                # the perturbation in the same order as the client did.
                torch.manual_seed(int(seeds[j]))
                # Iterate parameters in same order, advancing global RNG.
                for n_, p_ in client.model.named_parameters():
                    if not p_.requires_grad:
                        continue
                    z = torch.normal(
                        mean=0.0, std=1.0,
                        size=p_.data.size(),
                        device=p_.data.device,
                        dtype=p_.data.dtype,
                    )
                    if n_ == name:
                        update.add_(z, alpha=wij * float(projected_grads[j]))
                        break
                else:
                    continue
            update.add_(param.data, alpha=decay)
            param.data.add_(update, alpha=-config.lr)
