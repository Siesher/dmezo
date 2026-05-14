"""Frozen snapshot of the pre-refactor ``consensus_via_updates``.

Used only by ``test_via_updates_matches_legacy_implementation_on_tiny_model``
in test_consensus.py for one-time semantic equivalence verification.

DELETE THIS FILE after the bridge test has passed against the refactored
implementation (see docs/07-audit-harden.md Task 6).
"""

from __future__ import annotations

import numpy as np
import torch

from dmezo.federated.client import ClientState
from dmezo.mezo.step import MeZOConfig


def consensus_via_updates_legacy(
    clients: list[ClientState],
    W: np.ndarray,
    seeds: list[int],
    projected_grads: list[float],
    config: MeZOConfig,
) -> None:
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
                if ("bias" not in lname)
                and ("layer_norm" not in lname)
                and ("layernorm" not in lname)
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
                        mean=0.0,
                        std=1.0,
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
