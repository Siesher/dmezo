"""Per-client federated MeZO logic.

Each client owns a copy of the model weights (or LoRA adapters) and runs
local MeZO steps on its local data shard. Between rounds, clients exchange
``(seed, projected_grad)`` pairs with their neighbors per the mixing matrix.

This is in-process simulation: we represent all clients as objects in one
Python process, sharing one GPU. A real federated deployment would replace
``ClientState`` with a network-aware variant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dmezo.mezo.nesterov import NesterovState, nesterov_lookahead_step, nesterov_step
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update


@dataclass
class ClientState:
    """State of one federated client.

    Attributes:
        client_id: 0-indexed client identifier.
        model: Local copy of the model. For LoRA-mode, ``model`` should be the
            full model with LoRA adapters and only LoRA params having requires_grad=True.
        dataloader: Iterable yielding local batches.
        mezo_config: MeZO hyperparameters for this client.
        nesterov_state: Optional Nesterov velocity state. If None, use plain MeZO.
        local_steps: Number of MeZO steps per federated round.
        rng: Per-client numpy Generator for seed sampling.
    """

    client_id: int
    model: nn.Module
    dataloader: DataLoader
    mezo_config: MeZOConfig
    local_steps: int = 1
    nesterov_state: NesterovState | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    _data_iter: object | None = None

    def _next_batch(self) -> dict:
        """Cycle through dataloader, restarting on exhaustion."""
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def local_round(
        self,
        loss_fn: Callable[[nn.Module, dict], torch.Tensor],
        *,
        apply: bool = True,
        round_idx: int = 0,
    ) -> list[tuple[int, float, float]]:
        """Execute ``local_steps`` MeZO steps locally.

        Args:
            loss_fn: Loss function ``(model, batch) -> scalar tensor``.
            apply: If True (default), apply each MeZO update in-place via
                ``mezo_update`` (or ``nesterov_step`` if ``nesterov_state`` is set).
                If False, return ``(seed, rho, loss_plus)`` triples without
                mutating parameters — use with ``consensus_via_updates``, which
                owns the eventual parameter mutation.
            round_idx: Current federated round index. Forwarded to
                ``NesterovState.update_schedule`` so β can be β(t) when a
                schedule is configured. Default 0 (no-op when schedule is off).

        Returns:
            List of ``(seed, projected_grad, loss_plus)`` from each local step.
        """
        # Update β from schedule before any nesterov_step / nesterov_lookahead_step.
        if self.nesterov_state is not None:
            self.nesterov_state.update_schedule(round_idx)
        history: list[tuple[int, float, float]] = []
        for _ in range(self.local_steps):
            batch = self._next_batch()
            if apply and self.nesterov_state is not None and self.nesterov_state.look_ahead:
                # Look-ahead Nesterov: MeZO probes at theta + beta*v, not theta.
                # The function handles the shift, mezo step, rollback, and
                # velocity update in one call.
                seed, rho, loss_plus = nesterov_lookahead_step(
                    self.model,
                    self.nesterov_state,
                    batch,
                    loss_fn,
                    self.mezo_config,
                    rng=self.rng,
                )
            else:
                seed, rho, loss_plus = mezo_step(
                    self.model, batch, loss_fn, self.mezo_config, rng=self.rng
                )
                if apply:
                    if self.nesterov_state is not None:
                        nesterov_step(
                            self.model,
                            self.nesterov_state,
                            seed=seed,
                            projected_grad=rho,
                            lr=self.mezo_config.lr,
                            weight_decay=self.mezo_config.weight_decay,
                        )
                    else:
                        mezo_update(
                            self.model,
                            seed=seed,
                            projected_grad=rho,
                            config=self.mezo_config,
                        )
            history.append((seed, rho, loss_plus))
        return history
