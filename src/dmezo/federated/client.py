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

from dmezo.mezo.nesterov import (
    NesterovState,
    md_nesterov_step,
    nesterov_lookahead_step,
    nesterov_step,
)
from dmezo.mezo.step import (
    AdaptiveClipState,
    MeZOConfig,
    md_mezo_step,
    md_mezo_update,
    mezo_step,
    mezo_update,
)


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
        adaptive_clip_state: B1 — optional running-quantile ρ-clipper. When set,
            its ``current_threshold()`` supersedes ``mezo_config.rho_clip`` for
            ``mezo_step``. Default None = use fixed ``mezo_config.rho_clip``.
    """

    client_id: int
    model: nn.Module
    dataloader: DataLoader
    mezo_config: MeZOConfig
    local_steps: int = 1
    nesterov_state: NesterovState | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    adaptive_clip_state: AdaptiveClipState | None = None
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
        K = max(1, int(getattr(self.mezo_config, "k_directions", 1)))
        for _ in range(self.local_steps):
            batch = self._next_batch()
            if K > 1:
                # Multi-direction MeZO (MD-D-MeZO-N path).
                # Look-ahead Nesterov is not yet supported with K>1 — explicit
                # error rather than silent fallback for clarity.
                if (
                    apply
                    and self.nesterov_state is not None
                    and self.nesterov_state.look_ahead
                ):
                    raise NotImplementedError(
                        "Look-ahead Nesterov + K>1 (MD) is not yet implemented. "
                        "Either set look_ahead=False (heavy-ball) or k_directions=1."
                    )
                seeds_list, rhos_list, loss_plus = md_mezo_step(
                    self.model, batch, loss_fn, self.mezo_config, rng=self.rng
                )
                # For per-step history we log the MEAN ρ (aggregated estimator).
                rho_mean = sum(rhos_list) / len(rhos_list)
                seed_repr = seeds_list[0]  # first seed for logging only
                if apply:
                    if self.nesterov_state is not None:
                        md_nesterov_step(
                            self.model,
                            self.nesterov_state,
                            seeds=seeds_list,
                            projected_grads=rhos_list,
                            lr=self.mezo_config.lr,
                            weight_decay=self.mezo_config.weight_decay,
                        )
                    else:
                        md_mezo_update(
                            self.model,
                            seeds=seeds_list,
                            projected_grads=rhos_list,
                            config=self.mezo_config,
                        )
                history.append((seed_repr, rho_mean, loss_plus))
            elif apply and self.nesterov_state is not None and self.nesterov_state.look_ahead:
                # Look-ahead Nesterov: MeZO probes at theta + beta*v, not theta.
                # NOTE: adaptive clip + DP currently NOT wired into the look-ahead
                # path (separate sub-step within nesterov_lookahead_step). Future work.
                seed, rho, loss_plus = nesterov_lookahead_step(
                    self.model,
                    self.nesterov_state,
                    batch,
                    loss_fn,
                    self.mezo_config,
                    rng=self.rng,
                )
                history.append((seed, rho, loss_plus))
            else:
                # B1: adaptive clip — compute threshold from running window
                # BEFORE this step, so the threshold uses prior |ρ| history only
                # (current ρ not yet in the buffer).
                rho_clip_override = None
                if self.adaptive_clip_state is not None:
                    rho_clip_override = self.adaptive_clip_state.current_threshold()
                # D2: DP noise — taken from config; override unused (use config).
                seed, rho, loss_plus = mezo_step(
                    self.model, batch, loss_fn, self.mezo_config,
                    rng=self.rng,
                    rho_clip_override=rho_clip_override,
                )
                # B1: append this step's |ρ| to the running window (for the
                # NEXT step's threshold). Note: this is the POST-clip (and
                # post-DP if enabled) ρ — that's intentional, the threshold
                # tracks the distribution of values that actually went into
                # the update.
                if self.adaptive_clip_state is not None:
                    self.adaptive_clip_state.update(abs(rho))
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
                        # B5: drift detection — check AFTER applying the step,
                        # using loss_plus as a proxy for the current trajectory
                        # health. If drift detected, the velocity buffer is
                        # zeroed (caller can read state.n_resets for diagnostics).
                        self.nesterov_state.check_drift_and_reset(loss_plus)
                    else:
                        mezo_update(
                            self.model,
                            seed=seed,
                            projected_grad=rho,
                            config=self.mezo_config,
                        )
                history.append((seed, rho, loss_plus))
        return history
