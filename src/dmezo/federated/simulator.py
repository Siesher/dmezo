"""In-process simulator for decentralized federated MeZO.

Drives the main D-MeZO loop:

    for round in range(num_rounds):
        # 1. Each client does local_steps MeZO steps.
        for client in clients:
            history = client.local_round(loss_fn)

        # 2. Consensus mixing per W.
        if mode == "weight_avg":
            consensus_via_weights(clients, W)
        elif mode == "update_share":
            consensus_via_updates(clients, W, seeds, rhos, config)

        # 3. Optional eval.
        if round % eval_every == 0:
            metrics = evaluate(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from dmezo.federated.client import ClientState
from dmezo.federated.consensus import consensus_via_updates, consensus_via_weights
from dmezo.federated.topology import MixingMatrix


@dataclass
class SimulatorConfig:
    """Configuration for the federated simulator.

    Attributes:
        num_rounds: Total federated rounds.
        consensus_mode: "weight_avg" (full parameter exchange) or
            "update_share" (communication-efficient seed+scalar exchange).
        eval_every: Evaluate every N rounds. 0 disables eval.
        log_every: Log per-round summary every N rounds.
    """

    num_rounds: int = 100
    consensus_mode: str = "update_share"
    eval_every: int = 10
    log_every: int = 1


def run_simulation(
    clients: List[ClientState],
    topology: MixingMatrix,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: SimulatorConfig,
    eval_fn: Optional[Callable[[nn.Module, int], Dict[str, float]]] = None,
    logger: Optional[Callable[[Dict], None]] = None,
) -> List[Dict]:
    """Run the federated training loop.

    Args:
        clients: All client states. All clients must share parameter shapes.
        topology: Mixing matrix.
        loss_fn: Loss function applied to model + batch.
        config: Simulator config.
        eval_fn: Optional callable ``(model, round) -> dict`` for eval. Called
            on client 0's model.
        logger: Optional logger callable that receives per-round dicts.

    Returns:
        List of per-round logs.
    """
    if topology.n != len(clients):
        raise ValueError(
            f"Topology has {topology.n} nodes but {len(clients)} clients provided"
        )

    logs: List[Dict] = []
    for r in range(config.num_rounds):
        round_log: Dict = {"round": r}

        # 1. Local MeZO steps.
        all_seeds: List[int] = []
        all_rhos: List[float] = []
        all_losses: List[float] = []
        for c in clients:
            history = c.local_round(loss_fn)
            # For update_share mode, only local_steps=1 is currently supported.
            last_seed, last_rho, last_loss = history[-1]
            all_seeds.append(last_seed)
            all_rhos.append(last_rho)
            all_losses.append(last_loss)

        round_log["mean_local_loss"] = float(np.mean(all_losses))
        round_log["mean_projected_grad"] = float(np.mean(all_rhos))

        # 2. Consensus.
        if config.consensus_mode == "weight_avg":
            consensus_via_weights(clients, topology.W)
        elif config.consensus_mode == "update_share":
            # For update_share, clients already applied their local update during
            # local_round(). We additionally exchange (seed, rho) pairs and
            # apply the *neighbor* contribution as an extra step.
            #
            # Simpler interpretation here: skip local_update for update_share by
            # using nesterov_state=None and rolling the consensus into one step.
            # To keep this skeleton simple, current implementation calls
            # consensus_via_updates which assumes the local step has NOT been
            # applied yet. Callers should configure clients with
            # local_steps=0-style behavior — see scripts/04 for usage.
            consensus_via_updates(
                clients, topology.W, all_seeds, all_rhos, clients[0].mezo_config
            )
        elif config.consensus_mode == "none":
            pass  # Pure local training, baseline.
        else:
            raise ValueError(f"Unknown consensus_mode={config.consensus_mode!r}")

        # 3. Eval.
        if eval_fn is not None and config.eval_every > 0 and (r + 1) % config.eval_every == 0:
            eval_metrics = eval_fn(clients[0].model, r)
            round_log.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        # 4. Log.
        if config.log_every > 0 and (r + 1) % config.log_every == 0:
            if logger is not None:
                logger(round_log)

        logs.append(round_log)

    return logs
