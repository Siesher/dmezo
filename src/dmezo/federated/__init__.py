"""Federated/decentralized layer: clients, topology, consensus, simulator.

NOTE: We deliberately do NOT eagerly import ``client``, ``consensus``,
``simulator`` from this __init__ because they pull in torch. Importing
``dmezo.federated`` should be lightweight enough to use topology in
environments without torch (e.g. for testing graph properties).

Use explicit imports for the torch-dependent parts:

    from dmezo.federated.client import ClientState
    from dmezo.federated.simulator import run_simulation
"""

from dmezo.federated.topology import (
    MixingMatrix,
    complete_graph,
    random_regular,
    ring_graph,
    spectral_gap,
    star_graph,
)

__all__ = [
    "MixingMatrix",
    "complete_graph",
    "random_regular",
    "ring_graph",
    "spectral_gap",
    "star_graph",
]
