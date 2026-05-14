"""IID / non-IID partitioning of a dataset across federated clients."""

from __future__ import annotations

from typing import List

import numpy as np


def iid_partition(
    num_examples: int,
    num_clients: int,
    seed: int = 0,
) -> List[np.ndarray]:
    """Uniformly random shuffle, split into equal contiguous chunks.

    Args:
        num_examples: Total number of examples.
        num_clients: Number of clients.
        seed: RNG seed.

    Returns:
        List of length ``num_clients``; each entry is a numpy array of indices.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_examples)
    return np.array_split(perm, num_clients)


def label_skew_partition(
    labels: np.ndarray,
    num_clients: int,
    classes_per_client: int = 1,
    seed: int = 0,
) -> List[np.ndarray]:
    """Each client gets examples from only a few classes (extreme non-IID).

    For a binary task with num_clients=2 and classes_per_client=1, this gives
    client 0 all class-0 examples and client 1 all class-1 examples — a worst
    case for FedAvg.

    Args:
        labels: 1D array of class labels.
        num_clients: Number of clients.
        classes_per_client: How many classes each client gets.
        seed: RNG seed.

    Returns:
        List of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    if classes_per_client > len(unique_labels):
        raise ValueError("classes_per_client > number of classes")

    # Assign each client a set of classes.
    client_classes = []
    pool = list(unique_labels)
    for _ in range(num_clients):
        assigned = rng.choice(pool, size=classes_per_client, replace=False).tolist()
        client_classes.append(assigned)

    # Distribute examples per assigned class evenly across clients that own it.
    label_to_clients: dict = {lbl: [] for lbl in unique_labels}
    for ci, cls_list in enumerate(client_classes):
        for cls in cls_list:
            label_to_clients[cls].append(ci)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        rng.shuffle(idx)
        owners = label_to_clients[lbl]
        if not owners:
            continue
        chunks = np.array_split(idx, len(owners))
        for ci, chunk in zip(owners, chunks):
            client_indices[ci].extend(chunk.tolist())

    return [np.array(idx) for idx in client_indices]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
) -> List[np.ndarray]:
    """Dirichlet-based heterogeneous partition (standard non-IID benchmark).

    For each class, samples a Dirichlet(alpha) over clients and splits that
    class's examples accordingly. Small alpha => more heterogeneous.

    Args:
        labels: 1D array of class labels.
        num_clients: Number of clients.
        alpha: Dirichlet concentration. Common: 0.1 (very heterogeneous),
            0.5 (mild), 10.0 (close to IID).
        seed: RNG seed.

    Returns:
        List of index arrays.
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, splits)
        for ci, chunk in enumerate(chunks):
            client_indices[ci].extend(chunk.tolist())
    return [np.array(idx) for idx in client_indices]
