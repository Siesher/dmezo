"""Unit tests for src/dmezo/data/partition.py.

Conventions follow the federated learning literature:
- IID partition: disjoint uniform-random chunks (McMahan et al. 2017, FedAvg).
- Dirichlet partition: per-class Dirichlet(alpha) over clients (Hsu et al. 2019,
  "Measuring the Effects of Non-Identical Data Distribution for Federated Visual
  Classification"; standard non-IID benchmark with alpha in {0.1, 0.5, 1.0, 10}).
- Label-skew partition: each client owns only a subset of classes (extreme worst
  case for FedAvg).
"""

from __future__ import annotations

import numpy as np
import pytest

from dmezo.data.partition import (
    dirichlet_partition,
    iid_partition,
    label_skew_partition,
)


# ---------------------------------------------------------------------------
# iid_partition
# ---------------------------------------------------------------------------


class TestIIDPartition:
    def test_returns_one_array_per_client(self):
        parts = iid_partition(num_examples=100, num_clients=4, seed=42)
        assert len(parts) == 4

    def test_partitions_are_disjoint(self):
        """Each example index must appear in exactly one client's slice."""
        parts = iid_partition(num_examples=100, num_clients=4, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(set(all_idx.tolist())), "duplicate indices across clients"

    def test_partitions_cover_full_index_set(self):
        """Union of all client partitions == {0, ..., n-1}."""
        parts = iid_partition(num_examples=100, num_clients=4, seed=42)
        all_idx = sorted(np.concatenate(parts).tolist())
        assert all_idx == list(range(100))

    def test_partitions_are_balanced(self):
        """For n_examples=100, n_clients=4: every client should get 25."""
        parts = iid_partition(num_examples=100, num_clients=4, seed=42)
        sizes = [len(p) for p in parts]
        assert sizes == [25, 25, 25, 25]

    def test_unbalanced_examples_distribute_within_one(self):
        """For n_examples=103, n_clients=4: sizes within 1 of each other."""
        parts = iid_partition(num_examples=103, num_clients=4, seed=42)
        sizes = sorted(len(p) for p in parts)
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == 103

    def test_determinism(self):
        """Same seed -> same partition."""
        p1 = iid_partition(num_examples=100, num_clients=4, seed=42)
        p2 = iid_partition(num_examples=100, num_clients=4, seed=42)
        for a, b in zip(p1, p2):
            assert np.array_equal(a, b)

    def test_different_seeds_give_different_partitions(self):
        p1 = iid_partition(num_examples=100, num_clients=4, seed=42)
        p2 = iid_partition(num_examples=100, num_clients=4, seed=43)
        assert not all(np.array_equal(a, b) for a, b in zip(p1, p2))


# ---------------------------------------------------------------------------
# dirichlet_partition
# ---------------------------------------------------------------------------


class TestDirichletPartition:
    @pytest.fixture
    def binary_labels(self):
        """1000 examples, 50/50 class split (binary, like SST-2)."""
        rng = np.random.default_rng(0)
        labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
        rng.shuffle(labels)
        return labels

    def test_returns_one_array_per_client(self, binary_labels):
        parts = dirichlet_partition(binary_labels, num_clients=4, alpha=0.5, seed=42)
        assert len(parts) == 4

    def test_partitions_cover_full_index_set(self, binary_labels):
        parts = dirichlet_partition(binary_labels, num_clients=4, alpha=0.5, seed=42)
        all_idx = sorted(np.concatenate(parts).tolist())
        assert all_idx == list(range(len(binary_labels)))

    def test_partitions_are_disjoint(self, binary_labels):
        parts = dirichlet_partition(binary_labels, num_clients=4, alpha=0.5, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(set(all_idx.tolist()))

    def test_low_alpha_produces_skewed_label_distributions(self, binary_labels):
        """alpha=0.1 -> at least one client has highly skewed class ratios.

        With alpha=0.1 (very heterogeneous, per Hsu 2019 conventions) on a
        2-class problem with 4 clients, we expect at least one client whose
        class-1 fraction is outside [0.3, 0.7] (vs. true ratio of 0.5).
        """
        parts = dirichlet_partition(binary_labels, num_clients=4, alpha=0.1, seed=42)
        class1_fractions = []
        for idx in parts:
            if len(idx) == 0:
                continue
            class1_fractions.append(binary_labels[idx].mean())
        assert any(f < 0.3 or f > 0.7 for f in class1_fractions), (
            f"alpha=0.1 should produce label skew, got fractions {class1_fractions}"
        )

    def test_high_alpha_approximates_iid(self, binary_labels):
        """alpha=100 -> per-client label fractions all near 0.5 (true global ratio)."""
        parts = dirichlet_partition(binary_labels, num_clients=4, alpha=100.0, seed=42)
        for idx in parts:
            if len(idx) == 0:
                continue
            frac = binary_labels[idx].mean()
            assert 0.35 < frac < 0.65, f"alpha=100 should be ~IID, got class-1 frac {frac:.3f}"

    def test_determinism(self, binary_labels):
        p1 = dirichlet_partition(binary_labels, num_clients=4, alpha=0.5, seed=42)
        p2 = dirichlet_partition(binary_labels, num_clients=4, alpha=0.5, seed=42)
        for a, b in zip(p1, p2):
            assert np.array_equal(sorted(a.tolist()), sorted(b.tolist()))


# ---------------------------------------------------------------------------
# label_skew_partition
# ---------------------------------------------------------------------------


class TestLabelSkewPartition:
    @pytest.fixture
    def binary_labels(self):
        labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
        return labels

    def test_returns_one_array_per_client(self, binary_labels):
        parts = label_skew_partition(binary_labels, num_clients=2, classes_per_client=1, seed=42)
        assert len(parts) == 2

    def test_two_clients_one_class_each_gives_pure_clients(self, binary_labels):
        """With 2 clients and classes_per_client=1, each client should see ONLY one class.

        This is the canonical worst-case non-IID setup for binary tasks.
        """
        parts = label_skew_partition(binary_labels, num_clients=2, classes_per_client=1, seed=42)
        per_client_classes = [set(binary_labels[idx].tolist()) for idx in parts if len(idx) > 0]
        for cls_set in per_client_classes:
            assert len(cls_set) == 1, f"client should have one class, has {cls_set}"

    def test_raises_when_classes_per_client_exceeds_unique_classes(self, binary_labels):
        with pytest.raises(ValueError, match="classes_per_client"):
            label_skew_partition(binary_labels, num_clients=2, classes_per_client=5, seed=42)
