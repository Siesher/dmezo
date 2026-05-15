"""Tests for build_partitioned_loaders in src/dmezo/data/superglue.py.

Validates that per-client DataLoaders are constructed correctly under different
partitioning regimes. Uses a synthetic HF Dataset (via Dataset.from_dict) to
avoid downloading GLUE/SuperGLUE during unit tests.

Convention: SST-2 'label' column with 0/1 entries, 'sentence' column with raw
text — matches the HF schema that build_sst2_loader consumes.
"""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from dmezo.data.superglue import build_partitioned_loaders

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Tiny tokenizer for testing — gpt2 is small (~500KB) and ships padding-safe."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def synthetic_sst2(monkeypatch):
    """Patch datasets.load_dataset so build_sst2_loader receives our synthetic data.

    Returns 1000 examples with 50/50 class split; sentences are short and
    deterministic so tokenization is fast.
    """
    sentences = []
    labels = []
    for i in range(500):
        sentences.append(f"good movie number {i}")
        labels.append(1)
    for i in range(500):
        sentences.append(f"bad movie number {i}")
        labels.append(0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(1000)
    sentences = [sentences[i] for i in perm]
    labels = [labels[i] for i in perm]
    ds = Dataset.from_dict({"sentence": sentences, "label": labels, "idx": list(range(1000))})

    def fake_load_dataset(name, subset, split, **kwargs):
        return ds

    import dmezo.data.superglue as sg

    monkeypatch.setattr(sg, "_load_raw_dataset", lambda task, split: ds)
    return ds


# ---------------------------------------------------------------------------
# IID mode
# ---------------------------------------------------------------------------


class TestPartitionedLoadersIID:
    def test_returns_one_loader_per_client(self, tokenizer, synthetic_sst2):
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="iid",
            num_examples=200,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        assert len(loaders) == 4

    def test_iid_loaders_have_disjoint_indices(self, tokenizer, synthetic_sst2):
        """Across clients, the underlying dataset rows must not overlap."""
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="iid",
            num_examples=200,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        # Each loader wraps a _SST2Dataset whose .data is a subset of the raw.
        # We compare 'idx' column values.
        per_client_idx = []
        for dl in loaders:
            per_client_idx.append(set(dl.dataset.data["idx"]))
        all_idx = set().union(*per_client_idx)
        # Disjoint: total size == sum of per-client sizes.
        assert sum(len(s) for s in per_client_idx) == len(all_idx)

    def test_iid_loaders_cover_subsampled_pool(self, tokenizer, synthetic_sst2):
        """Union of per-client indices == the num_examples subsample."""
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="iid",
            num_examples=200,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        total = sum(len(dl.dataset) for dl in loaders)
        assert total == 200

    def test_iid_balanced_when_num_examples_divisible(self, tokenizer, synthetic_sst2):
        """200 examples, 4 clients -> 50 each."""
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="iid",
            num_examples=200,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        sizes = sorted(len(dl.dataset) for dl in loaders)
        assert sizes == [50, 50, 50, 50]


# ---------------------------------------------------------------------------
# Dirichlet mode
# ---------------------------------------------------------------------------


class TestPartitionedLoadersDirichlet:
    def test_dirichlet_low_alpha_produces_label_skew(self, tokenizer, synthetic_sst2):
        """alpha=0.1 should give at least one client with skewed label distribution."""
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="dirichlet",
            partition_kwargs={"alpha": 0.1},
            num_examples=400,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        fractions = []
        for dl in loaders:
            labels = np.array(dl.dataset.data["label"])
            if len(labels) == 0:
                continue
            fractions.append(labels.mean())
        assert any(f < 0.3 or f > 0.7 for f in fractions), (
            f"Dirichlet(0.1) should skew labels, got {fractions}"
        )

    def test_dirichlet_high_alpha_is_iid_like(self, tokenizer, synthetic_sst2):
        """alpha=100 should give all clients near 50/50."""
        loaders = build_partitioned_loaders(
            task="sst2",
            tokenizer=tokenizer,
            n_clients=4,
            partition_mode="dirichlet",
            partition_kwargs={"alpha": 100.0},
            num_examples=400,
            batch_size=8,
            max_length=64,
            seed=42,
        )
        for dl in loaders:
            labels = np.array(dl.dataset.data["label"])
            if len(labels) == 0:
                continue
            frac = labels.mean()
            assert 0.30 < frac < 0.70, f"alpha=100 should be ~IID, got {frac:.3f}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPartitionedLoadersErrors:
    def test_unknown_partition_mode_raises(self, tokenizer, synthetic_sst2):
        with pytest.raises(ValueError, match="partition_mode"):
            build_partitioned_loaders(
                task="sst2",
                tokenizer=tokenizer,
                n_clients=4,
                partition_mode="bogus_mode",
                num_examples=100,
                batch_size=8,
                max_length=64,
                seed=42,
            )

    def test_unknown_task_raises(self, tokenizer):
        with pytest.raises(ValueError, match="task"):
            build_partitioned_loaders(
                task="bogus_task",
                tokenizer=tokenizer,
                n_clients=4,
                partition_mode="iid",
                num_examples=100,
                batch_size=8,
                max_length=64,
                seed=42,
            )
