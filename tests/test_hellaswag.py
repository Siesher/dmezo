"""Tests for HellaSwag loader and 4-way multi-choice evaluator.

Uses a synthetic HF dataset (via :class:`datasets.Dataset.from_dict`) so the
tests don't need network access. The synthetic rows mimic the real HellaSwag
schema (ctx_a, ctx_b, endings, label) closely enough that the loader code path
is identical to production.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer

from dmezo.data.hellaswag import (
    _HellaSwagDataset,
    build_hellaswag_loader,
    evaluate_hellaswag_accuracy,
    format_hellaswag_example,
    format_hellaswag_prompt,
)
from dmezo.data.superglue import _collate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_hellaswag_rows(n: int = 8) -> Dataset:
    """Build a synthetic HellaSwag dataset matching the HF schema."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "ind": i,
                "activity_label": "test activity",
                "ctx_a": f"A person is doing thing {i}.",
                "ctx_b": "Then",
                "ctx": f"A person is doing thing {i}. Then",
                "endings": [
                    "the cat ran away quickly.",
                    "the dog barked loudly twice.",
                    "the bird flew into the sky.",
                    "the fish jumped out of water.",
                ],
                "source_id": f"src_{i}",
                "split": "train",
                "split_type": "indomain",
                "label": str(i % 4),  # HF stores label as str
            }
        )
    return Dataset.from_list(rows)


def _get_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestHellaSwagFormatter:
    def test_prompt_concats_ctx_a_and_ctx_b(self):
        prompt = format_hellaswag_prompt("Hello world.", "Then")
        assert prompt == "Hello world. Then"

    def test_full_example_appends_ending_with_space(self):
        full = format_hellaswag_example("Hello world.", "Then", "we eat.")
        assert full == "Hello world. Then we eat."


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestHellaSwagDataset:
    def test_dataset_size_matches_input(self):
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(8)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        assert len(ds) == 8

    def test_item_has_correct_keys_and_dtypes(self):
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(4)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        item = ds[0]
        assert set(item) == {"input_ids", "attention_mask", "labels"}
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        # Sequence dim consistent across the three tensors.
        n = item["input_ids"].size(0)
        assert item["attention_mask"].size(0) == n
        assert item["labels"].size(0) == n

    def test_labels_masked_on_prompt_tokens(self):
        """Prompt-region of `labels` must be -100; ending-region must have real ids."""
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(4)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        for i in range(4):
            item = ds[i]
            labels = item["labels"]
            # At least one masked token at the start (prompt is non-empty).
            assert (labels[:1] == -100).all(), f"row {i}: first token must be -100"
            # At least one unmasked token (ending is non-empty).
            assert (labels != -100).any(), f"row {i}: ending must have unmasked tokens"

    def test_string_label_is_cast_to_int(self):
        """HF stores label as str — _HellaSwagDataset must tolerate this."""
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(4)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        # No crash on indexing means str→int cast worked.
        for i in range(4):
            _ = ds[i]


# ---------------------------------------------------------------------------
# Loader tests (uses _load_raw_dataset monkeypatch)
# ---------------------------------------------------------------------------


class TestHellaSwagLoader:
    def test_loader_yields_batches_with_correct_shapes(self, monkeypatch):
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(8)
        import dmezo.data.hellaswag as hs
        import dmezo.data.superglue as sg

        monkeypatch.setattr(sg, "_load_raw_dataset", lambda task, split: raw)
        # The loader imports _load_raw_dataset from sg; the monkeypatch above
        # is the one that takes effect because hs imports it lazily.
        monkeypatch.setattr(hs, "_load_raw_dataset", lambda task, split: raw, raising=False)

        loader = build_hellaswag_loader(
            tok, split="train", batch_size=2, max_length=64, shuffle=False
        )
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == 2
        assert batch["input_ids"].size(1) == batch["labels"].size(1)


# ---------------------------------------------------------------------------
# Evaluator tests (4-way multi-choice)
# ---------------------------------------------------------------------------


class _ScriptedHellaSwagLM(nn.Module):
    """Tiny scripted model that prefers a specific ending index.

    Returns low loss when the input contains the preferred ending's first
    distinguishing token, high loss otherwise. Lets us test that the evaluator
    correctly identifies argmin.
    """

    def __init__(self, tokenizer, preferred_ending: str):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.tokenizer = tokenizer
        # Encode the preferred ending's first content token (skip leading space).
        ids = tokenizer.encode(" " + preferred_ending, add_special_tokens=False)
        self.preferred_first_id = ids[0] if ids else None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        score = torch.tensor(1.0, dtype=torch.float32)
        if labels is not None and self.preferred_first_id is not None:
            for row in labels:
                non_masked = row[row != -100]
                if non_masked.numel() == 0:
                    continue
                first = int(non_masked[0].item())
                if first == self.preferred_first_id:
                    score = torch.tensor(0.1, dtype=torch.float32)
                    break

        class _Out:
            loss = score

        return _Out()

    def eval(self):
        return self

    def parameters(self, recurse=True):
        return iter([self.dummy.weight])


class TestEvaluateHellaSwagAccuracy:
    def test_scripted_model_picks_preferred_ending(self):
        """If the scripted model strongly prefers ending 0, accuracy equals the
        fraction of examples whose gold label is 0."""
        tok = _get_tokenizer()
        # Build a small dataset where gold labels rotate 0,1,2,3,0,1,2,3.
        raw = _synthetic_hellaswag_rows(8)
        ds = _HellaSwagDataset(raw, tok, max_length=64)

        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=lambda b: _collate(b, tok.pad_token_id)
        )

        # The model prefers "the cat ran away quickly." (ending index 0).
        model = _ScriptedHellaSwagLM(tok, preferred_ending="the cat ran away quickly.")
        acc = evaluate_hellaswag_accuracy(model, loader, max_batches=10)
        # 2/8 examples have gold label 0 (rotation: 0,1,2,3,0,1,2,3).
        assert abs(acc - 2 / 8) < 1e-6, f"expected 0.25, got {acc:.4f}"

    def test_returns_zero_for_empty_loader(self):
        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(0)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=lambda b: _collate(b, tok.pad_token_id)
        )
        model = _ScriptedHellaSwagLM(tok, preferred_ending="anything")
        acc = evaluate_hellaswag_accuracy(model, loader, max_batches=10)
        assert acc == 0.0


# ---------------------------------------------------------------------------
# Dispatch tests — task="hellaswag" routed through evaluate_classification_accuracy
# ---------------------------------------------------------------------------


class TestHellaSwagDispatch:
    def test_evaluate_classification_accuracy_routes_hellaswag(self):
        """task='hellaswag' should dispatch to evaluate_hellaswag_accuracy."""
        from dmezo.data.superglue import evaluate_classification_accuracy

        tok = _get_tokenizer()
        raw = _synthetic_hellaswag_rows(4)
        ds = _HellaSwagDataset(raw, tok, max_length=64)
        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=lambda b: _collate(b, tok.pad_token_id)
        )
        model = _ScriptedHellaSwagLM(tok, preferred_ending="the cat ran away quickly.")
        acc = evaluate_classification_accuracy(model, loader, task="hellaswag", max_batches=10)
        # Gold labels: 0, 1, 2, 3 -> only the first is "ending 0" -> 1/4.
        assert abs(acc - 1 / 4) < 1e-6, f"expected 0.25, got {acc:.4f}"

    def test_hellaswag_registered_in_task_loaders(self):
        from dmezo.data.superglue import TASK_DATASETS, TASK_DEFAULTS, TASK_LOADERS

        assert "hellaswag" in TASK_LOADERS
        assert "hellaswag" in TASK_DATASETS
        assert "hellaswag" in TASK_DEFAULTS
        # 4-way classification → defaults should accept multi-token endings.
        assert TASK_DEFAULTS["hellaswag"]["max_length"] >= 128
