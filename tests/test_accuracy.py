"""Tests for evaluate_classification_accuracy in src/dmezo/data/superglue.py.

For prompt-completion framing (Malladi 2023 / MeZO style), accuracy is computed
by scoring each candidate label suffix and picking the lower-loss one. This is
exactly what eval-time generation would do; the metric matches downstream
classification behavior, unlike training loss which only measures the gold
suffix's likelihood.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer

from dmezo.data.superglue import evaluate_classification_accuracy


class _ScriptedLM(nn.Module):
    """Minimal HF-style model that returns a deterministic loss based on which
    label suffix is in the input.

    Pretends to "prefer" class-1 (label=1) — i.e. returns a LOWER loss when the
    " positive" suffix is in the input than when " negative" is. This means
    the model classifies every example as class-1.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.tokenizer = tokenizer
        # Token id for " positive" — first non-pad token after the prompt.
        self.positive_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
        self.negative_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Walk the label tensor — wherever the label suffix token is " positive",
        # return a low loss (0.1); wherever it's " negative", return high (1.0).
        # `labels` has -100 on prompt tokens and the real ids on suffix tokens.
        score = torch.tensor(0.5, dtype=torch.float32)
        if labels is not None:
            for row in labels:
                # Find the first non-(-100) token — that's the label suffix.
                non_masked = row[row != -100]
                if non_masked.numel() == 0:
                    continue
                first = int(non_masked[0].item())
                if first == self.positive_id:
                    score = torch.tensor(0.1, dtype=torch.float32)
                    break
                elif first == self.negative_id:
                    score = torch.tensor(1.0, dtype=torch.float32)
                    break

        class _Out:
            loss = score

        return _Out()

    def eval(self):
        return self

    def parameters(self, recurse=True):
        return iter([self.dummy.weight])


def _make_dataloader(tokenizer, labels: list[int]):
    """Build a tiny SST-2-like loader where each example has a chosen label."""
    sentences = [f"example {i}" for i in range(len(labels))]
    ds = Dataset.from_dict(
        {"sentence": sentences, "label": labels, "idx": list(range(len(labels)))}
    )
    from torch.utils.data import DataLoader

    from dmezo.data.superglue import _collate, _SST2Dataset

    tds = _SST2Dataset(ds, tokenizer, max_length=32)
    return DataLoader(
        tds, batch_size=2, shuffle=False, collate_fn=lambda b: _collate(b, tokenizer.pad_token_id)
    )


class TestEvaluateClassificationAccuracy:
    def test_model_that_always_says_positive_scores_correctly(self):
        """A scripted model that always prefers " positive" should get accuracy
        equal to the fraction of class-1 examples in the dataset."""
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        labels = [1, 1, 0, 0, 1]  # 3/5 positive
        loader = _make_dataloader(tok, labels)
        model = _ScriptedLM(tok)

        acc = evaluate_classification_accuracy(model, loader, task="sst2", max_batches=10)
        # Model always picks positive => correct on class-1 examples only
        assert abs(acc - 3 / 5) < 1e-6, f"expected 0.6, got {acc:.4f}"

    def test_returns_zero_for_empty_loader(self):
        """No batches -> accuracy is 0 (or NaN). We return 0 for safety."""
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        loader = _make_dataloader(tok, [])
        model = _ScriptedLM(tok)
        acc = evaluate_classification_accuracy(model, loader, task="sst2", max_batches=10)
        # Either 0.0 or NaN is acceptable; we want a defined non-crashing return.
        assert acc == 0.0 or (acc != acc), f"expected 0.0 or NaN for empty loader, got {acc}"

    def test_unknown_task_raises(self):
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        loader = _make_dataloader(tok, [0, 1])
        model = _ScriptedLM(tok)
        import pytest

        with pytest.raises(ValueError, match="task"):
            evaluate_classification_accuracy(model, loader, task="bogus", max_batches=1)
