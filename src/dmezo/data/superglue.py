"""SuperGLUE / SST-2 data preparation for MeZO.

For Day 1 sanity check we use SST-2 (binary sentiment classification) framed as
a causal LM task: prompt the model with the sentence + a label suffix and
measure cross-entropy on the label tokens only.

This is exactly the framing used in Malladi 2023 for OPT models.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


SST2_TEMPLATE = (
    "Review: {sentence}\n"
    "Sentiment:"
)
SST2_LABEL_WORDS = {0: " negative", 1: " positive"}


def format_sst2_example(sentence: str, label: int | None = None) -> str:
    """Format an SST-2 example as a prompt (optionally with the label)."""
    prompt = SST2_TEMPLATE.format(sentence=sentence)
    if label is not None:
        return prompt + SST2_LABEL_WORDS[label]
    return prompt


class _SST2Dataset(Dataset):
    """Tokenized SST-2 with the prompt-completion framing.

    Each item returns a dict with input_ids, attention_mask, labels (with -100
    on prompt tokens so the loss is computed only on the label suffix).
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        sentence = row["sentence"]
        label = int(row["label"])
        prompt = format_sst2_example(sentence, label=None)
        label_text = SST2_LABEL_WORDS[label]
        full = prompt + label_text

        full_ids = self.tokenizer(
            full, max_length=self.max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]
        prompt_ids = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]

        labels = full_ids.clone()
        # Mask prompt tokens so they don't contribute to the loss.
        labels[: prompt_ids.size(0)] = -100
        attention_mask = torch.ones_like(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _collate(batch: List[dict], pad_token_id: int) -> dict:
    """Pad a batch to the longest sequence."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    out_ids, out_mask, out_labels = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        out_ids.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        out_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )
        out_labels.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(out_ids),
        "attention_mask": torch.stack(out_mask),
        "labels": torch.stack(out_labels),
    }


def build_sst2_loader(
    tokenizer,
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    num_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader for SST-2 with MeZO-style framing.

    Args:
        tokenizer: HF tokenizer.
        split: ``"train"`` or ``"validation"``.
        batch_size: Batch size.
        max_length: Max token length.
        shuffle: Shuffle for training.
        num_examples: Subsample to this many examples (None = full split).
        seed: Seed for subsampling.

    Returns:
        Torch DataLoader.
    """
    from datasets import load_dataset

    raw = load_dataset("glue", "sst2", split=split)
    if num_examples is not None and num_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(num_examples))

    ds = _SST2Dataset(raw, tokenizer, max_length=max_length)
    collate_fn = lambda b: _collate(b, pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0
    )


def causal_lm_loss(model, batch: dict) -> torch.Tensor:
    """Standard causal-LM loss for use with MeZO step.

    Assumes ``batch`` has ``input_ids``, ``attention_mask``, ``labels`` and the
    model returns ``outputs.loss`` when given ``labels`` (HF default).
    """
    batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
    outputs = model(**batch)
    return outputs.loss
