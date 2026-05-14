"""SuperGLUE / GLUE data preparation for MeZO.

Tasks framed as causal LM completion: prompt the model with the input + a label
suffix and measure cross-entropy on the label tokens only. This is the framing
used in Malladi 2023 (MeZO) for OPT models.

Supported tasks:
    - sst2: binary sentiment (GLUE)
    - boolq: yes/no question answering on a passage (SuperGLUE)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

SST2_TEMPLATE = "Review: {sentence}\nSentiment:"
SST2_LABEL_WORDS = {0: " negative", 1: " positive"}

BOOLQ_TEMPLATE = "Passage: {passage}\nQuestion: {question}?\nAnswer:"
BOOLQ_LABEL_WORDS = {0: " No", 1: " Yes"}


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


def _collate(batch: list[dict], pad_token_id: int) -> dict:
    """Pad a batch to the longest sequence."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    out_ids, out_mask, out_labels = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        out_ids.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        out_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        out_labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
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


def format_boolq_example(passage: str, question: str, label: int | None = None) -> str:
    """Format a BoolQ example as a prompt (optionally with the label)."""
    prompt = BOOLQ_TEMPLATE.format(passage=passage, question=question)
    if label is not None:
        return prompt + BOOLQ_LABEL_WORDS[label]
    return prompt


class _BoolQDataset(Dataset):
    """Tokenized BoolQ with prompt-completion framing for MeZO.

    BoolQ entry shape (HF SuperGLUE): {passage, question, label, idx}.
    Loss is computed only on the " Yes" / " No" label suffix.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        passage = row["passage"]
        question = row["question"]
        label = int(row["label"])
        prompt = format_boolq_example(passage, question, label=None)
        label_text = BOOLQ_LABEL_WORDS[label]
        full = prompt + label_text

        full_ids = self.tokenizer(
            full, max_length=self.max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]
        prompt_ids = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]

        labels = full_ids.clone()
        labels[: prompt_ids.size(0)] = -100
        attention_mask = torch.ones_like(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_boolq_loader(
    tokenizer,
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
    num_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader for SuperGLUE BoolQ with MeZO-style framing.

    BoolQ passages are long — default max_length=512 captures most of them.
    Default batch_size=4 to keep total token budget similar to SST-2 (bs=8 × 256).
    """
    from datasets import load_dataset

    raw = load_dataset("super_glue", "boolq", split=split, trust_remote_code=True)
    if num_examples is not None and num_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(num_examples))

    ds = _BoolQDataset(raw, tokenizer, max_length=max_length)
    collate_fn = lambda b: _collate(b, pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0
    )


# Task dispatch: ``cfg["data"]["task"]`` -> dataloader factory function.
TASK_LOADERS = {
    "sst2": build_sst2_loader,
    "boolq": build_boolq_loader,
}


def build_loader_for_task(task: str, **kwargs) -> DataLoader:
    """Dispatch to the right loader factory based on task name."""
    if task not in TASK_LOADERS:
        raise ValueError(f"Unknown task {task!r}. Supported: {sorted(TASK_LOADERS)}")
    return TASK_LOADERS[task](**kwargs)
