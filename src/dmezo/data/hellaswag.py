"""HellaSwag (Zellers et al. 2019): 4-way commonsense reasoning.

Dataset: https://huggingface.co/datasets/hellaswag
Task: given a context (ctx_a + ctx_b) and 4 candidate endings, pick the most
plausible continuation.

Framing for MeZO (following Malladi 2023 §G for multi-choice tasks):

- **Training:** use only the ground-truth ending. Build
  ``prompt = ctx_a + " " + ctx_b``, ``full = prompt + " " + endings[label]``.
  Cross-entropy loss is computed on ending tokens only (prompt tokens masked
  with -100). This is the same prompt-completion framing as SST-2/BoolQ in
  ``superglue.py``, except the suffix is multi-token and example-specific.
- **Evaluation:** score all 4 candidate completions independently (mean loss
  over ending tokens) and predict ``argmin_i loss_i``. See
  :func:`evaluate_hellaswag_accuracy`.

This module deliberately re-uses ``_collate`` from ``superglue`` to keep
batching behaviour consistent (pad to longest in batch with ``pad_token_id``
on inputs and ``-100`` on labels).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

HELLASWAG_TEMPLATE = "{ctx_a} {ctx_b}"


def format_hellaswag_prompt(ctx_a: str, ctx_b: str) -> str:
    """Build the HellaSwag prompt (context only, no ending)."""
    return HELLASWAG_TEMPLATE.format(ctx_a=ctx_a, ctx_b=ctx_b)


def format_hellaswag_example(ctx_a: str, ctx_b: str, ending: str) -> str:
    """Build the full prompt-completion: ``ctx_a + ctx_b + " " + ending``."""
    return f"{format_hellaswag_prompt(ctx_a, ctx_b)} {ending}"


class _HellaSwagDataset(Dataset):
    """Tokenised HellaSwag train-style: ground-truth ending only.

    Each item is a dict ``(input_ids, attention_mask, labels)`` with ``-100`` on
    the prompt portion so loss is computed only on ending tokens.

    For 4-way multi-choice eval over *all* endings, use
    :func:`evaluate_hellaswag_accuracy` which builds candidates on the fly from
    the underlying HF rows (exposed via ``self.data``).
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 256) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        ctx_a = row["ctx_a"]
        ctx_b = row["ctx_b"]
        endings = row["endings"]
        # HellaSwag stores label as str ("0"/"1"/"2"/"3") in HF; tolerate both.
        label = int(row["label"])
        prompt = format_hellaswag_prompt(ctx_a, ctx_b)
        full = format_hellaswag_example(ctx_a, ctx_b, endings[label])

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


def build_hellaswag_loader(
    tokenizer,
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    num_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader for HellaSwag with MeZO-style framing.

    Args:
        tokenizer: HF tokenizer. ``tokenizer.pad_token_id`` must be set.
        split: ``"train"`` or ``"validation"`` (HellaSwag ``test`` labels are
            withheld).
        batch_size: DataLoader batch size. Default 4 keeps token budget similar
            to BoolQ (long passages).
        max_length: Max token length per example. 256 fits most HellaSwag
            contexts; bump to 384 for the long-context subset.
        shuffle: Shuffle for training.
        num_examples: Subsample to this many examples (``None`` = full split).
        seed: RNG seed for subsampling.

    Returns:
        Torch DataLoader yielding ``(input_ids, attention_mask, labels)`` dicts.
    """
    from dmezo.data.superglue import _collate, _load_raw_dataset

    raw = _load_raw_dataset("hellaswag", split)
    if num_examples is not None and num_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(num_examples))

    ds = _HellaSwagDataset(raw, tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None; set tokenizer.pad_token first")
    collate_fn = lambda b: _collate(b, pad_token_id=pad_id)  # noqa: E731
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )


@torch.inference_mode()
def evaluate_hellaswag_accuracy(
    model,
    dataloader,
    max_batches: int = 20,
) -> float:
    """4-way multi-choice accuracy on HellaSwag.

    For each example, scores all 4 candidate endings by mean ending-token loss
    and predicts ``argmin_i loss_i``. The dataloader's underlying dataset must
    expose ``.data`` (raw HF rows) so we can recover all 4 endings and the gold
    label per row.

    This is the standard HellaSwag eval protocol used by EleutherAI's
    lm-evaluation-harness (Gao et al. 2021).

    Args:
        model: HF causal-LM model. Should be in eval mode; we temporarily put
            it in eval() and restore the original training mode on exit.
        dataloader: DataLoader over a :class:`_HellaSwagDataset`.
        max_batches: Stop after this many batches (matches ``evaluate_loss``
            convention so accuracy and loss are computed over the same pool).

    Returns:
        Float in ``[0.0, 1.0]`` — fraction correct. Returns ``0.0`` on an empty
        loader.
    """
    underlying = dataloader.dataset
    tokenizer = underlying.tokenizer
    max_length = underlying.max_length
    raw = underlying.data

    n_correct = 0
    n_total = 0
    batches_seen = 0
    was_training = model.training
    model.eval()
    try:
        cursor = 0
        for batch in dataloader:
            if batches_seen >= max_batches:
                break
            bs = batch["input_ids"].size(0)
            for i in range(bs):
                idx = cursor + i
                if idx >= len(raw):
                    break
                row = raw[idx]
                gold_label = int(row["label"])
                ctx_a, ctx_b = row["ctx_a"], row["ctx_b"]
                endings = row["endings"]
                prompt = format_hellaswag_prompt(ctx_a, ctx_b)

                best_cls = 0
                best_loss = float("inf")
                for cls, ending in enumerate(endings):
                    candidate = format_hellaswag_example(ctx_a, ctx_b, ending)
                    full_ids = tokenizer(
                        candidate, max_length=max_length, truncation=True, return_tensors="pt"
                    ).input_ids
                    prompt_ids = tokenizer(
                        prompt, max_length=max_length, truncation=True, return_tensors="pt"
                    ).input_ids
                    labels = full_ids.clone()
                    labels[:, : prompt_ids.size(1)] = -100
                    attn = torch.ones_like(full_ids)
                    device = next(model.parameters()).device
                    out = model(
                        input_ids=full_ids.to(device),
                        attention_mask=attn.to(device),
                        labels=labels.to(device),
                    )
                    loss_val = float(out.loss.item())
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_cls = cls
                if best_cls == gold_label:
                    n_correct += 1
                n_total += 1
            cursor += bs
            batches_seen += 1
    finally:
        if was_training:
            model.train()
    return n_correct / n_total if n_total > 0 else 0.0
