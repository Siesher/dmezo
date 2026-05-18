"""MathLogicQA (Russian symbolic logic + arithmetic, part of MERA).

Dataset: https://huggingface.co/datasets/ai-forever/MERA (config: ``mathlogicqa``)
Paper: MERA — Multimodal Evaluation for Russian-language Architectures
(Fenogenova et al. 2024).

Task: 4-way multi-choice symbolic logic / math reasoning in Russian. Each
example has a question and four candidate answers labeled A/B/C/D; the gold
label is the letter of the correct answer.

Examples (paraphrased):
    text:    "Если x + 3 = 7, чему равно x?"
    option_a: "2"
    option_b: "3"
    option_c: "4"
    option_d: "5"
    outputs: "C"

This task is the strongest reasoning stress-test in our suite because:

1. Russian language (cross-lingual generality with our English-only tasks).
2. Symbolic logic + arithmetic (cannot be solved by retrieval / lexical
   pattern matching the way SST-2 can).
3. 4-way multi-choice (matches HellaSwag infra in :mod:`dmezo.data.hellaswag`).

**Framing.** Unlike HellaSwag where candidates are example-specific multi-token
endings, MathLogicQA candidates are fixed letters {А, Б, В, Г} in the Cyrillic
script (matches the Russian prompt) — the model "answers" by predicting one
letter. This is identical in structure to SST-2's {" positive", " negative"}
framing, just 4-way instead of 2-way.

For training (MeZO step): use the ground-truth letter only; loss masked to
the suffix token. For evaluation: score all four letter candidates by per-
token log-likelihood, predict argmin loss. The single-token suffix makes
this 4× cheaper than HellaSwag's full-ending scoring.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

MATHLOGICQA_TEMPLATE = (
    "Задача: {text}\n"
    "А) {a}\n"
    "Б) {b}\n"
    "В) {c}\n"
    "Г) {d}\n"
    "Ответ:"
)

# Russian-letter labels chosen so the prompt and the suffix are linguistically
# consistent. Qwen3-class tokenizers encode these as single tokens.
MATHLOGICQA_LABEL_WORDS = {0: " А", 1: " Б", 2: " В", 3: " Г"}

# Map from HF gold-label strings (typically "A"/"B"/"C"/"D" in MERA) to our
# 0/1/2/3 indices. We accept both Latin and Cyrillic forms so the loader is
# robust to schema variations.
_GOLD_TO_INDEX = {
    "A": 0, "B": 1, "C": 2, "D": 3,
    "А": 0, "Б": 1, "В": 2, "Г": 3,
    "1": 0, "2": 1, "3": 2, "4": 3,
}


def format_mathlogicqa_prompt(text: str, a: str, b: str, c: str, d: str) -> str:
    """Build the MathLogicQA prompt (no answer suffix)."""
    return MATHLOGICQA_TEMPLATE.format(text=text, a=a, b=b, c=c, d=d)


def format_mathlogicqa_example(text: str, a: str, b: str, c: str, d: str, label: int) -> str:
    """Build the full prompt + ground-truth letter suffix."""
    return format_mathlogicqa_prompt(text, a, b, c, d) + MATHLOGICQA_LABEL_WORDS[label]


def _gold_to_index(gold: str) -> int:
    """Convert HF gold-label string to our 0/1/2/3 index.

    Raises:
        ValueError: if ``gold`` is not a recognised label.
    """
    g = str(gold).strip()
    if g in _GOLD_TO_INDEX:
        return _GOLD_TO_INDEX[g]
    raise ValueError(
        f"Unrecognised MathLogicQA gold label {gold!r}. "
        f"Expected one of {sorted(_GOLD_TO_INDEX)}."
    )


def _extract_fields(row: dict) -> tuple[str, str, str, str, str, int]:
    """Pull (text, a, b, c, d, label_index) out of a MERA row.

    MERA stores task-specific fields under ``inputs`` (dict) and the gold
    answer under ``outputs`` (str). Tolerant to two known schema variations:

    - ``row['inputs']['text'/'option_a'/...]`` + ``row['outputs']`` (MERA v1.0)
    - ``row['text']`` + ``row['option_a']`` + ... + ``row['answer']``
      (flattened parquet variant)
    """
    if "inputs" in row and isinstance(row["inputs"], dict):
        inp = row["inputs"]
        text = inp.get("text", "")
        a = inp.get("option_a", "")
        b = inp.get("option_b", "")
        c = inp.get("option_c", "")
        d = inp.get("option_d", "")
        gold = row.get("outputs", "")
    else:
        text = row.get("text", "")
        a = row.get("option_a", "")
        b = row.get("option_b", "")
        c = row.get("option_c", "")
        d = row.get("option_d", "")
        gold = row.get("answer", row.get("outputs", row.get("label", "")))
    return text, a, b, c, d, _gold_to_index(gold)


class _MathLogicQADataset(Dataset):
    """Tokenised MathLogicQA for MeZO: prompt + ground-truth letter suffix.

    Loss is computed only on the single answer token (А/Б/В/Г), prompt tokens
    masked with ``-100``.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 256) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        text, a, b, c, d, label = _extract_fields(row)
        prompt = format_mathlogicqa_prompt(text, a, b, c, d)
        suffix = MATHLOGICQA_LABEL_WORDS[label]
        full = prompt + suffix

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


def build_mathlogicqa_loader(
    tokenizer,
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    num_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader for MathLogicQA with MeZO-style framing.

    Args:
        tokenizer: HF tokenizer. ``tokenizer.pad_token_id`` must be set.
        split: ``"train"`` or ``"validation"``. (MERA test labels are public
            but eval is via the leaderboard; for our purposes, use ``train``
            for fitting and ``validation`` for our own eval.)
        batch_size: DataLoader batch size.
        max_length: Max token length per example. 256 fits most MathLogicQA
            prompts (options are short numbers / logic statements).
        shuffle: Shuffle for training.
        num_examples: Subsample to this many examples (``None`` = full split).
        seed: RNG seed for subsampling.

    Returns:
        Torch DataLoader yielding ``(input_ids, attention_mask, labels)``.
    """
    from dmezo.data.superglue import _collate, _load_raw_dataset

    raw = _load_raw_dataset("mathlogicqa", split)
    if num_examples is not None and num_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(num_examples))

    ds = _MathLogicQADataset(raw, tokenizer, max_length=max_length)
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
