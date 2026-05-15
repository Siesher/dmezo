"""SuperGLUE / GLUE data preparation for MeZO.

Tasks framed as causal LM completion: prompt the model with the input + a label
suffix and measure cross-entropy on the label tokens only. This is the framing
used in Malladi 2023 (MeZO) for OPT models.

Supported tasks:
    - sst2: binary sentiment (GLUE)
    - boolq: yes/no question answering on a passage (SuperGLUE)
"""

from __future__ import annotations

import numpy as np
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

# Task -> (Dataset class, label column name) for partitioned-loader factory.
TASK_DATASETS = {
    "sst2": (_SST2Dataset, "label"),
    "boolq": (_BoolQDataset, "label"),
}

# Task -> default DataLoader kwargs for partitioned mode.
TASK_DEFAULTS = {
    "sst2": {"batch_size": 8, "max_length": 256},
    "boolq": {"batch_size": 4, "max_length": 512},
}


def build_loader_for_task(task: str, **kwargs) -> DataLoader:
    """Dispatch to the right loader factory based on task name."""
    if task not in TASK_LOADERS:
        raise ValueError(f"Unknown task {task!r}. Supported: {sorted(TASK_LOADERS)}")
    return TASK_LOADERS[task](**kwargs)


def _load_raw_dataset(task: str, split: str):
    """Load the raw HF dataset for a given task and split.

    Centralised so tests can monkeypatch this function and inject synthetic data
    without hitting the network.
    """
    from datasets import load_dataset

    if task == "sst2":
        return load_dataset("glue", "sst2", split=split)
    if task == "boolq":
        return load_dataset("super_glue", "boolq", split=split, trust_remote_code=True)
    raise ValueError(f"Unknown task {task!r}. Supported: {sorted(TASK_DATASETS)}")


def build_partitioned_loaders(
    task: str,
    tokenizer,
    n_clients: int,
    partition_mode: str = "iid",
    partition_kwargs: dict | None = None,
    split: str = "train",
    batch_size: int | None = None,
    max_length: int | None = None,
    num_examples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[DataLoader]:
    """Build per-client DataLoaders with a disjoint partition of the train split.

    Partition modes (federated learning literature):
        - ``"iid"``: uniform-random disjoint chunks (McMahan et al. 2017,
          "Communication-Efficient Learning of Deep Networks from Decentralized
          Data").
        - ``"dirichlet"``: per-class Dirichlet(alpha) over clients (Hsu et al.
          2019, "Measuring the Effects of Non-Identical Data Distribution for
          Federated Visual Classification"). Requires
          ``partition_kwargs={"alpha": float}``. Common values: 0.1 (very
          heterogeneous), 0.5 (mild), 10.0 (close to IID).
        - ``"label_skew"``: each client owns ``classes_per_client`` classes
          only — extreme non-IID worst case. Requires
          ``partition_kwargs={"classes_per_client": int}``.

    The returned loaders are disjoint at the example level: every example from
    the (sub)sampled train pool is owned by exactly one client.

    Args:
        task: Task name (``"sst2"`` or ``"boolq"``).
        tokenizer: HF tokenizer (must have ``pad_token_id``).
        n_clients: Number of federated clients.
        partition_mode: One of ``"iid"``, ``"dirichlet"``, ``"label_skew"``.
        partition_kwargs: Mode-specific options (``alpha`` for Dirichlet,
            ``classes_per_client`` for label-skew).
        split: HF split name (default ``"train"``).
        batch_size: DataLoader batch size; defaults to task convention.
        max_length: Max tokens per example; defaults to task convention.
        num_examples: Subsample the train pool to this size before partitioning.
            ``None`` = use full split.
        shuffle: Whether each client's loader shuffles its examples.
        seed: RNG seed for subsampling and partitioning.

    Returns:
        List of ``n_clients`` DataLoaders, in client-id order.

    Raises:
        ValueError: On unknown ``task`` or ``partition_mode``.
    """
    if task not in TASK_DATASETS:
        raise ValueError(f"Unknown task {task!r}. Supported: {sorted(TASK_DATASETS)}")

    dataset_cls, label_col = TASK_DATASETS[task]
    defaults = TASK_DEFAULTS[task]
    bs = batch_size if batch_size is not None else defaults["batch_size"]
    ml = max_length if max_length is not None else defaults["max_length"]
    pk = dict(partition_kwargs or {})

    raw = _load_raw_dataset(task, split)
    if num_examples is not None and num_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(num_examples))

    n_total = len(raw)
    if partition_mode == "iid":
        from dmezo.data.partition import iid_partition

        client_indices = iid_partition(n_total, n_clients, seed=seed)
    elif partition_mode == "dirichlet":
        from dmezo.data.partition import dirichlet_partition

        labels = np.asarray(raw[label_col])
        alpha = float(pk.get("alpha", 0.5))
        client_indices = dirichlet_partition(labels, n_clients, alpha=alpha, seed=seed)
    elif partition_mode == "label_skew":
        from dmezo.data.partition import label_skew_partition

        labels = np.asarray(raw[label_col])
        cpc = int(pk.get("classes_per_client", 1))
        client_indices = label_skew_partition(labels, n_clients, classes_per_client=cpc, seed=seed)
    else:
        raise ValueError(
            f"Unknown partition_mode {partition_mode!r}. "
            "Supported: 'iid', 'dirichlet', 'label_skew'."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None; set tokenizer.pad_token first")

    def make_collate(pid: int):
        return lambda batch: _collate(batch, pad_token_id=pid)

    loaders: list[DataLoader] = []
    for ci in range(n_clients):
        idx = client_indices[ci]
        if len(idx) == 0:
            # A client may get zero examples under extreme low-alpha Dirichlet on
            # tiny pools. Keep the client_id -> loader mapping consistent by
            # constructing an empty loader.
            subset = raw.select([])
        else:
            subset = raw.select(idx.tolist())
        ds = dataset_cls(subset, tokenizer, max_length=ml)
        loaders.append(
            DataLoader(
                ds,
                batch_size=bs,
                shuffle=shuffle,
                collate_fn=make_collate(pad_id),
                num_workers=0,
            )
        )
    return loaders
