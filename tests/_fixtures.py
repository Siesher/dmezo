"""Shared CPU-only fixtures for federated tests.

Not auto-collected by pytest (underscore prefix). Imported by test_*.py files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dmezo.federated.client import ClientState
from dmezo.mezo.step import MeZOConfig


@dataclass
class _TinyCausalOutput:
    """Minimal HF-style output container with a single ``.loss`` attribute."""

    loss: torch.Tensor


class TinyCausalLM(nn.Module):
    """Tiny causal-LM-like module compatible with ``causal_lm_loss``.

    Forward signature matches what ``causal_lm_loss`` in data/superglue.py
    expects: keyword args ``input_ids``, ``attention_mask``, ``labels``; output
    has a ``.loss`` attribute.

    Total params: ~vocab_size * hidden * 2 + linear bias ≈ 2K params with
    defaults. Small enough to fit on CPU for ~ms-fast forward.
    """

    def __init__(self, vocab_size: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002 (kept for API parity)
        labels: torch.Tensor | None = None,
    ) -> _TinyCausalOutput:
        h = self.embed(input_ids)
        logits = self.proj(h)
        if labels is None:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return _TinyCausalOutput(loss=zero)
        # Standard HF causal-LM shift: predict token i+1 from position i.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return _TinyCausalOutput(loss=loss)


def make_tiny_causal_lm(seed: int = 0, vocab_size: int = 32, hidden: int = 16) -> TinyCausalLM:
    """Deterministically initialise a TinyCausalLM with a given seed."""
    torch.manual_seed(seed)
    return TinyCausalLM(vocab_size=vocab_size, hidden=hidden)


class _SyntheticDataset(Dataset):
    def __init__(self, num_examples: int, seq_len: int, vocab_size: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab_size, (num_examples, seq_len), generator=g)
        self.labels = self.input_ids.clone()
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def synthetic_token_loader(
    num_examples: int = 32,
    batch_size: int = 4,
    seq_len: int = 8,
    vocab_size: int = 32,
    seed: int = 0,
) -> DataLoader:
    """Synthetic (input_ids, labels) batches for tests. CPU, deterministic."""
    ds = _SyntheticDataset(
        num_examples=num_examples, seq_len=seq_len, vocab_size=vocab_size, seed=seed
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def make_tiny_clients(
    n: int,
    *,
    mezo_lr: float = 1e-3,
    mezo_eps: float = 1e-3,
    weight_decay: float = 0.0,
    seed_offset: int = 0,
    same_init: bool = False,
) -> list[ClientState]:
    """Create ``n`` ClientState objects with TinyCausalLM models and synthetic data.

    Args:
        n: Number of clients.
        mezo_lr / mezo_eps / weight_decay: MeZO config passed to each client.
        seed_offset: Added to the per-client seed for model init.
        same_init: If True, all clients use seed 0 (parameters identical at start).

    Returns:
        List of ClientState, one per client.
    """
    clients: list[ClientState] = []
    cfg = MeZOConfig(lr=mezo_lr, eps=mezo_eps, weight_decay=weight_decay)
    for i in range(n):
        model_seed = seed_offset if same_init else seed_offset + i
        model = make_tiny_causal_lm(seed=model_seed)
        for p in model.parameters():
            p.requires_grad_(True)
        loader = synthetic_token_loader(seed=100 + i)
        clients.append(
            ClientState(
                client_id=i,
                model=model,
                dataloader=loader,
                mezo_config=cfg,
                rng=np.random.default_rng(1000 + i),
            )
        )
    return clients
