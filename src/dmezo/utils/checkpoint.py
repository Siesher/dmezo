"""Lightweight checkpoint save/load for MeZO state (no optimizer to save)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    step: int,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Save model state_dict + step + metadata.

    For MeZO, there is no optimizer state to save, which keeps checkpoints
    small (just model weights). For Nesterov, save the velocities separately.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta_path = p.with_suffix(".json")
    payload = {"step": step, **(metadata or {})}
    meta_path.write_text(json.dumps(payload, default=str, indent=2))


def load_checkpoint(path: str | Path, model: nn.Module, strict: bool = True) -> Dict[str, Any]:
    """Load model state_dict and return metadata (if present)."""
    p = Path(path)
    state = torch.load(p, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    meta_path = p.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}
