"""Lightweight JSONL logger for experiments + console logger setup."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str = "dmezo", level: int = logging.INFO) -> logging.Logger:
    """Console logger with timestamp + level + name."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


class JSONLLogger:
    """Append one JSON object per line to a log file. Robust to crashes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on first open so we start fresh.
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        """Write one record as a JSON line and flush immediately."""
        line = json.dumps(record, default=_json_default, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(obj):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)
