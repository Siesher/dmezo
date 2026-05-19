"""Composite figure: ε(t) schedule ablation × {vanilla, D-MeZO-N} × {0.6B, 1.7B}.

2x2 panel showing eval-loss-at-clean-θ trajectories. Each cell contrasts
3 schedules: const_1e-3 (Princeton), exp_decay 1e-3→1e-4 (refine-below),
exp_decay 1e-2→1e-3 (warmup-style). Rows = variant (vanilla vs D-MeZO-N),
columns = model scale (Qwen3-0.6B vs Qwen3-1.7B).

Annotates each schedule trajectory with the final drop %; clear visual of
(a) ε(t) ordering preservation across variants and scales,
(b) D-MeZO-N's clip+momentum overhead on benign tasks,
(c) shrinking warmup penalty at larger scale.

Usage::

    python scripts/compose_fig15_dmezo_schedule.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.fig15")

CELLS = [
    # (row, col, variant, slug, label)
    (0, 0, "vanilla", "Qwen_Qwen3-0p6B",  "Vanilla MeZO · Qwen3-0.6B"),
    (0, 1, "vanilla", "Qwen_Qwen3-1p7B",  "Vanilla MeZO · Qwen3-1.7B"),
    (1, 0, "dmezo_n", "Qwen_Qwen3-0p6B",  "D-MeZO-N · Qwen3-0.6B"),
    (1, 1, "dmezo_n", "Qwen_Qwen3-1p7B",  "D-MeZO-N · Qwen3-1.7B"),
]

SCHED_COLOURS = {
    "const_1e-3":             "#1f77b4",
    "exp_decay_1e-3_to_1e-4": "#2ca02c",
    "exp_decay_1e-2_to_1e-3": "#d62728",
}


def _load(variant: str, slug: str) -> dict:
    path = ROOT / "experiments" / "diagnostics" / f"eps_schedule_{variant}_{slug}.json"
    if not path.exists():
        # The vanilla 0.6B / 0.8B runs were saved under the old naming
        # (no variant tag) by ablate_eps_schedule.py. Fall back to that file
        # for the vanilla 0.6B case.
        alt = ROOT / "experiments" / "diagnostics" / f"eps_schedule_{slug}.json"
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"{path} (and fallback {alt}) missing")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    out_path = ROOT / "docs" / "figures" / "fig15_eps_schedule_dmezo_composite.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
            "lines.linewidth": 1.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)

    for (row, col, variant, slug, label) in CELLS:
        ax = axes[row, col]
        d = _load(variant, slug)
        # Restrict to 3 schedules of interest (older JSONs may have 5).
        keep = ["const_1e-3", "exp_decay_1e-3_to_1e-4", "exp_decay_1e-2_to_1e-3"]
        for sched_name in keep:
            if sched_name not in d["runs"]:
                continue
            r = d["runs"][sched_name]
            evals = np.array(r["eval_losses"])
            steps = np.array(r["eval_steps"])
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            col_hex = SCHED_COLOURS[sched_name]
            ax.plot(steps, evals, "o-", color=col_hex, markersize=4.5, alpha=0.9,
                    label=f"{sched_name}  ({drop:+.1f}%)")

        ax.set_title(label, fontsize=10)
        if row == 1:
            ax.set_xlabel("MeZO step")
        if col == 0:
            ax.set_ylabel(r"eval loss at clean $\theta$")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "ε(t) schedule ablation across variants × scales:  ordering robust, warmup penalty shrinks with scale",
        fontsize=11, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
