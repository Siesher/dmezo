"""Composite Figure 14: cross-arch ε-schedule ablation.

Side-by-side comparison of 5 ε(t) schedules on Qwen3-0.6B (full-attn) and
Qwen3.5-0.8B (hybrid linear-attn). Loads eps_schedule_*.json for both models
and produces a 1x2 panel showing eval@clean-θ trajectories with the final
drop annotated per schedule.

Usage::

    python scripts/compose_fig14_eps_schedule.py
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
logger = logging.getLogger("dmezo.fig14")

MODELS = [
    {"slug": "Qwen_Qwen3-0p6B",   "label": "Qwen3-0.6B  (full-attention)"},
    {"slug": "Qwen_Qwen3p5-0p8B", "label": "Qwen3.5-0.8B  (hybrid linear-attention)"},
]

# Stable colour mapping per schedule across both panels.
SCHED_COLOURS = {
    "const_1e-3":             "#1f77b4",  # blue — Princeton baseline
    "exp_decay_1e-3_to_1e-4": "#2ca02c",  # green — classical SPSA refinement
    "exp_decay_1e-2_to_1e-3": "#ff7f0e",  # orange — mild Spall warmup
    "exp_decay_3e-2_to_1e-3": "#d62728",  # red — autotuner-style warmup
    "exp_grow_1e-3_to_1e-2":  "#9467bd",  # purple — anti-schedule
}


def _load(slug: str) -> dict:
    path = ROOT / "experiments" / "diagnostics" / f"eps_schedule_{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    out_path = ROOT / "docs" / "figures" / "fig14_eps_schedule_cross_arch.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
            "lines.linewidth": 1.6,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col_idx, m in enumerate(MODELS):
        ax = axes[col_idx]
        d = _load(m["slug"])
        for sched_name, r in d["runs"].items():
            evals = np.array(r["eval_losses"])
            steps = np.array(r["eval_steps"])
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            col = SCHED_COLOURS.get(sched_name, "#7f7f7f")
            ax.plot(steps, evals, "o-", color=col, markersize=4.5, alpha=0.9,
                    label=f"{sched_name}  ({drop:+.1f}%)")
        ax.set_xlabel("MeZO step")
        if col_idx == 0:
            ax.set_ylabel(r"eval loss at clean $\theta$  (fixed batch)")
        ax.set_title(m["label"], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "ε(t) schedule ablation: warmup-style and grow schedules both lose to constant ε=10⁻³",
        fontsize=11, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
