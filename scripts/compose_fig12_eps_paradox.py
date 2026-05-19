"""Composite Figure 12: autotuner score vs downstream eval — the ε paradox.

Loads autotuner JSONs (extended grid) and downstream-validation JSONs for both
models, draws a 2x2 figure:

    Top row    — autotuner J(ε) score across the ε grid (with divergence
                 markers and the picked ε*).
    Bottom row — eval-loss-at-clean-θ trajectory across 100 MeZO steps for
                 ε ∈ {Princeton default, autotuner-safe, autotuner-best}.
    Left col   — Qwen3-0.6B (full-attention).
    Right col  — Qwen3.5-0.8B (hybrid linear-attention).

Annotates each downstream panel with the final drop %, so the contradiction
between autotuner recommendation and actual training performance is visible
in a single figure.

Usage::

    python scripts/compose_fig12_eps_paradox.py
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
logger = logging.getLogger("dmezo.fig12")

MODELS = [
    {
        "slug": "Qwen_Qwen3-0p6B",
        "label": "Qwen3-0.6B  (full-attention)",
    },
    {
        "slug": "Qwen_Qwen3p5-0p8B",
        "label": "Qwen3.5-0.8B  (hybrid linear-attention)",
    },
]


def _load(prefix: str, slug: str) -> dict:
    path = ROOT / "experiments" / "diagnostics" / f"{prefix}_{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    out_path = ROOT / "docs" / "figures" / "fig12_eps_autotuner_paradox.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.6,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex="row")

    # Stable colour map for the per-ε downstream trajectories.
    EPS_COLOURS = {
        "1.00e-03": "#9467bd",  # purple — Princeton default
        "3.00e-02": "#2ca02c",  # green  — autotuner safe
        "1.00e-01": "#ff7f0e",  # orange — autotuner best (full-attn)
        "3.00e-01": "#d62728",  # red    — autotuner best (hybrid)
    }

    for col_idx, m in enumerate(MODELS):
        slug = m["slug"]
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # ---- Top row: autotuner J(ε) score.
        d_auto = _load("eps_warmup", slug)
        eps_vals = np.array([s["eps"] for s in d_auto["scores"].values()])
        j_norm = np.array([s["j_norm"] for s in d_auto["scores"].values()])
        diverged = np.array(
            [s.get("diverged", False) for s in d_auto["scores"].values()]
        )
        eps_star = d_auto["eps_star"]

        ax_top.semilogx(eps_vals, j_norm, "o-", color="#1f77b4", markersize=6,
                        label=r"$J(\varepsilon)$  (autotuner score)")
        if diverged.any():
            ax_top.scatter(
                eps_vals[diverged], np.zeros(diverged.sum()),
                marker="x", s=90, color="#d62728", zorder=5,
                label="diverged",
            )
        ax_top.axvline(1e-3, color="#9467bd", linestyle=":", linewidth=1.5,
                       alpha=0.9, label=r"Princeton  $\varepsilon=10^{-3}$")
        ax_top.axvline(eps_star, color="black", linestyle="--", linewidth=1.3,
                       alpha=0.7, label=rf"autotuner $\varepsilon^* = {eps_star:.0e}$")
        ax_top.set_ylabel(r"$J(\varepsilon)$  (normalised bias + var)" if col_idx == 0 else "")
        ax_top.set_xlabel(r"$\varepsilon$")
        ax_top.set_title(f"(a) Autotuner score — {m['label']}", fontsize=10)
        ax_top.legend(loc="upper right", fontsize=8)

        # ---- Bottom row: downstream eval-loss trajectory.
        d_val = _load("eps_validate", slug)
        for eps_str, r in d_val["runs"].items():
            evals = np.array(r["eval_losses"])
            steps = np.array(r.get("eval_steps", np.arange(len(evals)) * 10))
            col = EPS_COLOURS.get(eps_str, "#7f7f7f")
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100
            tag = "Princeton" if eps_str == "1.00e-03" else "autotune"
            label = rf"$\varepsilon={r['eps']:.0e}$  ({tag}, {drop:+.1f}%)"
            ax_bot.plot(steps, evals, "o-", color=col, label=label,
                        markersize=4.5, alpha=0.9)

        ax_bot.set_xlabel("MeZO step")
        ax_bot.set_ylabel(r"eval loss at clean $\theta$" if col_idx == 0 else "")
        ax_bot.set_title(f"(b) Downstream training — {m['label']}", fontsize=10)
        ax_bot.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "ε autotuner vs downstream: variance proxy predicts the WRONG ε",
        fontsize=12, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
