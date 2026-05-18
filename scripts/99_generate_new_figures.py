"""Generate fig6 (cross-domain trajectories) + fig7 (cross-task summary).

fig6 — two-panel trajectory plot showing:
  Left:  HellaSwag (Qwen3-4B) — vanilla DIVERGES, D-MeZO-N RESCUES
  Right: MathLogicQA (Qwen3.5-4B-Base) — vanilla converges, D-MeZO-N safe-tracks

fig7 — horizontal bar chart of accuracy gain (Δacc vs centralized vanilla)
across the four task domains.

Data points are extracted from the Colab logs reproduced in the paper §5.5
and §5.6 tables. Single-seed for HellaSwag and MathLogicQA (note in caption).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "lines.linewidth": 2.0,
    }
)

# Color scheme matching existing paper figures.
COL_VANILLA = "#d62728"   # red — vanilla baseline
COL_DMEZON = "#2ca02c"    # green — D-MeZO-N v1
COL_INIT = "#888888"      # grey — init reference


# ----------------------------------------------------------------------------
# Trajectory data extracted from Colab logs (eval at every 100 rounds).
# ----------------------------------------------------------------------------

# HellaSwag / Qwen3-4B / 2026-05-18, MLflow runs 561706... and bae89d...
HELLASWAG = {
    "rounds": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "central_loss": [2.5691, 2.5778, 2.5700, 2.6068, 2.6317, 2.6182, 2.6347, 2.6817, 2.6839, 2.7375, 2.7112],
    "federated_loss": [2.5691, 2.5363, 2.5183, 2.5153, 2.5078, 2.4983, 2.4967, 2.4954, 2.4853, 2.4899, 2.4959],
    "central_acc": [0.6625, None, None, None, None, None, None, None, None, None, 0.6375],
    "federated_acc": [0.6625, 0.6750, 0.6750, 0.6750, 0.6875, 0.6750, 0.6875, 0.6875, 0.7000, 0.6875, 0.7000],
}

# MathLogicQA / Qwen3.5-4B-Base / 2026-05-18, MLflow runs 07e355... and 76a8b5...
MATHLOGICQA = {
    "rounds": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "central_loss": [2.8493, 2.0029, 1.7475, 1.6407, 1.5523, 1.5524, 1.5281, 1.4709, 1.4790, 1.4863, 1.4331],
    "federated_loss": [2.8493, 1.8947, 1.7013, 1.6459, 1.5956, 1.5825, 1.5868, 1.5419, 1.5469, 1.5480, 1.5155],
    "central_acc": [0.3750, None, None, None, None, None, None, None, None, None, 0.3750],
    "federated_acc": [0.3750, 0.3625, 0.3750, 0.3750, 0.3625, 0.4125, 0.3625, 0.4000, 0.3625, 0.3500, 0.3875],
}


# ----------------------------------------------------------------------------
# fig6 — cross-domain trajectories (rescue vs safe-track)
# ----------------------------------------------------------------------------


def figure6_cross_domain_trajectories():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- Left: HellaSwag (rescue pattern)
    rounds = HELLASWAG["rounds"]
    axL.axhline(2.5691, color=COL_INIT, linestyle=":", linewidth=1.0, label="init (both)")
    axL.plot(rounds, HELLASWAG["central_loss"], "o-", color=COL_VANILLA,
             markersize=5, label="Centralized vanilla MeZO")
    axL.plot(rounds, HELLASWAG["federated_loss"], "s-", color=COL_DMEZON,
             markersize=5, label="Federated D-MeZO-N v1")
    axL.annotate("DIVERGES\n(loss +5.5%,\nacc −2.5pp)", xy=(900, 2.7375),
                 xytext=(700, 2.78), fontsize=9, color=COL_VANILLA,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COL_VANILLA, lw=1.0))
    axL.annotate("RESCUES\n(loss −2.85%,\nacc +3.75pp)", xy=(900, 2.4899),
                 xytext=(700, 2.40), fontsize=9, color=COL_DMEZON,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COL_DMEZON, lw=1.0))
    axL.set_xlabel("Round (= MeZO step for centralized)")
    axL.set_ylabel("Eval loss")
    axL.set_title("(a) HellaSwag / Qwen3-4B — RESCUE regime")
    axL.legend(loc="lower left")
    axL.set_xlim(-30, 1030)

    # ---- Right: MathLogicQA (safe-track pattern)
    rounds = MATHLOGICQA["rounds"]
    axR.axhline(2.8493, color=COL_INIT, linestyle=":", linewidth=1.0, label="init (both)")
    axR.plot(rounds, MATHLOGICQA["central_loss"], "o-", color=COL_VANILLA,
             markersize=5, label="Centralized vanilla MeZO")
    axR.plot(rounds, MATHLOGICQA["federated_loss"], "s-", color=COL_DMEZON,
             markersize=5, label="Federated D-MeZO-N v1")
    axR.annotate("converges\n(loss −49.7%,\nacc unchanged)", xy=(1000, 1.4331),
                 xytext=(700, 2.0), fontsize=9, color=COL_VANILLA,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COL_VANILLA, lw=1.0))
    axR.annotate("safe-tracks\n(loss −46.8%,\nacc +1.25pp)", xy=(1000, 1.5155),
                 xytext=(700, 2.5), fontsize=9, color=COL_DMEZON,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COL_DMEZON, lw=1.0))
    axR.set_xlabel("Round")
    axR.set_ylabel("Eval loss")
    axR.set_title("(b) MathLogicQA / Qwen3.5-4B-Base — SAFE-TRACK regime")
    axR.legend(loc="upper right")
    axR.set_xlim(-30, 1030)

    plt.tight_layout()
    out = OUT / "fig6_cross_domain_trajectories.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ----------------------------------------------------------------------------
# fig7 — cross-task summary (accuracy gain bar chart)
# ----------------------------------------------------------------------------


def figure7_cross_task_summary():
    """Horizontal bar chart: Δ(D-MeZO-N − vanilla) accuracy across task domains.

    Note: SST-2 results are reported as final-loss reduction (the task converges
    quickly on accuracy); HellaSwag / MathLogicQA report final-acc Δ pp.
    """
    tasks = [
        ("SST-2\n(en, sentiment)", "Qwen3-4B", "6.5", "loss reduction", "speedup"),
        ("HellaSwag\n(en, commonsense)", "Qwen3-4B", "6.25", "pp accuracy", "rescue"),
        ("MathLogicQA\n(ru, symbolic)", "Qwen3.5-4B-Base", "1.25", "pp accuracy", "safe-track"),
    ]
    # Δ values
    deltas = [6.5, 6.25, 1.25]
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    labels = [
        f"{t[0]}\n[{t[1]}]\n({t[4]})" for t in tasks
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, deltas, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    # Annotate
    for bar, t in zip(bars, tasks):
        w = bar.get_width()
        unit = "% loss" if t[3].startswith("loss") else "pp acc"
        ax.text(w + 0.15, bar.get_y() + bar.get_height() / 2,
                f"+{w} {unit}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Δ (D-MeZO-N v1) − (centralized vanilla MeZO)")
    ax.set_title("Cross-task improvement of D-MeZO-N v1 (same recipe: β-decay 0.9→0 + ρ-clip=50)")
    ax.set_xlim(0, 9)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    out = OUT / "fig7_cross_task_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    figure6_cross_domain_trajectories()
    figure7_cross_task_summary()
