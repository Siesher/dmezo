"""Master results table figure (fig20) for D-MeZO-N paper.

Consolidated landscape view of ALL numerical evidence in one figure:
    - §5.2 Day 5 federated grid (4 cells, 2 seeds)
    - §5.4 Day 8 D-MeZO-N v1 vs control (single seed)
    - §5.5 HellaSwag rescue (single seed, multi-seed pending)
    - §5.6 MathLogicQA Pareto (single seed)
    - §6.8 joint sweep best cells

Each row has:
    - Section
    - Setup string
    - Metric value(s)
    - Tier badge (Robust / Tentative / Negative)
    - Evidence column with n_seeds + replication

Designed as table-style figure for at-a-glance review. Matplotlib-based,
exported at high DPI for paper use.

Usage::

    python scripts/compose_fig20_master_results.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.fig20")


# Each row: (section, setup, metric, tier, evidence)
ROWS = [
    {
        "section": "§5.2",
        "setup":   "Day 5 federated 2×2 grid\n(topology × distribution)\nQwen3.5-4B-Base / SST-2",
        "metric":  "ring+IID: 0.1271 ± 0.0014\nring+Dir: 0.1402 ± 0.0029\ncomplete+IID: 0.1348 ± 0.0051\ncomplete+Dir: 0.1507 ± 0.0089",
        "tier":    "robust",
        "evidence": "2 seeds × 4 cells = 8 runs\n(partition variance entangled)",
    },
    {
        "section": "§5.4",
        "setup":   "Day 8 R1d D-MeZO-N v1\nvs ablation control (no momentum)\nring+Dir worst cell, seed=42",
        "metric":  "R1d: 0.1291\ncontrol: 0.1373\nimprovement: 6.0%",
        "tier":    "tentative",
        "evidence": "1 seed paired comparison\n(bit-exact for ablation)",
    },
    {
        "section": "§5.5",
        "setup":   "HellaSwag rescue\nQwen3-4B (standard transformer)\n1000 rounds",
        "metric":  "vanilla: 2.5691 → 2.7112  (+5.5% loss, divergence)\nD-MeZO-N: 2.5691 → 2.4959  (−2.85% loss)\nΔacc: +6.25pp absolute",
        "tier":    "tentative",
        "evidence": "1 seed; SE_acc on 200-eval ≈ ±3.4pp\nmulti-seed pending (Section 19)",
    },
    {
        "section": "§5.6",
        "setup":   "MathLogicQA Pareto\nQwen3.5-4B-Base hybrid\nRussian symbolic reasoning",
        "metric":  "vanilla: −49.7% loss / +0pp acc\nD-MeZO-N: −46.8% loss / +1.25pp acc\n(Pareto trade-off, not dominance)",
        "tier":    "tentative",
        "evidence": "1 seed\nExplicit Pareto framing in §5.6",
    },
    {
        "section": "§6.4",
        "setup":   "Batch-variance CLT failure\nQwen3-0.6B / SST-2, fixed z\nB ∈ {1, 2, 4, 8, 16, 32}",
        "metric":  "Observed σ / CLT-expected ratio\n1.55× (B=2) → 3.43× (B=32)\nmonotonic deviation from 1/√B",
        "tier":    "robust",
        "evidence": "6-point monotonic trend\nDirect cross-B comparison",
    },
    {
        "section": "§6.7",
        "setup":   "ε-autotuner failure\nQwen3-0.6B + Qwen3.5-0.8B\nautotuner picks ε* ∈ {1e-1, 3e-1}",
        "metric":  "Princeton ε=1e-3 downstream drop: −60.8% / −63.1%\nautotuner ε* drop: −13.9% / +5.3% (loss UP)\n3-6× worse",
        "tier":    "robust",
        "evidence": "Cross-arch × cross-stage\n4 independent confirmations",
    },
    {
        "section": "§6.7",
        "setup":   "Warmup ε(t) systematically loses\n5 schedules × cross-arch\n+ cross-variant",
        "metric":  "Warmup ε(t) loses const ε=1e-3 in\n16+ cells out of 16 examined\n(SST-2 + HellaSwag + 4B)",
        "tier":    "robust",
        "evidence": "Cross-arch + cross-variant + cross-task\nmechanism: Taylor validity boundary",
    },
    {
        "section": "§6.7+",
        "setup":   "Richardson 4-pt + 6-pt\nfalsification of higher-order rescue\nQwen3-0.6B + Qwen3.5-0.8B",
        "metric":  "Unit tests on quintic: PASS (analytic ε² and ε⁴ cancel)\nLLM downstream: dominated by 2-pt at Princeton ε",
        "tier":    "negative",
        "evidence": "128 tests including\nanalytic finite-diff verification",
    },
    {
        "section": "§6.8",
        "setup":   "Joint lr × ε × variant sweep\nQwen3.5-4B-Base / HellaSwag\n24 cells (4 lr × 3 sched × 2 var) × 500 steps",
        "metric":  "Best vanilla: η=1e-6 + const ε=1e-3 → +1.0% drop\nBest D-MeZO-N: η=3e-7 + const → +0.3% drop\nbroad lr=3e-6 catastrophic",
        "tier":    "tentative",
        "evidence": "1 seed × 24 cells\nNear-saturation task; gap inside SE",
    },
]

TIER_COLOURS = {
    "robust":      ("#d1f0d8", "#2ca02c", "[Robust]"),
    "tentative":   ("#fff4d1", "#ff7f0e", "[Tentative]"),
    "exploratory": ("#e8e8e8", "#7f7f7f", "[Exploratory]"),
    "negative":    ("#f5d6d6", "#d62728", "[Negative]"),
}


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    n_rows = len(ROWS)
    fig_height = 1.0 + n_rows * 1.05
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.995, "D-MeZO-N: Master Results Table — All Numerical Evidence at a Glance",
            fontsize=13, fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.965,
            "Each row: section + setup + metric + statistical-rigor tier + evidence quality.\n"
            "Robust findings cross-replicated; Tentative single-seed; Negative = clean falsification of a stated hypothesis.",
            fontsize=8.5, ha="center", va="top", color="#444", style="italic")

    # Column boundaries
    col_x = {
        "section":  (0.01, 0.07),
        "setup":    (0.075, 0.32),
        "metric":   (0.325, 0.62),
        "tier":     (0.625, 0.74),
        "evidence": (0.745, 0.99),
    }

    # Column headers
    header_y = 0.935
    for col, (x0, x1) in col_x.items():
        ax.text((x0 + x1) / 2, header_y, col.upper(), fontsize=9.5, fontweight="bold",
                ha="center", va="center", color="#333")
    # Separator under header
    ax.plot([0.01, 0.99], [header_y - 0.012, header_y - 0.012], color="#888", linewidth=0.8)

    # Rows
    row_height = 0.91 / (n_rows + 1)
    for i, row in enumerate(ROWS):
        y_top = header_y - 0.025 - i * row_height
        y_bot = y_top - row_height + 0.008
        y_mid = (y_top + y_bot) / 2
        # Alternate row shading
        if i % 2 == 0:
            ax.add_patch(patches.Rectangle((0.005, y_bot - 0.003), 0.99, row_height - 0.003,
                                            facecolor="#f8f8f8", edgecolor="none", zorder=0))
        # Section (left-aligned)
        ax.text(col_x["section"][0] + 0.005, y_mid, row["section"],
                fontsize=9, fontweight="bold", ha="left", va="center", color="#1f77b4")
        # Setup
        ax.text(col_x["setup"][0] + 0.005, y_mid, row["setup"],
                fontsize=8.2, ha="left", va="center", color="#222")
        # Metric (monospace)
        ax.text(col_x["metric"][0] + 0.005, y_mid, row["metric"],
                fontsize=8.0, ha="left", va="center", color="#222", family="monospace")
        # Tier badge
        fill_col, border_col, tier_label = TIER_COLOURS[row["tier"]]
        tier_box = patches.FancyBboxPatch(
            (col_x["tier"][0] + 0.005, y_mid - 0.018),
            col_x["tier"][1] - col_x["tier"][0] - 0.015, 0.036,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            linewidth=1.0, edgecolor=border_col, facecolor=fill_col,
        )
        ax.add_patch(tier_box)
        ax.text((col_x["tier"][0] + col_x["tier"][1]) / 2, y_mid,
                tier_label, fontsize=8.5, fontweight="bold",
                ha="center", va="center", color=border_col)
        # Evidence
        ax.text(col_x["evidence"][0] + 0.005, y_mid, row["evidence"],
                fontsize=8.2, ha="left", va="center", color="#333", style="italic")

    # Bottom footer
    ax.text(0.5, 0.005,
            "All Tentative rows benefit from multi-seed validation. "
            "See docs/robustness_matrix.md for full statistical classification.",
            fontsize=8.0, ha="center", va="bottom", color="#666", style="italic")

    out_path = ROOT / "docs" / "figures" / "fig20_master_results.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
