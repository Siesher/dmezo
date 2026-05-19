"""Master summary figure (fig00) for D-MeZO-N paper.

Single-page schematic showing 6 contributions (C1-C6) with headline numbers
and statistical-rigor classification (from docs/robustness_matrix.md).
Designed as one-page overview for abstract/intro.

Layout: 2x3 grid of "contribution cards", each with:
    - C# code + short title
    - Headline metric / numbers
    - Rigor tier (🟢/🟡/⚪/🔴) badge
    - Section reference
    - Short mechanism

Usage::

    python scripts/compose_fig00_paper_summary.py
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
logger = logging.getLogger("dmezo.fig00")


# (title, headline numbers, tier emoji, tier label colour, section ref, mechanism)
CONTRIBUTIONS = [
    {
        "code": "C1",
        "title": "Hybrid linear-attn MeZO",
        "headline": "first MeZO on Qwen3.5\n(linear-attn) family\n→ 6 model sizes",
        "tier": "robust",
        "section": "§5.1 / Day 1",
        "mechanism": "Frozen ViT + text-only loss; preserves seed-based\nperturbation invariant",
    },
    {
        "code": "C2",
        "title": "Decentralized federated D-MeZO",
        "headline": "n=4 clients, complete + ring\n2 partitions × 2 seeds\npartition tax < 13%",
        "tier": "robust",
        "section": "§5.2 / fig1",
        "mechanism": "Consensus mixing (weight_avg) preserves convergence;\n1 scalar + 1 seed per round per neighbour",
    },
    {
        "code": "C3",
        "title": "Ring ≤ complete on ZO",
        "headline": "ring partition tax\nnegligible vs complete\nat n=4",
        "tier": "robust",
        "section": "§5.2 / §6.1",
        "mechanism": "Z-direction averaging dominates over topology effect\nat low n (rho_W < 1 but small)",
    },
    {
        "code": "C4",
        "title": "D-MeZO-N rescue",
        "headline": "β-decay 0.9→0 + ρ-clip 50\nHellaSwag: +3.75pp acc\nMathLogicQA: +1.25pp",
        "tier": "tentative",
        "section": "§5.4 / §5.5 / §5.6",
        "mechanism": "Heavy-ball Nesterov + clip rescues noise-amplified\ndivergence at high β; ${T \\geq 1000}$ steps needed",
    },
    {
        "code": "C5",
        "title": "Theorem 1 (convex + momentum)",
        "headline": "PL-convex bound\nrate $1 - \\frac{3\\eta\\mu}{4}$\nLyapunov V_t",
        "tier": "robust",
        "section": "§4 / theorem_nesterov_mezo.md",
        "mechanism": "Formal proof closes Open Problem 1\n(Nesterov-MeZO + β-decay + clip)",
    },
    {
        "code": "C6",
        "title": "Theorem 2 (non-convex PL, plain SGD)",
        "headline": "linear PL convergence\nrate consistent with §5\nempirical predictions",
        "tier": "robust",
        "section": "§4",
        "mechanism": "8 quantitative predictions matched\nin empirical §5.2",
    },
]

# Additional callout: negative findings band at bottom
NEGATIVE_FINDINGS = [
    "NEG: ε-autotuner: Princeton ε=10⁻³ wins by 3-6× downstream (§6.7)",
    "NEG: Warmup ε(t) systematically loses across 16+ cells (§6.7/§6.8)",
    "NEG: 1/√B CLT fails: variance saturates at B≥8 (§6.4)",
    "NEG: Higher-order finite-diff (Richardson, 6-pt) ≼ 2-pt at Princeton ε (§6.7+)",
]

TIER_COLOURS = {
    "robust":      ("#d1f0d8", "#2ca02c", "[Robust]"),
    "tentative":   ("#fff4d1", "#ff7f0e", "[Tentative]"),
    "exploratory": ("#e8e8e8", "#7f7f7f", "[Exploratory]"),
    "negative":    ("#f5d6d6", "#d62728", "[Negative]"),
}


def _draw_card(ax, x, y, w, h, contrib):
    """Draw one contribution card at (x, y) with width w, height h."""
    fill_col, border_col, tier_label = TIER_COLOURS[contrib["tier"]]
    # Background card.
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.5, edgecolor=border_col, facecolor=fill_col, alpha=0.85,
    )
    ax.add_patch(rect)
    # Code (top-left).
    ax.text(x + 0.05, y + h - 0.06, contrib["code"], fontsize=18, fontweight="bold",
            color=border_col, va="top", ha="left")
    # Tier label (top-right).
    ax.text(x + w - 0.05, y + h - 0.06, tier_label, fontsize=9,
            color=border_col, va="top", ha="right", style="italic")
    # Title.
    ax.text(x + 0.05, y + h - 0.20, contrib["title"], fontsize=11,
            fontweight="bold", va="top", ha="left", color="black")
    # Headline numbers.
    ax.text(x + 0.05, y + h - 0.36, contrib["headline"], fontsize=10,
            va="top", ha="left", color="black", family="monospace")
    # Mechanism (italic, bottom).
    ax.text(x + 0.05, y + 0.10, contrib["mechanism"], fontsize=8.5,
            va="bottom", ha="left", color="#444", style="italic", wrap=True)
    # Section reference (bottom-right).
    ax.text(x + w - 0.05, y + 0.04, contrib["section"], fontsize=8.5,
            va="bottom", ha="right", color="#666", family="monospace")


def main() -> int:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10}
    )

    fig = plt.figure(figsize=(15, 11))
    # Main grid: 2 rows × 3 cols of contributions + bottom bar for negatives.
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[0.12, 0.62, 0.20],
        hspace=0.05,
    )
    # Row 0: title
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.7,
        "D-MeZO-N: Decentralized Federated Memory-Efficient ZO Optimization with Nesterov Acceleration",
        fontsize=14, fontweight="bold", ha="center", va="center",
    )
    ax_title.text(
        0.5, 0.30,
        "Six contributions (C1–C6) with empirical and theoretical evidence — single-page summary",
        fontsize=10.5, ha="center", va="center", color="#444", style="italic",
    )

    # Row 1: 6 contribution cards in 2x3 grid
    ax_cards = fig.add_subplot(gs[1])
    ax_cards.set_xlim(0, 3)
    ax_cards.set_ylim(0, 2)
    ax_cards.axis("off")
    card_w, card_h = 0.95, 0.9
    for i, c in enumerate(CONTRIBUTIONS):
        col = i % 3
        row = 1 - (i // 3)  # invert: top row = first 3
        x = col + (1 - card_w) / 2
        y = row + (1 - card_h) / 2
        _draw_card(ax_cards, x, y, card_w, card_h, c)

    # Row 2: negative findings + caveat
    ax_neg = fig.add_subplot(gs[2])
    ax_neg.axis("off")
    ax_neg.set_xlim(0, 1)
    ax_neg.set_ylim(0, 1)
    ax_neg.add_patch(patches.FancyBboxPatch(
        (0.02, 0.10), 0.96, 0.78,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="#d62728", facecolor="#fdf2f2", alpha=0.6,
    ))
    ax_neg.text(0.06, 0.78,
                "Useful negative findings (paper §6.4 / §6.7 / §6.8):",
                fontsize=10.5, fontweight="bold", ha="left", va="top", color="#a04040")
    for i, txt in enumerate(NEGATIVE_FINDINGS):
        ax_neg.text(0.08, 0.62 - i * 0.13, "• " + txt, fontsize=9.5,
                    ha="left", va="top", color="#2d2d2d")

    # Bottom-most footer
    fig.text(
        0.5, 0.005,
        "See docs/experiments_summary.md for full timeline, docs/robustness_matrix.md for statistical-rigor classification.",
        fontsize=8.5, ha="center", va="bottom", color="#666", style="italic",
    )

    out_path = ROOT / "docs" / "figures" / "fig00_paper_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
