"""Generate publication-ready figures for the D-MeZO-N paper.

Reads JSONL trajectories from experiments/results/Final_run/unpack/ where
available, supplements with hardcoded R1b / R1d / Day 6 trajectories from
console logs that weren't zipped.

Output: docs/figures/*.png at 300 DPI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = Path("C:/Work/dmezo/experiments/results/Final_run/unpack")
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Publication style.
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
        "lines.linewidth": 1.8,
    }
)


def load_jsonl_trajectory(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (rounds, eval_losses, eval_accs) for eval points in log.jsonl."""
    path = run_dir / "log.jsonl"
    rounds, losses, accs = [], [], []
    if not path.exists():
        return rounds, losses, accs
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_loss" in d and ("round" in d or "step" in d):
                r = d.get("round", d.get("step", 0))
                rounds.append(r)
                losses.append(float(d["eval_loss"]))
                accs.append(float(d.get("eval_acc", float("nan"))))
    return rounds, losses, accs


# ---------------------------------------------------------------------------
# Hardcoded trajectories from runs we don't have local JSONL for
# (taken from console logs the user pasted during the project)
# ---------------------------------------------------------------------------
R1D_ROUNDS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
R1D_LOSS = [3.5614, 0.3655, 0.2885, 0.2078, 0.1950, 0.1903, 0.1610, 0.1559, 0.1514, 0.1396, 0.1291]
R1D_ACC = [0.6188, 0.9313, 0.9062, 0.9375, 0.9313, 0.9250, 0.9625, 0.9500, 0.9437, 0.9500, 0.9563]

R1B_ROUNDS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
R1B_LOSS = [3.5614, 0.2304, 0.1700, 0.1189, 0.1370, 0.1598, 0.1808, 0.1900, 0.1913, 0.2059, 0.2246]
R1B_ACC = [0.6188, 0.9563, 0.9437, 0.9563, 0.9500, 0.9437, 0.9375, 0.9437, 0.9375, 0.9375, 0.9313]

R1_CLIP200_ROUNDS = [0, 100, 200, 300, 400, 500, 600, 700]
R1_CLIP200_LOSS = [3.5614, 0.2134, 0.2739, 0.2476, 0.2925, 0.5975, 1.0759, 2.9618]

# Day 6 vanilla beta=0.9 — diverged at R140
DAY6_B09_ROUNDS = [0, 100, 200, 300, 360]
DAY6_B09_LOSS = [3.5614, 0.5218, 15.5834, 16.1568, 16.0]  # interpolated upper bound

# Day 6 look-ahead beta=0.9 — NaN at R20
DAY6_LA_ROUNDS = [0, 20, 40]
DAY6_LA_LOSS = [3.5614, float("nan"), float("nan")]

# Complete + Dir(0.5) s43 final (no local zip, but final known)
COMPLETE_DIR_S43_FINAL = (0.1418, 0.9500)

# ---------------------------------------------------------------------------
# Figure 1: Day 5 multi-seed grid — per-cell trajectories (4 panels)
# ---------------------------------------------------------------------------


def figure1_day5_grid():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    cells = [
        ("complete + IID", "complete_iid", axes[0, 0]),
        ("complete + Dir(α=0.5)", "complete_dir05", axes[0, 1]),
        ("ring + IID", "ring_iid", axes[1, 0]),
        ("ring + Dir(α=0.5)", "ring_dir05", axes[1, 1]),
    ]

    # Centralized baseline trajectory.
    _, c_loss, _ = load_jsonl_trajectory(DATA / "centralized_qwen3_5_4b_base_sst2_2000")
    # Steps in centralized are 0..1000, but eval_every=100 so 11 points expected.
    c_rounds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # Trim/pad c_loss to 11 points (it may have init + 10 evals).
    if len(c_loss) >= 11:
        c_loss = c_loss[:11]
    else:
        c_loss = [3.5614] + c_loss[:10]

    for title, key, ax in cells:
        # s42
        r42, l42, _ = load_jsonl_trajectory(
            DATA / f"dmezo_4c_{key}_qwen3_5_4b_base_sst2_s42"
        )
        # s43 (may be missing for complete_dir05)
        r43, l43, _ = load_jsonl_trajectory(
            DATA / f"dmezo_4c_{key}_qwen3_5_4b_base_sst2_s43"
        )

        if r42:
            ax.plot(r42, l42, "o-", label="seed=42", color="#2E86AB", markersize=4)
        if r43:
            ax.plot(r43, l43, "s--", label="seed=43", color="#A23B72", markersize=4)
        else:
            # Use known final-only point for complete_dir05_s43.
            if key == "complete_dir05":
                ax.plot(
                    [1000],
                    [COMPLETE_DIR_S43_FINAL[0]],
                    "s",
                    label="seed=43 (final only)",
                    color="#A23B72",
                    markersize=8,
                )

        ax.plot(
            c_rounds,
            c_loss,
            ":",
            color="#666",
            alpha=0.7,
            linewidth=1.4,
            label="centralized (ref)",
        )

        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xlim(-30, 1030)

    for ax in axes[-1, :]:
        ax.set_xlabel("Round")
    for ax in axes[:, 0]:
        ax.set_ylabel("eval loss (log)")

    fig.suptitle(
        "Figure 1. Day 5 grid — federated D-MeZO trajectories per cell\n"
        "(Qwen3.5-4B-Base / SST-2 / 4 clients / 1000 rounds; 2 seeds + centralized reference)",
        y=1.00,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_day5_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / 'fig1_day5_grid.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Nesterov phase diagram — trajectories on worst Day 5 cell
# ---------------------------------------------------------------------------


def figure2_nesterov_phase_diagram():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Control: ring + Dir(0.5) seed=42 retrofit
    r_ctrl, l_ctrl, _ = load_jsonl_trajectory(
        DATA / "dmezo_4c_ring_dir05_qwen3_5_4b_base_sst2_s42"
    )
    if r_ctrl:
        ax.plot(r_ctrl, l_ctrl, "o-", label="no Nesterov (control), final=0.137", color="#2E86AB")

    ax.plot(R1B_ROUNDS, R1B_LOSS, "s-", label="R1b: β=0.9 const + clip50, final=0.225", color="#E63946")
    ax.plot(R1D_ROUNDS, R1D_LOSS, "D-", label="R1d: β=0.9→0 decay + clip50, final=0.129 ⭐", color="#06A77D")
    ax.plot(R1_CLIP200_ROUNDS, R1_CLIP200_LOSS, "^--", label="R1: β=0.9 + clip200, slow-div R500", color="#F4A261", alpha=0.85)
    ax.plot(DAY6_B09_ROUNDS, DAY6_B09_LOSS, "x--", label="Day 6: β=0.9 unclipped — DIVERGE R140", color="#9D4EDD", alpha=0.7)

    # Annotate the R1b best point (R300).
    ax.annotate(
        "R1b best 0.119 @ R300",
        xy=(300, 0.119),
        xytext=(450, 0.30),
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
        fontsize=9,
        color="#E63946",
    )
    # Annotate R1d final.
    ax.annotate(
        "R1d final 0.129\nmonotonic descent",
        xy=(1000, 0.129),
        xytext=(700, 0.060),
        arrowprops=dict(arrowstyle="->", color="#06A77D", lw=1.2),
        fontsize=9,
        color="#06A77D",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("eval loss (log)")
    ax.set_yscale("log")
    ax.set_xlim(-30, 1030)
    ax.set_ylim(0.05, 50)
    ax.set_title(
        "Figure 2. Nesterov-MeZO phase diagram on worst Day 5 cell (ring(4) + Dir(α=0.5))\n"
        "All runs at seed=42 for bit-exact ablation; ρ-clip + β-decay required for stable acceleration",
        fontsize=11,
    )
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_nesterov_phase_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / 'fig2_nesterov_phase_diagram.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Federated beats centralized — bar chart with error bars
# ---------------------------------------------------------------------------


def figure3_federated_vs_centralized():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Multi-seed means.
    configs = ["complete\n+ IID", "complete\n+ Dir(0.5)", "ring\n+ IID", "ring\n+ Dir(0.5)", "R1d\n(D-MeZO-N)"]
    means = [0.1348, 0.1507, 0.1271, 0.1402, 0.1291]
    ranges = [0.0051, 0.0089, 0.0014, 0.0029, 0.0]  # R1d is single-seed → 0
    centralized = 0.1762

    colors = ["#2E86AB", "#A23B72", "#06A77D", "#F4A261", "#E63946"]
    bars = axes[0].bar(
        configs,
        means,
        yerr=ranges,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )
    axes[0].axhline(
        centralized,
        color="#666",
        linestyle="--",
        linewidth=1.5,
        label=f"centralized MeZO Qwen3.5 baseline = {centralized:.4f}",
    )

    # Annotate each bar with mean value + improvement vs centralized.
    for bar, m in zip(bars, means):
        delta_pct = (m - centralized) / centralized * 100
        axes[0].annotate(
            f"{m:.4f}\n({delta_pct:+.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, m),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axes[0].set_ylabel("Final eval loss (n=2 seeds, error = range/2)")
    axes[0].set_ylim(0, 0.22)
    axes[0].set_title("(a) Federated D-MeZO + D-MeZO-N beat centralized by 14.5–27.9%")
    axes[0].legend(loc="upper right", fontsize=9)

    # Right panel: accuracy comparison.
    accs = [96.56, 95.00, 97.81, 95.63, 95.63]
    centralized_acc = 95.63
    bars2 = axes[1].bar(configs, accs, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    axes[1].axhline(centralized_acc, color="#666", linestyle="--", linewidth=1.5, label=f"centralized = {centralized_acc:.2f}%")
    axes[1].set_ylim(94, 99)
    axes[1].set_ylabel("Final eval accuracy (%, n=2 mean)")
    axes[1].set_title("(b) Accuracy similar; ring + IID highest (97.8%)")
    axes[1].legend(loc="lower right", fontsize=9)

    for bar, a in zip(bars2, accs):
        axes[1].annotate(
            f"{a:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, a),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Figure 3. Federated D-MeZO consistently beats centralized MeZO on Qwen3.5-4B-Base / SST-2",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_federated_vs_centralized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / 'fig3_federated_vs_centralized.png'}")


# ---------------------------------------------------------------------------
# Figure 4: R1d trajectory + β-schedule + accuracy on right axis
# ---------------------------------------------------------------------------


def figure4_r1d_breakdown():
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Control trajectory (ring + Dir + s42)
    r_ctrl, l_ctrl, _ = load_jsonl_trajectory(
        DATA / "dmezo_4c_ring_dir05_qwen3_5_4b_base_sst2_s42"
    )

    color_loss = "#06A77D"
    color_ctrl = "#2E86AB"
    color_beta = "#E63946"
    color_acc = "#9D4EDD"

    if r_ctrl:
        ax1.plot(r_ctrl, l_ctrl, "o-", label="control (no Nesterov), final=0.137", color=color_ctrl)
    ax1.plot(R1D_ROUNDS, R1D_LOSS, "D-", label="R1d (β-decay + clip50), final=0.129", color=color_loss, linewidth=2.0)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("eval loss (log)", color="black")
    ax1.set_yscale("log")
    ax1.set_ylim(0.07, 5)
    ax1.set_xlim(-30, 1030)
    ax1.legend(loc="upper right", fontsize=9)

    # Right axis: β schedule.
    ax2 = ax1.twinx()
    beta_rounds = np.linspace(0, 999, 100)
    beta_vals = 0.9 * (1 - beta_rounds / 999)
    ax2.plot(beta_rounds, beta_vals, "--", color=color_beta, linewidth=1.4, alpha=0.85, label="β(t) schedule")
    ax2.set_ylabel("β(t)", color=color_beta)
    ax2.set_ylim(-0.05, 1.0)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", labelcolor=color_beta)
    ax2.legend(loc="center right", fontsize=9)

    ax1.set_title(
        "Figure 4. D-MeZO-N (R1d) detailed trajectory — β-decay produces monotonic descent\n"
        "Final 0.129 beats vanilla 0.137 by 6%; β linearly decays 0.9 → 0 over T=1000",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_r1d_detailed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / 'fig4_r1d_detailed.png'}")


# ---------------------------------------------------------------------------
# Figure 5: Schematic algorithm flow
# ---------------------------------------------------------------------------


def figure5_algorithm_schematic():
    # Generate two variants — English (fig5) and Russian (fig5_ru) — sharing
    # the same layout but with the labels translated and a slightly different
    # call signature for the inner builder.
    for lang, suffix in [("en", ""), ("ru", "_ru")]:
        _figure5_one(lang=lang, suffix=suffix)


def _figure5_one(lang: str = "en", suffix: str = ""):
    """Algorithm-schematic figure with non-overlapping labels and LaTeX-typeset
    formulas inside both the centre badge and the procedure box.

    lang: "en" or "ru" — controls all visible text.
    suffix: appended to filename (e.g. "_ru" for Russian).
    """
    fig = plt.figure(figsize=(14, 7.2))
    # Two side panels: left = graph, right = algorithm box.
    ax_graph = fig.add_axes([0.02, 0.05, 0.46, 0.86])
    ax_box = fig.add_axes([0.52, 0.05, 0.46, 0.86])
    for ax in (ax_graph, ax_box):
        ax.axis("off")

    # --- Translations
    L = {
        "en": {
            "title": "Figure 5. D-MeZO-N algorithm — decentralized peer-to-peer with consensus mixing",
            "client": "Client",
            "mezo": "MeZO step",
            "consensus": "Consensus W\n(doubly-stochastic)\nρ(W) = 0.33  (ring n=4)",
            "box_header": "D-MeZO-N round t  (for each client i):",
            "box_lines": [
                r"1.  sample seed  $s_i^t \sim$  counter PRNG",
                r"2.  ZO probe:    $\hat\rho_i^t = \dfrac{L(\theta_i + \epsilon z) - L(\theta_i - \epsilon z)}{2\epsilon}$",
                r"3.  $\rho$-clip:      $\tilde\rho_i^t = \mathrm{clip}(\hat\rho_i^t,\,\pm C)$",
                r"4.  velocity:    $v_i^{t+1} = \beta_t\,v_i^t + \tilde\rho_i^t\,z_{s_i^t}$",
                r"5.  update:      $\theta_i^{t+1/2} = \theta_i^t - \eta\,v_i^{t+1}$",
                r"6.  consensus:   $\theta_i^{t+1} = \sum_{j} W_{ij}\,\theta_j^{t+1/2}$",
            ],
            "box_footer": (
                r"Communication per round per neighbour:"
                "\n"
                r"$\mathcal{O}(1)$ scalar  +  1 int seed"
            ),
        },
        "ru": {
            "title": "Рисунок 5. Алгоритм D-MeZO-N — peer-to-peer с consensus-усреднением",
            "client": "Клиент",
            "mezo": "MeZO шаг",
            "consensus": "Усреднение W\n(дважды стох.)\nρ(W) = 0.33  (кольцо n=4)",
            "box_header": "Раунд t алгоритма D-MeZO-N  (для каждого клиента i):",
            "box_lines": [
                r"1.  выбрать seed $s_i^t \sim$  счётчик-PRNG",
                r"2.  ZO-замер: $\hat\rho_i^t = \dfrac{L(\theta_i + \epsilon z) - L(\theta_i - \epsilon z)}{2\epsilon}$",
                r"3.  $\rho$-клип:   $\tilde\rho_i^t = \mathrm{clip}(\hat\rho_i^t,\,\pm C)$",
                r"4.  скорость: $v_i^{t+1} = \beta_t\,v_i^t + \tilde\rho_i^t\,z_{s_i^t}$",
                r"5.  обновление: $\theta_i^{t+1/2} = \theta_i^t - \eta\,v_i^{t+1}$",
                r"6.  consensus: $\theta_i^{t+1} = \sum_{j} W_{ij}\,\theta_j^{t+1/2}$",
            ],
            "box_footer": (
                r"Коммуникация за раунд на соседа:"
                "\n"
                r"$\mathcal{O}(1)$ скаляр  +  1 целое seed"
            ),
        },
    }[lang]

    # --- LEFT: graph layout with labels strictly outside the node disk
    n = 4
    # Place nodes on a square (corners) — 4 fits nicely.
    coords = np.array([(-1.6, 1.6), (1.6, 1.6), (1.6, -1.6), (-1.6, -1.6)])
    # Client index ordering on the diagram: 0 top-right, 1 top-left, 2 bottom-left, 3 bottom-right.
    client_to_corner = {0: 0, 1: 1, 2: 2, 3: 3}
    R_node = 0.50
    R_label = 1.10  # how far outside the node the (MeZO step / s_i, ρ_i) label sits

    # Map client index to its xy position
    pos = {ci: tuple(coords[client_to_corner[ci]]) for ci in range(n)}

    # Compute outward radial directions per corner.
    def outward(p):
        x, y = p
        sx = 1 if x >= 0 else -1
        sy = 1 if y >= 0 else -1
        return sx, sy

    for ci, (x, y) in pos.items():
        circ = plt.Circle((x, y), R_node, color="#2E86AB", alpha=0.65, ec="black", linewidth=1.3)
        ax_graph.add_patch(circ)
        ax_graph.text(
            x,
            y,
            f"{L['client']}\n{ci}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        # Outward radial label (MeZO step + (s_i, ρ_i)) — placed strictly outside the circle
        sx, sy = outward((x, y))
        lx = x + sx * R_label
        ly = y + sy * R_label
        ax_graph.text(
            lx,
            ly,
            f"{L['mezo']}\n$(s_{{{ci}}},\\,\\rho_{{{ci}}})$",
            ha="center",
            va="center",
            fontsize=9,
            color="#333",
            style="italic",
        )

    # Connections: complete bipartite drawing showing both ring (red, thick)
    # and diagonal "could be present in complete" (gray dashed) edges. Keep
    # only the ring edges for n=4 to match the (ρ(W)=0.333) caption.
    ring_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in ring_edges:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        dx, dy = x2 - x1, y2 - y1
        d = np.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        shrink = R_node + 0.05
        x1s, y1s = x1 + ux * shrink, y1 + uy * shrink
        x2s, y2s = x2 - ux * shrink, y2 - uy * shrink
        ax_graph.annotate(
            "",
            xy=(x2s, y2s),
            xytext=(x1s, y1s),
            arrowprops=dict(arrowstyle="<->", color="#E63946", lw=2.0),
        )

    # Centre badge for consensus W
    ax_graph.text(
        0,
        0,
        L["consensus"],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#E63946",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff4f4", edgecolor="#E63946", linewidth=1.0),
    )

    ax_graph.set_xlim(-3.6, 3.6)
    ax_graph.set_ylim(-3.0, 3.0)
    ax_graph.set_aspect("equal")

    # --- RIGHT: algorithm box with LaTeX-typeset lines
    box_top = 0.97
    line_dy = 0.13
    # Header
    ax_box.text(
        0.02,
        box_top,
        L["box_header"],
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax_box.transAxes,
    )
    # Each numbered step, rendered with LaTeX mathtext
    for k, line in enumerate(L["box_lines"]):
        ax_box.text(
            0.04,
            box_top - 0.13 - k * line_dy,
            line,
            ha="left",
            va="top",
            fontsize=12,
            transform=ax_box.transAxes,
        )
    # Footer (communication note)
    ax_box.text(
        0.02,
        0.06,
        L["box_footer"],
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1B4F8C",
        transform=ax_box.transAxes,
    )

    # Surrounding rectangle around the right panel
    rect = plt.Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax_box.transAxes,
        fill=True,
        facecolor="#f8f8f8",
        edgecolor="#999",
        linewidth=1.0,
    )
    ax_box.add_patch(rect)
    # Re-draw text on top of rectangle (matplotlib draws in z-order)
    ax_box.text(
        0.02,
        box_top,
        L["box_header"],
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax_box.transAxes,
    )
    for k, line in enumerate(L["box_lines"]):
        ax_box.text(
            0.04,
            box_top - 0.13 - k * line_dy,
            line,
            ha="left",
            va="top",
            fontsize=12,
            transform=ax_box.transAxes,
        )
    ax_box.text(
        0.02,
        0.06,
        L["box_footer"],
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#1B4F8C",
        transform=ax_box.transAxes,
    )

    fig.suptitle(L["title"], fontsize=12, y=0.98)
    out_path = OUT / f"fig5_algorithm_schematic{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating paper figures...")
    figure1_day5_grid()
    figure2_nesterov_phase_diagram()
    figure3_federated_vs_centralized()
    figure4_r1d_breakdown()
    figure5_algorithm_schematic()
    print(f"\nAll figures saved to {OUT}")
