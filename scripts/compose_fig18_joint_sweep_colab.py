"""Composite Figure 18: vanilla vs D-MeZO-N joint lr × ε sweep on HellaSwag.

Loads sweep_lr_eps_hellaswag_{variant}_Qwen_Qwen3p5-4B-Base.json (Colab Blackwell
runs, 500 steps × 100-example eval × 25-step cadence) and produces a 2-row ×
4-column composite:
    Row 0 = vanilla MeZO            (control)
    Row 1 = D-MeZO-N (clip50 + β-decay 0.9→0)
    Columns = lr ∈ {1e-7, 3e-7, 1e-6, 3e-6}

Each panel overlays the three ε(t) schedules (const_1e-3, decay 1e-3→1e-4,
warmup 1e-2→1e-3) showing eval-loss-at-clean-θ vs MeZO step. Final loss /
acc annotated in legend. Catastrophic-divergence cells (lr=3e-6) get a
clipped y-axis with the actual L_final printed.

Usage::

    python scripts/compose_fig18_joint_sweep_colab.py
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
logger = logging.getLogger("dmezo.fig18")

SCHED_COLOURS = {
    "const_1e-3":             "#1f77b4",  # blue   — Princeton
    "exp_decay_1e-3_to_1e-4": "#2ca02c",  # green  — refine-below
    "exp_decay_1e-2_to_1e-3": "#d62728",  # red    — warmup-style
}
SCHED_LABELS = {
    "const_1e-3":             r"const $\varepsilon=10^{-3}$",
    "exp_decay_1e-3_to_1e-4": r"decay $10^{-3}\!\to\!10^{-4}$",
    "exp_decay_1e-2_to_1e-3": r"warmup $10^{-2}\!\to\!10^{-3}$",
}
VARIANT_LABEL = {
    "vanilla": "Vanilla MeZO",
    "dmezo_n": r"D-MeZO-N (clip 50, $\beta\!: 0.9\to 0$)",
}


def _load(variant: str) -> dict:
    p = ROOT / "experiments" / "diagnostics" / f"sweep_lr_eps_hellaswag_{variant}_Qwen_Qwen3p5-4B-Base.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
            "lines.linewidth": 1.4,
        }
    )

    out_path = ROOT / "docs" / "figures" / "fig18_joint_sweep_colab.png"
    data = {v: _load(v) for v in ("vanilla", "dmezo_n")}

    LRS = [1e-7, 3e-7, 1e-6, 3e-6]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)

    for row, variant in enumerate(("vanilla", "dmezo_n")):
        d = data[variant]
        for col, lr in enumerate(LRS):
            ax = axes[row, col]
            # Determine y-axis clipping based on whether any cell diverged catastrophically.
            cell_keys = [f"lr={lr:.0e}|sched={s}" for s in SCHED_COLOURS]
            cells = [d["cells"][k] for k in cell_keys if k in d["cells"]]
            max_L_final = max(c["eval_losses"][-1] for c in cells)
            # If any final loss exceeds 3.0, the trajectory has diverged catastrophically.
            # Clip to [1.9, 3.0] and annotate with text the actual final value.
            clip_y = max_L_final > 3.0
            ylim_top = 3.0 if clip_y else 2.30

            for sched in SCHED_COLOURS:
                key = f"lr={lr:.0e}|sched={sched}"
                if key not in d["cells"]:
                    continue
                r = d["cells"][key]
                steps = np.array(r["eval_steps"])
                losses = np.array(r["eval_losses"])
                L0, Lf = losses[0], losses[-1]
                drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
                acc_f = r["eval_accs"][-1]
                col_hex = SCHED_COLOURS[sched]
                label = SCHED_LABELS[sched]
                tag = f"  ({drop:+.1f}%, acc {acc_f:.2f})"
                if Lf > 3.0:
                    tag = f"  ({drop:+.0f}%, L={Lf:.1f}, acc {acc_f:.2f})"
                # Clip plotted losses to the y-range so divergence doesn't squish other curves.
                plot_losses = np.minimum(losses, ylim_top - 0.02)
                ax.plot(steps, plot_losses, "o-", color=col_hex, markersize=3.5,
                        alpha=0.9, label=label + tag)
                # Mark where trajectory leaves the visible range.
                exits = np.where(losses > ylim_top - 0.02)[0]
                if len(exits) > 0 and clip_y:
                    first_exit = exits[0]
                    ax.scatter([steps[first_exit]], [ylim_top - 0.05], marker="^",
                               color=col_hex, s=60, zorder=5)

            ax.set_ylim(1.9, ylim_top)
            if row == 0:
                ax.set_title(rf"$\eta = {lr:.0e}$", fontsize=10)
            if row == 1:
                ax.set_xlabel("MeZO step")
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\neval loss")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
            # Random-chance reference for HellaSwag 4-way: log(4)=1.386. Doesn't fit our y-axis (>1.9), skip.

    fig.suptitle(
        "Joint lr × ε × variant sweep on HellaSwag / Qwen3.5-4B-Base "
        "(500 steps, eval@clean θ on 100 examples)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)

    # ---- Summary table to stdout for verification.
    logger.info("=" * 96)
    logger.info(f"{'lr':<8}{'sched':<28}{'V drop%':>10}{'D drop%':>10}{'V acc':>8}{'D acc':>8}")
    for lr in LRS:
        for sched in SCHED_COLOURS:
            key = f"lr={lr:.0e}|sched={sched}"
            v = data["vanilla"]["cells"].get(key)
            n = data["dmezo_n"]["cells"].get(key)
            if v is None or n is None:
                continue
            v_drop = (v["eval_losses"][0] - v["eval_losses"][-1]) / v["eval_losses"][0] * 100
            n_drop = (n["eval_losses"][0] - n["eval_losses"][-1]) / n["eval_losses"][0] * 100
            v_acc = v["eval_accs"][-1]
            n_acc = n["eval_accs"][-1]
            logger.info(
                f"{lr:<8.0e}{sched:<28}{v_drop:>+9.1f}%{n_drop:>+9.1f}%{v_acc:>8.3f}{n_acc:>8.3f}"
            )
    logger.info("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
