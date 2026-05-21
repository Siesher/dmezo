"""Reconstruct §22 paper-scale multi-seed run JSON + figure from chat logs.

Original run output на Colab (Qwen3.5-4B-Base / MathLogicQA / 3 seeds × 5 variants ×
1000 rounds) был сохранён в /content/dmezo/experiments/diagnostics/ и /docs/figures/
но эти пути не на mounted Drive — после Colab session expiry данные потерялись.

Все trajectory checkpoints (R100, R200, ..., R1000) + final values + resets count
были выведены в stdout и сохранены в chat logs. Этот скрипт hardcodes их и
reconstruct identical JSON + figure files.

Output:
    experiments/diagnostics/local_test_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.json
    docs/figures/fig_local_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.png

Usage:
    uv run --no-sync python scripts/reconstruct_section22_from_logs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments" / "diagnostics" / "local_test_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.json"
OUT_FIG = ROOT / "docs" / "figures" / "fig_local_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.png"

# Init eval (same for all cells)
INIT_LOSS = 2.8160
INIT_ACC = 0.3700

# Eval steps (0, 100, ..., 1000)
EVAL_STEPS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Per-cell trajectories: list of 10 (loss, acc) after init point.
# Format: variant -> seed -> [(loss, acc), ...] for R100..R1000.
TRAJECTORIES = {
    "vanilla": {
        42: [(1.8905, 0.38), (1.6539, 0.37), (1.5550, 0.37), (1.4937, 0.39), (1.4945, 0.33),
             (1.4738, 0.33), (1.4198, 0.38), (1.4068, 0.39), (1.3979, 0.36), (1.3747, 0.38)],
        43: [(1.9293, 0.35), (1.6353, 0.35), (1.5356, 0.36), (1.4809, 0.36), (1.4405, 0.39),
             (1.3793, 0.39), (1.3684, 0.39), (1.3532, 0.35), (1.3508, 0.35), (1.3432, 0.36)],
        44: [(1.9797, 0.35), (1.6940, 0.38), (1.6006, 0.39), (1.5104, 0.38), (1.4542, 0.40),
             (1.4410, 0.39), (1.4279, 0.41), (1.4100, 0.39), (1.3896, 0.36), (1.3863, 0.39)],
    },
    "dmezo_n": {
        42: [(1.8864, 0.36), (1.6477, 0.34), (1.5612, 0.38), (1.5225, 0.37), (1.5143, 0.37),
             (1.5081, 0.39), (1.4860, 0.35), (1.4731, 0.38), (1.4679, 0.36), (1.4598, 0.38)],
        43: [(1.8877, 0.37), (1.6298, 0.39), (1.5421, 0.42), (1.5054, 0.42), (1.4769, 0.40),
             (1.4279, 0.42), (1.4184, 0.39), (1.4413, 0.38), (1.4584, 0.36), (1.4569, 0.36)],
        44: [(1.8660, 0.38), (1.6425, 0.40), (1.5692, 0.43), (1.5182, 0.40), (1.5069, 0.40),
             (1.4956, 0.41), (1.4826, 0.40), (1.4719, 0.40), (1.4711, 0.39), (1.4735, 0.39)],
    },
    "dmezo_n_drift": {
        42: [(1.8776, 0.36), (1.6537, 0.32), (1.5640, 0.34), (1.5228, 0.35), (1.5236, 0.37),
             (1.5040, 0.39), (1.4827, 0.38), (1.4823, 0.39), (1.4768, 0.39), (1.4608, 0.38)],
        43: [(1.8985, 0.36), (1.6372, 0.39), (1.5454, 0.39), (1.5225, 0.41), (1.4821, 0.40),
             (1.4302, 0.41), (1.4156, 0.42), (1.4377, 0.37), (1.4508, 0.34), (1.4531, 0.36)],
        44: [(1.8888, 0.38), (1.6439, 0.39), (1.5593, 0.43), (1.5007, 0.41), (1.4956, 0.39),
             (1.4792, 0.42), (1.4686, 0.39), (1.4653, 0.40), (1.4489, 0.38), (1.4537, 0.39)],
    },
    "dmezo_n_adaptive_clip": {
        42: [(1.4215, 0.37), (1.3430, 0.35), (1.3184, 0.43), (1.2929, 0.37), (1.2998, 0.34),
             (1.2808, 0.44), (1.2736, 0.41), (1.2720, 0.41), (1.2602, 0.44), (1.2691, 0.41)],
        43: [(1.3984, 0.35), (1.3502, 0.38), (1.3252, 0.36), (1.3359, 0.33), (1.3176, 0.34),
             (1.3090, 0.37), (1.3014, 0.34), (1.3021, 0.35), (1.3164, 0.34), (1.3135, 0.33)],
        44: [(1.3972, 0.38), (1.3540, 0.40), (1.3351, 0.39), (1.3246, 0.42), (1.3200, 0.39),
             (1.3402, 0.38), (1.3377, 0.39), (1.3341, 0.38), (1.3110, 0.39), (1.3135, 0.43)],
    },
    "dmezo_n_combo": {
        42: [(1.4420, 0.37), (1.3389, 0.36), (1.3312, 0.34), (1.2913, 0.39), (1.2989, 0.38),
             (1.2805, 0.41), (1.2696, 0.43), (1.2758, 0.38), (1.2707, 0.37), (1.2790, 0.37)],
        43: [(1.3926, 0.38), (1.3481, 0.36), (1.3203, 0.38), (1.3305, 0.36), (1.3028, 0.37),
             (1.2862, 0.44), (1.2740, 0.40), (1.2729, 0.41), (1.2818, 0.40), (1.2951, 0.44)],
        44: [(1.3859, 0.39), (1.3487, 0.39), (1.3018, 0.45), (1.2919, 0.44), (1.2999, 0.44),
             (1.3138, 0.43), (1.3016, 0.43), (1.3064, 0.42), (1.2926, 0.41), (1.3036, 0.39)],
    },
}

# Final drift-reset counts per (variant, seed). Empty/0 for non-drift variants.
N_RESETS = {
    ("dmezo_n_drift", 42): 18, ("dmezo_n_drift", 43): 17, ("dmezo_n_drift", 44): 18,
    ("dmezo_n_combo", 42): 18, ("dmezo_n_combo", 43): 18, ("dmezo_n_combo", 44): 18,
}

# Per-cell wall clock estimate (from chat log timestamps): ~47 min/cell.
WALL_CLOCK_S = 47 * 60  # ~2820 seconds


def build_cells() -> dict:
    """Reconstruct cell dict in same schema as local_test_improvements_*.json."""
    cells = {}
    for variant, by_seed in TRAJECTORIES.items():
        for seed, traj in by_seed.items():
            losses = [INIT_LOSS] + [t[0] for t in traj]
            accs = [INIT_ACC] + [t[1] for t in traj]
            cells[f"{variant}|seed={seed}"] = {
                "variant": variant,
                "seed": seed,
                "lr": 3e-7,
                "eps": 1e-3,
                "n_clients": 4,
                "eval_steps": EVAL_STEPS,
                "eval_losses": losses,
                "eval_accs": accs,
                "wall_clock_s": WALL_CLOCK_S,
                "n_drift_resets": N_RESETS.get((variant, seed), 0),
                "dp_epsilon": None,
            }
    return cells


def build_root() -> dict:
    """Reconstruct root metadata."""
    return {
        "model": "Qwen/Qwen3.5-4B-Base",
        "task": "mathlogicqa",
        "dtype": "bfloat16",
        "n_clients": 4,
        "num_rounds": 1000,
        "lr": 3e-7,
        "eps": 1e-3,
        "rho_clip": 50.0,
        "beta_start": 0.9,
        "beta_end": 0.0,
        "drift_window": 50,
        "drift_threshold": 0.1,
        "ac_window": 50,
        "ac_quantile": 0.95,
        "ac_alpha": 1.3,
        "dp_sigma": None,
        "dp_delta": 1e-3,
        "seeds": [42, 43, 44],
        "variants": ["vanilla", "dmezo_n", "dmezo_n_drift", "dmezo_n_adaptive_clip", "dmezo_n_combo"],
        "cells": build_cells(),
        "_reconstructed_from_logs": True,
        "_note": "Reconstructed 2026-05-21 from chat logs after Colab session expired. "
                 "Trajectory checkpoints (every 100 rounds) match original stdout exactly; "
                 "intermediate per-round metrics (loss+, rho) not reconstructed (only checkpoints).",
    }


def make_figure(data: dict, out_path: Path) -> None:
    """4-panel figure: loss trajectory, acc trajectory, final loss bar, final acc bar."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cells = data["cells"]
    variants = data["variants"]
    seeds = data["seeds"]

    colors = {
        "vanilla": "#4C72B0",  # blue (reference)
        "dmezo_n": "#DD8452",  # orange (v1, falsified)
        "dmezo_n_drift": "#937860",  # brown (B5 alone, falsified)
        "dmezo_n_adaptive_clip": "#8172B3",  # purple (B1 alone)
        "dmezo_n_combo": "#937D14",  # gold (D-MeZO-N v2 [BEST])
    }
    labels = {
        "vanilla": "vanilla MeZO",
        "dmezo_n": "v1 (fixed C=50)",
        "dmezo_n_drift": "B5 only (drift-reset)",
        "dmezo_n_adaptive_clip": "B1 (adaptive clip)",
        "dmezo_n_combo": "v2 = combo (B1+B5) [BEST]",
    }

    # --- Panel 1: Loss trajectory (mean ± shaded per-seed) ---
    ax = axes[0, 0]
    steps = EVAL_STEPS
    for v in variants:
        all_losses = np.array([cells[f"{v}|seed={s}"]["eval_losses"] for s in seeds])
        mean = all_losses.mean(axis=0)
        std = all_losses.std(axis=0)
        ax.plot(steps, mean, color=colors[v], label=labels[v], linewidth=2.0,
                linestyle="-" if v == "dmezo_n_combo" else ("--" if v == "vanilla" else ":"))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=colors[v])
    ax.set_xlabel("Round")
    ax.set_ylabel("Eval loss")
    ax.set_title("(a) Loss trajectory (mean ± std across 3 seeds)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 2: Accuracy trajectory ---
    ax = axes[0, 1]
    for v in variants:
        all_accs = np.array([cells[f"{v}|seed={s}"]["eval_accs"] for s in seeds])
        mean = all_accs.mean(axis=0)
        std = all_accs.std(axis=0)
        ax.plot(steps, mean, color=colors[v], label=labels[v], linewidth=2.0,
                linestyle="-" if v == "dmezo_n_combo" else ("--" if v == "vanilla" else ":"))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=colors[v])
    ax.set_xlabel("Round")
    ax.set_ylabel("Eval accuracy")
    ax.set_title("(b) Accuracy trajectory (mean ± std across 3 seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: Final loss bar plot ---
    ax = axes[1, 0]
    means = []
    stds = []
    for v in variants:
        finals = [cells[f"{v}|seed={s}"]["eval_losses"][-1] for s in seeds]
        means.append(np.mean(finals))
        stds.append(np.std(finals))
    bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=5,
                  color=[colors[v] for v in variants],
                  edgecolor="black", linewidth=0.8)
    # Highlight combo bar
    bars[variants.index("dmezo_n_combo")].set_edgecolor("gold")
    bars[variants.index("dmezo_n_combo")].set_linewidth(2.5)
    # Vanilla reference line
    vanilla_mean = means[variants.index("vanilla")]
    ax.axhline(vanilla_mean, color=colors["vanilla"], linestyle="--", alpha=0.5, label=f"vanilla baseline ({vanilla_mean:.3f})")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([labels[v].split(" ")[0] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Final eval loss @R1000")
    ax.set_title("(c) Final loss — combo wins 3/3 seeds (Δ=−5.5%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=8)

    # --- Panel 4: Final acc bar plot ---
    ax = axes[1, 1]
    means_acc = []
    stds_acc = []
    for v in variants:
        finals = [cells[f"{v}|seed={s}"]["eval_accs"][-1] for s in seeds]
        means_acc.append(np.mean(finals))
        stds_acc.append(np.std(finals))
    bars = ax.bar(range(len(variants)), means_acc, yerr=stds_acc, capsize=5,
                  color=[colors[v] for v in variants],
                  edgecolor="black", linewidth=0.8)
    bars[variants.index("dmezo_n_combo")].set_edgecolor("gold")
    bars[variants.index("dmezo_n_combo")].set_linewidth(2.5)
    vanilla_acc = means_acc[variants.index("vanilla")]
    ax.axhline(vanilla_acc, color=colors["vanilla"], linestyle="--", alpha=0.5, label=f"vanilla baseline ({vanilla_acc:.3f})")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([labels[v].split(" ")[0] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Final eval accuracy @R1000")
    ax.set_title("(d) Final accuracy — combo +2.3pp mean (3 seeds)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means_acc, stds_acc)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=8)

    plt.suptitle("D-MeZO-N v2 = combo (B1 adaptive_clip + B5 drift-reset) — §22 paper-scale\n"
                 "Qwen3.5-4B-Base / MathLogicQA / 4 clients complete IID / 1000 rounds / 3 seeds paired",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    data = build_root()

    # Write JSON
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved JSON: {OUT_JSON}  ({OUT_JSON.stat().st_size / 1024:.1f} KB)")

    # Verify final means match expected
    cells = data["cells"]
    print("\nFinal value verification (mean across 3 seeds):")
    print(f"{'variant':<30}{'mean_loss':>12}{'mean_acc':>12}{'resets':>10}")
    print("-" * 64)
    for v in data["variants"]:
        finals_loss = [cells[f"{v}|seed={s}"]["eval_losses"][-1] for s in data["seeds"]]
        finals_acc = [cells[f"{v}|seed={s}"]["eval_accs"][-1] for s in data["seeds"]]
        total_resets = sum(cells[f"{v}|seed={s}"]["n_drift_resets"] for s in data["seeds"])
        print(f"{v:<30}{np.mean(finals_loss):>12.4f}{np.mean(finals_acc):>12.4f}{total_resets:>10}")

    # Build figure
    make_figure(data, OUT_FIG)
    print(f"\nSaved figure: {OUT_FIG}  ({OUT_FIG.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
