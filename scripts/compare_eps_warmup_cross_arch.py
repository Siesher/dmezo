"""Cross-architecture comparison of ε warmup autotuner results.

Loads ``eps_warmup_*.json`` for the models listed below and produces a 2x2
publication-style figure (fig10) comparing bias-proxy, variance, j_norm, and
gradient SNR across architectures.

Usage::

    python scripts/compare_eps_warmup_cross_arch.py
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
logger = logging.getLogger("dmezo.eps_compare")

# Mapping: pretty label -> JSON file slug used by diagnose_eps_warmup.py
MODELS: dict[str, dict] = {
    "Qwen3-0.6B (full-attn)": {
        "slug": "Qwen_Qwen3-0p6B",
        "color": "#d62728",
        "marker": "o",
    },
    "Qwen3.5-0.8B (hybrid linear-attn)": {
        "slug": "Qwen_Qwen3p5-0p8B",
        "color": "#1f77b4",
        "marker": "s",
    },
}


def _load(slug: str) -> dict:
    path = ROOT / "experiments" / "diagnostics" / f"eps_warmup_{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run diagnose_eps_warmup.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    out_path = ROOT / "docs" / "figures" / "fig10_eps_warmup_cross_arch.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax_bias, ax_var = axes[0]
    ax_j, ax_snr = axes[1]

    summary_rows: list[str] = []

    for label, meta in MODELS.items():
        d = _load(meta["slug"])
        scores = d["scores"]
        probes = d["probes"]
        eps_vals = np.array([s["eps"] for s in scores.values()])
        biases = np.array([s["bias"] for s in scores.values()])
        vars_ = np.array([s["var"] for s in scores.values()])
        j_norm = np.array([s["j_norm"] for s in scores.values()])
        # diverged flag is present in newer JSONs; older ones default to False.
        diverged = np.array([s.get("diverged", False) for s in scores.values()])
        rho_means = np.array([probes[k]["rho_mean"] for k in scores.keys()])
        rho_stds = np.array([probes[k]["rho_std"] for k in scores.keys()])
        snr = np.abs(rho_means) / (rho_stds + 1e-12)

        col = meta["color"]
        mk = meta["marker"]
        eps_star = d["eps_star"]
        eps_star_idx = d["eps_star_idx"]

        ax_bias.loglog(eps_vals, biases, marker=mk, color=col, label=label, markersize=7)
        ax_var.loglog(eps_vals, np.sqrt(vars_), marker=mk, color=col, label=label, markersize=7)
        ax_j.semilogx(eps_vals, j_norm, marker=mk, color=col, label=label, markersize=7)
        ax_snr.loglog(eps_vals, snr, marker=mk, color=col, label=label, markersize=7)

        # Mark each arch's ε* with a thin vertical line in matching colour.
        for ax in (ax_bias, ax_var, ax_j, ax_snr):
            ax.axvline(eps_star, color=col, linestyle="--", linewidth=1.0, alpha=0.35)

        # Mark divergence region with cross markers (per-axis, at ymin).
        if diverged.any():
            div_eps = eps_vals[diverged]
            for ax in (ax_bias, ax_var, ax_snr):
                ymin = ax.get_ylim()[0] if ax.has_data() else 1e-3
                ax.scatter(
                    div_eps, [ymin] * len(div_eps),
                    marker="x", s=80, color=col, alpha=0.7,
                    label="_nolegend_",
                )
            ax_j.scatter(
                div_eps, [0.0] * len(div_eps),
                marker="x", s=80, color=col, alpha=0.7,
                label="_nolegend_",
            )

        summary_rows.append(
            f"  {label:40s}  baseline L0={d['baseline_loss']:.3f}  "
            f"eps*={eps_star:.2e}  std[rho]@eps*={np.sqrt(vars_[eps_star_idx]):.2f}  "
            f"SNR@eps*={snr[eps_star_idx]:.3f}  diverged@{[f'{e:.0e}' for e in eps_vals[diverged]] or 'none'}"
        )

    # Reference line: Princeton default ε = 1e-3.
    for ax in (ax_bias, ax_var, ax_j, ax_snr):
        ax.axvline(
            1e-3,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            label="_nolegend_",
        )

    ax_bias.set_xlabel(r"$\varepsilon$")
    ax_bias.set_ylabel(r"$|E[z^\top H z]|$  (Taylor-2 proxy)")
    ax_bias.set_title("(a) Taylor-2 magnitude vs ε")
    ax_bias.legend(loc="upper right", fontsize=8)
    ax_bias.text(
        1e-3, ax_bias.get_ylim()[1] * 0.5,
        " Princeton\n default",
        fontsize=8, color="black", alpha=0.7,
    )

    ax_var.set_xlabel(r"$\varepsilon$")
    ax_var.set_ylabel(r"std$[\hat\rho]$  (estimator variance, $\sqrt{\cdot}$)")
    ax_var.set_title("(b) Estimator variance vs ε")
    ax_var.legend(loc="upper right", fontsize=8)

    ax_j.set_xlabel(r"$\varepsilon$")
    ax_j.set_ylabel(r"$J(\varepsilon)$  (min-max norm. of bias + var)")
    ax_j.set_title("(c) Autotuner trade-off score")
    ax_j.legend(loc="upper right", fontsize=8)

    ax_snr.set_xlabel(r"$\varepsilon$")
    ax_snr.set_ylabel(r"$|E[\hat\rho]| / $std$[\hat\rho]$  (signal-to-noise)")
    ax_snr.set_title("(d) Gradient signal-to-noise vs ε")
    ax_snr.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Cross-architecture ε autotuner sweep "
        "(SST-2, B=4, fp16, n_probes=30)",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved %s", out_path)
    logger.info("")
    logger.info("=" * 88)
    logger.info("Cross-architecture autotuner summary:")
    for row in summary_rows:
        logger.info(row)
    logger.info("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
