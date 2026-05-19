"""Ablation: Richardson 4-point vs standard 2-point central difference.

Hypothesis (from paper §6.7): the dominant failure mode of large-ε MeZO in
fp16 is the O(ε²) third-Taylor bias. Richardson extrapolation cancels this
term exactly to leading order (residual O(ε⁴)), so Richardson should
converge at large ε where 2-point fails.

Setup (compute-equivalent comparison):
    For each ε ∈ {1e-3, 3e-3, 1e-2, 3e-2}, run:
        * 2-point at N steps      (2N forward passes)
        * Richardson at N/2 steps (2N forward passes — same compute)
    Both share batches + z-seeds across method axis (each method sees the
    same training data sequence; only the gradient estimator differs).

    Plus a "step-equal" track for diagnostic clarity:
        * Richardson at N steps (4N forwards — 2× compute, isolates the
          per-step quality of the estimator).

Output:
    experiments/diagnostics/richardson_vs_2point_{model_short}.json
    docs/figures/fig17_richardson_vs_2point_{model_short}.png

Usage::

    python scripts/ablate_richardson_vs_2point.py --model Qwen/Qwen3-0.6B \\
        --epsilons 1e-3 3e-3 1e-2 3e-2 --steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmezo.data.superglue import (  # noqa: E402
    _SST2Dataset,
    _collate,
    _load_raw_dataset,
    causal_lm_loss,
)
from dmezo.mezo.step import (  # noqa: E402
    MeZOConfig,
    mezo_step,
    mezo_step_richardson,
    mezo_update,
)
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.richardson")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[1e-3, 3e-3, 1e-2, 3e-2])
    p.add_argument("--steps", type=int, default=100,
                   help="Steps for 2-point. Richardson half-budget uses N/2.")
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--task", type=str, default="sst2")
    p.add_argument("--eval-every", type=int, default=10)
    return p.parse_args()


def _eval_loss(model, batch) -> float:
    with torch.inference_mode():
        out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _train(
    *,
    model_id: str,
    dtype: torch.dtype,
    method: str,        # "2point" | "richardson_step_eq" | "richardson_compute_eq"
    eps: float,
    lr: float,
    n_steps: int,
    batches: list[dict],
    z_seeds: list[int],
    seed: int,
    eval_every: int,
) -> dict:
    """Run ``n_steps`` MeZO updates at fixed ε with the chosen estimator."""
    torch.manual_seed(seed)
    model, _ = load_causal_lm(model_id, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    cfg = MeZOConfig(lr=lr, eps=eps)
    step_fn = mezo_step_richardson if method.startswith("richardson") else mezo_step
    forwards_per_step = 4 if method.startswith("richardson") else 2

    eval_batch = batches[0]
    L0 = _eval_loss(model, eval_batch)
    train_losses: list[float] = [L0]
    rhos: list[float] = [0.0]
    eval_losses: list[float] = [L0]
    eval_steps: list[int] = [0]
    diverged = False
    t0 = time.time()

    for step in range(n_steps):
        one_shot = np.random.Generator(np.random.PCG64(z_seeds[step]))
        try:
            seed_used, rho, loss_plus = step_fn(
                model, batches[step], causal_lm_loss, cfg, rng=one_shot,
            )
        except RuntimeError as e:
            logger.warning(f"{method}|eps={eps:.0e} step {step}: {e!r}")
            diverged = True
            break
        if not np.isfinite(loss_plus) or not np.isfinite(rho):
            logger.warning(f"{method}|eps={eps:.0e} step {step}: NaN/inf")
            diverged = True
            break
        mezo_update(model, seed_used, rho, cfg)
        train_losses.append(float(loss_plus))
        rhos.append(float(rho))
        if (step + 1) % eval_every == 0:
            eL = _eval_loss(model, eval_batch)
            eval_losses.append(eL)
            eval_steps.append(step + 1)
            logger.info(
                f"{method:<24s} eps={eps:.0e} step={step + 1:3d}/{n_steps} "
                f"rho={rho:+8.2f} loss+={loss_plus:6.3f} eval@theta={eL:.4f}"
            )

    wall = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return {
        "method": method,
        "eps": float(eps), "lr": float(lr),
        "steps_completed": len(train_losses) - 1,
        "forwards_per_step": forwards_per_step,
        "total_forwards": (len(train_losses) - 1) * forwards_per_step,
        "diverged": bool(diverged),
        "train_losses": train_losses, "rhos": rhos,
        "eval_losses": eval_losses, "eval_steps": eval_steps,
        "wall_clock_s": float(wall),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"richardson_vs_2point_{model_short}.json"
    fig_path = fig_dir / f"fig17_richardson_vs_2point_{model_short}.png"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]

    logger.info("Pre-building shared data + z-seed schedule...")
    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = _load_raw_dataset(args.task, split="train")
    ds = _SST2Dataset(raw, tokenizer, max_length=args.max_length)
    rng_data = np.random.default_rng(args.seed)
    batches = [
        _collate(
            [ds[i] for i in rng_data.choice(len(ds), size=args.batch_size, replace=False).tolist()],
            pad_token_id=tokenizer.pad_token_id,
        )
        for _ in range(args.steps)
    ]
    rng_z = np.random.default_rng(args.seed + 1)
    z_seeds = [int(rng_z.integers(0, 2**31 - 1)) for _ in range(args.steps)]
    logger.info(f"  {len(batches)} batches, {len(z_seeds)} z-seeds")

    # Run grid: for each eps, run three configurations.
    cells: dict[str, dict] = {}
    for eps in args.epsilons:
        for method in ("2point", "richardson_step_eq", "richardson_compute_eq"):
            key = f"{method}|eps={eps:.0e}"
            n_steps = args.steps if method != "richardson_compute_eq" else args.steps // 2
            logger.info(f"=== {key}  ({n_steps} steps) ===")
            cells[key] = _train(
                model_id=args.model, dtype=dtype, method=method,
                eps=eps, lr=args.lr, n_steps=n_steps,
                batches=batches, z_seeds=z_seeds,
                seed=args.seed, eval_every=args.eval_every,
            )

    out = {
        "model": args.model, "dtype": args.dtype, "task": args.task,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "lr": args.lr, "n_steps_2point": args.steps,
        "n_steps_richardson_compute_eq": args.steps // 2,
        "epsilons": args.epsilons, "seed": args.seed,
        "cells": cells,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Plot: 2 columns × len(epsilons) rows... no, simpler: 1 row × len(eps) cols.
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
         "lines.linewidth": 1.5}
    )
    n_eps = len(args.epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(4.0 * n_eps, 4.5), sharey=True)
    if n_eps == 1:
        axes = [axes]
    method_colours = {
        "2point": "#d62728",
        "richardson_step_eq": "#2ca02c",
        "richardson_compute_eq": "#1f77b4",
    }
    method_labels = {
        "2point": f"2-point (N={args.steps})",
        "richardson_step_eq": f"Richardson step-eq (N={args.steps}, 2× compute)",
        "richardson_compute_eq": f"Richardson compute-eq (N={args.steps // 2})",
    }

    for col_idx, eps in enumerate(args.epsilons):
        ax = axes[col_idx]
        for method in ("2point", "richardson_step_eq", "richardson_compute_eq"):
            key = f"{method}|eps={eps:.0e}"
            r = cells[key]
            steps = np.array(r["eval_steps"])
            evals = np.array(r["eval_losses"])
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            tag = method_labels[method] + f"  ({drop:+.1f}%)"
            if r["diverged"]:
                tag += " DIVERGED"
            ax.plot(steps, evals, "o-", color=method_colours[method],
                    markersize=4.5, label=tag, alpha=0.9)
        ax.set_xlabel("MeZO step")
        if col_idx == 0:
            ax.set_ylabel(r"eval loss at clean $\theta$")
        ax.set_title(rf"$\varepsilon = {eps:.0e}$", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Richardson 4-point vs 2-point central diff — {args.model} / {args.task} "
        f"(lr={args.lr}, B={args.batch_size}, {args.dtype})",
        fontsize=11, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary.
    logger.info("=" * 100)
    logger.info("Richardson vs 2-point — eval@clean-θ (lower L_final = better):")
    logger.info(f"  {'eps':<8}{'method':<28}{'steps':>7}{'fwds':>7}{'L_init':>9}{'L_final':>9}{'drop':>8}")
    for key, r in cells.items():
        L0 = r["eval_losses"][0]
        Lf = r["eval_losses"][-1]
        n = r["steps_completed"]
        fwds = r["total_forwards"]
        drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
        tag = "DIVERGED" if r["diverged"] else f"{drop:+.1f}%"
        logger.info(
            f"  {r['eps']:<8.0e}{r['method']:<28}{n:>7}{fwds:>7}"
            f"{L0:>9.3f}{Lf:>9.3f}{tag:>8}"
        )
    logger.info("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
