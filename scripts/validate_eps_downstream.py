"""Downstream validation of the ε autotuner choice.

The autotuner picks ε* by minimizing a bias-variance trade-off proxy. But the
proxy can be confounded by floating-point roundoff at small ε. This script
runs SHORT MeZO training (100 steps) at multiple ε values on the same model +
same batches + same z-sequence, so any difference in loss trajectory is
attributable PURELY to ε.

Goal: confirm (or refute) that the autotuner's ε* gives faster / more stable
convergence than the Princeton default ε=1e-3.

Output:
    - experiments/diagnostics/eps_validate_{model_short}.json (trajectories)
    - docs/figures/fig11_eps_validate_{model_short}.png (loss-vs-step plot)

Usage::

    python scripts/validate_eps_downstream.py \\
        --model Qwen/Qwen3-0.6B \\
        --epsilons 1e-3 3e-2 1e-1 \\
        --steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.eps_validate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--epsilons", type=float, nargs="+", default=[1e-3, 3e-2, 1e-1],
        help="ε values to compare. Loss trajectories run sequentially.",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-7, help="Same lr for all ε runs.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-every", type=int, default=10,
        help="Sample eval-loss at clean θ every N steps (smaller = denser figure)",
    )
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--task", type=str, default="sst2")
    return p.parse_args()


def _eval_loss(model, batch) -> float:
    """No-grad forward pass returning scalar loss (fp32-upcast)."""
    with torch.inference_mode():
        out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _train_one_eps(
    *,
    model_id: str,
    dtype: torch.dtype,
    eps: float,
    lr: float,
    steps: int,
    batches: list[dict],
    z_seeds: list[int],
    seed: int,
    eval_every: int = 10,
) -> dict:
    """Run ``steps`` MeZO updates at fixed ε on a freshly-loaded model.

    Args:
        model_id: HF model id (e.g. "Qwen/Qwen3-0.6B").
        dtype: torch dtype for model weights.
        eps: perturbation magnitude.
        lr: learning rate.
        steps: number of MeZO steps.
        batches: pre-built list of length ``steps`` of input batches.
        z_seeds: pre-built list of length ``steps`` of int seeds for z direction.
            Identical sequence used across all ε runs → trajectories differ
            ONLY due to ε (not stochasticity).
        seed: torch seed (controls model init randomness for reproducibility).

    Returns:
        Dict with per-step loss, projected_grad, wall-clock metadata, and
        a ``diverged`` flag set if any loss is NaN/inf.
    """
    torch.manual_seed(seed)
    model, _ = load_causal_lm(model_id, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    cfg = MeZOConfig(lr=lr, eps=eps)

    losses: list[float] = []
    rhos: list[float] = []
    eval_losses: list[float] = []
    eval_steps: list[int] = []
    diverged = False
    t0 = time.time()
    eval_batch = batches[0]  # fixed-batch eval probe for tracking
    L0 = _eval_loss(model, eval_batch)
    eval_losses.append(L0)
    eval_steps.append(0)
    losses.append(L0)
    rhos.append(0.0)

    for step in range(steps):
        batch = batches[step]
        z_seed = z_seeds[step]
        # We feed the seed into mezo_step's rng by faking a one-shot generator.
        one_shot = np.random.Generator(np.random.PCG64(z_seed))
        try:
            seed_used, rho, loss_plus = mezo_step(model, batch, causal_lm_loss, cfg, rng=one_shot)
        except RuntimeError as e:
            logger.warning(f"eps={eps:.2e} step {step}: mezo_step raised {e!r}")
            diverged = True
            break
        if not np.isfinite(loss_plus) or not np.isfinite(rho):
            logger.warning(f"eps={eps:.2e} step {step}: NaN/inf in forward")
            diverged = True
            break
        mezo_update(model, seed_used, rho, cfg)
        losses.append(float(loss_plus))
        rhos.append(float(rho))
        if (step + 1) % eval_every == 0:
            eval_L = _eval_loss(model, eval_batch)
            eval_losses.append(eval_L)
            eval_steps.append(step + 1)
            logger.info(
                f"eps={eps:.2e}  step={step + 1:4d}  loss+={loss_plus:.4f}  "
                f"rho={rho:+.2f}  eval@fixed-batch={eval_L:.4f}"
            )

    wall = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return {
        "eps": float(eps),
        "lr": float(lr),
        "steps_completed": len(losses) - 1,  # exclude initial L0
        "diverged": bool(diverged),
        "losses": losses,
        "rhos": rhos,
        "eval_losses": eval_losses,
        "eval_steps": eval_steps,
        "wall_clock_s": float(wall),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"eps_validate_{model_short}.json"
    fig_path = fig_dir / f"fig11_eps_validate_{model_short}.png"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]

    # Pre-build the SHARED training schedule: same batches + same z-seeds for all ε runs.
    logger.info("Pre-building shared data + z-seed schedule...")
    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = _load_raw_dataset(args.task, split="train")
    ds = _SST2Dataset(raw, tokenizer, max_length=args.max_length)
    rng_data = np.random.default_rng(args.seed)
    batches: list[dict] = []
    for _ in range(args.steps):
        idx = rng_data.choice(len(ds), size=args.batch_size, replace=False).tolist()
        items = [ds[i] for i in idx]
        batches.append(_collate(items, pad_token_id=tokenizer.pad_token_id))
    rng_z = np.random.default_rng(args.seed + 1)
    z_seeds: list[int] = [int(rng_z.integers(0, 2**31 - 1)) for _ in range(args.steps)]
    logger.info(f"  {len(batches)} batches, {len(z_seeds)} z-seeds pre-built")

    # Run each ε sequentially (fresh model each time).
    runs: dict[str, dict] = {}
    for eps in args.epsilons:
        logger.info(f"=== eps = {eps:.2e}  (lr={args.lr}, steps={args.steps}) ===")
        runs[f"{eps:.2e}"] = _train_one_eps(
            model_id=args.model,
            dtype=dtype,
            eps=eps,
            lr=args.lr,
            steps=args.steps,
            batches=batches,
            z_seeds=z_seeds,
            seed=args.seed,
            eval_every=args.eval_every,
        )

    out = {
        "model": args.model,
        "dtype": args.dtype,
        "task": args.task,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "runs": runs,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Plot.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axL, axR = axes
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(args.epsilons) - 1, 1)) for i in range(len(args.epsilons))]

    for (eps_str, r), col in zip(runs.items(), colors):
        if r["diverged"] and len(r.get("eval_losses", [])) <= 1:
            logger.warning(f"  eps={eps_str}: diverged before first eval, skipping plot")
            continue
        label = f"ε={r['eps']:.0e}"
        if r["diverged"]:
            label += "  (DIVERGED)"

        # Primary panel: eval@clean-θ on a fixed batch. This is the ONLY
        # interpretable convergence signal — loss+ is noise-floor dominated
        # by ε²·tr(H) and grows with ε independent of actual progress.
        eval_steps = np.array(r.get("eval_steps", []))
        eval_losses = np.array(r["eval_losses"])
        axL.plot(eval_steps, eval_losses, "o-", color=col, label=label, alpha=0.9,
                 markersize=5)

        # Secondary panel: perturbed-loss trajectory (smoothed). Useful for
        # noise-floor diagnostics, NOT for judging convergence.
        loss = np.array(r["losses"])
        if len(loss) > 20:
            w = 10
            smooth = np.convolve(loss, np.ones(w) / w, mode="valid")
            axR.plot(np.arange(len(smooth)) + w // 2, smooth, "-", color=col,
                     label=label, alpha=0.85)

    axL.set_xlabel("MeZO step")
    axL.set_ylabel(r"eval loss at clean $\theta$  (fixed batch)")
    axL.set_title("(a) Convergence at clean θ  (objective metric)")
    axL.legend(loc="upper right", fontsize=9)
    axR.set_xlabel("MeZO step")
    axR.set_ylabel(r"loss$(\theta + \varepsilon z)$  (running mean, w=10)")
    axR.set_title("(b) Perturbed-loss noise floor  (diagnostic only)")
    axR.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Downstream ε validation on {args.model}  "
        f"({args.task}, B={args.batch_size}, lr={args.lr}, fp16)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary (use eval@clean-θ as the meaningful metric, NOT loss+).
    logger.info("=" * 80)
    logger.info("Downstream validation summary  (eval-loss at CLEAN θ on fixed batch):")
    for eps_str, r in runs.items():
        n = r["steps_completed"]
        evals = r.get("eval_losses", [])
        if r["diverged"]:
            tag = f"DIVERGED@step{n}"
        elif len(evals) < 2:
            tag = "no-eval-samples"
        else:
            initial = evals[0]
            final = evals[-1]
            drop = (initial - final) / initial * 100 if initial != 0 else 0.0
            tag = f"L0={initial:.3f} -> L_final={final:.3f}  (drop={drop:+.1f}%)"
        logger.info(f"  eps={eps_str:>9s}  steps={n:>3d}/{args.steps}  {tag}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
