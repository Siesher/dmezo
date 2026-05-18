"""Batch-size variance diagnostic for the MeZO ρ estimator.

Produces ``docs/figures/fig8_batch_variance.png`` — a 6-panel histogram showing
the distribution of the per-step projected-gradient estimate

.. math::

    \\hat\\rho(B) = \\frac{L_B(\\theta + \\epsilon z) - L_B(\\theta - \\epsilon z)}{2\\epsilon}

for different mini-batch sizes ``B ∈ {1, 2, 4, 8, 16, 32}``, with model
parameters and perturbation direction ``z`` held fixed. The plot empirically
validates the central-limit prediction that ``std[ρ̂(B)] ∝ 1/√B``.

Defaults are tuned for a fast local run on a Turing/Blackwell-class GPU with
Qwen3-0.6B:

- 200 random batches per B value (1200 ρ samples in total)
- Single fixed seed for ``z``
- Cached model on disk (~1.2 GB FP16)

Output: figure + JSON dump of (B → list of ρ samples) under
``experiments/diagnostics/``.

Usage::

    python scripts/diagnose_batch_variance.py
    python scripts/diagnose_batch_variance.py --model Qwen/Qwen3-0.6B --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmezo.data.superglue import _SST2Dataset, _collate, _load_raw_dataset, causal_lm_loss  # noqa: E402
from dmezo.mezo.perturbation import perturb_parameters  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.diagnose")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                   help="HF model name (default: Qwen3-0.6B, fits 8 GB VRAM)")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--task", type=str, default="sst2")
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of random batches to sample per B value")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--z-seed", type=int, default=12345,
                   help="Seed for the fixed perturbation direction z")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out-dir", type=str,
                   default=str(ROOT / "experiments" / "diagnostics"))
    p.add_argument("--fig-path", type=str,
                   default=str(ROOT / "docs" / "figures" / "fig8_batch_variance.png"))
    return p.parse_args()


@torch.inference_mode()
def _forward_loss_value(model, batch) -> float:
    """Compute scalar loss with autograd off and explicit fp32 cast."""
    out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _sample_rho(model, named_params, dataset, indices, tokenizer, z_seed, eps) -> float:
    """One ρ̂ sample on the given index set (one mini-batch)."""
    items = [dataset[i] for i in indices]
    batch = _collate(items, pad_token_id=tokenizer.pad_token_id)
    # +eps z
    perturb_parameters(named_params, seed=z_seed, scaling_factor=+1.0, eps=eps)
    l_plus = _forward_loss_value(model, batch)
    # -2 eps z  → at θ - eps z
    perturb_parameters(named_params, seed=z_seed, scaling_factor=-2.0, eps=eps)
    l_minus = _forward_loss_value(model, batch)
    # restore
    perturb_parameters(named_params, seed=z_seed, scaling_factor=+1.0, eps=eps)
    return (l_plus - l_minus) / (2.0 * eps)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load model + tokenizer.
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    logger.info(f"Loading {args.model} in {dtype}...")
    model, tokenizer = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load dataset.
    logger.info(f"Loading task={args.task}...")
    raw = _load_raw_dataset(args.task, split="train")
    ds = _SST2Dataset(raw, tokenizer, max_length=args.max_length)
    n_total = len(ds)
    logger.info(f"Pool size: {n_total}")

    # ---- Sample ρ for each B.
    rng = np.random.default_rng(0)
    samples: dict[int, list[float]] = {}
    for B in args.batch_sizes:
        rhos: list[float] = []
        logger.info(f"Sampling B={B}: {args.n_samples} batches × bs={B}...")
        for k in range(args.n_samples):
            idx = rng.choice(n_total, size=B, replace=False).tolist()
            rho = _sample_rho(model, named, ds, idx, tokenizer, args.z_seed, args.eps)
            rhos.append(rho)
            if (k + 1) % 50 == 0:
                logger.info(f"  B={B}: {k+1}/{args.n_samples}, std so far={np.std(rhos):.4f}")
        samples[B] = rhos
        logger.info(f"B={B}: mean={np.mean(rhos):+.4f}  std={np.std(rhos):.4f}")

    # ---- Save raw samples.
    json_path = out_dir / f"batch_variance_{args.model.replace('/', '_')}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "task": args.task,
                "eps": args.eps,
                "z_seed": args.z_seed,
                "n_samples": args.n_samples,
                "samples": {str(B): rhos for B, rhos in samples.items()},
                "stats": {
                    str(B): {"mean": float(np.mean(rhos)), "std": float(np.std(rhos))}
                    for B, rhos in samples.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved raw samples to {json_path}")

    # ---- Build figure.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
        }
    )
    n_panels = len(args.batch_sizes)
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5 * rows), sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Shared x-axis range for visual comparability.
    all_rhos = np.concatenate([np.asarray(r) for r in samples.values()])
    x_lim = (np.quantile(all_rhos, 0.005), np.quantile(all_rhos, 0.995))
    bins = np.linspace(x_lim[0], x_lim[1], 30)

    stds = []
    for ax, B in zip(axes_flat, args.batch_sizes):
        rhos = np.asarray(samples[B])
        ax.hist(rhos, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(np.mean(rhos), color="#d62728", linewidth=1.4, label=f"mean = {np.mean(rhos):+.3f}")
        std = float(np.std(rhos))
        stds.append(std)
        ax.set_title(f"B = {B}    σ = {std:.4f}", fontsize=11)
        ax.set_xlabel(r"$\hat\rho$")
        ax.set_ylabel("count")
        ax.set_xlim(x_lim)
        ax.legend(fontsize=9, loc="upper right")

    # Hide unused subplots.
    for ax in axes_flat[len(args.batch_sizes):]:
        ax.axis("off")

    # Theoretical std curve in the title/super-suptitle area.
    sigma_0 = stds[0]
    expected_stds = [sigma_0 / np.sqrt(B / args.batch_sizes[0]) for B in args.batch_sizes]
    ratio_text = "   ".join(
        f"B={B}: σ={s:.4f} (expect {e:.4f})"
        for B, s, e in zip(args.batch_sizes, stds, expected_stds)
    )
    fig.suptitle(
        f"ρ̂ distribution vs mini-batch size B "
        f"({args.model}, {args.dtype}, n={args.n_samples}/B, fixed z @seed={args.z_seed})\n"
        f"{ratio_text}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Print 1/√B-scaling check.
    logger.info("std vs theoretical 1/√B scaling (rel to B={}):".format(args.batch_sizes[0]))
    for B, s, e in zip(args.batch_sizes, stds, expected_stds):
        ratio = s / e if e > 0 else float("nan")
        logger.info(f"  B={B}: observed σ={s:.4f}, expected σ={e:.4f}, ratio={ratio:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
