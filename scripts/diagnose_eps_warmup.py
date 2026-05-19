"""Warmup-based ε autotuner for MeZO.

For each candidate perturbation magnitude ε in a user-supplied grid, probe
the gradient estimator N times with FRESH random directions ``z_k`` (one per
probe, NOT reused across probes — otherwise direction-noise inflates the
estimate). For each ε compute three statistics:

1. **Variance proxy** ``Var[ρ̂(ε)]`` — direct estimator noise.
2. **Bias proxy** via second-order Taylor::

       Δ_2 ≈ (L(θ+εz) + L(θ-εz) - 2 L(θ)) / ε²  ~  z^T H z

   Mean of ``Δ_2`` across probes estimates ``E[z^T H z] = tr(H)``, which is
   ε-independent in the linear regime. **Deviation from the constant** at
   different ε reveals Taylor-nonlinearity.

3. **Combined trade-off score**::

       J(ε) = bias_proxy(ε)² + variance_proxy(ε)

   Optimal ε* minimizes ``J``. Caveat: bias_proxy uses the same loss
   measurements that go into ρ̂, so the two terms are correlated; the
   minimum is a useful operating point but not a precise theoretical optimum.

Output (always):

- ``docs/figures/fig9_eps_warmup_{model_short}.png`` (300 DPI 2-panel:
  bias vs ε and variance vs ε with the trade-off marked).
- ``experiments/diagnostics/eps_warmup_{model_short}.json`` (raw samples,
  fully reproducible).
- Stdout: recommended ε* with its bias/variance breakdown.

Usage::

    # Local sanity on the smallest model
    python scripts/diagnose_eps_warmup.py --model Qwen/Qwen3-0.6B

    # Cross-arch sweep (one call per arch)
    python scripts/diagnose_eps_warmup.py --model Qwen/Qwen3.5-0.8B
    python scripts/diagnose_eps_warmup.py --model Qwen/Qwen3-4B          # Colab
    python scripts/diagnose_eps_warmup.py --model Qwen/Qwen3.5-4B-Base   # Colab
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
from transformers import AutoTokenizer  # noqa: F401  — used inside loader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmezo.data.superglue import (  # noqa: E402
    _SST2Dataset,
    _collate,
    _load_raw_dataset,
    causal_lm_loss,
)
from dmezo.mezo.perturbation import perturb_parameters  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.eps_autotune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="HF model id (default fits 8 GB VRAM)",
    )
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--task", type=str, default="sst2")
    p.add_argument(
        "--eps-candidates", type=float, nargs="+",
        default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        help="Grid of ε values to probe (log-spaced is recommended)",
    )
    p.add_argument(
        "--n-probes", type=int, default=30,
        help="Number of fresh-z probes per ε candidate",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--out-dir", type=str,
        default=str(ROOT / "experiments" / "diagnostics"),
    )
    p.add_argument(
        "--fig-dir", type=str,
        default=str(ROOT / "docs" / "figures"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def _forward_loss(model, batch) -> float:
    """Loss measurement with autograd off and explicit fp32 upcast."""
    out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _probe_eps(
    model, named_params, batch, *, eps: float, z_seed: int
) -> tuple[float, float, float]:
    """Run one ε-probe at fixed batch. Returns (L_plus, L_minus, L0).

    L0 = baseline (no perturbation), only computed once per batch outside;
    we re-use it across all (eps, z_seed) combinations for this batch.
    Here we just return (L_plus, L_minus) and the caller pairs with cached L0.
    """
    perturb_parameters(named_params, seed=z_seed, scaling_factor=+1.0, eps=eps)
    L_plus = _forward_loss(model, batch)
    perturb_parameters(named_params, seed=z_seed, scaling_factor=-2.0, eps=eps)
    L_minus = _forward_loss(model, batch)
    perturb_parameters(named_params, seed=z_seed, scaling_factor=+1.0, eps=eps)
    return L_plus, L_minus, 0.0  # L0 placeholder, filled by caller


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"eps_warmup_{model_short}.json"
    fig_path = fig_dir / f"fig9_eps_warmup_{model_short}.png"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]
    logger.info(f"Loading {args.model} in {dtype}...")
    model, tokenizer = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading task={args.task}...")
    raw = _load_raw_dataset(args.task, split="train")
    ds = _SST2Dataset(raw, tokenizer, max_length=args.max_length)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=args.batch_size, replace=False).tolist()
    items = [ds[i] for i in idx]
    batch = _collate(items, pad_token_id=tokenizer.pad_token_id)

    L0 = _forward_loss(model, batch)
    logger.info(f"Baseline L₀ = {L0:.6f}")

    # ---- Probe each ε candidate with fresh z per probe.
    # Divergence handling: a probe is INVALID if L_plus or L_minus is NaN/inf
    # (catastrophic perturbation knocks logits out of representable fp16 range).
    # An ε candidate is marked "diverged" if any probe is invalid, since even
    # a single NaN means we are past the safe-perturbation cliff and any
    # statistics from this candidate are unreliable / misleading.
    raw_data: dict[str, dict] = {}
    for eps in args.eps_candidates:
        rhos: list[float] = []
        bias_proxies: list[float] = []
        diverged = False
        for probe in range(args.n_probes):
            z_seed = int(rng.integers(0, 2**31 - 1))
            L_plus, L_minus, _ = _probe_eps(model, named, batch, eps=eps, z_seed=z_seed)
            if not (np.isfinite(L_plus) and np.isfinite(L_minus)):
                diverged = True
                # Still collect the remaining probes so users can inspect the JSON,
                # but mark the ε as diverged so it is excluded from j_score.
                rhos.append(float("nan"))
                bias_proxies.append(float("nan"))
                continue
            rho = (L_plus - L_minus) / (2.0 * eps)
            # 2nd-order Taylor proxy: (L+ + L- - 2 L0) / ε² ≈ z^T H z (sign-invariant)
            taylor_2 = (L_plus + L_minus - 2.0 * L0) / (eps**2)
            rhos.append(rho)
            bias_proxies.append(taylor_2)
        rhos_arr = np.asarray(rhos)
        bias_arr = np.asarray(bias_proxies)
        raw_data[f"{eps:.2e}"] = {
            "eps": float(eps),
            "rhos": rhos_arr.tolist(),
            "bias_proxies": bias_arr.tolist(),
            "rho_mean": float(np.nanmean(rhos_arr)) if not diverged else float("nan"),
            "rho_std": float(np.nanstd(rhos_arr)) if not diverged else float("nan"),
            "bias_mean": float(np.nanmean(bias_arr)) if not diverged else float("nan"),
            "bias_std": float(np.nanstd(bias_arr)) if not diverged else float("nan"),
            "diverged": bool(diverged),
        }
        if diverged:
            logger.warning(
                f"eps={eps:.2e}: DIVERGED (forward returned NaN/inf for at least one probe) — "
                "excluded from j_score selection"
            )
        else:
            logger.info(
                f"eps={eps:.2e}: rho mean={rhos_arr.mean():+.3f} std={rhos_arr.std():.3f}  "
                f"bias-proxy mean={bias_arr.mean():+.3f} std={bias_arr.std():.3f}"
            )

    # ---- Compute trade-off score per ε (diverged candidates excluded).
    score_table = {}
    for k, v in raw_data.items():
        if v["diverged"]:
            score_table[k] = {
                "eps": v["eps"],
                "bias": float("nan"),
                "var": float("nan"),
                "diverged": True,
            }
            continue
        bias_strength = abs(v["bias_mean"])  # |E[z^T H z]| (Taylor-2 magnitude)
        var = v["rho_std"] ** 2
        score_table[k] = {"eps": v["eps"], "bias": bias_strength, "var": var, "diverged": False}

    eps_vals = np.array([s["eps"] for s in score_table.values()])
    biases = np.array([s["bias"] for s in score_table.values()])
    vars_ = np.array([s["var"] for s in score_table.values()])
    valid_mask = np.array([not s["diverged"] for s in score_table.values()])
    if not valid_mask.any():
        raise RuntimeError(
            "All ε candidates diverged. Reduce the upper range of --eps-candidates."
        )

    # Min-max normalize over VALID candidates only so neither dimension dominates.
    def _norm_valid(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=np.float64)
        if mask.any():
            lo = float(x[mask].min())
            hi = float(x[mask].max())
            rng = hi - lo
            out[mask] = (x[mask] - lo) / (rng + 1e-12)
        return out

    j_score = _norm_valid(biases, valid_mask) + _norm_valid(vars_, valid_mask)
    # nanargmin: ignores NaN entries (diverged candidates).
    eps_star_idx = int(np.nanargmin(j_score))
    eps_star = float(eps_vals[eps_star_idx])

    # ---- Save raw + scores.
    out_data = {
        "model": args.model,
        "dtype": args.dtype,
        "task": args.task,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_probes": args.n_probes,
        "baseline_loss": L0,
        "probes": raw_data,
        "scores": {k: {**s, "j_norm": float(j_score[i])} for i, (k, s) in enumerate(score_table.items())},
        "eps_star": eps_star,
        "eps_star_idx": eps_star_idx,
    }
    json_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    logger.info(f"Saved raw + scores to {json_path}")

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
            "lines.linewidth": 1.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axL, axR = axes[0], axes[1]

    # Left: bias proxy + variance proxy vs ε.
    axL.semilogx(eps_vals, biases, "o-", color="#d62728", label=r"$|E[z^T H z]|$ (bias proxy)")
    axL2 = axL.twinx()
    axL2.semilogx(eps_vals, np.sqrt(vars_), "s-", color="#1f77b4", label=r"std$[\hat\rho]$ (variance proxy)")
    axL.set_xlabel(r"$\varepsilon$")
    axL.set_ylabel(r"$|E[z^T H z]|$ (bias)", color="#d62728")
    axL2.set_ylabel(r"std$[\hat\rho]$ (variance, $\sqrt{\cdot}$)", color="#1f77b4")
    axL.tick_params(axis="y", labelcolor="#d62728")
    axL2.tick_params(axis="y", labelcolor="#1f77b4")
    axL.axvline(eps_star, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"recommended ε* = {eps_star:.2e}")
    axL.legend(loc="upper left", fontsize=9)
    axL2.legend(loc="upper right", fontsize=9)
    axL.set_title("(a) Bias-variance breakdown vs ε")

    # Right: combined trade-off J(ε).
    axR.semilogx(eps_vals, j_score, "o-", color="#2ca02c", label=r"$J(\varepsilon)$ (normalized bias + var)")
    axR.axvline(eps_star, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"ε* = {eps_star:.2e}  (J = {j_score[eps_star_idx]:.3f})")
    axR.set_xlabel(r"$\varepsilon$")
    axR.set_ylabel(r"$J(\varepsilon) = \tilde{\rm bias} + \tilde{\rm var}$ (min-max normalised)")
    axR.legend(loc="upper center", fontsize=9)
    axR.set_title("(b) Combined trade-off score")

    fig.suptitle(
        f"ε warmup autotuner on {args.model} ({args.dtype}, B={args.batch_size}, n_probes={args.n_probes})",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # NOTE: avoid Unicode in print() — Windows cp1251 stdout chokes on ε / ρ̂.
    # The JSON file (saved above) preserves the precise values.
    logger.info("")
    logger.info("=" * 64)
    logger.info(f"  Recommended eps* = {eps_star:.2e}  (idx={eps_star_idx})")
    logger.info(
        f"  At eps*: bias-proxy={biases[eps_star_idx]:+.3f}  "
        f"std[rho]={np.sqrt(vars_[eps_star_idx]):.3f}"
    )
    is_default = abs(eps_star - 1e-3) < 1e-10
    logger.info(
        "  Princeton default (1e-3) for reference: "
        + ("OPTIMAL" if is_default else "differs from autotune choice")
    )
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
