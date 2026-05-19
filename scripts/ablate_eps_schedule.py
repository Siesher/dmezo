"""Ablation: does ε scheduling help vs constant ε in fp16 MeZO?

Per-step time-varying ε(t) is NOT exposed in canonical MeZOConfig — this script
injects ε(t) into mezo_step by building a fresh per-step config inside the
training loop. If a winner emerges, we'd then promote it to MeZOConfig + tests.

Schedules tested:
    const_1e-3                    Princeton baseline (control)
    exp_decay_1e-2_to_1e-3       mild Spall-style finite-diff decay
    exp_decay_3e-2_to_1e-3       aggressive autotuner-then-refine
    exp_decay_1e-3_to_1e-4       classical SPSA refinement below Princeton
    exp_grow_1e-3_to_1e-2        anti-schedule control (predicted to fail
                                  per the §6.7 ε^2 ∇^3 L bias mechanism)

Per-step ε computed as exponential interpolation in log-space:
    ε(t) = ε_start · (ε_end / ε_start)^(t / (N-1))

Fair-comparison protocol mirrors validate_eps_downstream.py:
    - Same model + lr + batches + z-seeds across all schedules.
    - Differences in trajectory are attributable PURELY to the ε schedule.

Outputs:
    experiments/diagnostics/eps_schedule_{model_short}.json
    docs/figures/fig13_eps_schedule_{model_short}.png
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
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.eps_schedule")


# ---------------------------------------------------------------------------
# Schedule definitions.
# Each entry: (name, eps_start, eps_end). Interpolation is exponential in
# log-space, which is the natural metric for ε (since ε is multiplicative
# in the finite-difference formula).
# ---------------------------------------------------------------------------
SCHEDULES: dict[str, tuple[float, float]] = {
    "const_1e-3":             (1e-3, 1e-3),
    "exp_decay_1e-2_to_1e-3": (1e-2, 1e-3),
    "exp_decay_3e-2_to_1e-3": (3e-2, 1e-3),
    "exp_decay_1e-3_to_1e-4": (1e-3, 1e-4),
    "exp_grow_1e-3_to_1e-2":  (1e-3, 1e-2),
}


def _build_eps_schedule(name: str, n_steps: int) -> np.ndarray:
    """Return ε(t) array of length n_steps for the named schedule."""
    eps_start, eps_end = SCHEDULES[name]
    if n_steps == 1:
        return np.array([eps_start])
    # Exponential interpolation in log-space:
    log_start = np.log(eps_start)
    log_end = np.log(eps_end)
    t_frac = np.linspace(0.0, 1.0, n_steps)
    return np.exp(log_start + (log_end - log_start) * t_frac)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--schedules", type=str, nargs="+", default=list(SCHEDULES.keys()),
        choices=list(SCHEDULES.keys()),
    )
    p.add_argument("--steps", type=int, default=100)
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


def _train_one_schedule(
    *,
    model_id: str,
    dtype: torch.dtype,
    schedule_name: str,
    eps_schedule: np.ndarray,
    lr: float,
    batches: list[dict],
    z_seeds: list[int],
    seed: int,
    eval_every: int,
) -> dict:
    """Train for ``len(eps_schedule)`` steps with per-step ε(t)."""
    torch.manual_seed(seed)
    model, _ = load_causal_lm(model_id, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    base_cfg = MeZOConfig(lr=lr, eps=float(eps_schedule[0]))
    eval_batch = batches[0]
    L0 = _eval_loss(model, eval_batch)
    losses: list[float] = [L0]
    rhos: list[float] = [0.0]
    eval_losses: list[float] = [L0]
    eval_steps: list[int] = [0]
    eps_log: list[float] = [float(eps_schedule[0])]
    diverged = False
    t0 = time.time()

    for step in range(len(eps_schedule)):
        eps_t = float(eps_schedule[step])
        cfg_t = replace(base_cfg, eps=eps_t)
        one_shot = np.random.Generator(np.random.PCG64(z_seeds[step]))
        try:
            seed_used, rho, loss_plus = mezo_step(
                model, batches[step], causal_lm_loss, cfg_t, rng=one_shot,
            )
        except RuntimeError as e:
            logger.warning(f"{schedule_name} step {step}: {e!r}")
            diverged = True
            break
        if not np.isfinite(loss_plus) or not np.isfinite(rho):
            logger.warning(f"{schedule_name} step {step}: NaN/inf in forward")
            diverged = True
            break
        mezo_update(model, seed_used, rho, cfg_t)
        losses.append(float(loss_plus))
        rhos.append(float(rho))
        eps_log.append(eps_t)
        if (step + 1) % eval_every == 0:
            eval_L = _eval_loss(model, eval_batch)
            eval_losses.append(eval_L)
            eval_steps.append(step + 1)
            logger.info(
                f"{schedule_name:<28s}  step={step + 1:4d}  eps={eps_t:.2e}  "
                f"loss+={loss_plus:.4f}  rho={rho:+.2f}  eval={eval_L:.4f}"
            )

    wall = time.time() - t0
    del model
    torch.cuda.empty_cache()
    return {
        "schedule": schedule_name,
        "eps_start": float(eps_schedule[0]),
        "eps_end": float(eps_schedule[-1]),
        "lr": float(lr),
        "steps_completed": len(losses) - 1,
        "diverged": bool(diverged),
        "losses": losses,
        "rhos": rhos,
        "eval_losses": eval_losses,
        "eval_steps": eval_steps,
        "eps_log": eps_log,
        "wall_clock_s": float(wall),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"eps_schedule_{model_short}.json"
    fig_path = fig_dir / f"fig13_eps_schedule_{model_short}.png"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]

    # Pre-build shared schedule.
    logger.info("Pre-building shared data + z-seed schedule...")
    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = _load_raw_dataset(args.task, split="train")
    ds = _SST2Dataset(raw, tokenizer, max_length=args.max_length)
    rng_data = np.random.default_rng(args.seed)
    batches = [
        _collate([ds[i] for i in rng_data.choice(len(ds), size=args.batch_size, replace=False).tolist()],
                 pad_token_id=tokenizer.pad_token_id)
        for _ in range(args.steps)
    ]
    rng_z = np.random.default_rng(args.seed + 1)
    z_seeds = [int(rng_z.integers(0, 2**31 - 1)) for _ in range(args.steps)]
    logger.info(f"  {len(batches)} batches, {len(z_seeds)} z-seeds pre-built")

    runs: dict[str, dict] = {}
    for sched_name in args.schedules:
        eps_arr = _build_eps_schedule(sched_name, args.steps)
        logger.info(
            f"=== {sched_name}: eps[0]={eps_arr[0]:.2e} -> eps[N-1]={eps_arr[-1]:.2e} ==="
        )
        runs[sched_name] = _train_one_schedule(
            model_id=args.model, dtype=dtype, schedule_name=sched_name,
            eps_schedule=eps_arr, lr=args.lr, batches=batches, z_seeds=z_seeds,
            seed=args.seed, eval_every=args.eval_every,
        )

    out = {
        "model": args.model, "dtype": args.dtype, "task": args.task,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "lr": args.lr, "steps": args.steps, "seed": args.seed,
        "schedule_defs": {k: {"start": v[0], "end": v[1]} for k, v in SCHEDULES.items()},
        "runs": runs,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Plot: 2-panel composite.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
            "lines.linewidth": 1.6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_eps, ax_eval = axes
    cmap = plt.get_cmap("tab10")

    sorted_runs = list(runs.items())
    for i, (name, r) in enumerate(sorted_runs):
        col = cmap(i % 10)
        eps_log = np.array(r["eps_log"])
        steps_axis = np.arange(len(eps_log))
        ax_eps.semilogy(steps_axis, eps_log, "-", color=col, label=name, alpha=0.9)

        evals = np.array(r["eval_losses"])
        eval_steps = np.array(r["eval_steps"])
        L0, Lf = evals[0], evals[-1]
        drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
        tag = " (DIVERGED)" if r["diverged"] else ""
        ax_eval.plot(eval_steps, evals, "o-", color=col, markersize=4.5,
                     label=f"{name}  ({drop:+.1f}%){tag}", alpha=0.9)

    ax_eps.set_xlabel("MeZO step")
    ax_eps.set_ylabel(r"$\varepsilon(t)$  (log scale)")
    ax_eps.set_title("(a) ε(t) schedules")
    ax_eps.legend(loc="lower left", fontsize=8)

    ax_eval.set_xlabel("MeZO step")
    ax_eval.set_ylabel(r"eval loss at clean $\theta$  (fixed batch)")
    ax_eval.set_title("(b) Convergence at clean θ")
    ax_eval.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"ε-schedule ablation on {args.model}  (SST-2, lr={args.lr}, B={args.batch_size}, fp16)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary.
    logger.info("=" * 88)
    logger.info(f"ε-schedule ablation on {args.model}  (eval@clean-θ):")
    summary = []
    for name, r in runs.items():
        evals = r["eval_losses"]
        n = r["steps_completed"]
        if r["diverged"]:
            tag = f"DIVERGED@step{n}"
            drop = float("nan")
        else:
            L0 = evals[0]
            Lf = evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            tag = f"L0={L0:.3f} -> L_final={Lf:.3f}  (drop={drop:+.1f}%)"
        summary.append((name, drop))
        logger.info(f"  {name:<28s}  steps={n:>3d}/{args.steps}  {tag}")
    logger.info("=" * 88)
    summary.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else float("-inf"))
    winner = summary[0]
    logger.info(f"WINNER by eval-loss drop: {winner[0]}  ({winner[1]:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
