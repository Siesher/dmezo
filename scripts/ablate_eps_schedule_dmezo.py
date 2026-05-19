"""Ablation: does ε(t) scheduling help in D-MeZO-N (Nesterov + ρ-clip + β-decay)?

The vanilla MeZO ablation (`ablate_eps_schedule.py`) showed:
    1. Warmup-style schedules (large → small) systematically lose.
    2. Constant ε=1e-3 is robust.
    3. Refinement 1e-3 → 1e-4 ties or beats const on hybrid.

But our deployed algorithm is D-MeZO-N — heavy-ball Nesterov with ρ-clipping
(C=50) and linear β-decay (0.9 → 0). Three potential interactions:
    (a) Momentum smooths the per-step direction noise → might rescue
        large-ε warmup phase that vanilla cannot survive.
    (b) ρ-clip caps the noise contribution, also potentially helping
        large-ε early steps.
    (c) β-decay shifts the dynamics from acceleration-dominated (early)
        to plain-SGD-dominated (late), interacting with ε(t) phases.

This script supports BOTH:
    --variant vanilla   — replicates ablate_eps_schedule.py for direct check
    --variant dmezo_n   — Nesterov heavy-ball + ρ-clip + linear β-decay

Per-step ε(t) is injected by rebuilding MeZOConfig each step. NesterovState
schedules β internally over ``--steps`` rounds.

Outputs:
    experiments/diagnostics/eps_schedule_{variant}_{model_short}.json
    docs/figures/fig15_eps_schedule_{variant}_{model_short}.png
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
from dmezo.mezo.nesterov import NesterovState, nesterov_step  # noqa: E402
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.eps_schedule_dmezo")

SCHEDULES: dict[str, tuple[float, float]] = {
    "const_1e-3":             (1e-3, 1e-3),
    "exp_decay_1e-3_to_1e-4": (1e-3, 1e-4),
    "exp_decay_1e-2_to_1e-3": (1e-2, 1e-3),
    "exp_decay_3e-2_to_1e-3": (3e-2, 1e-3),
    "exp_grow_1e-3_to_1e-2":  (1e-3, 1e-2),
}


def _build_eps_schedule(name: str, n_steps: int) -> np.ndarray:
    eps_start, eps_end = SCHEDULES[name]
    if n_steps == 1:
        return np.array([eps_start])
    log_start, log_end = np.log(eps_start), np.log(eps_end)
    t_frac = np.linspace(0.0, 1.0, n_steps)
    return np.exp(log_start + (log_end - log_start) * t_frac)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--variant", type=str, default="dmezo_n",
        choices=["vanilla", "dmezo_n"],
        help="vanilla = plain MeZO; dmezo_n = Nesterov β-decay + ρ-clip C=50",
    )
    p.add_argument(
        "--schedules", type=str, nargs="+",
        default=["const_1e-3", "exp_decay_1e-3_to_1e-4", "exp_decay_1e-2_to_1e-3"],
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
    # D-MeZO-N hyperparameters (only used when --variant dmezo_n).
    p.add_argument("--rho-clip", type=float, default=50.0,
                   help="ρ-clip threshold (only for dmezo_n). 0 = disable.")
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    return p.parse_args()


def _eval_loss(model, batch) -> float:
    with torch.inference_mode():
        out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _train_one_schedule(
    *,
    args,
    dtype: torch.dtype,
    schedule_name: str,
    eps_schedule: np.ndarray,
    batches: list[dict],
    z_seeds: list[int],
) -> dict:
    """Train ``len(eps_schedule)`` steps with per-step ε(t).

    Centralized client. For ``variant=dmezo_n``, applies Nesterov heavy-ball
    update with ρ-clip and linear β-decay over the run horizon.
    """
    torch.manual_seed(args.seed)
    model, _ = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    rho_clip = args.rho_clip if (args.variant == "dmezo_n" and args.rho_clip > 0) else None
    base_cfg = MeZOConfig(lr=args.lr, eps=float(eps_schedule[0]), rho_clip=rho_clip)

    nesterov: NesterovState | None = None
    if args.variant == "dmezo_n":
        nesterov = NesterovState(
            beta=args.beta_start,
            look_ahead=False,
            beta_end=args.beta_end,
            num_rounds_total=args.steps,
        )

    eval_batch = batches[0]
    L0 = _eval_loss(model, eval_batch)
    losses: list[float] = [L0]
    rhos: list[float] = [0.0]
    eval_losses: list[float] = [L0]
    eval_steps: list[int] = [0]
    eps_log: list[float] = [float(eps_schedule[0])]
    beta_log: list[float] = [nesterov.beta if nesterov is not None else 0.0]
    diverged = False
    t0 = time.time()

    for step in range(len(eps_schedule)):
        eps_t = float(eps_schedule[step])
        cfg_t = replace(base_cfg, eps=eps_t)
        if nesterov is not None:
            nesterov.update_schedule(step)
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
            logger.warning(f"{schedule_name} step {step}: NaN/inf")
            diverged = True
            break
        if nesterov is not None:
            nesterov_step(model, nesterov, seed_used, rho, cfg_t.lr,
                          weight_decay=cfg_t.weight_decay)
        else:
            mezo_update(model, seed_used, rho, cfg_t)
        losses.append(float(loss_plus))
        rhos.append(float(rho))
        eps_log.append(eps_t)
        beta_log.append(nesterov.beta if nesterov is not None else 0.0)
        if (step + 1) % args.eval_every == 0:
            eval_L = _eval_loss(model, eval_batch)
            eval_losses.append(eval_L)
            eval_steps.append(step + 1)
            beta_str = f" β={nesterov.beta:.3f}" if nesterov is not None else ""
            logger.info(
                f"{args.variant}|{schedule_name:<24s} step={step + 1:4d}  "
                f"eps={eps_t:.2e}{beta_str}  rho={rho:+.2f}  eval={eval_L:.4f}"
            )

    wall = time.time() - t0
    del model
    if nesterov is not None:
        del nesterov
    torch.cuda.empty_cache()
    return {
        "schedule": schedule_name, "variant": args.variant,
        "eps_start": float(eps_schedule[0]), "eps_end": float(eps_schedule[-1]),
        "lr": float(args.lr), "rho_clip": rho_clip,
        "beta_start": args.beta_start if args.variant == "dmezo_n" else None,
        "beta_end": args.beta_end if args.variant == "dmezo_n" else None,
        "steps_completed": len(losses) - 1, "diverged": bool(diverged),
        "losses": losses, "rhos": rhos,
        "eval_losses": eval_losses, "eval_steps": eval_steps,
        "eps_log": eps_log, "beta_log": beta_log,
        "wall_clock_s": float(wall),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"eps_schedule_{args.variant}_{model_short}.json"
    fig_path = fig_dir / f"fig15_eps_schedule_{args.variant}_{model_short}.png"

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

    runs: dict[str, dict] = {}
    for sched_name in args.schedules:
        eps_arr = _build_eps_schedule(sched_name, args.steps)
        logger.info(
            f"=== {args.variant} | {sched_name}: "
            f"ε[0]={eps_arr[0]:.2e} → ε[N-1]={eps_arr[-1]:.2e} ==="
        )
        runs[sched_name] = _train_one_schedule(
            args=args, dtype=dtype, schedule_name=sched_name,
            eps_schedule=eps_arr, batches=batches, z_seeds=z_seeds,
        )

    out = {
        "model": args.model, "variant": args.variant, "dtype": args.dtype,
        "task": args.task, "batch_size": args.batch_size, "max_length": args.max_length,
        "lr": args.lr, "steps": args.steps, "seed": args.seed,
        "rho_clip": args.rho_clip if args.variant == "dmezo_n" else None,
        "beta_start": args.beta_start if args.variant == "dmezo_n" else None,
        "beta_end": args.beta_end if args.variant == "dmezo_n" else None,
        "schedule_defs": {k: {"start": v[0], "end": v[1]} for k, v in SCHEDULES.items()},
        "runs": runs,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Plot.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
            "lines.linewidth": 1.6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_sched, ax_eval = axes
    cmap = plt.get_cmap("tab10")
    for i, (name, r) in enumerate(runs.items()):
        col = cmap(i % 10)
        steps_axis = np.arange(len(r["eps_log"]))
        ax_sched.semilogy(steps_axis, r["eps_log"], "-", color=col, label=name, alpha=0.9)
        evals = np.array(r["eval_losses"])
        eval_steps = np.array(r["eval_steps"])
        L0, Lf = evals[0], evals[-1]
        drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
        tag = " (DIVERGED)" if r["diverged"] else ""
        ax_eval.plot(eval_steps, evals, "o-", color=col, markersize=4.5,
                     label=f"{name}  ({drop:+.1f}%){tag}", alpha=0.9)

    ax_sched.set_xlabel("MeZO step"); ax_sched.set_ylabel(r"$\varepsilon(t)$  (log)")
    ax_sched.set_title("(a) ε(t) schedules")
    ax_sched.legend(loc="lower left", fontsize=8)
    ax_eval.set_xlabel("MeZO step"); ax_eval.set_ylabel(r"eval loss at clean $\theta$")
    ax_eval.set_title("(b) Convergence at clean θ")
    ax_eval.legend(loc="upper right", fontsize=8)

    var_tag = "vanilla MeZO" if args.variant == "vanilla" else f"D-MeZO-N (clip={args.rho_clip}, β:{args.beta_start}→{args.beta_end})"
    fig.suptitle(
        f"ε-schedule ablation — {var_tag}  on {args.model}  "
        f"(lr={args.lr}, B={args.batch_size}, fp16)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    logger.info("=" * 88)
    logger.info(f"{args.variant} on {args.model} — eval@clean-θ:")
    for name, r in runs.items():
        evals = r["eval_losses"]
        n = r["steps_completed"]
        if r["diverged"]:
            tag = f"DIVERGED@step{n}"
        else:
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            tag = f"L0={L0:.3f} → L_f={Lf:.3f}  (drop={drop:+.1f}%)"
        logger.info(f"  {name:<28s}  steps={n:>3d}/{args.steps}  {tag}")
    logger.info("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
