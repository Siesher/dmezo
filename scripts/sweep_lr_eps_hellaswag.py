"""Joint lr × ε-schedule sweep on HellaSwag / Qwen3.5-4B-Base.

3×3 grid: lr ∈ {1e-7, 3e-7, 1e-6} × ε_schedule ∈ {const_1e-3, decay 1e-3→1e-4,
warmup 1e-2→1e-3}. Vanilla MeZO (no ρ-clip, no momentum) — focus on the
ε(t) × lr interaction without D-MeZO-N overhead.

HellaSwag = 4-way commonsense reasoning. Eval metric: train-loss (cheap) +
4-way accuracy (every 25 steps, more expensive — 4× forwards per example).

Fair comparison: shared batches + z-seeds across all (lr, ε) cells. Each cell
loads a fresh model from HF cache for clean starting state.

Outputs:
    experiments/diagnostics/sweep_lr_eps_hellaswag_{model_short}.json
    docs/figures/fig16_sweep_lr_eps_hellaswag_{model_short}.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmezo.data.hellaswag import (  # noqa: E402
    _HellaSwagDataset,
    build_hellaswag_loader,
    evaluate_hellaswag_accuracy,
)
from dmezo.data.superglue import _collate, _load_raw_dataset, causal_lm_loss  # noqa: E402
from dmezo.mezo.nesterov import NesterovState, nesterov_step  # noqa: E402
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.sweep")


SCHEDULES: dict[str, tuple[float, float]] = {
    "const_1e-3":             (1e-3, 1e-3),
    "exp_decay_1e-3_to_1e-4": (1e-3, 1e-4),
    "exp_decay_1e-2_to_1e-3": (1e-2, 1e-3),
}


def _build_eps_schedule(name: str, n_steps: int) -> np.ndarray:
    s, e = SCHEDULES[name]
    if n_steps == 1:
        return np.array([s])
    return np.exp(np.log(s) + (np.log(e) - np.log(s)) * np.linspace(0, 1, n_steps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B-Base")
    p.add_argument(
        "--variant", type=str, default="vanilla",
        choices=["vanilla", "dmezo_n"],
        help="vanilla = plain MeZO; dmezo_n = Nesterov beta-decay + rho-clip C=50",
    )
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-7, 3e-7, 1e-6])
    p.add_argument(
        "--schedules", type=str, nargs="+",
        default=["const_1e-3", "exp_decay_1e-3_to_1e-4", "exp_decay_1e-2_to_1e-3"],
        choices=list(SCHEDULES.keys()),
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=10,
                   help="HellaSwag eval batches (each scores 4 endings)")
    # D-MeZO-N hyperparameters.
    p.add_argument("--rho-clip", type=float, default=50.0,
                   help="rho-clip threshold (dmezo_n only). 0 = disable.")
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    # Output controls (useful for parallel runs on Colab).
    p.add_argument("--output-tag", type=str, default=None,
                   help="Optional suffix on JSON/figure filenames (e.g. 'seed43')")
    return p.parse_args()


def _train_loss(model, batch) -> float:
    with torch.inference_mode():
        out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _run_one_cell(
    *,
    args,
    dtype: torch.dtype,
    lr: float,
    schedule_name: str,
    eps_schedule: np.ndarray,
    batches: list[dict],
    z_seeds: list[int],
    eval_loader,
) -> dict:
    """Train ``len(eps_schedule)`` steps for one (lr, ε-schedule) combo.

    Supports ``args.variant in {'vanilla', 'dmezo_n'}``. For dmezo_n we add
    ρ-clipping (``args.rho_clip``) and Nesterov heavy-ball with linear
    β-decay (``args.beta_start`` → ``args.beta_end`` over the run horizon).
    """
    torch.manual_seed(args.seed)
    model, _ = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    rho_clip = args.rho_clip if (args.variant == "dmezo_n" and args.rho_clip > 0) else None
    base_cfg = MeZOConfig(lr=lr, eps=float(eps_schedule[0]), rho_clip=rho_clip)

    nesterov: NesterovState | None = None
    if args.variant == "dmezo_n":
        nesterov = NesterovState(
            beta=args.beta_start,
            look_ahead=False,
            beta_end=args.beta_end,
            num_rounds_total=args.steps,
        )

    eval_batch = batches[0]
    L0 = _train_loss(model, eval_batch)
    acc0 = evaluate_hellaswag_accuracy(model, eval_loader, max_batches=args.eval_batches)
    train_losses: list[float] = [L0]
    eval_losses: list[float] = [L0]
    accs: list[float] = [acc0]
    eval_steps: list[int] = [0]
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
            logger.warning(f"lr={lr:.0e}|{schedule_name} step {step}: {e!r}")
            diverged = True
            break
        if not np.isfinite(loss_plus) or not np.isfinite(rho):
            logger.warning(f"lr={lr:.0e}|{schedule_name} step {step}: NaN/inf")
            diverged = True
            break
        if nesterov is not None:
            nesterov_step(model, nesterov, seed_used, rho, cfg_t.lr,
                          weight_decay=cfg_t.weight_decay)
        else:
            mezo_update(model, seed_used, rho, cfg_t)
        train_losses.append(float(loss_plus))
        beta_log.append(nesterov.beta if nesterov is not None else 0.0)
        if (step + 1) % args.eval_every == 0:
            eL = _train_loss(model, eval_batch)
            acc = evaluate_hellaswag_accuracy(model, eval_loader, max_batches=args.eval_batches)
            eval_losses.append(eL)
            accs.append(acc)
            eval_steps.append(step + 1)
            beta_str = f" beta={nesterov.beta:.3f}" if nesterov is not None else ""
            logger.info(
                f"{args.variant}|lr={lr:.0e}|{schedule_name:<24s} step={step + 1:3d} "
                f"eps={eps_t:.2e}{beta_str} rho={rho:+.1f} eval_loss={eL:.4f} acc={acc:.3f}"
            )

    wall = time.time() - t0
    del model
    if nesterov is not None:
        del nesterov
    torch.cuda.empty_cache()
    return {
        "variant": args.variant,
        "lr": float(lr), "schedule": schedule_name,
        "eps_start": float(eps_schedule[0]), "eps_end": float(eps_schedule[-1]),
        "rho_clip": rho_clip,
        "beta_start": args.beta_start if args.variant == "dmezo_n" else None,
        "beta_end": args.beta_end if args.variant == "dmezo_n" else None,
        "steps_completed": len(train_losses) - 1,
        "diverged": bool(diverged),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "eval_accs": accs,
        "eval_steps": eval_steps,
        "beta_log": beta_log,
        "wall_clock_s": float(wall),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    json_path = out_dir / f"sweep_lr_eps_hellaswag_{args.variant}_{model_short}{suffix}.json"
    fig_path = fig_dir / f"fig16_sweep_lr_eps_hellaswag_{args.variant}_{model_short}{suffix}.png"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]

    # Pre-build shared train batches + z-seeds.
    logger.info("Pre-building shared data + z-seed schedule (HellaSwag)...")
    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_train = _load_raw_dataset("hellaswag", "train")
    ds_train = _HellaSwagDataset(raw_train, tokenizer, max_length=args.max_length)
    rng_data = np.random.default_rng(args.seed)
    batches = [
        _collate(
            [ds_train[i] for i in rng_data.choice(len(ds_train), size=args.batch_size, replace=False).tolist()],
            pad_token_id=tokenizer.pad_token_id,
        )
        for _ in range(args.steps)
    ]
    rng_z = np.random.default_rng(args.seed + 1)
    z_seeds = [int(rng_z.integers(0, 2**31 - 1)) for _ in range(args.steps)]
    eval_loader = build_hellaswag_loader(
        tokenizer, split="validation", batch_size=args.batch_size,
        max_length=args.max_length, shuffle=False, num_examples=200, seed=args.seed,
    )
    logger.info(f"  train: {len(batches)} batches; eval: {args.eval_batches} batches × 4 endings")

    # ---- Run lrs × schedules grid.
    cells: dict[str, dict] = {}
    grid_total = len(args.lrs) * len(args.schedules)
    grid_done = 0
    for lr, sched in product(args.lrs, args.schedules):
        grid_done += 1
        key = f"lr={lr:.0e}|sched={sched}"
        eps_arr = _build_eps_schedule(sched, args.steps)
        logger.info(
            f"=== [{grid_done}/{grid_total}] variant={args.variant}, lr={lr:.0e}, sched={sched} "
            f"(ε[0]={eps_arr[0]:.2e}→ε[N-1]={eps_arr[-1]:.2e}) ==="
        )
        cells[key] = _run_one_cell(
            args=args, dtype=dtype, lr=lr, schedule_name=sched,
            eps_schedule=eps_arr, batches=batches, z_seeds=z_seeds,
            eval_loader=eval_loader,
        )

    out = {
        "model": args.model, "variant": args.variant, "dtype": args.dtype,
        "task": "hellaswag",
        "batch_size": args.batch_size, "max_length": args.max_length,
        "steps": args.steps, "seed": args.seed,
        "lrs": args.lrs, "schedules": args.schedules,
        "rho_clip": args.rho_clip if args.variant == "dmezo_n" else None,
        "beta_start": args.beta_start if args.variant == "dmezo_n" else None,
        "beta_end": args.beta_end if args.variant == "dmezo_n" else None,
        "schedule_defs": {k: {"start": v[0], "end": v[1]} for k, v in SCHEDULES.items()},
        "cells": cells,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Composite figure: 2 rows × len(lrs) cols.
    # Row 0 = train loss, Row 1 = HellaSwag accuracy.
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
         "lines.linewidth": 1.5}
    )
    sched_colours = {
        "const_1e-3":             "#1f77b4",
        "exp_decay_1e-3_to_1e-4": "#2ca02c",
        "exp_decay_1e-2_to_1e-3": "#d62728",
    }
    ncols = len(args.lrs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8), sharex=True)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col_idx, lr in enumerate(args.lrs):
        ax_loss = axes[0, col_idx]
        ax_acc = axes[1, col_idx]
        for sched in args.schedules:
            key = f"lr={lr:.0e}|sched={sched}"
            if key not in cells:
                continue
            r = cells[key]
            col = sched_colours.get(sched, "#7f7f7f")
            steps = np.array(r["eval_steps"])
            losses = np.array(r["eval_losses"])
            accs = np.array(r["eval_accs"])
            L0, Lf = losses[0], losses[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            acc0, accf = accs[0], accs[-1]
            tag_loss = f"{sched}  (Δ={drop:+.1f}%)"
            tag_acc = f"{sched}  (acc {acc0:.2f}→{accf:.2f})"
            ax_loss.plot(steps, losses, "o-", color=col, markersize=5, label=tag_loss)
            ax_acc.plot(steps, accs, "o-", color=col, markersize=5, label=tag_acc)

        ax_loss.set_title(f"lr = {lr:.0e}", fontsize=10)
        if col_idx == 0:
            ax_loss.set_ylabel("HellaSwag eval loss")
            ax_acc.set_ylabel("HellaSwag 4-way acc")
        ax_acc.set_xlabel("MeZO step")
        ax_acc.axhline(0.25, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_loss.legend(loc="upper right", fontsize=8)
        ax_acc.legend(loc="lower right", fontsize=8)

    var_tag = (
        "vanilla MeZO"
        if args.variant == "vanilla"
        else f"D-MeZO-N (clip={args.rho_clip}, beta:{args.beta_start}->{args.beta_end})"
    )
    fig.suptitle(
        f"lr × ε-schedule sweep — {var_tag} on {args.model} / HellaSwag "
        f"(B={args.batch_size}, {args.dtype}, {args.steps} steps)",
        fontsize=11, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary.
    logger.info("=" * 96)
    logger.info(f"lr × ε-schedule sweep on {args.model} / HellaSwag:")
    logger.info(f"  {'lr':<10}{'schedule':<28}{'L_init':>9}{'L_final':>9}{'Δ%':>7}{'acc_init':>10}{'acc_final':>11}")
    for key, r in cells.items():
        evals = r["eval_losses"]
        accs = r["eval_accs"]
        n = r["steps_completed"]
        if r["diverged"]:
            tag = f"DIVERGED@step{n}"
            logger.info(f"  {r['lr']:<10.0e}{r['schedule']:<28}  {tag}")
        else:
            L0, Lf = evals[0], evals[-1]
            drop = (L0 - Lf) / L0 * 100 if L0 != 0 else 0.0
            a0, af = accs[0], accs[-1]
            logger.info(
                f"  {r['lr']:<10.0e}{r['schedule']:<28}{L0:9.3f}{Lf:9.3f}"
                f"{drop:>+6.1f}%{a0:>10.3f}{af:>11.3f}"
            )
    logger.info("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
