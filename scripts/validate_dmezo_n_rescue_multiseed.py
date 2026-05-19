"""Multi-seed validation of §5.5 D-MeZO-N rescue claim.

§5.5 reported D-MeZO-N rescue on Qwen3-4B / HellaSwag (vanilla diverges,
D-MeZO-N converges, +3.75 pp acc). The original run was SINGLE SEED. This
script re-validates with 3 seeds × 2 variants, 1000 steps, 500-example eval
— to establish 95% CI on Δacc(D-MeZO-N − vanilla) and either:

(a) confirm rescue effect statistically (CI excludes 0)
(b) downgrade C4 contribution if CI includes 0

Design (matches §5.5 conditions as closely as possible):
    Model:     Qwen/Qwen3-4B (standard transformer, NOT Qwen3.5 hybrid)
    Task:      HellaSwag (4-way commonsense reasoning)
    Steps:     1000
    lr:        3e-7 (canonical Qwen3-4B lr per project memory)
    Schedule:  const eps=1e-3
    D-MeZO-N:  rho_clip C=50, beta-decay 0.9 -> 0
    Variants:  vanilla, dmezo_n
    Seeds:     42, 43, 44
    Eval:      every 100 steps on 500 HellaSwag validation examples
               (SE on acc ~= sqrt(p(1-p)/500) ~= 0.02 for p=0.7)

Total: 6 cells × ~25 min on Blackwell ~= 2.5 hours.

Outputs:
    experiments/diagnostics/validate_multiseed_qwen3-4b_hellaswag.json
    docs/figures/fig19_multiseed_validation.png (paired-seed accuracy with CIs)

Usage::

    python scripts/validate_dmezo_n_rescue_multiseed.py \\
        --seeds 42 43 44 --steps 1000 --eval-every 100 --eval-batches 125

(eval-batches=125 × batch-size=4 = 500 examples; HellaSwag scoring does
4 forwards per example -> ~2000 forwards per eval, manageable on Blackwell)
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
logger = logging.getLogger("dmezo.validate_multiseed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--variants", type=str, nargs="+", default=["vanilla", "dmezo_n"],
                   choices=["vanilla", "dmezo_n"])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=125,
                   help="125*B=4 = 500 examples; SE_acc~=0.02 at p=0.7")
    # D-MeZO-N hyperparameters.
    p.add_argument("--rho-clip", type=float, default=50.0)
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    return p.parse_args()


def _train_loss(model, batch) -> float:
    with torch.inference_mode():
        out = causal_lm_loss(model, batch)
    return float(out.detach().float().item())


def _run_one_cell(*, args, variant: str, seed: int, batches, z_seeds, eval_loader):
    torch.manual_seed(seed)
    model, _ = load_causal_lm(args.model, dtype=getattr(torch, args.dtype),
                              use_flash_attention=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    rho_clip = args.rho_clip if (variant == "dmezo_n" and args.rho_clip > 0) else None
    cfg = MeZOConfig(lr=args.lr, eps=args.eps, rho_clip=rho_clip)

    nesterov = None
    if variant == "dmezo_n":
        nesterov = NesterovState(beta=args.beta_start, look_ahead=False,
                                 beta_end=args.beta_end,
                                 num_rounds_total=args.steps)

    eval_batch = batches[0]
    L0 = _train_loss(model, eval_batch)
    acc0 = evaluate_hellaswag_accuracy(model, eval_loader, max_batches=args.eval_batches)
    eval_losses = [L0]; accs = [acc0]; eval_steps = [0]
    diverged = False
    t0 = time.time()

    for step in range(args.steps):
        if nesterov is not None:
            nesterov.update_schedule(step)
        one_shot = np.random.Generator(np.random.PCG64(z_seeds[step]))
        try:
            seed_used, rho, loss_plus = mezo_step(
                model, batches[step], causal_lm_loss, cfg, rng=one_shot,
            )
        except RuntimeError as e:
            logger.warning(f"{variant}|seed={seed} step {step}: {e!r}")
            diverged = True
            break
        if not np.isfinite(loss_plus) or not np.isfinite(rho):
            logger.warning(f"{variant}|seed={seed} step {step}: NaN/inf")
            diverged = True
            break
        if nesterov is not None:
            nesterov_step(model, nesterov, seed_used, rho, cfg.lr,
                          weight_decay=cfg.weight_decay)
        else:
            mezo_update(model, seed_used, rho, cfg)
        if (step + 1) % args.eval_every == 0:
            eL = _train_loss(model, eval_batch)
            acc = evaluate_hellaswag_accuracy(model, eval_loader, max_batches=args.eval_batches)
            eval_losses.append(eL); accs.append(acc); eval_steps.append(step + 1)
            beta_str = f" beta={nesterov.beta:.3f}" if nesterov is not None else ""
            logger.info(
                f"{variant}|seed={seed} step={step + 1:4d}{beta_str} rho={rho:+.1f} "
                f"eval_loss={eL:.4f} acc={acc:.4f}"
            )

    wall = time.time() - t0
    del model
    if nesterov is not None:
        del nesterov
    torch.cuda.empty_cache()
    return {
        "variant": variant, "seed": int(seed),
        "lr": args.lr, "eps": args.eps,
        "diverged": bool(diverged),
        "eval_losses": eval_losses, "eval_accs": accs, "eval_steps": eval_steps,
        "wall_clock_s": float(wall),
    }


def _bootstrap_ci(differences: list[float], n_boot: int = 10000, ci: float = 0.95):
    """Bootstrap percentile CI on the mean of paired differences."""
    if len(differences) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(differences, size=len(differences), replace=True)
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"validate_multiseed_{model_short}_hellaswag.json"
    fig_path = fig_dir / f"fig19_multiseed_validation_{model_short}.png"

    dtype = getattr(torch, args.dtype)
    logger.info("Pre-building train batches + z-seeds per seed...")

    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_train = _load_raw_dataset("hellaswag", "train")
    ds_train = _HellaSwagDataset(raw_train, tokenizer, max_length=args.max_length)

    # Per-seed batch + z schedule. Variants share the schedule within seed.
    seed_data: dict[int, dict] = {}
    for s in args.seeds:
        rng_data = np.random.default_rng(s)
        batches = [
            _collate(
                [ds_train[i] for i in rng_data.choice(len(ds_train), size=args.batch_size, replace=False).tolist()],
                pad_token_id=tokenizer.pad_token_id,
            )
            for _ in range(args.steps)
        ]
        rng_z = np.random.default_rng(s + 1)
        z_seeds = [int(rng_z.integers(0, 2**31 - 1)) for _ in range(args.steps)]
        seed_data[s] = {"batches": batches, "z_seeds": z_seeds}

    eval_loader = build_hellaswag_loader(
        tokenizer, split="validation", batch_size=args.batch_size,
        max_length=args.max_length, shuffle=False, num_examples=500, seed=0,
    )
    logger.info(f"  train: per-seed {args.steps} batches; eval: {args.eval_batches} batches x 4")

    cells: dict[str, dict] = {}
    grid_total = len(args.seeds) * len(args.variants)
    done = 0
    for s in args.seeds:
        for variant in args.variants:
            done += 1
            key = f"{variant}|seed={s}"
            logger.info(f"=== [{done}/{grid_total}] {key} ===")
            cells[key] = _run_one_cell(
                args=args, variant=variant, seed=s,
                batches=seed_data[s]["batches"], z_seeds=seed_data[s]["z_seeds"],
                eval_loader=eval_loader,
            )

    # ---- Paired-seed analysis: per seed, compute Delta_acc = acc(dmezo_n) - acc(vanilla) at the final step.
    paired = []
    for s in args.seeds:
        v = cells.get(f"vanilla|seed={s}")
        d = cells.get(f"dmezo_n|seed={s}")
        if v is None or d is None:
            continue
        delta_acc = d["eval_accs"][-1] - v["eval_accs"][-1]
        delta_loss = v["eval_losses"][-1] - d["eval_losses"][-1]  # positive = D-MeZO-N has lower loss
        paired.append({"seed": s, "delta_acc": delta_acc, "delta_loss": delta_loss,
                       "acc_v": v["eval_accs"][-1], "acc_d": d["eval_accs"][-1],
                       "L_v": v["eval_losses"][-1], "L_d": d["eval_losses"][-1]})

    mean_delta_acc = float(np.mean([p["delta_acc"] for p in paired])) if paired else float("nan")
    ci_lo, ci_hi = _bootstrap_ci([p["delta_acc"] for p in paired])

    out = {
        "model": args.model, "task": "hellaswag", "dtype": args.dtype,
        "steps": args.steps, "lr": args.lr, "eps": args.eps,
        "seeds": args.seeds, "variants": args.variants,
        "eval_batches": args.eval_batches, "eval_examples": args.eval_batches * args.batch_size,
        "rho_clip": args.rho_clip, "beta_start": args.beta_start, "beta_end": args.beta_end,
        "cells": cells,
        "paired_analysis": {
            "per_seed": paired,
            "mean_delta_acc": mean_delta_acc,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "n_seeds": len(paired),
        },
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ---- Figure: per-seed trajectories with mean curves + CI shaded.
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
         "lines.linewidth": 1.5}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax_loss, ax_acc = axes
    colours = {"vanilla": "#d62728", "dmezo_n": "#1f77b4"}
    seed_styles = ["-", "--", ":"]

    for variant in args.variants:
        all_steps = None
        loss_curves = []
        acc_curves = []
        for s in args.seeds:
            r = cells.get(f"{variant}|seed={s}")
            if r is None:
                continue
            steps = np.array(r["eval_steps"])
            all_steps = steps
            loss_curves.append(np.array(r["eval_losses"]))
            acc_curves.append(np.array(r["eval_accs"]))
        if not loss_curves:
            continue
        loss_arr = np.stack(loss_curves)  # (n_seeds, n_evals)
        acc_arr = np.stack(acc_curves)
        m_loss, s_loss = loss_arr.mean(axis=0), loss_arr.std(axis=0)
        m_acc, s_acc = acc_arr.mean(axis=0), acc_arr.std(axis=0)

        col = colours[variant]
        ax_loss.plot(all_steps, m_loss, "-", color=col, linewidth=2.0,
                     label=f"{variant} (mean of {len(loss_curves)} seeds)")
        ax_loss.fill_between(all_steps, m_loss - s_loss, m_loss + s_loss,
                             color=col, alpha=0.2)
        # per-seed thin lines
        for i, lc in enumerate(loss_curves):
            ax_loss.plot(all_steps, lc, color=col, linewidth=0.7, alpha=0.5,
                         linestyle=seed_styles[i % 3])

        ax_acc.plot(all_steps, m_acc, "-", color=col, linewidth=2.0,
                    label=f"{variant} (mean)")
        ax_acc.fill_between(all_steps, m_acc - s_acc, m_acc + s_acc,
                            color=col, alpha=0.2)
        for i, ac in enumerate(acc_curves):
            ax_acc.plot(all_steps, ac, color=col, linewidth=0.7, alpha=0.5,
                        linestyle=seed_styles[i % 3])

    ax_loss.set_xlabel("MeZO step"); ax_loss.set_ylabel("HellaSwag eval loss")
    ax_loss.set_title("(a) Loss trajectories (mean ± 1 std across seeds)")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_acc.set_xlabel("MeZO step"); ax_acc.set_ylabel("HellaSwag 4-way accuracy")
    ax_acc.set_title("(b) Accuracy trajectories")
    ax_acc.legend(loc="lower right", fontsize=9)
    # Annotate CI on the figure.
    if paired:
        annotation = (
            f"Mean Δacc (D-MeZO-N − vanilla, paired by seed) = {mean_delta_acc:+.4f}\n"
            f"95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]\n"
            f"n = {len(paired)} seeds"
        )
        ax_acc.text(0.02, 0.98, annotation, transform=ax_acc.transAxes,
                    fontsize=9, va="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.suptitle(
        f"§5.5 multi-seed validation: D-MeZO-N rescue on {args.model} / HellaSwag\n"
        f"({len(args.seeds)} seeds × {args.steps} steps, lr={args.lr}, ε={args.eps}, "
        f"eval {args.eval_batches*args.batch_size}-example pool)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary.
    logger.info("=" * 88)
    logger.info(f"Multi-seed validation §5.5: D-MeZO-N rescue on {args.model} / HellaSwag")
    logger.info(f"{'seed':<6}{'L_van':>9}{'L_d':>9}{'acc_van':>10}{'acc_d':>9}{'Δ_acc':>10}")
    for p in paired:
        logger.info(
            f"{p['seed']:<6}{p['L_v']:>9.4f}{p['L_d']:>9.4f}"
            f"{p['acc_v']:>10.4f}{p['acc_d']:>9.4f}{p['delta_acc']:>+10.4f}"
        )
    logger.info(f"  mean Δacc = {mean_delta_acc:+.4f}  95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    logger.info(f"  CI excludes zero (statistically significant): "
                f"{'YES — RESCUE CONFIRMED' if ci_lo > 0 else 'NO — INSIDE NOISE BAND' if ci_hi > 0 and ci_lo < 0 else 'YES — D-MeZO-N WORSE'}")
    logger.info("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
