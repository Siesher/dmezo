"""DP-MeZO sigma sweep: privacy/utility frontier for D-MeZO-N + Gaussian noise.

Maps the (eps_dp, delta)-DP privacy budget to model utility (final loss / accuracy) by
running D-MeZO-N with multiple Gaussian noise scales sigma added to the clipped
projected gradient. Each sigma gives a different eps_dp via:

    eps_dp = C * sqrt(2 ln(1.25/delta)) / sigma        (Dwork-Roth 2014, Gaussian mechanism)

where C = ρ-clip threshold = L2 sensitivity bound, delta is the failure probability.

Default sweep on our standard setup (C=50, delta=10^-3):

    sigma=0.5   -> eps_dp=378   (no privacy, baseline)
    sigma=2.0   -> eps_dp=94    (trivial)
    sigma=5.0   -> eps_dp=38    (trivial)
    sigma=10.0  -> eps_dp=19    (weak)
    sigma=18.9  -> eps_dp=10    (medium — paper claim threshold)
    sigma=50.0  -> eps_dp=4     (medium-strong)

A "publishable" DP claim requires reaching eps_dp ≤ 10 (sigma ≥ 19) with utility loss
within ~10% of the no-DP baseline. This script measures that empirically.

Outputs:
    experiments/diagnostics/sweep_dp_sigma_<model>_<task>.json
    docs/figures/fig_sweep_dp_sigma_trajectories_<model>_<task>.png
    docs/figures/fig_sweep_dp_sigma_frontier_<model>_<task>.png

Usage:
    .venv\\Scripts\\python scripts\\sweep_dp_sigma.py `
        --model Qwen/Qwen3.5-0.8B `
        --task mathlogicqa `
        --seeds 42 43 `
        --sigmas 0.5 2 5 10 19 50 `
        --num-rounds 200

Compute estimate (Qwen3.5-0.8B local Blackwell w/ fla):
    - 6 sigma values + 2 baselines = 8 variants
    - x 2 seeds x 200 rounds = 16 cells
    - x ~3.5 min/cell ≈ ~55 min total
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
    build_partitioned_loaders,
    causal_lm_loss,
    evaluate_classification_accuracy,
)
from dmezo.federated.client import ClientState  # noqa: E402
from dmezo.federated.simulator import SimulatorConfig, run_simulation  # noqa: E402
from dmezo.federated.topology import complete_graph  # noqa: E402
from dmezo.mezo.nesterov import NesterovState  # noqa: E402
from dmezo.mezo.step import MeZOConfig, dp_epsilon_from_sigma  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.dp_sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-0.8B",
        help="Default Qwen3.5-0.8B fits on local Blackwell w/ fla.",
    )
    p.add_argument(
        "--task", type=str, default="mathlogicqa",
        choices=["sst2", "boolq", "mathlogicqa", "hellaswag"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--sigmas", type=float, nargs="+",
        default=[0.5, 2.0, 5.0, 10.0, 19.0, 50.0],
        help="Gaussian noise sigma values to sweep. eps_dp = C*sqrt(2 ln(1.25/delta))/sigma.",
    )
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--num-train-examples", type=int, default=500)
    p.add_argument("--num-eval-examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--rho-clip", type=float, default=50.0,
                   help="Fixed clip threshold (= L2 sensitivity bound for DP)")
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    p.add_argument("--dp-delta", type=float, default=1e-3,
                   help="DP delta for eps_dp accounting in summary table.")
    p.add_argument(
        "--include-baselines", action="store_true", default=True,
        help="Also run vanilla + dmezo_n no-DP for reference.",
    )
    return p.parse_args()


@torch.inference_mode()
def _eval_loss(model, dataloader, max_batches: int = 20) -> float:
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        loss = causal_lm_loss(model, batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _build_eval_loader(args, tokenizer):
    if args.task == "mathlogicqa":
        from dmezo.data.mathlogicqa import build_mathlogicqa_loader
        return build_mathlogicqa_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    if args.task == "hellaswag":
        from dmezo.data.hellaswag import build_hellaswag_loader
        return build_hellaswag_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    from dmezo.data.superglue import build_loader_for_task
    return build_loader_for_task(
        args.task, tokenizer=tokenizer, split="validation",
        batch_size=args.batch_size, max_length=args.max_length, shuffle=False,
        num_examples=args.num_eval_examples, seed=0,
    )


def _run_cell(args, *, variant: str, seed: int, dtype, sigma: float | None = None) -> dict:
    """One training cell. variant in {'vanilla', 'dmezo_n', 'dp'}.

    For variant='dp', sigma must be provided.
    """
    torch.manual_seed(seed)
    label = variant if variant != "dp" else f"dp_sigma={sigma:.2f}"
    logger.info(f"  Building {args.n_clients} clients for {label}|seed={seed}...")
    models = []
    tokenizer = None
    for ci in range(args.n_clients):
        m, tok = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
        m.eval()
        if sum(1 for p in m.parameters() if not p.requires_grad) == 0:
            for p in m.parameters():
                p.requires_grad_(True)
        models.append(m)
        tokenizer = tok

    client_loaders = build_partitioned_loaders(
        task=args.task, tokenizer=tokenizer, n_clients=args.n_clients,
        partition_mode="iid", batch_size=args.batch_size,
        max_length=args.max_length, num_examples=args.num_train_examples,
        shuffle=True, seed=seed,
    )
    eval_loader = _build_eval_loader(args, tokenizer)

    base_kwargs = dict(lr=args.lr, eps=args.eps)
    if variant == "vanilla":
        cfg = MeZOConfig(**base_kwargs, rho_clip=None)
        nesterov_factory = lambda: None  # noqa: E731
    elif variant == "dmezo_n":
        cfg = MeZOConfig(**base_kwargs, rho_clip=args.rho_clip)
        nesterov_factory = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
        )
    elif variant == "dp":
        cfg = MeZOConfig(**base_kwargs, rho_clip=args.rho_clip, dp_sigma=float(sigma))
        nesterov_factory = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    clients = [
        ClientState(
            client_id=ci, model=models[ci], dataloader=client_loaders[ci],
            mezo_config=cfg, local_steps=1,
            nesterov_state=nesterov_factory(),
            rng=np.random.default_rng(seed + ci),
        )
        for ci in range(args.n_clients)
    ]
    topology = complete_graph(args.n_clients)

    L0 = _eval_loss(clients[0].model, eval_loader, max_batches=args.eval_batches)
    A0 = evaluate_classification_accuracy(
        clients[0].model, eval_loader, task=args.task, max_batches=args.eval_batches
    )
    eval_steps = [0]; eval_losses = [L0]; eval_accs = [A0]
    logger.info(f"  init: eval_loss={L0:.4f}  acc={A0:.4f}")
    t0 = time.time()

    def eval_fn(model, rnd):
        L = _eval_loss(model, eval_loader, max_batches=args.eval_batches)
        A = evaluate_classification_accuracy(
            model, eval_loader, task=args.task, max_batches=args.eval_batches
        )
        return {"loss": L, "acc": A}

    def round_logger(rl):
        r = rl["round"]
        el = rl.get("eval_loss"); ea = rl.get("eval_acc")
        if el is not None:
            eval_steps.append(r + 1)
            eval_losses.append(el)
            eval_accs.append(ea if ea is not None else float("nan"))
        if (r + 1) % args.eval_every == 0:
            logger.info(
                f"  {label}|s={seed} r={r+1:4d}/{args.num_rounds} "
                f"loss+={rl.get('mean_local_loss', 0):.3f} "
                f"rho={rl.get('mean_projected_grad', 0):+.2f} "
                f"eval={el:.4f} acc={ea:.4f}"
            )

    sim_cfg = SimulatorConfig(
        num_rounds=args.num_rounds, consensus_mode="weight_avg",
        eval_every=args.eval_every, log_every=args.eval_every,
    )
    run_simulation(clients, topology, causal_lm_loss, sim_cfg,
                    eval_fn=eval_fn, logger=round_logger)
    wall = time.time() - t0

    dp_eps = (
        dp_epsilon_from_sigma(sigma, args.rho_clip, args.dp_delta)
        if variant == "dp" else None
    )

    for m in models:
        del m
    torch.cuda.empty_cache()

    return {
        "variant": variant, "seed": int(seed), "sigma": sigma,
        "lr": args.lr, "eps": args.eps, "rho_clip": args.rho_clip,
        "n_clients": args.n_clients,
        "eval_steps": eval_steps, "eval_losses": eval_losses, "eval_accs": eval_accs,
        "wall_clock_s": float(wall),
        "dp_epsilon": dp_eps,
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"sweep_dp_sigma_{model_short}_{args.task}.json"
    fig_traj = fig_dir / f"fig_sweep_dp_sigma_trajectories_{model_short}_{args.task}.png"
    fig_front = fig_dir / f"fig_sweep_dp_sigma_frontier_{model_short}_{args.task}.png"

    dtype = getattr(torch, args.dtype)

    cells = {}
    plan = []
    if args.include_baselines:
        plan.append(("vanilla", None))
        plan.append(("dmezo_n", None))
    for sigma in args.sigmas:
        plan.append(("dp", float(sigma)))

    total = len(plan) * len(args.seeds)
    done = 0
    for variant, sigma in plan:
        for s in args.seeds:
            done += 1
            label = variant if variant != "dp" else f"dp_sigma={sigma:.2f}"
            logger.info(f"=== [{done}/{total}] {label}|seed={s} ===")
            key = f"{label}|seed={s}"
            cells[key] = _run_cell(args, variant=variant, seed=s, dtype=dtype, sigma=sigma)

    out = {
        "model": args.model, "task": args.task, "dtype": args.dtype,
        "n_clients": args.n_clients, "num_rounds": args.num_rounds,
        "lr": args.lr, "eps": args.eps, "rho_clip": args.rho_clip,
        "beta_start": args.beta_start, "beta_end": args.beta_end,
        "dp_delta": args.dp_delta, "sigmas": args.sigmas,
        "seeds": args.seeds, "cells": cells,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # ------------------------------ Trajectory plot.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_acc = axes
    # Color scheme: baselines distinct + viridis for sigma sweep.
    cm = plt.cm.viridis
    sigma_colours = {sigma: cm(0.1 + 0.7 * (i / max(len(args.sigmas) - 1, 1)))
                     for i, sigma in enumerate(args.sigmas)}

    def _plot_variant(label_key: str, color: str, display: str, ax_l, ax_a):
        losses, accs, steps = [], [], None
        for s in args.seeds:
            r = cells.get(f"{label_key}|seed={s}")
            if r is None:
                continue
            steps = np.array(r["eval_steps"])
            losses.append(np.array(r["eval_losses"]))
            accs.append(np.array(r["eval_accs"]))
        if not losses:
            return
        loss_arr = np.stack(losses); acc_arr = np.stack(accs)
        m_l, s_l = loss_arr.mean(axis=0), loss_arr.std(axis=0)
        m_a, s_a = acc_arr.mean(axis=0), acc_arr.std(axis=0)
        ax_l.plot(steps, m_l, "-", color=color, linewidth=2.0, label=display)
        ax_l.fill_between(steps, m_l - s_l, m_l + s_l, color=color, alpha=0.15)
        ax_a.plot(steps, m_a, "-", color=color, linewidth=2.0, label=display)
        ax_a.fill_between(steps, m_a - s_a, m_a + s_a, color=color, alpha=0.15)

    if args.include_baselines:
        _plot_variant("vanilla", "#d62728", "Vanilla (no DP)", ax_loss, ax_acc)
        _plot_variant("dmezo_n", "#1f77b4", "D-MeZO-N (no DP)", ax_loss, ax_acc)
    for sigma in args.sigmas:
        eps = dp_epsilon_from_sigma(sigma, args.rho_clip, args.dp_delta)
        col = sigma_colours[sigma]
        _plot_variant(f"dp_sigma={sigma:.2f}", col,
                      f"sigma={sigma}, eps_dp={eps:.1f}", ax_loss, ax_acc)
    ax_loss.set_xlabel("MeZO round"); ax_loss.set_ylabel(f"{args.task} eval loss")
    ax_loss.set_title("(a) Loss trajectories (mean +/- 1 std)")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_acc.set_xlabel("MeZO round"); ax_acc.set_ylabel(f"{args.task} accuracy")
    ax_acc.set_title("(b) Accuracy trajectories")
    ax_acc.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"DP-MeZO sigma sweep: {args.model} / {args.task} / {len(args.seeds)} seeds, "
        f"C={args.rho_clip}, delta={args.dp_delta}",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_traj, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved trajectory figure to {fig_traj}")

    # ------------------------------ Privacy/utility frontier.
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss2, ax_acc2 = axes2
    eps_vals = []; losses_final = []; accs_final = []; sigmas_used = []
    loss_stds = []; acc_stds = []
    for sigma in args.sigmas:
        eps = dp_epsilon_from_sigma(sigma, args.rho_clip, args.dp_delta)
        lfs = [cells[f"dp_sigma={sigma:.2f}|seed={s}"]["eval_losses"][-1]
               for s in args.seeds if f"dp_sigma={sigma:.2f}|seed={s}" in cells]
        afs = [cells[f"dp_sigma={sigma:.2f}|seed={s}"]["eval_accs"][-1]
               for s in args.seeds if f"dp_sigma={sigma:.2f}|seed={s}" in cells]
        if not lfs:
            continue
        eps_vals.append(eps)
        losses_final.append(np.mean(lfs))
        loss_stds.append(np.std(lfs))
        accs_final.append(np.mean(afs))
        acc_stds.append(np.std(afs))
        sigmas_used.append(sigma)
    ax_loss2.errorbar(eps_vals, losses_final, yerr=loss_stds, marker="o",
                      capsize=4, linewidth=1.5, color="#1f77b4", label="D-MeZO-N + DP")
    if args.include_baselines:
        # Reference lines for no-DP baselines.
        vanilla_losses = [cells[f"vanilla|seed={s}"]["eval_losses"][-1]
                          for s in args.seeds if f"vanilla|seed={s}" in cells]
        dmezo_losses = [cells[f"dmezo_n|seed={s}"]["eval_losses"][-1]
                        for s in args.seeds if f"dmezo_n|seed={s}" in cells]
        if vanilla_losses:
            ax_loss2.axhline(np.mean(vanilla_losses), color="#d62728", linestyle="--",
                              label=f"Vanilla baseline ({np.mean(vanilla_losses):.3f})")
        if dmezo_losses:
            ax_loss2.axhline(np.mean(dmezo_losses), color="#2ca02c", linestyle="--",
                              label=f"D-MeZO-N no-DP ({np.mean(dmezo_losses):.3f})")
    ax_loss2.set_xscale("log")
    ax_loss2.set_xlabel("Privacy budget eps_dp (lower = stronger privacy)")
    ax_loss2.set_ylabel(f"Final {args.task} eval loss")
    ax_loss2.set_title("(a) Privacy/Utility frontier — Loss")
    ax_loss2.legend(loc="best", fontsize=9)
    ax_loss2.invert_xaxis()  # stronger privacy (smaller eps_dp) goes RIGHT
    # Annotate each point with sigma.
    for sigma, eps, loss in zip(sigmas_used, eps_vals, losses_final):
        ax_loss2.annotate(f"sigma={sigma}", (eps, loss), textcoords="offset points",
                          xytext=(5, 5), fontsize=8)

    ax_acc2.errorbar(eps_vals, accs_final, yerr=acc_stds, marker="o",
                     capsize=4, linewidth=1.5, color="#1f77b4", label="D-MeZO-N + DP")
    if args.include_baselines:
        vanilla_accs = [cells[f"vanilla|seed={s}"]["eval_accs"][-1]
                        for s in args.seeds if f"vanilla|seed={s}" in cells]
        dmezo_accs = [cells[f"dmezo_n|seed={s}"]["eval_accs"][-1]
                      for s in args.seeds if f"dmezo_n|seed={s}" in cells]
        if vanilla_accs:
            ax_acc2.axhline(np.mean(vanilla_accs), color="#d62728", linestyle="--",
                             label=f"Vanilla baseline ({np.mean(vanilla_accs):.3f})")
        if dmezo_accs:
            ax_acc2.axhline(np.mean(dmezo_accs), color="#2ca02c", linestyle="--",
                             label=f"D-MeZO-N no-DP ({np.mean(dmezo_accs):.3f})")
    ax_acc2.set_xscale("log")
    ax_acc2.set_xlabel("Privacy budget eps_dp (lower = stronger privacy)")
    ax_acc2.set_ylabel(f"Final {args.task} accuracy")
    ax_acc2.set_title("(b) Privacy/Utility frontier — Accuracy")
    ax_acc2.legend(loc="best", fontsize=9)
    ax_acc2.invert_xaxis()
    for sigma, eps, acc in zip(sigmas_used, eps_vals, accs_final):
        ax_acc2.annotate(f"sigma={sigma}", (eps, acc), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)

    fig2.suptitle(
        f"DP-MeZO Privacy/Utility frontier: {args.model} / {args.task}\n"
        f"({len(args.seeds)} seeds x {args.num_rounds} rounds, C={args.rho_clip})",
        fontsize=11, y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(fig_front, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved frontier figure to {fig_front}")

    # ------------------------------ Summary table.
    logger.info("=" * 110)
    logger.info(f"DP-MeZO sweep: {args.model} / {args.task}")
    logger.info(
        f"{'variant':<18}{'sigma':>8}{'epsilon':>10}"
        f"{'mean_loss':>12}{'std_loss':>11}{'mean_acc':>11}{'std_acc':>10}"
    )
    logger.info("-" * 110)
    if args.include_baselines:
        for name in ("vanilla", "dmezo_n"):
            lfs = [cells[f"{name}|seed={s}"]["eval_losses"][-1] for s in args.seeds
                   if f"{name}|seed={s}" in cells]
            afs = [cells[f"{name}|seed={s}"]["eval_accs"][-1] for s in args.seeds
                   if f"{name}|seed={s}" in cells]
            if lfs:
                logger.info(
                    f"{name:<18}{'-':>8}{'inf':>10}"
                    f"{np.mean(lfs):>12.4f}{np.std(lfs):>11.4f}"
                    f"{np.mean(afs):>11.4f}{np.std(afs):>10.4f}"
                )
    for sigma in args.sigmas:
        eps = dp_epsilon_from_sigma(sigma, args.rho_clip, args.dp_delta)
        lfs = [cells[f"dp_sigma={sigma:.2f}|seed={s}"]["eval_losses"][-1]
               for s in args.seeds if f"dp_sigma={sigma:.2f}|seed={s}" in cells]
        afs = [cells[f"dp_sigma={sigma:.2f}|seed={s}"]["eval_accs"][-1]
               for s in args.seeds if f"dp_sigma={sigma:.2f}|seed={s}" in cells]
        if lfs:
            logger.info(
                f"{'dp_mezo_n':<18}{sigma:>8.2f}{eps:>10.2f}"
                f"{np.mean(lfs):>12.4f}{np.std(lfs):>11.4f}"
                f"{np.mean(afs):>11.4f}{np.std(afs):>10.4f}"
            )
    logger.info("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
