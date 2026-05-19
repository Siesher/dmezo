"""Federated multi-seed validation of §5.5 D-MeZO-N rescue claim.

§5.5 single-seed result was on the FEDERATED setup (4c complete IID), not
centralized. ``validate_dmezo_n_rescue_multiseed.py`` accidentally tested
the centralized version — that's a useful ablation (momentum+clip alone
insufficient), but doesn't validate the actual §5.5 claim.

This script re-runs the EXACT §5.5 setup with 3 seeds:
    Model:        Qwen/Qwen3-4B (standard transformer)
    Task:         HellaSwag (4-way commonsense reasoning)
    Federation:   n=4 clients, complete topology, IID partition, weight_avg
    Rounds:       1000
    lr:           3e-7
    eps:          1e-3
    D-MeZO-N v1:  beta-decay 0.9 -> 0, rho-clip C=50
    Variants:     vanilla (no momentum, no clip), dmezo_n (full recipe)
    Seeds:        42, 43, 44
    Eval:         every 100 rounds on 500-example HellaSwag validation pool

Outputs:
    experiments/diagnostics/validate_multiseed_fed_qwen3-4b_hellaswag.json
    docs/figures/fig19b_multiseed_federated_validation.png

Compute: ~5-6 hours on Blackwell (4 clients * 1000 rounds * 2 fwd =
8000 forwards per cell, + eval = ~50 min/cell, 6 cells total).
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

from dmezo.data.hellaswag import build_hellaswag_loader, evaluate_hellaswag_accuracy  # noqa: E402
from dmezo.data.mathlogicqa import build_mathlogicqa_loader  # noqa: E402
from dmezo.data.superglue import (  # noqa: E402
    build_partitioned_loaders,
    causal_lm_loss,
    evaluate_classification_accuracy,
)
from dmezo.federated.client import ClientState  # noqa: E402
from dmezo.federated.simulator import SimulatorConfig, run_simulation  # noqa: E402
from dmezo.federated.topology import complete_graph  # noqa: E402
from dmezo.mezo.nesterov import NesterovState  # noqa: E402
from dmezo.mezo.step import MeZOConfig  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.validate_fed_multiseed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--task", type=str, default="hellaswag",
                   choices=["hellaswag", "mathlogicqa"],
                   help="hellaswag (default): English commonsense; "
                        "mathlogicqa: Russian symbolic logic (MERA)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--variants", type=str, nargs="+", default=["vanilla", "dmezo_n"],
                   choices=["vanilla", "dmezo_n"])
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--num-train-examples", type=int, default=2000)
    p.add_argument("--num-eval-examples", type=int, default=500,
                   help="500 examples -> SE_acc ~= 0.02 at p=0.7")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=125)
    # D-MeZO-N hyperparameters.
    p.add_argument("--rho-clip", type=float, default=50.0)
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    return p.parse_args()


@torch.inference_mode()
def _eval_loss(model, dataloader, max_batches: int = 20) -> float:
    """Mean cross-entropy loss over up to ``max_batches`` of the eval loader."""
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        loss = causal_lm_loss(model, batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _run_one_cell(*, args, variant: str, seed: int, dtype):
    """Run one federated (variant, seed) cell — 4 clients × 1000 rounds."""
    torch.manual_seed(seed)

    # Build 4 fresh client models — load_causal_lm has no random init for
    # pretrained, so they're identical at start.
    logger.info(f"  loading {args.n_clients} client model copies of {args.model}...")
    models = []
    tokenizer = None
    for ci in range(args.n_clients):
        m, tok = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
        m.eval()
        # Respect loader-set requires_grad (V-L vision branch may be frozen).
        n_pre_frozen = sum(1 for p in m.parameters() if not p.requires_grad)
        if n_pre_frozen == 0:
            for p in m.parameters():
                p.requires_grad_(True)
        models.append(m)
        tokenizer = tok

    # Data: IID partitioned train + eval pool. Task-specific eval loader builder.
    client_loaders = build_partitioned_loaders(
        task=args.task, tokenizer=tokenizer, n_clients=args.n_clients,
        partition_mode="iid", batch_size=args.batch_size, max_length=args.max_length,
        num_examples=args.num_train_examples, shuffle=True, seed=seed,
    )
    if args.task == "hellaswag":
        eval_loader = build_hellaswag_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    elif args.task == "mathlogicqa":
        eval_loader = build_mathlogicqa_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    else:
        raise ValueError(f"Unknown task {args.task!r}")

    # MeZO config (shared across clients).
    rho_clip = args.rho_clip if (variant == "dmezo_n" and args.rho_clip > 0) else None
    mezo_cfg = MeZOConfig(lr=args.lr, eps=args.eps, rho_clip=rho_clip)

    # Per-client Nesterov state (None for vanilla).
    clients = []
    for ci in range(args.n_clients):
        ns = None
        if variant == "dmezo_n":
            ns = NesterovState(
                beta=args.beta_start, look_ahead=False,
                beta_end=args.beta_end, num_rounds_total=args.num_rounds,
            )
        clients.append(ClientState(
            client_id=ci, model=models[ci], dataloader=client_loaders[ci],
            mezo_config=mezo_cfg, local_steps=1, nesterov_state=ns,
            rng=np.random.default_rng(seed + ci),
        ))

    # Topology: complete graph, weight_avg consensus (required for Nesterov).
    topology = complete_graph(args.n_clients)

    # Eval at t=0.
    eval_steps = [0]
    eval_losses = [_eval_loss(clients[0].model, eval_loader, max_batches=args.eval_batches)]
    eval_accs = [evaluate_classification_accuracy(clients[0].model, eval_loader,
                                                       task=args.task,
                                                       max_batches=args.eval_batches)]
    logger.info(f"  init: eval_loss={eval_losses[0]:.4f}  acc={eval_accs[0]:.4f}")

    t0 = time.time()

    def eval_fn(model, rnd):
        ev = _eval_loss(model, eval_loader, max_batches=args.eval_batches)
        acc = evaluate_classification_accuracy(model, eval_loader, task=args.task,
                                                max_batches=args.eval_batches)
        return {"loss": ev, "acc": acc}

    def round_logger(round_log):
        r = round_log["round"]
        eval_loss = round_log.get("eval_loss")
        eval_acc = round_log.get("eval_acc")
        if eval_loss is not None:
            eval_steps.append(r + 1)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc if eval_acc is not None else float("nan"))
        if (r + 1) % args.eval_every == 0:
            beta_str = ""
            if clients[0].nesterov_state is not None:
                beta_str = f" beta={clients[0].nesterov_state.beta:.3f}"
            logger.info(
                f"  {variant}|seed={seed} round={r + 1:4d}/{args.num_rounds}{beta_str} "
                f"loss+={round_log.get('mean_local_loss', 0):.3f} "
                f"rho={round_log.get('mean_projected_grad', 0):+.2f} "
                f"eval_loss={eval_loss:.4f} acc={eval_acc:.4f}"
            )

    sim_cfg = SimulatorConfig(
        num_rounds=args.num_rounds, consensus_mode="weight_avg",
        eval_every=args.eval_every, log_every=args.eval_every,
    )
    run_simulation(clients, topology, causal_lm_loss, sim_cfg, eval_fn=eval_fn, logger=round_logger)

    wall = time.time() - t0
    for m in models:
        del m
    torch.cuda.empty_cache()
    return {
        "variant": variant, "seed": int(seed),
        "lr": args.lr, "eps": args.eps,
        "n_clients": args.n_clients, "consensus": "weight_avg",
        "topology": "complete", "partition": "iid",
        "eval_steps": eval_steps, "eval_losses": eval_losses, "eval_accs": eval_accs,
        "wall_clock_s": float(wall),
    }


def _bootstrap_ci(differences, n_boot=10000, ci=0.95):
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
    json_path = out_dir / f"validate_multiseed_fed_{model_short}_{args.task}.json"
    fig_path = fig_dir / f"fig19b_multiseed_federated_{model_short}_{args.task}.png"

    dtype = getattr(torch, args.dtype)

    cells = {}
    grid_total = len(args.seeds) * len(args.variants)
    done = 0
    for s in args.seeds:
        for variant in args.variants:
            done += 1
            key = f"{variant}|seed={s}"
            logger.info(f"=== [{done}/{grid_total}] {key} ===")
            cells[key] = _run_one_cell(args=args, variant=variant, seed=s, dtype=dtype)

    # ---- Paired analysis.
    paired = []
    for s in args.seeds:
        v = cells.get(f"vanilla|seed={s}")
        d = cells.get(f"dmezo_n|seed={s}")
        if v is None or d is None:
            continue
        delta_acc = d["eval_accs"][-1] - v["eval_accs"][-1]
        delta_loss = v["eval_losses"][-1] - d["eval_losses"][-1]
        paired.append({"seed": s, "delta_acc": delta_acc, "delta_loss": delta_loss,
                       "acc_v": v["eval_accs"][-1], "acc_d": d["eval_accs"][-1],
                       "L_v": v["eval_losses"][-1], "L_d": d["eval_losses"][-1]})

    mean_delta_acc = float(np.mean([p["delta_acc"] for p in paired])) if paired else float("nan")
    ci_lo, ci_hi = _bootstrap_ci([p["delta_acc"] for p in paired])

    out = {
        "model": args.model, "task": args.task, "dtype": args.dtype,
        "n_clients": args.n_clients, "consensus": "weight_avg", "topology": "complete",
        "partition": "iid", "num_rounds": args.num_rounds, "lr": args.lr, "eps": args.eps,
        "seeds": args.seeds, "variants": args.variants,
        "eval_examples": args.num_eval_examples,
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

    # ---- Figure.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False,
                          "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax_loss, ax_acc = axes
    colours = {"vanilla": "#d62728", "dmezo_n": "#1f77b4"}
    seed_styles = ["-", "--", ":"]
    for variant in args.variants:
        loss_curves, acc_curves, all_steps = [], [], None
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
        loss_arr = np.stack(loss_curves)
        acc_arr = np.stack(acc_curves)
        m_loss, s_loss = loss_arr.mean(axis=0), loss_arr.std(axis=0)
        m_acc, s_acc = acc_arr.mean(axis=0), acc_arr.std(axis=0)
        col = colours[variant]
        ax_loss.plot(all_steps, m_loss, "-", color=col, linewidth=2.0,
                     label=f"{variant} (mean of {len(loss_curves)} seeds)")
        ax_loss.fill_between(all_steps, m_loss - s_loss, m_loss + s_loss, color=col, alpha=0.2)
        for i, lc in enumerate(loss_curves):
            ax_loss.plot(all_steps, lc, color=col, linewidth=0.7, alpha=0.5,
                         linestyle=seed_styles[i % 3])
        ax_acc.plot(all_steps, m_acc, "-", color=col, linewidth=2.0, label=f"{variant} (mean)")
        ax_acc.fill_between(all_steps, m_acc - s_acc, m_acc + s_acc, color=col, alpha=0.2)
        for i, ac in enumerate(acc_curves):
            ax_acc.plot(all_steps, ac, color=col, linewidth=0.7, alpha=0.5,
                        linestyle=seed_styles[i % 3])
    ax_loss.set_xlabel("MeZO round"); ax_loss.set_ylabel(f"{args.task} eval loss")
    ax_loss.set_title("(a) Loss trajectories (mean ± 1 std)")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_acc.set_xlabel("MeZO round"); ax_acc.set_ylabel(f"{args.task} 4-way accuracy")
    ax_acc.set_title("(b) Accuracy trajectories")
    ax_acc.legend(loc="lower right", fontsize=9)
    if paired:
        annotation = (
            f"Mean Δacc (D-MeZO-N − vanilla, paired by seed) = {mean_delta_acc:+.4f}\n"
            f"95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]\n"
            f"n = {len(paired)} seeds  |  {args.n_clients} clients × complete × IID"
        )
        ax_acc.text(0.02, 0.98, annotation, transform=ax_acc.transAxes, fontsize=9,
                    va="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    task_label = {"hellaswag": "HellaSwag", "mathlogicqa": "MathLogicQA (RU)"}[args.task]
    fig.suptitle(
        f"Federated multi-seed validation: D-MeZO-N on {args.model} / {task_label}\n"
        f"({len(args.seeds)} seeds × {args.num_rounds} rounds, n={args.n_clients} clients complete IID, "
        f"lr={args.lr}, ε={args.eps}, eval {args.num_eval_examples}-example pool)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # ---- Summary.
    logger.info("=" * 88)
    logger.info(f"Federated multi-seed validation: D-MeZO-N on {args.model} / {args.task}")
    logger.info(f"{'seed':<6}{'L_van':>9}{'L_d':>9}{'acc_van':>10}{'acc_d':>9}{'Δ_acc':>10}")
    for p in paired:
        logger.info(
            f"{p['seed']:<6}{p['L_v']:>9.4f}{p['L_d']:>9.4f}"
            f"{p['acc_v']:>10.4f}{p['acc_d']:>9.4f}{p['delta_acc']:>+10.4f}"
        )
    logger.info(f"  mean Δacc = {mean_delta_acc:+.4f}  95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    verdict = (
        "YES -- FEDERATED RESCUE CONFIRMED" if ci_lo > 0
        else "NO -- INSIDE NOISE BAND" if ci_hi > 0 and ci_lo < 0
        else "YES -- D-MeZO-N WORSE"
    )
    logger.info(f"  CI excludes zero: {verdict}")
    logger.info("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
