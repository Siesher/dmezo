"""Head-to-head comparison: D-MeZO-N vs FedKSeed vs vanilla D-MeZO.

This is the critical empirical comparison missing from the paper as of 2026-05-20.
FedKSeed (Qin et al. 2024, ICML) is the closest published competitor to D-MeZO-N
in the federated zeroth-order LLM fine-tuning space.

FedKSeed core mechanism (paraphrased from arXiv:2312.06353):
    1. Server maintains a small SHARED seed pool S = {s^(1), ..., s^(K)}.
    2. Each round t, server samples ONE shared seed s_t and broadcasts to all clients.
    3. Each client i computes rho_i^t = MeZO projected gradient using z_{s_t} on its
       local data shard. Note: every client uses the SAME z direction.
    4. Server aggregates: rho_bar^t = (1/n) sum_i rho_i^t.
    5. Server broadcasts rho_bar^t; all clients apply IDENTICAL update:
       theta <- theta - eta * rho_bar^t * z_{s_t}.

Key differences vs D-MeZO-N:
    | Aspect              | D-MeZO-N                  | FedKSeed                  |
    |---------------------|---------------------------|---------------------------|
    | Directions / round  | n unique z_i (per client) | 1 shared z (server seed)  |
    | Variance reduction  | 1/n on sigma_z + sigma_d  | 1/n on sigma_d only       |
    | Topology            | Decentralized (W)         | Star (central server)     |
    | Momentum            | Heavy-ball + beta-decay   | None                      |
    | rho-clipping        | Yes (C=50)                | No                        |
    | Communication       | O(n) scalars (peer-peer)  | O(n) scalars (to server)  |

For paired comparison we keep all hyperparameters identical (lr, eps, n_clients,
num_rounds, data partition); only the variant logic differs.

Outputs:
    experiments/diagnostics/head_to_head_fedkseed_{model}_{task}.json
    docs/figures/fig20_head_to_head_fedkseed.png

Expected compute on Blackwell:
    ~45 min per cell (1000 rounds, 4 clients, Qwen3.5-4B)
    3 seeds * 3 variants = 9 cells -> ~6.75 hours total.
    For quick run reduce to 2 seeds (4.5 hours) or 500 rounds (3.5 hours).

Reference:
    Qin, Z., Chen, D., Qian, B., Ding, B., Li, Y., Deng, S. (2024).
    Federated Full-Parameter Tuning of Billion-Sized Language Models with
    Communication Cost under 18 Kilobytes. ICML 2024. arXiv:2312.06353.
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

from dmezo.data.hellaswag import build_hellaswag_loader  # noqa: E402
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
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.head_to_head")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B-Base")
    p.add_argument(
        "--task", type=str, default="mathlogicqa",
        choices=["hellaswag", "mathlogicqa"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--variants", type=str, nargs="+",
        default=["vanilla", "dmezo_n", "fedkseed"],
        choices=["vanilla", "dmezo_n", "fedkseed"],
    )
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--num-train-examples", type=int, default=500)
    p.add_argument("--num-eval-examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    # D-MeZO-N hyperparameters.
    p.add_argument("--rho-clip", type=float, default=50.0)
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
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
    if args.task == "hellaswag":
        return build_hellaswag_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    if args.task == "mathlogicqa":
        return build_mathlogicqa_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    raise ValueError(f"Unknown task {args.task!r}")


def _load_clients(args, seed, dtype):
    """Build n client model copies + IID data partition."""
    logger.info(f"  loading {args.n_clients} client model copies of {args.model}...")
    models = []
    tokenizer = None
    for ci in range(args.n_clients):
        m, tok = load_causal_lm(args.model, dtype=dtype, use_flash_attention=False)
        m.eval()
        n_pre_frozen = sum(1 for p in m.parameters() if not p.requires_grad)
        if n_pre_frozen == 0:
            for p in m.parameters():
                p.requires_grad_(True)
        models.append(m)
        tokenizer = tok
    client_loaders = build_partitioned_loaders(
        task=args.task, tokenizer=tokenizer, n_clients=args.n_clients,
        partition_mode="iid", batch_size=args.batch_size, max_length=args.max_length,
        num_examples=args.num_train_examples, shuffle=True, seed=seed,
    )
    eval_loader = _build_eval_loader(args, tokenizer)
    return models, client_loaders, eval_loader


def _eval_metrics(model, eval_loader, args):
    ev = _eval_loss(model, eval_loader, max_batches=args.eval_batches)
    acc = evaluate_classification_accuracy(
        model, eval_loader, task=args.task, max_batches=args.eval_batches
    )
    return ev, acc


def _run_one_cell_dmezo(*, args, variant: str, seed: int, dtype) -> dict:
    """vanilla or dmezo_n via existing simulator. Identical to validate_*.py."""
    torch.manual_seed(seed)
    models, client_loaders, eval_loader = _load_clients(args, seed, dtype)

    rho_clip = args.rho_clip if (variant == "dmezo_n" and args.rho_clip > 0) else None
    mezo_cfg = MeZOConfig(lr=args.lr, eps=args.eps, rho_clip=rho_clip)

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

    topology = complete_graph(args.n_clients)

    eval_steps = [0]
    L0, A0 = _eval_metrics(clients[0].model, eval_loader, args)
    eval_losses = [L0]
    eval_accs = [A0]
    logger.info(f"  init: eval_loss={L0:.4f}  acc={A0:.4f}")

    t0 = time.time()

    def eval_fn(model, rnd):
        L, A = _eval_metrics(model, eval_loader, args)
        return {"loss": L, "acc": A}

    def round_logger(rl):
        r = rl["round"]
        el = rl.get("eval_loss")
        ea = rl.get("eval_acc")
        if el is not None:
            eval_steps.append(r + 1)
            eval_losses.append(el)
            eval_accs.append(ea if ea is not None else float("nan"))
        if (r + 1) % args.eval_every == 0:
            beta_str = ""
            if clients[0].nesterov_state is not None:
                beta_str = f" beta={clients[0].nesterov_state.beta:.3f}"
            logger.info(
                f"  {variant}|seed={seed} round={r+1:4d}/{args.num_rounds}{beta_str} "
                f"loss+={rl.get('mean_local_loss', 0):.3f} "
                f"rho={rl.get('mean_projected_grad', 0):+.2f} "
                f"eval_loss={el:.4f} acc={ea:.4f}"
            )

    sim_cfg = SimulatorConfig(
        num_rounds=args.num_rounds, consensus_mode="weight_avg",
        eval_every=args.eval_every, log_every=args.eval_every,
    )
    run_simulation(clients, topology, causal_lm_loss, sim_cfg,
                    eval_fn=eval_fn, logger=round_logger)

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


class _FixedRng:
    """Mimic numpy Generator returning a fixed shared seed.

    FedKSeed uses ONE shared direction per round. To get clients to use the
    same seed (and hence the same z), we hand each client a fake RNG that
    returns the round-specific shared seed.
    """

    def __init__(self, seed: int):
        self._seed = int(seed)

    def integers(self, lo, hi):  # noqa: D401
        return self._seed


def _run_one_cell_fedkseed(*, args, seed: int, dtype) -> dict:
    """FedKSeed: shared per-round seed, scalar averaging, no momentum/clip.

    Custom inner loop (does NOT use run_simulation) to control the seed
    assignment precisely. Star topology mathematically equivalent to
    complete-graph weight_avg here (all clients identical after each round).
    """
    torch.manual_seed(seed)
    models, client_loaders, eval_loader = _load_clients(args, seed, dtype)

    # FedKSeed has NO momentum, NO rho-clip, NO weight decay (paper default).
    mezo_cfg = MeZOConfig(lr=args.lr, eps=args.eps, rho_clip=None)

    # Per-client data iterators (so each client sees own data shard).
    client_iters = [iter(loader) for loader in client_loaders]

    def _next_batch(ci):
        nonlocal client_iters
        try:
            return next(client_iters[ci])
        except StopIteration:
            client_iters[ci] = iter(client_loaders[ci])
            return next(client_iters[ci])

    # Shared seed RNG (deterministic from run seed).
    shared_rng = np.random.default_rng(seed * 1000)

    eval_steps = [0]
    L0, A0 = _eval_metrics(models[0], eval_loader, args)
    eval_losses = [L0]
    eval_accs = [A0]
    logger.info(f"  init: eval_loss={L0:.4f}  acc={A0:.4f}")

    t0 = time.time()
    local_losses_window: list[float] = []
    rhos_window: list[float] = []

    for r in range(args.num_rounds):
        # 1. Server samples shared seed for this round.
        shared_seed = int(shared_rng.integers(0, 2**31 - 1))

        # 2. Each client computes rho_i with shared z (no per-client randomness in z).
        rhos = []
        losses = []
        for ci in range(args.n_clients):
            batch = _next_batch(ci)
            seed_back, rho_i, loss_plus = mezo_step(
                models[ci], batch, causal_lm_loss, mezo_cfg,
                rng=_FixedRng(shared_seed),
            )
            assert seed_back == shared_seed, "FedKSeed: client used wrong seed"
            rhos.append(rho_i)
            losses.append(loss_plus)

        # 3. Server averages.
        rho_bar = float(np.mean(rhos))

        # 4. All clients apply IDENTICAL update (same seed, same rho_bar).
        for ci in range(args.n_clients):
            mezo_update(models[ci], shared_seed, rho_bar, mezo_cfg)

        local_losses_window.append(float(np.mean(losses)))
        rhos_window.append(rho_bar)

        # 5. Eval periodically.
        if (r + 1) % args.eval_every == 0:
            el, ea = _eval_metrics(models[0], eval_loader, args)
            eval_steps.append(r + 1)
            eval_losses.append(el)
            eval_accs.append(ea)
            mean_loss = float(np.mean(local_losses_window))
            mean_rho = float(np.mean(rhos_window))
            logger.info(
                f"  fedkseed|seed={seed} round={r+1:4d}/{args.num_rounds} "
                f"loss+={mean_loss:.3f} rho={mean_rho:+.2f} "
                f"eval_loss={el:.4f} acc={ea:.4f}"
            )
            local_losses_window.clear()
            rhos_window.clear()

    wall = time.time() - t0
    for m in models:
        del m
    torch.cuda.empty_cache()
    return {
        "variant": "fedkseed", "seed": int(seed),
        "lr": args.lr, "eps": args.eps,
        "n_clients": args.n_clients, "consensus": "fedkseed (shared seed + scalar avg)",
        "topology": "star (equivalent to complete weight_avg here)",
        "partition": "iid",
        "eval_steps": eval_steps, "eval_losses": eval_losses, "eval_accs": eval_accs,
        "wall_clock_s": float(wall),
    }


def _bootstrap_ci(values, n_boot=10000, ci=0.95):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def _paired_analysis(cells, variant_a, variant_b, seeds):
    """Bootstrap CI on (A - B) paired by seed."""
    per_seed = []
    for s in seeds:
        a = cells.get(f"{variant_a}|seed={s}")
        b = cells.get(f"{variant_b}|seed={s}")
        if a is None or b is None:
            continue
        per_seed.append({
            "seed": s,
            "delta_acc": a["eval_accs"][-1] - b["eval_accs"][-1],
            "delta_loss": b["eval_losses"][-1] - a["eval_losses"][-1],
            "acc_a": a["eval_accs"][-1], "acc_b": b["eval_accs"][-1],
            "L_a": a["eval_losses"][-1], "L_b": b["eval_losses"][-1],
        })
    mean_dacc = (
        float(np.mean([p["delta_acc"] for p in per_seed])) if per_seed else float("nan")
    )
    ci_lo, ci_hi = _bootstrap_ci([p["delta_acc"] for p in per_seed])
    return {
        "per_seed": per_seed, "mean_delta_acc": mean_dacc,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi, "n_seeds": len(per_seed),
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"head_to_head_fedkseed_{model_short}_{args.task}.json"
    fig_path = fig_dir / f"fig20_head_to_head_fedkseed_{model_short}_{args.task}.png"

    dtype = getattr(torch, args.dtype)

    cells = {}
    grid_total = len(args.seeds) * len(args.variants)
    done = 0
    for s in args.seeds:
        for variant in args.variants:
            done += 1
            key = f"{variant}|seed={s}"
            logger.info(f"=== [{done}/{grid_total}] {key} ===")
            if variant == "fedkseed":
                cells[key] = _run_one_cell_fedkseed(args=args, seed=s, dtype=dtype)
            else:
                cells[key] = _run_one_cell_dmezo(
                    args=args, variant=variant, seed=s, dtype=dtype
                )

    # Paired analyses for all pairs of variants.
    paired = {}
    if "dmezo_n" in args.variants and "vanilla" in args.variants:
        paired["dmezo_n_vs_vanilla"] = _paired_analysis(
            cells, "dmezo_n", "vanilla", args.seeds
        )
    if "fedkseed" in args.variants and "vanilla" in args.variants:
        paired["fedkseed_vs_vanilla"] = _paired_analysis(
            cells, "fedkseed", "vanilla", args.seeds
        )
    if "dmezo_n" in args.variants and "fedkseed" in args.variants:
        paired["dmezo_n_vs_fedkseed"] = _paired_analysis(
            cells, "dmezo_n", "fedkseed", args.seeds
        )

    out = {
        "model": args.model, "task": args.task, "dtype": args.dtype,
        "n_clients": args.n_clients,
        "num_rounds": args.num_rounds, "lr": args.lr, "eps": args.eps,
        "seeds": args.seeds, "variants": args.variants,
        "eval_examples": args.num_eval_examples,
        "rho_clip": args.rho_clip, "beta_start": args.beta_start, "beta_end": args.beta_end,
        "cells": cells, "paired_analyses": paired,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # Figure: 3-panel comparison.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax_loss, ax_acc = axes
    colours = {"vanilla": "#d62728", "dmezo_n": "#1f77b4", "fedkseed": "#2ca02c"}
    labels = {"vanilla": "Vanilla D-MeZO", "dmezo_n": "D-MeZO-N (ours)",
              "fedkseed": "FedKSeed (Qin 2024)"}

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
        lbl = labels[variant]
        ax_loss.plot(all_steps, m_loss, "-", color=col, linewidth=2.0,
                     label=f"{lbl} (n={len(loss_curves)} seeds)")
        ax_loss.fill_between(all_steps, m_loss - s_loss, m_loss + s_loss,
                              color=col, alpha=0.2)
        ax_acc.plot(all_steps, m_acc, "-", color=col, linewidth=2.0,
                    label=f"{lbl}")
        ax_acc.fill_between(all_steps, m_acc - s_acc, m_acc + s_acc,
                             color=col, alpha=0.2)

    ax_loss.set_xlabel("MeZO round"); ax_loss.set_ylabel(f"{args.task} eval loss")
    ax_loss.set_title("(a) Loss trajectories (mean ± 1 std across seeds)")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_acc.set_xlabel("MeZO round"); ax_acc.set_ylabel(f"{args.task} accuracy")
    ax_acc.set_title("(b) Accuracy trajectories (mean ± 1 std)")
    ax_acc.legend(loc="lower right", fontsize=9)

    if paired:
        annotation_lines = []
        for name, pa in paired.items():
            if pa["n_seeds"] >= 1:
                annotation_lines.append(
                    f"Δacc {name}: {pa['mean_delta_acc']:+.4f} "
                    f"CI[{pa['ci95_lo']:+.4f},{pa['ci95_hi']:+.4f}]"
                )
        if annotation_lines:
            ax_acc.text(
                0.02, 0.98, "\n".join(annotation_lines),
                transform=ax_acc.transAxes, fontsize=8,
                va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            )

    task_label = {"hellaswag": "HellaSwag", "mathlogicqa": "MathLogicQA (RU)"}[args.task]
    fig.suptitle(
        f"Head-to-head: D-MeZO-N vs FedKSeed vs vanilla D-MeZO\n"
        f"{args.model} / {task_label} / {len(args.seeds)} seeds × {args.num_rounds} rounds / "
        f"n={args.n_clients} clients IID / lr={args.lr}, ε={args.eps}",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # Summary log.
    logger.info("=" * 90)
    logger.info(f"Head-to-head: {args.model} / {args.task} / {len(args.seeds)} seeds")
    logger.info(f"{'variant':<14}{'mean_loss':>12}{'std_loss':>11}{'mean_acc':>12}{'std_acc':>11}")
    for variant in args.variants:
        losses_f = [
            cells[f"{variant}|seed={s}"]["eval_losses"][-1]
            for s in args.seeds if f"{variant}|seed={s}" in cells
        ]
        accs_f = [
            cells[f"{variant}|seed={s}"]["eval_accs"][-1]
            for s in args.seeds if f"{variant}|seed={s}" in cells
        ]
        if losses_f:
            logger.info(
                f"{variant:<14}{np.mean(losses_f):>12.4f}{np.std(losses_f):>11.4f}"
                f"{np.mean(accs_f):>12.4f}{np.std(accs_f):>11.4f}"
            )
    for name, pa in paired.items():
        logger.info(
            f"  paired {name}: Δacc = {pa['mean_delta_acc']:+.4f}  "
            f"95% CI [{pa['ci95_lo']:+.4f}, {pa['ci95_hi']:+.4f}]  "
            f"n_seeds={pa['n_seeds']}"
        )
    logger.info("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
