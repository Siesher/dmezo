"""Local hypothesis-testing for D-MeZO-N improvements (B5 / B1 / D2).

Designed for **RTX 5070 Ti Blackwell (17 GB VRAM)** with Qwen3-0.6B / SST-2 —
quick iteration on a small model to test that:
    1. Improvements don't break anything (smoke).
    2. Improvements show the predicted directional effect vs vanilla / D-MeZO-N v1.

This is NOT the Colab head-to-head experiment (which uses Qwen3.5-4B-Base) —
it's a faster local replicate for hypothesis validation before burning compute.

Variants compared:
    - vanilla:                  plain MeZO (no momentum, no clip)
    - dmezo_n:                  D-MeZO-N v1 (β-decay 0.9→0 + fixed clip C=50)
    - dmezo_n_drift:            B5 — D-MeZO-N + drift reset
    - dmezo_n_adaptive_clip:    B1 — D-MeZO-N + adaptive clip (no fixed C)
    - dmezo_n_dp:               D2 — D-MeZO-N + DP noise σ=0.5 (mild privacy)

Each variant runs ≤ 2 seeds × 500 rounds = ~10-15 min per cell on Blackwell
local. Total: ~50-75 min for full sweep on 2 seeds × 5 variants.

Output:
    experiments/diagnostics/local_test_improvements_<task>.json
    docs/figures/fig_local_improvements_<task>.png

Usage (PowerShell):
    .venv\\Scripts\\python scripts\\local_test_improvements.py `
        --task sst2 --num-rounds 500 --seeds 42 43

Important: use `uv run --no-sync ...` if you need to invoke via uv (CUDA torch
is not pinned in pyproject and a regular `uv run` overwrites it with CPU build).
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
from dmezo.mezo.step import AdaptiveClipState, MeZOConfig, dp_epsilon_from_sigma  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("dmezo.local_test")


VARIANTS = [
    "vanilla",
    "dmezo_n",
    "dmezo_n_drift",
    "dmezo_n_adaptive_clip",
    "dmezo_n_dp",
    # NEW: B1 + B5 combination — adaptive clip + drift-reset safety net.
    # Motivation (§6.6.2): adaptive_clip alone shows loss-better/acc-worse paradox
    # due to momentum overshoot once the tight clip is removed. Pairing it with
    # B5 drift-reset (velocity-zeroing on detected loss rise) should let us keep
    # the faster loss descent of adaptive clip while catching the overshoot.
    "dmezo_n_combo",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="Default is Qwen3-0.6B — fits comfortably on 17GB Blackwell with "
             "4 client copies. Use Qwen/Qwen3.5-0.8B for hybrid linear-attn test.",
    )
    p.add_argument(
        "--task", type=str, default="sst2",
        choices=["sst2", "boolq", "mathlogicqa", "hellaswag"],
        help="sst2/boolq: easy GLUE/SuperGLUE; mathlogicqa: Russian 4-way "
             "symbolic logic (MERA); hellaswag: English 4-way commonsense.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--variants", type=str, nargs="+", default=VARIANTS,
        help=f"Subset of {VARIANTS}",
    )
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--num-train-examples", type=int, default=1000)
    p.add_argument("--num-eval-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=20)
    # D-MeZO-N hyperparameters (used by all D-MeZO-N variants).
    p.add_argument("--rho-clip", type=float, default=50.0,
                   help="Fixed clip (used by dmezo_n + dmezo_n_drift + dmezo_n_dp)")
    p.add_argument("--beta-start", type=float, default=0.9)
    p.add_argument("--beta-end", type=float, default=0.0)
    # B5 drift-reset hyperparameters.
    p.add_argument("--drift-window", type=int, default=100,
                   help="Window for drift detection (rounds)")
    p.add_argument("--drift-threshold", type=float, default=0.05,
                   help="Min loss increase over window to trigger reset")
    # B1 adaptive clip hyperparameters.
    p.add_argument("--ac-window", type=int, default=50,
                   help="Adaptive-clip window (|ρ| samples)")
    p.add_argument("--ac-quantile", type=float, default=0.95)
    p.add_argument("--ac-alpha", type=float, default=1.3)
    # D2 DP-MeZO hyperparameters.
    p.add_argument("--dp-sigma", type=float, default=0.5,
                   help="Gaussian noise σ on ρ (post-clip). 0 disables.")
    p.add_argument("--dp-delta", type=float, default=1e-3,
                   help="DP δ for reporting ε in summary.")
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


def _build_clients(args, variant: str, seed: int, dtype):
    """Create n client model copies + data partition + variant-specific state."""
    torch.manual_seed(seed)
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
    # Eval loader: dedicated validation pool for the task.
    if args.task == "mathlogicqa":
        from dmezo.data.mathlogicqa import build_mathlogicqa_loader  # local import
        eval_loader = build_mathlogicqa_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    elif args.task == "hellaswag":
        from dmezo.data.hellaswag import build_hellaswag_loader  # local import
        eval_loader = build_hellaswag_loader(
            tokenizer, split="validation", batch_size=args.batch_size,
            max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )
    else:
        from dmezo.data.superglue import build_loader_for_task  # local import
        eval_loader = build_loader_for_task(
            args.task, tokenizer=tokenizer, split="validation",
            batch_size=args.batch_size, max_length=args.max_length, shuffle=False,
            num_examples=args.num_eval_examples, seed=0,
        )

    # Per-variant MeZO config + state.
    base_cfg_kwargs = dict(lr=args.lr, eps=args.eps)
    rho_clip = args.rho_clip if variant != "vanilla" else None

    if variant == "vanilla":
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=None)
        make_nesterov = lambda: None  # noqa: E731
        make_ac = lambda: None  # noqa: E731
    elif variant == "dmezo_n":
        # Baseline D-MeZO-N v1: β-decay + fixed clip.
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=rho_clip)
        make_nesterov = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
        )
        make_ac = lambda: None  # noqa: E731
    elif variant == "dmezo_n_drift":
        # B5: D-MeZO-N + drift-reset.
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=rho_clip)
        make_nesterov = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
            drift_window=args.drift_window, drift_threshold=args.drift_threshold,
        )
        make_ac = lambda: None  # noqa: E731
    elif variant == "dmezo_n_adaptive_clip":
        # B1: D-MeZO-N + adaptive clip (replaces fixed C).
        # NOTE: keep mezo_cfg.rho_clip=None so override takes precedence.
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=None)
        make_nesterov = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
        )
        make_ac = lambda: AdaptiveClipState(  # noqa: E731
            window=args.ac_window, quantile=args.ac_quantile, alpha=args.ac_alpha,
        )
    elif variant == "dmezo_n_dp":
        # D2: D-MeZO-N + DP noise + fixed clip (required for DP sensitivity bound).
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=rho_clip, dp_sigma=args.dp_sigma)
        make_nesterov = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
        )
        make_ac = lambda: None  # noqa: E731
    elif variant == "dmezo_n_combo":
        # B1 + B5: adaptive clip + drift-reset safety net.
        # adaptive_clip removes the tight fixed-C constraint (allows faster loss);
        # drift_reset catches the resulting momentum overshoot by zeroing velocity
        # when eval loss rises above threshold over the window.
        # No fixed rho_clip in config — override is dominant.
        mezo_cfg = MeZOConfig(**base_cfg_kwargs, rho_clip=None)
        make_nesterov = lambda: NesterovState(  # noqa: E731
            beta=args.beta_start, look_ahead=False,
            beta_end=args.beta_end, num_rounds_total=args.num_rounds,
            drift_window=args.drift_window, drift_threshold=args.drift_threshold,
        )
        make_ac = lambda: AdaptiveClipState(  # noqa: E731
            window=args.ac_window, quantile=args.ac_quantile, alpha=args.ac_alpha,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    clients = [
        ClientState(
            client_id=ci, model=models[ci], dataloader=client_loaders[ci],
            mezo_config=mezo_cfg, local_steps=1,
            nesterov_state=make_nesterov(),
            rng=np.random.default_rng(seed + ci),
            adaptive_clip_state=make_ac(),
        )
        for ci in range(args.n_clients)
    ]
    return clients, models, eval_loader


def _run_one_cell(*, args, variant: str, seed: int, dtype) -> dict:
    logger.info(f"  Building clients for {variant} | seed={seed}...")
    clients, models, eval_loader = _build_clients(args, variant, seed, dtype)

    topology = complete_graph(args.n_clients)

    L0 = _eval_loss(clients[0].model, eval_loader, max_batches=args.eval_batches)
    A0 = evaluate_classification_accuracy(
        clients[0].model, eval_loader, task=args.task, max_batches=args.eval_batches
    )
    eval_steps = [0]
    eval_losses = [L0]
    eval_accs = [A0]
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
        el = rl.get("eval_loss")
        ea = rl.get("eval_acc")
        if el is not None:
            eval_steps.append(r + 1)
            eval_losses.append(el)
            eval_accs.append(ea if ea is not None else float("nan"))
        if (r + 1) % args.eval_every == 0:
            beta_str = ""
            ns = clients[0].nesterov_state
            if ns is not None:
                beta_str = f" beta={ns.beta:.3f} resets={ns.n_resets}"
            acs = clients[0].adaptive_clip_state
            ac_str = ""
            if acs is not None:
                thr = acs.current_threshold()
                ac_str = f" ac_thr={thr:.2f}" if thr is not None else " ac_thr=N/A"
            logger.info(
                f"  {variant}|s={seed} r={r+1:4d}/{args.num_rounds}{beta_str}{ac_str} "
                f"loss+={rl.get('mean_local_loss', 0):.3f} "
                f"rho={rl.get('mean_projected_grad', 0):+.2f} "
                f"eval={el:.4f} acc={ea:.4f}"
            )

    sim_cfg = SimulatorConfig(
        num_rounds=args.num_rounds, consensus_mode="weight_avg",
        eval_every=args.eval_every, log_every=args.eval_every,
    )
    run_simulation(
        clients, topology, causal_lm_loss, sim_cfg,
        eval_fn=eval_fn, logger=round_logger,
    )
    wall = time.time() - t0

    n_resets = (
        clients[0].nesterov_state.n_resets
        if clients[0].nesterov_state is not None else 0
    )

    # DP epsilon report (only meaningful for dp variant; otherwise N/A).
    if variant == "dmezo_n_dp":
        dp_eps = dp_epsilon_from_sigma(
            sigma=args.dp_sigma, sensitivity=args.rho_clip, delta=args.dp_delta
        )
    else:
        dp_eps = None

    for m in models:
        del m
    torch.cuda.empty_cache()

    return {
        "variant": variant, "seed": int(seed),
        "lr": args.lr, "eps": args.eps,
        "n_clients": args.n_clients,
        "eval_steps": eval_steps, "eval_losses": eval_losses, "eval_accs": eval_accs,
        "wall_clock_s": float(wall),
        "n_drift_resets": int(n_resets),
        "dp_epsilon": dp_eps,
    }


def main() -> int:
    args = parse_args()
    out_dir = ROOT / "experiments" / "diagnostics"
    fig_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "p")
    json_path = out_dir / f"local_test_improvements_{model_short}_{args.task}.json"
    fig_path = fig_dir / f"fig_local_improvements_{model_short}_{args.task}.png"

    dtype = getattr(torch, args.dtype)

    cells = {}
    total = len(args.seeds) * len(args.variants)
    done = 0
    for s in args.seeds:
        for variant in args.variants:
            if variant not in VARIANTS:
                logger.warning(f"Skipping unknown variant: {variant}")
                continue
            done += 1
            key = f"{variant}|seed={s}"
            logger.info(f"=== [{done}/{total}] {key} ===")
            cells[key] = _run_one_cell(args=args, variant=variant, seed=s, dtype=dtype)

    out = {
        "model": args.model, "task": args.task, "dtype": args.dtype,
        "n_clients": args.n_clients, "num_rounds": args.num_rounds,
        "lr": args.lr, "eps": args.eps, "rho_clip": args.rho_clip,
        "beta_start": args.beta_start, "beta_end": args.beta_end,
        "drift_window": args.drift_window, "drift_threshold": args.drift_threshold,
        "ac_window": args.ac_window, "ac_quantile": args.ac_quantile,
        "ac_alpha": args.ac_alpha,
        "dp_sigma": args.dp_sigma, "dp_delta": args.dp_delta,
        "seeds": args.seeds, "variants": args.variants,
        "cells": cells,
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved JSON to {json_path}")

    # Figure: side-by-side loss + accuracy.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_acc = axes
    palette = {
        "vanilla": "#d62728",
        "dmezo_n": "#1f77b4",
        "dmezo_n_drift": "#2ca02c",
        "dmezo_n_adaptive_clip": "#9467bd",
        "dmezo_n_dp": "#ff7f0e",
        "dmezo_n_combo": "#17becf",
    }
    labels = {
        "vanilla": "Vanilla D-MeZO",
        "dmezo_n": "D-MeZO-N v1 (fixed C, β-decay)",
        "dmezo_n_drift": "+ B5 drift reset",
        "dmezo_n_adaptive_clip": "+ B1 adaptive clip",
        "dmezo_n_dp": f"+ D2 DP (σ={args.dp_sigma})",
        "dmezo_n_combo": "+ B1+B5 combo",
    }
    for variant in args.variants:
        if variant not in VARIANTS:
            continue
        loss_curves, acc_curves, all_steps = [], [], None
        for s in args.seeds:
            r = cells.get(f"{variant}|seed={s}")
            if r is None:
                continue
            all_steps = np.array(r["eval_steps"])
            loss_curves.append(np.array(r["eval_losses"]))
            acc_curves.append(np.array(r["eval_accs"]))
        if not loss_curves:
            continue
        loss_arr = np.stack(loss_curves)
        acc_arr = np.stack(acc_curves)
        m_loss, s_loss = loss_arr.mean(axis=0), loss_arr.std(axis=0)
        m_acc, s_acc = acc_arr.mean(axis=0), acc_arr.std(axis=0)
        col = palette[variant]
        lbl = labels[variant]
        ax_loss.plot(all_steps, m_loss, "-", color=col, linewidth=2.0, label=lbl)
        ax_loss.fill_between(all_steps, m_loss - s_loss, m_loss + s_loss,
                              color=col, alpha=0.2)
        ax_acc.plot(all_steps, m_acc, "-", color=col, linewidth=2.0, label=lbl)
        ax_acc.fill_between(all_steps, m_acc - s_acc, m_acc + s_acc,
                             color=col, alpha=0.2)

    ax_loss.set_xlabel("MeZO round"); ax_loss.set_ylabel(f"{args.task} eval loss")
    ax_loss.set_title(f"(a) Loss trajectories (mean ± 1 std, n={len(args.seeds)} seeds)")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_acc.set_xlabel("MeZO round"); ax_acc.set_ylabel(f"{args.task} accuracy")
    ax_acc.set_title("(b) Accuracy trajectories")
    ax_acc.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Local hypothesis test: D-MeZO-N improvements on {args.model} / {args.task}\n"
        f"{len(args.seeds)} seeds × {args.num_rounds} rounds, n={args.n_clients} clients IID, "
        f"lr={args.lr}, ε={args.eps}",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {fig_path}")

    # Summary table.
    logger.info("=" * 100)
    logger.info(
        f"Local hypothesis test: {args.model} / {args.task} / "
        f"{len(args.seeds)} seeds × {args.num_rounds} rounds"
    )
    logger.info(
        f"{'variant':<28}{'mean_loss':>12}{'std_loss':>11}"
        f"{'mean_acc':>11}{'std_acc':>10}{'resets':>9}{'DP ε':>10}"
    )
    for variant in args.variants:
        losses_f = [
            cells[f"{variant}|seed={s}"]["eval_losses"][-1]
            for s in args.seeds if f"{variant}|seed={s}" in cells
        ]
        accs_f = [
            cells[f"{variant}|seed={s}"]["eval_accs"][-1]
            for s in args.seeds if f"{variant}|seed={s}" in cells
        ]
        resets_total = sum(
            cells[f"{variant}|seed={s}"]["n_drift_resets"]
            for s in args.seeds if f"{variant}|seed={s}" in cells
        )
        # DP ε is same per seed (deterministic from sigma/C/delta).
        dp_eps_vals = [
            cells[f"{variant}|seed={s}"]["dp_epsilon"]
            for s in args.seeds if f"{variant}|seed={s}" in cells
            and cells[f"{variant}|seed={s}"]["dp_epsilon"] is not None
        ]
        dp_str = f"{dp_eps_vals[0]:.2f}" if dp_eps_vals else "N/A"
        if losses_f:
            logger.info(
                f"{variant:<28}{np.mean(losses_f):>12.4f}{np.std(losses_f):>11.4f}"
                f"{np.mean(accs_f):>11.4f}{np.std(accs_f):>10.4f}"
                f"{resets_total:>9d}{dp_str:>10}"
            )
    logger.info("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
