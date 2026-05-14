"""Day 1: sanity check that MeZO converges on Qwen3-4B / SST-2.

This is the FIRST experiment to run. Goal: confirm that the canonical MeZO
optimizer drives down cross-entropy loss on SST-2 when applied to a modern
Qwen3 architecture (Malladi 2023 only validated MeZO on OPT/RoBERTa).

Success criteria:
    - Loss decreases by at least 30% over 500 MeZO steps.
    - Eval accuracy improves (random baseline = 50% for SST-2 binary).

Usage:
    python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running as a script from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmezo.data.superglue import build_sst2_loader, causal_lm_loss  # noqa: E402
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402
from dmezo.utils.config import load_yaml_config  # noqa: E402
from dmezo.utils.logging import JSONLLogger, setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config.output_dir (e.g. for Colab Drive path)",
    )
    return p.parse_args()


@torch.inference_mode()
def evaluate_loss(model, dataloader, max_batches: int = 20) -> float:
    """Compute mean loss over a few eval batches."""
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        loss = causal_lm_loss(model, batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    logger = setup_logger("dmezo.sanity")

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = JSONLLogger(output_dir / "log.jsonl")

    logger.info(f"Config: {cfg}")
    logger.info(f"Output dir: {output_dir}")

    # ---- Model
    model_name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    logger.info(f"Loading model {model_name} in {dtype}...")
    model, tokenizer = load_causal_lm(
        model_name=model_name,
        dtype=dtype,
        device_map="auto",
        use_flash_attention=cfg["model"].get("flash_attention", True),
    )
    # Make sure params have requires_grad (HF disables it sometimes).
    for p in model.parameters():
        p.requires_grad_(True)

    # ---- Data
    bs = cfg["data"].get("batch_size", 8)
    max_len = cfg["data"].get("max_length", 256)
    n_train = cfg["data"].get("num_train_examples", 1000)
    n_eval = cfg["data"].get("num_eval_examples", 200)

    logger.info("Building dataloaders...")
    train_loader = build_sst2_loader(
        tokenizer, split="train", batch_size=bs, max_length=max_len,
        shuffle=True, num_examples=n_train,
    )
    eval_loader = build_sst2_loader(
        tokenizer, split="validation", batch_size=bs, max_length=max_len,
        shuffle=False, num_examples=n_eval,
    )

    # ---- MeZO config
    mezo_cfg = MeZOConfig(
        lr=float(cfg["mezo"]["lr"]),
        eps=float(cfg["mezo"]["eps"]),
        weight_decay=float(cfg["mezo"].get("weight_decay", 0.0)),
    )
    n_steps = int(cfg["train"]["steps"])
    eval_every = int(cfg["train"]["eval_every"])
    log_every = int(cfg["train"].get("log_every", 20))

    # ---- Initial eval
    init_eval = evaluate_loss(model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20))
    logger.info(f"Initial eval loss: {init_eval:.4f}")
    jsonl.log({"step": 0, "eval_loss": init_eval, "phase": "init"})

    # ---- Training loop
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    train_iter = iter(train_loader)
    losses = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        seed, rho, loss_plus = mezo_step(model, batch, causal_lm_loss, mezo_cfg, rng=rng)
        mezo_update(model, seed=seed, projected_grad=rho, config=mezo_cfg)
        losses.append(loss_plus)

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = float(np.mean(losses[-log_every:]))
            logger.info(
                f"step={step}/{n_steps} loss={avg_loss:.4f} "
                f"rho={rho:+.4f} elapsed={elapsed:.1f}s"
            )
            jsonl.log({
                "step": step, "train_loss": avg_loss, "projected_grad": rho,
                "elapsed_sec": elapsed,
            })

        if step % eval_every == 0:
            ev = evaluate_loss(model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20))
            logger.info(f"  eval_loss={ev:.4f}")
            jsonl.log({"step": step, "eval_loss": ev, "phase": "eval"})

    # ---- Final eval
    final_eval = evaluate_loss(model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20))
    final_train = float(np.mean(losses[-50:]))
    logger.info(f"FINAL: train_loss={final_train:.4f} eval_loss={final_eval:.4f}")
    jsonl.log({"step": n_steps, "final_train_loss": final_train,
               "final_eval_loss": final_eval, "init_eval_loss": init_eval,
               "phase": "final"})

    # ---- Sanity verdict
    drop = (init_eval - final_eval) / max(init_eval, 1e-8)
    if drop > 0.10:
        logger.info(f"[PASS] Eval loss dropped by {drop * 100:.1f}% — MeZO works on {model_name}.")
    else:
        logger.warning(
            f"[FAIL?] Eval loss dropped only {drop * 100:.1f}%. "
            f"Investigate: lr, eps, num_steps, or architecture compatibility."
        )

    jsonl.close()


if __name__ == "__main__":
    main()
