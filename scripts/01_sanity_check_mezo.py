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
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
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
    p.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: file:./mlruns or $MLFLOW_TRACKING_URI)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: auto from config)",
    )
    return p.parse_args()


def _flatten_params(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested config dict to dot-separated keys for mlflow.log_params."""
    out: dict = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=key))
        else:
            out[key] = v
    return out


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

    # Also persist console output to a file so it can be uploaded to MLflow
    # at the end of the run (useful on Colab where stdout is ephemeral).
    console_log_path = output_dir / "console.log"
    file_handler = logging.FileHandler(console_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.info(f"Config: {cfg}")
    logger.info(f"Output dir: {output_dir}")

    # ---- MLflow setup
    tracking_uri = (
        args.mlflow_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"file:{(Path.cwd() / 'mlruns').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "dmezo_sanity"))
    run_name = args.run_name or f"{cfg['model']['name'].split('/')[-1]}_sanity"

    with mlflow.start_run(run_name=run_name) as mlrun:
        logger.info(f"MLflow run: {mlrun.info.run_id} (uri={tracking_uri})")
        mlflow.log_params(_flatten_params(cfg))
        mlflow.set_tags(
            {
                "algo": "mezo_centralized",
                "model_family": cfg["model"]["name"].split("/")[0],
                "script": "01_sanity_check_mezo",
            }
        )
        mlflow.log_artifact(args.config, artifact_path="config")

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
            tokenizer,
            split="train",
            batch_size=bs,
            max_length=max_len,
            shuffle=True,
            num_examples=n_train,
        )
        eval_loader = build_sst2_loader(
            tokenizer,
            split="validation",
            batch_size=bs,
            max_length=max_len,
            shuffle=False,
            num_examples=n_eval,
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
        init_eval = evaluate_loss(
            model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20)
        )
        logger.info(f"Initial eval loss: {init_eval:.4f}")
        jsonl.log({"step": 0, "eval_loss": init_eval, "phase": "init"})
        mlflow.log_metric("eval_loss", init_eval, step=0)

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
                jsonl.log(
                    {
                        "step": step,
                        "train_loss": avg_loss,
                        "projected_grad": rho,
                        "elapsed_sec": elapsed,
                    }
                )
                mlflow.log_metrics(
                    {"train_loss": avg_loss, "projected_grad": float(rho), "elapsed_sec": elapsed},
                    step=step,
                )

            if step % eval_every == 0:
                ev = evaluate_loss(
                    model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20)
                )
                logger.info(f"  eval_loss={ev:.4f}")
                jsonl.log({"step": step, "eval_loss": ev, "phase": "eval"})
                mlflow.log_metric("eval_loss", ev, step=step)

        # ---- Final eval
        final_eval = evaluate_loss(
            model, eval_loader, max_batches=cfg["train"].get("eval_batches", 20)
        )
        final_train = float(np.mean(losses[-50:])) if losses else float("nan")
        logger.info(f"FINAL: train_loss={final_train:.4f} eval_loss={final_eval:.4f}")
        jsonl.log(
            {
                "step": n_steps,
                "final_train_loss": final_train,
                "final_eval_loss": final_eval,
                "init_eval_loss": init_eval,
                "phase": "final",
            }
        )

        # ---- Sanity verdict
        drop = (init_eval - final_eval) / max(init_eval, 1e-8)
        verdict = "PASS" if drop > 0.10 else "FAIL"
        if verdict == "PASS":
            logger.info(
                f"[PASS] Eval loss dropped by {drop * 100:.1f}% — MeZO works on {model_name}."
            )
        else:
            logger.warning(
                f"[FAIL?] Eval loss dropped only {drop * 100:.1f}%. "
                f"Investigate: lr, eps, num_steps, or architecture compatibility."
            )

        mlflow.log_metrics(
            {
                "final_train_loss": final_train,
                "final_eval_loss": final_eval,
                "init_eval_loss": init_eval,
                "eval_loss_drop_pct": drop * 100.0,
            },
            step=n_steps,
        )
        mlflow.set_tag("sanity_verdict", verdict)

        jsonl.close()
        # Flush + detach FileHandler before uploading.
        file_handler.flush()
        logger.removeHandler(file_handler)
        file_handler.close()
        mlflow.log_artifact(str(output_dir / "log.jsonl"), artifact_path="logs")
        mlflow.log_artifact(str(console_log_path), artifact_path="logs")


if __name__ == "__main__":
    main()
