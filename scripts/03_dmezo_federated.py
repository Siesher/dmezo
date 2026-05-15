"""Day 4-5: n-client federated D-MeZO with consensus mixing.

Generic over number of clients and topology — Day 4 (`configs/dmezo_2c_*.yaml`)
runs with n=2, Day 5 with n=4. Loads ``n_clients`` separate copies of the same
pretrained model from HF cache (all clients start from identical pretrained
weights), partitions the train set IID (different seeds), and runs federated
MeZO with either ``weight_avg`` or ``update_share`` consensus.

Success criterion (Week 1 plan): federated final eval loss <= centralized
final eval loss × 1.1.

Usage:
    python scripts/03_dmezo_federated.py --config configs/dmezo_2c_complete_qwen3_4b_sst2.yaml
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

from dmezo.data.superglue import (  # noqa: E402
    build_loader_for_task,
    causal_lm_loss,
)
from dmezo.federated.client import ClientState  # noqa: E402
from dmezo.federated.simulator import SimulatorConfig, run_simulation  # noqa: E402
from dmezo.federated.topology import complete_graph, ring_graph  # noqa: E402
from dmezo.mezo.nesterov import NesterovState  # noqa: E402
from dmezo.mezo.step import MeZOConfig  # noqa: E402
from dmezo.models.loader import load_causal_lm  # noqa: E402
from dmezo.utils.config import load_yaml_config  # noqa: E402
from dmezo.utils.logging import JSONLLogger, setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--mlflow-uri", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def _flatten_params(cfg: dict, prefix: str = "") -> dict:
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
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        loss = causal_lm_loss(model, batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _build_topology(name: str, n: int):
    if name == "complete":
        return complete_graph(n)
    if name == "ring":
        return ring_graph(n)
    raise ValueError(f"Unknown topology {name!r} (supported: complete, ring)")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    logger = setup_logger("dmezo.federated")

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = JSONLLogger(output_dir / "log.jsonl")

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
    mlflow.set_experiment(cfg.get("experiment_name", "dmezo_federated"))
    n_clients = int(cfg["federated"]["num_clients"])
    topology_name = cfg["federated"]["topology"]
    consensus_mode = cfg["federated"]["consensus_mode"]
    short = cfg["model"]["name"].split("/")[-1]
    run_name = args.run_name or f"{short}_{n_clients}c_{topology_name}_{consensus_mode}"

    with mlflow.start_run(run_name=run_name) as mlrun:
        logger.info(f"MLflow run: {mlrun.info.run_id} (uri={tracking_uri})")
        mlflow.log_params(_flatten_params(cfg))
        mlflow.set_tags(
            {
                "algo": "dmezo_federated",
                "model_family": cfg["model"]["name"].split("/")[0],
                "script": "03_dmezo_federated",
                "n_clients": str(n_clients),
                "topology": topology_name,
                "consensus_mode": consensus_mode,
            }
        )
        mlflow.log_artifact(args.config, artifact_path="config")

        # ---- Model + tokenizer (loaded once for client 0, fresh for others to
        # avoid deep-copy issues with tied weights / V-L composite modules)
        model_name = cfg["model"]["name"]
        dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
        flash_attn = cfg["model"].get("flash_attention", False)

        logger.info(
            f"Loading model {model_name} for {n_clients} clients (this may take a moment)..."
        )
        models = []
        tokenizer = None
        for ci in range(n_clients):
            m, tok = load_causal_lm(
                model_name=model_name,
                dtype=dtype,
                device_map="auto",
                use_flash_attention=flash_attn,
            )
            # Respect loader-set requires_grad (e.g. V-L vision branch frozen).
            n_total = sum(1 for _ in m.parameters())
            n_pre_frozen = sum(1 for p in m.parameters() if not p.requires_grad)
            if n_pre_frozen == 0:
                for p in m.parameters():
                    p.requires_grad_(True)
            elif ci == 0:
                logger.info(
                    f"Models have {n_pre_frozen}/{n_total} params pre-frozen by loader; "
                    f"federated MeZO will only touch the remaining {n_total - n_pre_frozen}."
                )
            models.append(m)
            tokenizer = tok

        # ---- Data: per-client train loaders (IID partition via different seeds),
        # shared eval loader.
        task = cfg["data"].get("task", "sst2")
        bs = cfg["data"].get("batch_size", 8)
        max_len = cfg["data"].get("max_length", 256)
        n_train_total = cfg["data"].get("num_train_examples", 1000)
        n_eval = cfg["data"].get("num_eval_examples", 200)
        partition_mode = cfg["data"].get("partition_mode", "iid")
        partition_kwargs = cfg["data"].get("partition_kwargs", {}) or {}
        base_seed = int(cfg.get("seed", 42))

        logger.info(
            f"Building dataloaders for task={task!r}, partition={partition_mode!r} "
            f"({n_train_total} train examples, {n_clients} clients)..."
        )
        eval_loader = build_loader_for_task(
            task,
            tokenizer=tokenizer,
            split="validation",
            batch_size=bs,
            max_length=max_len,
            shuffle=False,
            num_examples=n_eval,
        )
        client_loaders = build_partitioned_loaders(
            task=task,
            tokenizer=tokenizer,
            n_clients=n_clients,
            partition_mode=partition_mode,
            partition_kwargs=partition_kwargs,
            batch_size=bs,
            max_length=max_len,
            num_examples=n_train_total,
            shuffle=True,
            seed=base_seed,
        )
        # Log per-client partition stats — useful to confirm Dirichlet/label-skew
        # asymmetry is actually present in the data, not just in the config flag.
        for ci, loader in enumerate(client_loaders):
            n = len(loader.dataset)
            try:
                labels = np.asarray(loader.dataset.data["label"])
                if labels.size > 0:
                    counts = np.bincount(labels)
                    dist_str = ", ".join(f"c{j}={c}" for j, c in enumerate(counts))
                else:
                    dist_str = "empty"
            except (KeyError, AttributeError):
                dist_str = "n/a"
            logger.info(f"  client {ci}: n={n}  label_dist=[{dist_str}]")
            mlflow.log_metric(f"client_{ci}_train_size", float(n), step=0)

        # ---- MeZO config (shared across clients)
        mezo_cfg = MeZOConfig(
            lr=float(cfg["mezo"]["lr"]),
            eps=float(cfg["mezo"]["eps"]),
            weight_decay=float(cfg["mezo"].get("weight_decay", 0.0)),
        )

        # ---- Construct ClientState objects
        nesterov_enabled = cfg.get("nesterov", {}).get("enabled", False)
        nesterov_beta = float(cfg.get("nesterov", {}).get("beta", 0.9))
        local_steps = int(cfg["federated"].get("local_steps", 1))

        clients = []
        for ci in range(n_clients):
            ns = NesterovState(beta=nesterov_beta) if nesterov_enabled else None
            clients.append(
                ClientState(
                    client_id=ci,
                    model=models[ci],
                    dataloader=client_loaders[ci],
                    mezo_config=mezo_cfg,
                    local_steps=local_steps,
                    nesterov_state=ns,
                    rng=np.random.default_rng(base_seed + ci),
                )
            )

        # ---- Topology
        topology = _build_topology(topology_name, n_clients)
        logger.info(f"Topology: {topology}")

        # ---- Initial eval (client 0 — all clients have identical init)
        eval_batches = cfg["train"].get("eval_batches", 20)
        init_eval = evaluate_loss(clients[0].model, eval_loader, max_batches=eval_batches)
        logger.info(f"Initial eval loss: {init_eval:.4f}")
        jsonl.log({"round": 0, "eval_loss": init_eval, "phase": "init"})
        mlflow.log_metric("eval_loss", init_eval, step=0)

        # ---- Callbacks for simulator
        t0 = time.time()

        def eval_fn(model, rnd):
            ev = evaluate_loss(model, eval_loader, max_batches=eval_batches)
            return {"loss": ev}

        def round_logger(round_log):
            elapsed = time.time() - t0
            r = round_log["round"]
            mean_loss = round_log.get("mean_local_loss", float("nan"))
            mean_rho = round_log.get("mean_projected_grad", float("nan"))
            eval_loss = round_log.get("eval_loss")
            msg = (
                f"round={r + 1}/{cfg['train']['num_rounds']} "
                f"local_loss={mean_loss:.4f} rho={mean_rho:+.4f} elapsed={elapsed:.1f}s"
            )
            if eval_loss is not None:
                msg += f"  eval={eval_loss:.4f}"
            logger.info(msg)
            entry = {
                "round": r + 1,
                "mean_local_loss": mean_loss,
                "mean_projected_grad": mean_rho,
                "elapsed_sec": elapsed,
            }
            if eval_loss is not None:
                entry["eval_loss"] = eval_loss
                mlflow.log_metric("eval_loss", eval_loss, step=r + 1)
            jsonl.log(entry)
            mlflow.log_metrics(
                {
                    "mean_local_loss": mean_loss,
                    "mean_projected_grad": float(mean_rho),
                    "elapsed_sec": elapsed,
                },
                step=r + 1,
            )

        sim_cfg = SimulatorConfig(
            num_rounds=int(cfg["train"]["num_rounds"]),
            consensus_mode=consensus_mode,
            eval_every=int(cfg["train"].get("eval_every", 100)),
            log_every=int(cfg["train"].get("log_every", 20)),
        )

        logger.info(
            f"Starting federated training: n_clients={n_clients} topology={topology_name} "
            f"consensus={consensus_mode} num_rounds={sim_cfg.num_rounds}"
        )
        run_simulation(
            clients=clients,
            topology=topology,
            loss_fn=causal_lm_loss,
            config=sim_cfg,
            eval_fn=eval_fn,
            logger=round_logger,
        )

        # ---- Final eval
        final_eval = evaluate_loss(clients[0].model, eval_loader, max_batches=eval_batches)
        logger.info(f"FINAL eval_loss (client 0): {final_eval:.4f}")
        drop = (init_eval - final_eval) / max(init_eval, 1e-8)
        verdict = "PASS" if drop > 0.10 else "FAIL"
        if verdict == "PASS":
            logger.info(
                f"[PASS] Eval loss dropped by {drop * 100:.1f}% — D-MeZO works on {model_name}."
            )
        else:
            logger.warning(f"[FAIL?] Eval loss dropped only {drop * 100:.1f}%.")

        jsonl.log(
            {
                "round": sim_cfg.num_rounds,
                "final_eval_loss": final_eval,
                "init_eval_loss": init_eval,
                "phase": "final",
            }
        )
        mlflow.log_metrics(
            {
                "final_eval_loss": final_eval,
                "init_eval_loss": init_eval,
                "eval_loss_drop_pct": drop * 100.0,
            },
            step=sim_cfg.num_rounds,
        )
        mlflow.set_tag("sanity_verdict", verdict)

        jsonl.close()
        file_handler.flush()
        logger.removeHandler(file_handler)
        file_handler.close()
        mlflow.log_artifact(str(output_dir / "log.jsonl"), artifact_path="logs")
        mlflow.log_artifact(str(console_log_path), artifact_path="logs")


if __name__ == "__main__":
    main()
