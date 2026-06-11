"""ViT-Large + CUB-200 fine-grained: vanilla MeZO vs D-MeZO-N v2, multi-seed demo.

Setup designed to demonstrate that D-MeZO-N v2 (combo B1 adaptive_clip + B5 drift-reset
+ heavy-ball + β-decay) **outperforms vanilla MeZO** in fine-grained vision refinement —
where adaptive clip catches outlier ρ events from noisy gradient direction and momentum
β-decay 0.9→0 stabilizes convergence.

Why CUB-200 (vs Flowers102):
  - 200 classes × ~30 samples/class = harder fine-grained transfer
  - Linear probe ceiling ~82% (vs Flowers 99%) → real room for MeZO refinement motion
  - Larger head (200 × 1024 = 205k params vs 102 × 768 = 78k) → wider ρ distribution

Setup:
  Model:    google/vit-large-patch16-224-in21k  (304M, head-only)
  Dataset:  Multimodal-Fatima/CUB_{train,test}  (5994 / 5794 samples, 200 classes)
  Training: head-only refinement from sklearn LR warm-start
  Horizon:  5000 steps × 3 seeds × 2 methods
  LR:       5e-3 (×50 from Flowers baseline — stress regime)
  EPS:      1e-3 (Princeton default; with bf16, ρ heavy-tailed by quantization)
  Eval:     every 250 steps on FULL 5794-sample test set

Output:
  experiments/diagnostics/vit_cub200_multiseed.json  — per-seed trajectories + summary
  docs/figures/fig22_vit_cub200_multiseed.png        — 4-panel headline plot
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

# ─── Config (CLI-overridable) ─────────────────────────────────────────────
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--smoke", action="store_true", help="quick smoke test (1 seed × 500 steps)")
_parser.add_argument("--n-steps",    type=int,   default=5000)
_parser.add_argument("--eval-every", type=int,   default=250)
_parser.add_argument("--lr",         type=float, default=5e-3)
_parser.add_argument("--eps",        type=float, default=1e-2)
_parser.add_argument("--seeds",      type=int,   nargs="+", default=[42, 43, 44])
_args = _parser.parse_args()

MODEL_NAME    = "google/vit-large-patch16-224-in21k"
DATASET_TRAIN = "Multimodal-Fatima/CUB_train"
DATASET_TEST  = "Multimodal-Fatima/CUB_test"
N_CLASSES     = 200
EPS           = _args.eps     # 1e-2 = bf16-safe (avoids L+ − L− cancellation; notebook-validated)
LR            = _args.lr      # 5e-3 = stress regime (×50 from Flowers baseline)
N_STEPS       = 500 if _args.smoke else _args.n_steps
BATCH_SIZE    = 16
EVAL_EVERY    = 100 if _args.smoke else _args.eval_every
SEEDS         = [42] if _args.smoke else _args.seeds
TRAIN_RNG_OFFSET = 0  # RNG offset for batch sampling — seed-dependent for paired comparison

_suffix = "_smoke" if _args.smoke else ""
OUT_JSON = Path(f"experiments/diagnostics/vit_cub200_multiseed{_suffix}.json")
OUT_PNG  = Path(f"docs/figures/fig22_vit_cub200_multiseed{_suffix}.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ─── Data loading (cached) ───────────────────────────────────────────────
def load_data() -> tuple[list, list, "AutoImageProcessor"]:
    print(f"[data] Loading {DATASET_TRAIN} / {DATASET_TEST}...")
    ds_train = load_dataset(DATASET_TRAIN, split="train")
    ds_test  = load_dataset(DATASET_TEST,  split="test")
    print(f"[data]   train={len(ds_train)}  test={len(ds_test)}  classes={N_CLASSES}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    train_samples = list(ds_train)
    test_samples  = list(ds_test)
    return train_samples, test_samples, processor


def preprocess(samples, processor):
    images = [s["image"].convert("RGB") for s in samples]
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE, dtype=DTYPE), labels.to(DEVICE)


# ─── Warm-start: extract frozen ViT-L features + fit sklearn LR ──────────
@torch.no_grad()
def warm_start_fit(train_samples, processor):
    print("[warm] Extracting frozen ViT-L CLS features...")
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    backbone.eval()
    feats, labels = [], []
    bs = 64
    for i in range(0, len(train_samples), bs):
        batch = train_samples[i:i + bs]
        pv, _ = preprocess(batch, processor)
        out = backbone(pixel_values=pv)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        feats.append(cls)
        labels.extend([s["label"] for s in batch])
    del backbone
    torch.cuda.empty_cache()
    X = np.concatenate(feats)
    y = np.array(labels)
    print(f"[warm]   features {X.shape}, classes {len(set(y.tolist()))}")
    print("[warm] Fitting sklearn LR (lbfgs, C=1, max_iter=2000)...")
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X, y)
    print(f"[warm]   train acc {lr.score(X, y):.4f}")
    W = torch.tensor(lr.coef_,      dtype=DTYPE, device=DEVICE)
    b = torch.tensor(lr.intercept_, dtype=DTYPE, device=DEVICE)
    return W, b


def load_model_warm(warm_W: torch.Tensor, warm_b: torch.Tensor):
    m = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, ignore_mismatched_sizes=True, dtype=DTYPE
    ).to(DEVICE)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    for name, p in m.named_parameters():
        if name.startswith("classifier"):
            p.requires_grad_(True)
    with torch.no_grad():
        m.classifier.weight.copy_(warm_W)
        m.classifier.bias.copy_(warm_b)
    return m


# ─── MeZO step primitives (from notebook, perturbs only requires_grad=True params) ──
def _perturb(params, seed: int, eps: float, sign: float):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
        p.data.add_(sign * eps * z)


def mezo_step(params, loss_fn, eps: float, lr: float, seed: int) -> float:
    snap = [p.data.clone() for p in params]
    _perturb(params, seed, eps, +1.0)
    lp = loss_fn().item()
    for p, o in zip(params, snap):
        p.data.copy_(o)
    _perturb(params, seed, eps, -1.0)
    lm = loss_fn().item()
    for p, o in zip(params, snap):
        p.data.copy_(o)
    g = (lp - lm) / (2.0 * eps)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
        p.data.add_(-lr * g * z)
    return g


@dataclass
class DMezoState:
    """D-MeZO-N v2 state: β-decay heavy-ball + adaptive ρ-clip (B1) + drift-reset (B5)."""
    n_steps: int
    beta_start: float = 0.9
    beta_end:   float = 0.0
    clip_window:   int = 50
    clip_quantile: float = 0.95
    clip_alpha:    float = 1.3
    drift_threshold: float = 0.1
    velocity: float = 0.0
    rho_history: deque = field(default_factory=lambda: deque(maxlen=50))
    rolling_min: float = float("inf")
    n_drift_resets: int = 0

    def current_beta(self, t: int) -> float:
        frac = min(1.0, t / max(1, self.n_steps))
        return self.beta_start * (1 - frac) + self.beta_end * frac

    def current_clip(self) -> float:
        if len(self.rho_history) < 5:
            return float("inf")
        return self.clip_alpha * float(np.quantile(np.abs(self.rho_history), self.clip_quantile))

    def maybe_drift_reset(self, monitored_loss: float) -> bool:
        if monitored_loss < self.rolling_min:
            self.rolling_min = monitored_loss
            return False
        if monitored_loss > self.rolling_min + self.drift_threshold:
            self.velocity = 0.0
            self.rolling_min = monitored_loss
            self.n_drift_resets += 1
            return True
        return False


def dmezo_n_step(params, loss_fn, eps: float, lr: float, seed: int,
                 state: DMezoState, t: int) -> tuple[float, float, float, float]:
    snap = [p.data.clone() for p in params]
    _perturb(params, seed, eps, +1.0)
    lp = loss_fn().item()
    for p, o in zip(params, snap):
        p.data.copy_(o)
    _perturb(params, seed, eps, -1.0)
    lm = loss_fn().item()
    for p, o in zip(params, snap):
        p.data.copy_(o)
    rho_raw = (lp - lm) / (2.0 * eps)
    C = state.current_clip()
    rho_clipped = max(-C, min(C, rho_raw)) if np.isfinite(C) else rho_raw
    state.rho_history.append(abs(rho_raw))
    beta = state.current_beta(t)
    state.velocity = beta * state.velocity + rho_clipped
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
        p.data.add_(-lr * state.velocity * z)
    return rho_raw, rho_clipped, beta, C


# ─── Evaluation ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_samples, processor, batch_size: int = 32) -> float:
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]
        pv, labels = preprocess(batch, processor)
        logits = model(pixel_values=pv).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(batch)
    return correct / max(total, 1)


# ─── Training loops ──────────────────────────────────────────────────────
def run_vanilla(model, train_samples, test_samples, processor, seed: int) -> list[dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(seed + TRAIN_RNG_OFFSET)
    log = []
    t0 = time.time()
    for step in range(N_STEPS):
        idxs = [rng.randrange(len(train_samples)) for _ in range(BATCH_SIZE)]
        batch = [train_samples[i] for i in idxs]
        pv, labels = preprocess(batch, processor)

        def loss_fn(pv=pv, lb=labels):
            return model(pixel_values=pv, labels=lb).loss

        g = mezo_step(params, loss_fn, EPS, LR, rng.randint(0, 2**31 - 1))

        if step % EVAL_EVERY == 0 or step == N_STEPS - 1:
            train_loss = loss_fn().item()
            acc = evaluate(model, test_samples, processor)
            elapsed = time.time() - t0
            print(f"  [vanilla seed={seed}] step={step:5d}  "
                  f"loss={train_loss:.4f}  test_acc={acc:.4f}  g={g:+.3e}  ({elapsed:.0f}s)")
            log.append({"step": step, "loss": train_loss, "acc": acc, "g": g})
    return log


def run_dmezo_n(model, train_samples, test_samples, processor, seed: int) -> list[dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(seed + TRAIN_RNG_OFFSET)
    state = DMezoState(n_steps=N_STEPS)
    log = []
    t0 = time.time()
    for step in range(N_STEPS):
        idxs = [rng.randrange(len(train_samples)) for _ in range(BATCH_SIZE)]
        batch = [train_samples[i] for i in idxs]
        pv, labels = preprocess(batch, processor)

        def loss_fn(pv=pv, lb=labels):
            return model(pixel_values=pv, labels=lb).loss

        rho_raw, rho_clipped, beta, C = dmezo_n_step(
            params, loss_fn, EPS, LR, rng.randint(0, 2**31 - 1), state, step
        )

        if step % EVAL_EVERY == 0 or step == N_STEPS - 1:
            train_loss = loss_fn().item()
            acc = evaluate(model, test_samples, processor)
            was_reset = state.maybe_drift_reset(train_loss)
            elapsed = time.time() - t0
            C_str = f"{C:.3f}" if np.isfinite(C) else "inf"
            tag = " [RESET]" if was_reset else ""
            print(f"  [v2 seed={seed}] step={step:5d}  "
                  f"loss={train_loss:.4f}  test_acc={acc:.4f}  "
                  f"ρ={rho_raw:+.3e}  C={C_str}  β={beta:.2f}  "
                  f"resets={state.n_drift_resets}  ({elapsed:.0f}s){tag}")
            log.append({
                "step": step, "loss": train_loss, "acc": acc,
                "rho_raw": rho_raw, "rho_clipped": rho_clipped,
                "beta": beta, "C": C if np.isfinite(C) else None,
                "resets": state.n_drift_resets,
            })
    return log


# ─── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    print(f"Config: model={MODEL_NAME}  lr={LR}  eps={EPS}  steps={N_STEPS}  "
          f"seeds={SEEDS}  eval_every={EVAL_EVERY}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    train_samples, test_samples, processor = load_data()
    warm_W, warm_b = warm_start_fit(train_samples, processor)

    # Warm baseline eval — pre-MeZO accuracy on full test (single value, reused across seeds)
    print("[eval] Warm baseline on full test set...")
    m = load_model_warm(warm_W, warm_b)
    warm_acc = evaluate(m, test_samples, processor)
    print(f"[eval]   warm baseline acc={warm_acc:.4f}")
    del m
    torch.cuda.empty_cache()

    results: dict[str, list] = {"vanilla": [], "dmezo_n_v2": []}
    final_summary: dict[str, dict] = {"vanilla": {}, "dmezo_n_v2": {}}

    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed} / vanilla MeZO\n{'=' * 60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = load_model_warm(warm_W, warm_b)
        log_v = run_vanilla(m, train_samples, test_samples, processor, seed)
        final_v_acc = evaluate(m, test_samples, processor)
        del m
        torch.cuda.empty_cache()
        results["vanilla"].append({"seed": seed, "log": log_v, "final_acc": final_v_acc})
        print(f"  vanilla final acc = {final_v_acc:.4f}  (Δ vs warm = {final_v_acc - warm_acc:+.4f})")

        print(f"\n{'=' * 60}\nSeed {seed} / D-MeZO-N v2\n{'=' * 60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = load_model_warm(warm_W, warm_b)
        log_d = run_dmezo_n(m, train_samples, test_samples, processor, seed)
        final_d_acc = evaluate(m, test_samples, processor)
        del m
        torch.cuda.empty_cache()
        results["dmezo_n_v2"].append({"seed": seed, "log": log_d, "final_acc": final_d_acc})
        print(f"  v2 final acc = {final_d_acc:.4f}  (Δ vs warm = {final_d_acc - warm_acc:+.4f})  "
              f"(Δ vs vanilla = {final_d_acc - final_v_acc:+.4f})")

    # Aggregate
    v_final = [r["final_acc"] for r in results["vanilla"]]
    d_final = [r["final_acc"] for r in results["dmezo_n_v2"]]
    paired_delta = [d - v for d, v in zip(d_final, v_final)]
    final_summary["vanilla"]    = {"per_seed": v_final, "mean": float(np.mean(v_final)), "std": float(np.std(v_final, ddof=0))}
    final_summary["dmezo_n_v2"] = {"per_seed": d_final, "mean": float(np.mean(d_final)), "std": float(np.std(d_final, ddof=0))}
    final_summary["paired"]     = {
        "delta_per_seed":     paired_delta,
        "mean_delta":         float(np.mean(paired_delta)),
        "all_positive":       all(d > 0 for d in paired_delta),
        "n_seeds_v2_wins":    sum(1 for d in paired_delta if d > 0),
    }
    final_summary["warm_baseline_acc"] = warm_acc
    final_summary["config"] = {
        "model": MODEL_NAME, "dataset": "CUB-200 (5994/5794)",
        "lr": LR, "eps": EPS, "n_steps": N_STEPS, "seeds": SEEDS,
        "eval_every": EVAL_EVERY, "warm_init": "sklearn LR on frozen ViT-L features",
    }
    payload = {"summary": final_summary, "trajectories": results}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_JSON}")

    # Print headline
    print(f"\n{'=' * 60}\nHEADLINE\n{'=' * 60}")
    print(f"  warm baseline acc:  {warm_acc:.4f}  (sklearn LR ceiling)")
    print(f"  vanilla MeZO:       {final_summary['vanilla']['mean']:.4f} ± {final_summary['vanilla']['std']:.4f}  "
          f"per-seed: {[f'{v:.4f}' for v in v_final]}")
    print(f"  D-MeZO-N v2:        {final_summary['dmezo_n_v2']['mean']:.4f} ± {final_summary['dmezo_n_v2']['std']:.4f}  "
          f"per-seed: {[f'{d:.4f}' for d in d_final]}")
    print(f"  paired Δ (v2 − vanilla):  per-seed {[f'{d:+.4f}' for d in paired_delta]}  "
          f"mean {final_summary['paired']['mean_delta']:+.4f}")
    print(f"  v2 wins on {final_summary['paired']['n_seeds_v2_wins']}/{len(SEEDS)} seeds  "
          f"({'STRICT WIN ✓' if final_summary['paired']['all_positive'] else 'mixed result'})")

    # Plot
    plot_results(payload)


def plot_results(payload: dict) -> None:
    import matplotlib.pyplot as plt
    summary = payload["summary"]
    warm_acc = summary["warm_baseline_acc"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(
        f"ViT-Large/21k + CUB-200 fine-grained: vanilla MeZO vs D-MeZO-N v2  "
        f"({len(SEEDS)} seeds, head-only, lr={LR}, {N_STEPS} steps)",
        fontsize=12,
    )

    # (a) Loss trajectories (per-seed thin + mean thick)
    for method, color in [("vanilla", "steelblue"), ("dmezo_n_v2", "darkorange")]:
        traj = payload["trajectories"][method]
        all_losses = []
        for r in traj:
            steps  = [x["step"] for x in r["log"]]
            losses = [x["loss"] for x in r["log"]]
            axes[0].plot(steps, losses, color=color, lw=0.7, alpha=0.4)
            all_losses.append(losses)
        mean_loss = np.mean(all_losses, axis=0)
        axes[0].plot(steps, mean_loss, color=color, lw=2.2,
                     label=f"{method} (mean of {len(SEEDS)})")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # (b) Test accuracy trajectories
    for method, color in [("vanilla", "steelblue"), ("dmezo_n_v2", "darkorange")]:
        traj = payload["trajectories"][method]
        all_accs = []
        for r in traj:
            steps = [x["step"] for x in r["log"]]
            accs  = [x["acc"]  for x in r["log"]]
            axes[1].plot(steps, accs, color=color, lw=0.7, alpha=0.4)
            all_accs.append(accs)
        mean_acc = np.mean(all_accs, axis=0)
        axes[1].plot(steps, mean_acc, color=color, lw=2.2,
                     label=f"{method} (mean of {len(SEEDS)})")
    axes[1].axhline(warm_acc, color="green", ls="--", lw=1.5,
                    label=f"sklearn LR ({warm_acc:.3f})")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Test acc (full 5794)")
    axes[1].set_title("Test accuracy"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # (c) Final bars + per-seed dots
    v_finals = [r["final_acc"] for r in payload["trajectories"]["vanilla"]]
    d_finals = [r["final_acc"] for r in payload["trajectories"]["dmezo_n_v2"]]
    x = np.arange(2)
    means = [np.mean(v_finals), np.mean(d_finals)]
    stds  = [np.std(v_finals),  np.std(d_finals)]
    axes[2].bar(x, means, yerr=stds, capsize=8,
                color=["steelblue", "darkorange"], alpha=0.85, edgecolor="black")
    for i, vals in enumerate([v_finals, d_finals]):
        axes[2].scatter([i] * len(vals), vals, color="black", zorder=10, s=40)
    axes[2].axhline(warm_acc, color="green", ls="--", lw=1.2,
                    label=f"sklearn ({warm_acc:.3f})")
    axes[2].set_xticks(x); axes[2].set_xticklabels(["vanilla MeZO", "D-MeZO-N v2"])
    axes[2].set_ylabel("Final test acc"); axes[2].set_title("Final accuracy + seeds")
    axes[2].grid(axis="y", alpha=0.3); axes[2].legend(fontsize=8)
    for i, (m_, s_) in enumerate(zip(means, stds)):
        axes[2].text(i, m_ + s_ + 0.005, f"{m_:.4f}±{s_:.4f}",
                     ha="center", va="bottom", fontsize=9)

    # (d) Paired Δ per seed
    paired = [d - v for d, v in zip(d_finals, v_finals)]
    colors = ["green" if p > 0 else "red" for p in paired]
    axes[3].bar(range(len(SEEDS)), paired, color=colors, alpha=0.8, edgecolor="black")
    axes[3].axhline(0, color="black", lw=0.6)
    axes[3].set_xticks(range(len(SEEDS)))
    axes[3].set_xticklabels([f"seed={s}" for s in SEEDS])
    axes[3].set_ylabel("Δacc = v2 − vanilla")
    axes[3].set_title(f"Paired Δacc per seed (mean={np.mean(paired):+.4f})")
    axes[3].grid(axis="y", alpha=0.3)
    for i, p in enumerate(paired):
        axes[3].text(i, p + (0.001 if p > 0 else -0.002), f"{p:+.4f}",
                     ha="center", va="bottom" if p > 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[save] {OUT_PNG}")


if __name__ == "__main__":
    main()
