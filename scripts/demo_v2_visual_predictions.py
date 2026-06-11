"""Visual demo: D-MeZO-N v2 corrects predictions where vanilla MeZO drifts.

Trains BOTH vanilla MeZO and D-MeZO-N v2 (combo B1 adaptive_clip + B5 drift-reset)
on ViT-Large + CUB-200 from the same sklearn warm-start, same seed. Then finds test
images where vanilla MeZO predicts WRONG bird species while D-MeZO-N v2 predicts
CORRECT — i.e. cases where combo's stabilization preserves what overshooting destroys.

Output:
  docs/figures/fig23_v2_visual_demo.png  — 3x3 grid of images + true/vanilla/v2 labels
  experiments/diagnostics/vit_cub200_visual_demo.json — full per-image predictions
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_NAME    = "google/vit-large-patch16-224-in21k"
DATASET_TRAIN = "Multimodal-Fatima/CUB_train"
DATASET_TEST  = "Multimodal-Fatima/CUB_test"
N_CLASSES     = 200
EPS           = 1e-2     # bf16-safe
LR            = 1e-3     # calibrated: 5e-3 catastrophically destabilizes combo (momentum × noise)
N_STEPS       = 5000     # at lr=1e-3 vanilla drifts mildly, combo should stabilize
BATCH_SIZE    = 16
EVAL_EVERY    = 500
SEED          = 42

OUT_JSON = Path("experiments/diagnostics/vit_cub200_visual_demo.json")
OUT_FIG  = Path("docs/figures/fig23_v2_visual_demo.png")
OUT_FIG_TOP3 = Path("docs/figures/fig23b_v2_visual_demo_top3.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ─── Data ────────────────────────────────────────────────────────────────
def load_data():
    print(f"[data] Loading CUB-200 from HF...")
    ds_train = load_dataset(DATASET_TRAIN, split="train")
    ds_test  = load_dataset(DATASET_TEST,  split="test")
    class_names = ds_train.features["label"].names
    print(f"[data]   train={len(ds_train)} test={len(ds_test)} classes={len(class_names)}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return list(ds_train), list(ds_test), class_names, processor


def preprocess(samples, processor):
    images = [s["image"].convert("RGB") for s in samples]
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE, dtype=DTYPE), labels.to(DEVICE)


# ─── Warm-start ──────────────────────────────────────────────────────────
@torch.no_grad()
def warm_start_fit(train_samples, processor):
    print("[warm] Extracting frozen ViT-L CLS features for sklearn LR fit...")
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    backbone.eval()
    feats, labels = [], []
    for i in range(0, len(train_samples), 64):
        batch = train_samples[i:i + 64]
        pv, _ = preprocess(batch, processor)
        out = backbone(pixel_values=pv)
        feats.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
        labels.extend([s["label"] for s in batch])
    del backbone
    torch.cuda.empty_cache()
    X, y = np.concatenate(feats), np.array(labels)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X, y)
    print(f"[warm]   sklearn train acc = {lr.score(X, y):.4f}")
    return (
        torch.tensor(lr.coef_,      dtype=DTYPE, device=DEVICE),
        torch.tensor(lr.intercept_, dtype=DTYPE, device=DEVICE),
    )


def load_model_warm(warm_W, warm_b):
    m = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, ignore_mismatched_sizes=True, dtype=DTYPE,
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


# ─── MeZO primitives ─────────────────────────────────────────────────────
def _perturb(params, seed, eps, sign):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device)
        p.data.add_(sign * eps * z)


def mezo_step(params, loss_fn, eps, lr, seed):
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
    n_steps: int
    beta_start: float = 0.9
    beta_end:   float = 0.0
    clip_quantile: float = 0.95
    clip_alpha:    float = 1.3
    drift_threshold: float = 0.1
    velocity: float = 0.0
    rho_history: deque = field(default_factory=lambda: deque(maxlen=50))
    rolling_min: float = float("inf")
    n_drift_resets: int = 0

    def current_beta(self, t):
        f = min(1.0, t / max(1, self.n_steps))
        return self.beta_start * (1 - f) + self.beta_end * f

    def current_clip(self):
        if len(self.rho_history) < 5:
            return float("inf")
        return self.clip_alpha * float(np.quantile(np.abs(self.rho_history), self.clip_quantile))

    def maybe_drift_reset(self, monitored_loss):
        if monitored_loss < self.rolling_min:
            self.rolling_min = monitored_loss
            return False
        if monitored_loss > self.rolling_min + self.drift_threshold:
            self.velocity = 0.0
            self.rolling_min = monitored_loss
            self.n_drift_resets += 1
            return True
        return False


def dmezo_n_step(params, loss_fn, eps, lr, seed, state, t):
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


# ─── Eval + train loops ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_samples, processor, batch_size=32):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i + batch_size]
        pv, labels = preprocess(batch, processor)
        preds = model(pixel_values=pv).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(batch)
    return correct / max(total, 1)


def train_vanilla(model, train_samples, test_samples, processor, seed):
    params = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(N_STEPS):
        idxs = [rng.randrange(len(train_samples)) for _ in range(BATCH_SIZE)]
        batch = [train_samples[i] for i in idxs]
        pv, lb = preprocess(batch, processor)

        def loss_fn(pv=pv, lb=lb):
            return model(pixel_values=pv, labels=lb).loss

        g = mezo_step(params, loss_fn, EPS, LR, rng.randint(0, 2**31 - 1))

        if step % EVAL_EVERY == 0:
            acc = evaluate(model, test_samples, processor)
            print(f"  [vanilla] step={step:5d}  acc={acc:.4f}  g={g:+.3e}  ({time.time()-t0:.0f}s)")


def train_dmezo_n(model, train_samples, test_samples, processor, seed):
    params = [p for p in model.parameters() if p.requires_grad]
    rng = random.Random(seed)
    state = DMezoState(n_steps=N_STEPS)
    t0 = time.time()
    for step in range(N_STEPS):
        idxs = [rng.randrange(len(train_samples)) for _ in range(BATCH_SIZE)]
        batch = [train_samples[i] for i in idxs]
        pv, lb = preprocess(batch, processor)

        def loss_fn(pv=pv, lb=lb):
            return model(pixel_values=pv, labels=lb).loss

        rho_raw, _, beta, C = dmezo_n_step(
            params, loss_fn, EPS, LR, rng.randint(0, 2**31 - 1), state, step
        )

        if step % EVAL_EVERY == 0:
            train_loss = loss_fn().item()
            acc = evaluate(model, test_samples, processor)
            was_reset = state.maybe_drift_reset(train_loss)
            tag = " [RESET]" if was_reset else ""
            C_str = f"{C:.3f}" if np.isfinite(C) else "inf"
            print(f"  [v2]      step={step:5d}  acc={acc:.4f}  ρ={rho_raw:+.3e}  "
                  f"C={C_str}  β={beta:.2f}  resets={state.n_drift_resets}  ({time.time()-t0:.0f}s){tag}")


# ─── Prediction analysis ─────────────────────────────────────────────────
@torch.no_grad()
def get_all_predictions(model, samples, processor, batch_size=32):
    model.eval()
    all_preds, all_confs, all_labels = [], [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        pv, lb = preprocess(batch, processor)
        logits = model(pixel_values=pv).logits.float()
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_confs.append(confs.cpu().numpy())
        all_labels.append(lb.cpu().numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_confs),
        np.concatenate(all_labels),
    )


def short_name(name: str, max_len: int = 22) -> str:
    """Truncate long species names for plot titles."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def find_disagreement_cases(v_preds, v_confs, d_preds, d_confs, labels):
    """vanilla WRONG & v2 RIGHT, ranked by combined drama (both confident)."""
    mask = (v_preds != labels) & (d_preds == labels)
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return []
    # Score: dramatic = vanilla wrong with high conf + v2 right with high conf
    scores = d_confs[candidates] + v_confs[candidates]
    return candidates[np.argsort(-scores)].tolist()


def plot_montage(test_samples, class_names, candidates, v_preds, v_confs,
                 d_preds, d_confs, labels, out_path, ncols=3, nrows=3, title_suffix=""):
    n = ncols * nrows
    picks = candidates[:n]
    if len(picks) < n:
        print(f"  Only {len(picks)} disagreement cases available, using all")
        nrows = max(1, (len(picks) + ncols - 1) // ncols)
        n = nrows * ncols
        picks = picks[:n]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(nrows, ncols)
    fig.suptitle(
        f"ViT-Large + CUB-200: D-MeZO-N v2 picks correct species where vanilla MeZO drifts{title_suffix}",
        fontsize=13,
    )
    for ax, idx in zip(axes.flat, picks):
        img = test_samples[idx]["image"].convert("RGB")
        true_name = short_name(class_names[labels[idx]])
        v_name    = short_name(class_names[v_preds[idx]])
        d_name    = short_name(class_names[d_preds[idx]])
        ax.imshow(img)
        ax.axis("off")
        title = (
            f"True: {true_name}\n"
            f"Vanilla: {v_name}  ({v_confs[idx]*100:.0f}%, wrong)\n"
            f"D-MeZO-N v2: {d_name}  ({d_confs[idx]*100:.0f}%, correct)"
        )
        ax.set_title(title, fontsize=9, pad=4, loc="left")
    # Hide unused axes
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


def main():
    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    print(f"Config: model={MODEL_NAME}  lr={LR}  eps={EPS}  steps={N_STEPS}  seed={SEED}")

    train_samples, test_samples, class_names, processor = load_data()
    warm_W, warm_b = warm_start_fit(train_samples, processor)

    # ── Warm baseline (sklearn LR ceiling on head-only refinement) ──
    print("\n[base] Warm baseline acc on full 5794 test set...")
    m_warm = load_model_warm(warm_W, warm_b)
    warm_acc = evaluate(m_warm, test_samples, processor)
    print(f"[base]   warm baseline = {warm_acc:.4f}")
    del m_warm
    torch.cuda.empty_cache()

    # ── Train vanilla MeZO ──
    print(f"\n{'=' * 60}\nTrain vanilla MeZO (seed={SEED}, {N_STEPS} steps)\n{'=' * 60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m_vanilla = load_model_warm(warm_W, warm_b)
    train_vanilla(m_vanilla, train_samples, test_samples, processor, SEED)
    vanilla_acc = evaluate(m_vanilla, test_samples, processor)
    print(f"  vanilla final acc = {vanilla_acc:.4f}  (Δ vs warm = {vanilla_acc - warm_acc:+.4f})")

    # Cache vanilla predictions
    print("[pred] Gathering vanilla predictions on full test...")
    v_preds, v_confs, labels = get_all_predictions(m_vanilla, test_samples, processor)
    del m_vanilla
    torch.cuda.empty_cache()

    # ── Train D-MeZO-N v2 ──
    print(f"\n{'=' * 60}\nTrain D-MeZO-N v2 (seed={SEED}, same config)\n{'=' * 60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m_v2 = load_model_warm(warm_W, warm_b)
    train_dmezo_n(m_v2, train_samples, test_samples, processor, SEED)
    v2_acc = evaluate(m_v2, test_samples, processor)
    print(f"  v2 final acc = {v2_acc:.4f}  (Δ vs warm = {v2_acc - warm_acc:+.4f})  "
          f"(Δ vs vanilla = {v2_acc - vanilla_acc:+.4f})")

    print("[pred] Gathering v2 predictions on full test...")
    d_preds, d_confs, _ = get_all_predictions(m_v2, test_samples, processor)
    del m_v2
    torch.cuda.empty_cache()

    # ── Find disagreement cases & save montage ──
    candidates = find_disagreement_cases(v_preds, v_confs, d_preds, d_confs, labels)
    print(f"\n[viz] Cases where vanilla WRONG & v2 RIGHT: {len(candidates)} / {len(labels)}")
    if len(candidates) == 0:
        print("[viz] No disagreement found — check hyperparameters")
        return
    print(f"[viz] Top 9 (by combined confidence):")
    for r, idx in enumerate(candidates[:9]):
        print(f"  #{r+1}  test_idx={idx}  "
              f"true={class_names[labels[idx]]!r}  "
              f"vanilla={class_names[v_preds[idx]]!r}({v_confs[idx]:.2f})  "
              f"v2={class_names[d_preds[idx]]!r}({d_confs[idx]:.2f})")

    plot_montage(test_samples, class_names, candidates, v_preds, v_confs,
                 d_preds, d_confs, labels, OUT_FIG, ncols=3, nrows=3)
    print(f"[viz] Saved 3x3 grid to {OUT_FIG}")
    plot_montage(test_samples, class_names, candidates, v_preds, v_confs,
                 d_preds, d_confs, labels, OUT_FIG_TOP3, ncols=3, nrows=1,
                 title_suffix=" (top 3)")
    print(f"[viz] Saved 1x3 strip to {OUT_FIG_TOP3}")

    # ── Save full per-image predictions for reproducibility ──
    payload = {
        "config": {"model": MODEL_NAME, "dataset": "CUB-200 (5994/5794)",
                   "lr": LR, "eps": EPS, "n_steps": N_STEPS, "seed": SEED},
        "warm_baseline_acc": warm_acc,
        "vanilla_final_acc": vanilla_acc,
        "v2_final_acc": v2_acc,
        "delta_v2_vs_vanilla": v2_acc - vanilla_acc,
        "n_disagreement_cases": len(candidates),
        "top_cases": [
            {
                "test_idx": int(idx),
                "true_class": class_names[labels[idx]],
                "vanilla_pred": class_names[v_preds[idx]],
                "vanilla_conf": float(v_confs[idx]),
                "v2_pred": class_names[d_preds[idx]],
                "v2_conf": float(d_confs[idx]),
            } for idx in candidates[:30]
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {OUT_JSON}")

    print(f"\n{'=' * 60}\nHEADLINE\n{'=' * 60}")
    print(f"  warm baseline       = {warm_acc:.4f}")
    print(f"  vanilla MeZO final  = {vanilla_acc:.4f}  (Δ vs warm = {vanilla_acc - warm_acc:+.4f})")
    print(f"  D-MeZO-N v2 final   = {v2_acc:.4f}  (Δ vs warm = {v2_acc - warm_acc:+.4f})")
    print(f"  STRICT WIN          = {v2_acc - vanilla_acc:+.4f} pp  "
          f"({(v2_acc - vanilla_acc) * 100:+.2f}pp)")
    print(f"  Disagreement images = {len(candidates)}  (vanilla wrong, v2 correct)")


if __name__ == "__main__":
    main()
