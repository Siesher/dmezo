# Robustness matrix — статистическая классификация findings

Расширение `paper_ru.md` §6.9 «Statistical caveats». Каждый empirical claim из paper классифицирован по уровню statistical rigor.

## Классификация

| Tier | Criteria |
|---|---|
| 🟢 **Robust** | Cross-replicated через ≥2 independent measurements (different archs, tasks, or stages); effect-size ≫ noise band; consistent direction |
| 🟡 **Tentative** | Single-seed positive evidence; effect-size on order of noise band; direction consistent with theory; multi-seed validation needed |
| ⚪ **Exploratory** | One-off observation, preliminary; not yet pattern-matched; useful as hypothesis |
| 🔴 **Negative** | Replicated failure / null result; falsification of a stated hypothesis |

## Robust findings (6 шт, +1 since 2026-05-21)

| Finding | Section | Replications | Effect size | Noise band | Why robust |
|---|---|---|---|---|---|
| ⭐ **D-MeZO-N v2 (combo) beats vanilla на MathLogicQA / Qwen3.5-4B-Base** (PROMOTED) | §5.6.2 | **3 seeds × 5 variants paired** | Δ loss = **−5.5% (3/3 same direction)**, Δ acc = **+2.3pp mean**, lowest std (0.010) | seed variance σ_loss ≈ 0.018 для vanilla; effect 4× выше noise band | 3/3 direction consistency loss; multi-seed validated combo beats vanilla. Companion findings: v1 (fixed C=50) **3/3 worse**; B5-alone **3/3 worse**. |
| ε autotuner fails downstream | §6.7 / fig9-12 | 4 (2 archs × 2 stages) | drop 60% vs 13-17% (3-6× factor) | small (autotuner deterministic ε* across seeds) | Cross-arch direction + huge effect-size; bias-proxy fundamentally measures wrong quantity (curvature, not gradient bias) |
| Warmup ε(t) systematically loses | §6.7 / fig13-15 + §6.8 | 16+ cells × 2 variants × cross-arch + 8 cells joint sweep | drop −42% to −147% vs +60-63% const | Consistent across 24+ cells | Cross-task + cross-arch + cross-variant; mechanism (irrecoverable early-step bias) consistent with §6.7 mechanism |
| Batch-variance CLT fails | §6.4 / fig8 | 1 setup (Qwen3-0.6B SST-2) but 6 B-levels | ratio 1.55× → 3.43× monotonic | within-B noise band tight | Monotonic 6-point trend; aligned with theory (z-noise dominates data-noise) |
| Day 5 federated 2×2 grid | §5.2 / fig1 | 2 seeds × 4 cells = 8 runs | partition tax < 13% consistent | seed variance ~2-3% | 2-seed replication on all cells; effect-size > seed variance |
| **v1 (fixed C=50) and B5-alone falsified** (NEW NEGATIVE, robust) | §5.6.2 | 3 seeds paired | v1: +7.0% loss 3/3 worse; B5-alone: +6.4% loss 3/3 worse | direction consistent, large effect | Multi-seed direction consistency; clean falsification |

Plus mathematical: **T1, T2, T3, T4 theorems** (formal proofs, not data-dependent — see `docs/theory_rigorous.md`).

## Tentative findings (2 шт, −2 promoted to Robust 2026-05-21)

| Finding | Section | Replication | Effect size | Risk |
|---|---|---|---|---|
| D-MeZO-N v2 rescue (HellaSwag) | §5.5 | 1 seed | +3.75 pp acc | Single seed; SE on 100-example acc ≈ ±0.045; effect roughly 1σ |
| §6.8 vanilla "wins" (lr=1e-6 + const) | §6.8 / fig18 | 1 seed × 24 cells | drop +1.0% best, gap 1.7 pp to nearest | Within seed variance; near-saturation task limits effect range |

**Validation plan**: `scripts/validate_dmezo_n_rescue_multiseed_federated.py` re-runs §5.5 HellaSwag rescue with 3 seeds + 500-example eval + bootstrap CI. Pending Colab compute (~5 h Blackwell).

**Promoted from Tentative to Robust (2026-05-21):**
- §5.6 D-MeZO-N v2 — now **Robust** with 3-seed paired evidence (was: 1 seed "safe-tracking +1.25pp")
- §5.4 R1d v1 — **Falsified** by multi-seed (moved to Negative findings)

## Exploratory observations (3 шт)

| Finding | Section | Why exploratory |
|---|---|---|
| Refine-below ε(t) (1e-3 → 1e-4) beats const on hybrid by +4.2pp | §6.7 follow-up | Single setup, marginal effect, not validated cross-task |
| K=3 vs K=1 Pareto trade-off (loss WORSE, acc BETTER) | §6.5 | 1 cell, mixed effect; useful as hyperparameter trade-off info |
| Richardson sweet spot at ε≈3e-3 (full-attn) | §6.7 supplement | Doesn't reproduce on hybrid; narrow window |

## Negative findings / null results (3 шт)

| Finding | Section | Hypothesis falsified |
|---|---|---|
| 6-point Romberg-Richardson ≼ 4-point ≼ 2-point at Princeton ε | §6.7 supplement / fig17 | "Higher-order finite-diff reduces variance + bias in fp16 MeZO" |
| Richardson 4-point doesn't rescue large ε | §6.7 supplement | "ε² bias is the bottleneck — Richardson cancels it → large ε works" |
| Autotuner ε* loses to Princeton 1e-3 in downstream | §6.7 / fig12 | "Variance reduction at larger ε translates to faster downstream training" |

Все три — clean falsifications with mechanistic explanation; **полезные** результаты для paper (показывают что наивные подходы не работают и **усиливают** Princeton MeZO default).

## Statistical considerations

### Eval noise estimates

| Pool size | n_examples | SE_acc at p=0.7 |
|---|---|---|
| Day 5 grid | 200 | ±0.032 |
| §6.7 sweeps | 100 (paired with z-seeds) | ±0.045 |
| §6.8 joint sweep | 100 | ±0.045 |
| §5.5 multi-seed validation (planned) | 500 | ±0.020 |

### Loss-drop noise estimates

Loss-drop intervals depend on starting loss. For HellaSwag (L_0 ≈ 2.1, near-saturation):
- 1pp accuracy change → ~0.03 loss change → ~1.5% relative drop
- Seed variance on loss-drop at 500 steps: ~2-3 pp from observed Day 5 multi-seed runs

### Recommended thresholds for confidence

| Confidence | Required |
|---|---|
| Strong claim | ≥3 seeds, CI excludes 0, paired comparison |
| Tentative claim | Single seed, effect-size ≥ 2× SE, direction consistent with theory |
| Speculation | Single seed, effect-size < 2× SE — label as **exploratory** |

## How to upgrade tentative → robust

1. **§5.5 D-MeZO-N rescue** (highest priority): `validate_dmezo_n_rescue_multiseed.py` Section 19 Colab → expected outcome: CI on Δacc that either excludes 0 (CONFIRMED) or includes 0 (DOWNGRADE to directional evidence).

2. **§5.6 MathLogicQA safe-tracking**: cheaper local re-run with 2 seeds; effect-size small but if direction consistent over 2 seeds, useful.

3. **§6.8 vanilla wins claim**: not paper-critical (already softened in §6.8 / §6.9); skip unless first two complete fast.

## Last updated

2026-05-19 — created during honesty pass (commit `b6019d6`) and organization pass (this file).
