# D-MeZO-N: Master experiments summary

Полный chronological + categorical обзор всех experimental runs в проекте. Дополняет `paper_ru.md` (содержательные тексты) и `robustness_matrix.md` (статистический rigor). Для navigation: ищи по разделам paper (§5.x, §6.x) или по дате.

**Notation**: 🟢 robust (cross-replicated), 🟡 tentative (single-seed positive), 🔴 negative/null finding, ⚪ exploratory.

## Day-by-day experiments

| # | Date | Section | Model | Task | Setup | Status | Key result |
|---|---|---|---|---|---|---|---|
| Day 1 | 2026-05-14 | §5.1 | Qwen3-4B | SST-2 | Vanilla sanity, 500 steps, Colab Blackwell | 🟢 | drop 88.1% / 2.4 min |
| Day 4 | 2026-05-15 | §5.2 | Qwen3-4B | SST-2 | 2c complete `weight_avg` federated D-MeZO | 🟡 | final 0.1793 vs centralized 0.17 |
| Day 5 | 2026-05-15 | §5.2 | Qwen3.5-4B-Base hybrid | SST-2 | 4c × {complete, ring} × {IID, Dir(0.5)} grid, 2 seeds | 🟢 | partition tax < 13%; first federated MeZO on linear-attn |
| Day 6 | 2026-05-15 | §5.4 | Qwen3.5-4B-Base | SST-2 worst cell | Nesterov ablation: β=0.9 без clip / clip200 / clip50 | 🟢 | β=0.9 diverges at R140 (noise-amp momentum) |
| Day 6b | 2026-05-15 | §5.4 | Qwen3.5-4B-Base | SST-2 | True look-ahead Nesterov | 🟢 | Look-ahead diverges 7× faster (NaN at R20) |
| Day 8 R1 | 2026-05-15 | §5.4 | Qwen3.5-4B-Base | SST-2 | clip200, single seed | 🟡 | R100 acc=92.5% momentum-accel; slow drift to R500 divergence |
| Day 8 R1b | 2026-05-15 | §5.4 | Qwen3.5-4B-Base | SST-2 | clip50 + const β=0.9 | 🟡 | best 0.119@R300 (3× speedup) + late drift to 0.225 |
| **Day 8 R1d** | 2026-05-15 | §5.4 | Qwen3.5-4B-Base | SST-2 | **D-MeZO-N v1**: β-decay 0.9→0 + clip50 | 🟡 | monotonic descent to 0.1291; beats vanilla 0.1381 by 6.5% |
| HellaSwag | 2026-05-18 | **§5.5** | **Qwen3-4B** | HellaSwag | D-MeZO-N v1, 1000 rounds, 4 clients | 🟡 | vanilla **diverges** (−2.5pp acc), D-MeZO-N **+3.75pp**; single seed |
| MathLogicQA | 2026-05-18 | §5.6 | Qwen3.5-4B-Base | MathLogicQA (Russian) | D-MeZO-N v1, 4 clients | 🟡 | safe-tracking +1.25pp acc; cross-lingual + cross-reasoning generality |
| K=3 ablation | 2026-05-18 | §6.5 | Qwen3.5-4B-Base | SST-2 worst cell | MD-D-MeZO-N K=3 vs K=1 | 🟢 | K=3 loss WORSE +41.6%, acc BETTER +1.25pp; Pareto trade-off |
| Batch variance | 2026-05-15 | §6.4 | Qwen3-0.6B | SST-2 | 100 batches × B ∈ {1,...,32}, fixed z | 🟢 | $1/\sqrt{B}$ CLT **fails**: ratio 1.55× (B=2) → 3.43× (B=32) |
| **§22 paper-scale multi-seed** | **2026-05-21** | **§5.6.2** | **Qwen3.5-4B-Base** | **MathLogicQA** | **3 seeds × 5 variants × 1000 rounds; 15-cell grid; ~12h Blackwell** | 🟢 | **D-MeZO-N v2 (combo) beats vanilla: Δ loss = −5.5% (3/3 same direction), Δ acc = +2.3pp mean. v1 (fixed C=50) and B5-alone robustly falsified (3/3 worse).** ⭐ First paper-scale multi-seed validated D-MeZO-N win |

## Diagnostic ablations (§6.7 family — 2026-05-19)

| # | Section | Model(s) | Setup | Status | Key result |
|---|---|---|---|---|---|
| ε autotuner | §6.7 / fig9-10 | Qwen3-0.6B + Qwen3.5-0.8B | Extended grid ε ∈ {1e-5...1.0}, 30 z-probes | 🟢 | Cross-arch consistent ε* = 1e-1 / 3e-1 |
| ε downstream validation | §6.7 / fig11-12 | same | 100 steps MeZO at multiple ε, eval@clean θ | 🟢 | Princeton 1e-3 **wins** by 3-6× on both archs |
| ε schedule (vanilla) | §6.7 / fig13-14 | Qwen3-0.6B + Qwen3.5-0.8B | 5 log-linear schedules × 100 steps | 🟢 | Warmup loses 16+ cells; refine-below ties const on full-attn, beats on hybrid (+4.2pp) |
| ε schedule (D-MeZO-N) | §6.7 / fig15 | Qwen3-0.6B + Qwen3-1.7B | 3 schedules × {vanilla, dmezo_n} | 🟢 | Same ε(t) ordering preserved under momentum + clip |
| Richardson 4-pt | §6.7 supplement / fig17 | Qwen3-0.6B + Qwen3.5-0.8B | 2-pt vs 4-pt step-eq + compute-eq | 🔴 | Doesn't rescue large ε; narrow sweet spot at ε≈3e-3 (+4.5pp full-attn) |
| 6-pt Romberg | §6.7 supplement / fig17 | Qwen3.5-0.8B | 2-pt vs 4-pt vs 6-pt | 🔴 | 6-pt dominated by 4-pt dominated by 2-pt at Princeton ε |

## Joint sweep (§6.8 — 2026-05-19)

| Cell | Variant | lr | Schedule | Steps | L_drop% | acc_final | Status |
|---|---|---|---|---|---|---|---|
| best vanilla | vanilla | 1e-6 | const 1e-3 | 500 | **+1.0%** | 0.67 | 🟡 |
| best D-MeZO-N | D-MeZO-N | 3e-7 | const 1e-3 | 500 | +0.3% | 0.69 | 🟡 |
| worst overall | D-MeZO-N | 3e-6 | decay→1e-4 | 500 | −631% | 0.28 | 🔴 catastrophic |

(полная 24-cell таблица — `paper_ru.md` §6.8 + Figure 18)

## Pending / future-work experiments

| # | Description | Owner | Compute estimate | Trigger |
|---|---|---|---|---|
| §5.5 multi-seed | 3 seeds × 1000 steps × 500-example eval; bootstrap CI on Δacc | `validate_dmezo_n_rescue_multiseed.py` | ~2.5h Blackwell | Submission rigor (CI executes/excludes 0) |
| §6.4 fix | Re-run `diagnose_batch_variance.py` after rng-reset fix | local Qwen3-0.6B | ~3 мин | Optional; current finding still valid |
| Multi-task §6.8 | Replay best cell on MathLogicQA / BoolQ for cross-task generality | local + Colab | ~2h | Only if first multi-seed positive |

## Compute budget tracking

Approximate Colab Pro+ usage (Blackwell 96 GB):

| Date | Activity | Hours |
|---|---|---|
| 2026-05-14 | Day 1 sanity + Day 4 federated | ~1 |
| 2026-05-15 | Day 5 grid (4 configs × 2 seeds = 8 runs) | ~2 |
| 2026-05-15 | Day 6 / Day 8 Nesterov ablations | ~2 |
| 2026-05-17 | Pre-flight + theory drafting | minimal |
| 2026-05-18 | HellaSwag + MathLogicQA + K=3 ablation | ~3 |
| 2026-05-19 | ε autotuner cross-arch (Qwen3-4B + Qwen3.5-4B-Base) | ~1 |
| 2026-05-19 | Joint sweep §6.8 (vanilla + D-MeZO-N × 12 cells each) | ~3 |
| 2026-05-19 | **§5.5 multi-seed validation (pending)** | ~2.5 |
| **Total** | | **~14h** |

Pro+ ежемесячный лимит ~600 compute units; среднее потребление **~25-40 units/h** в Blackwell. Спален ~350-560 units на этом проекте, остаток сохранён для финального poish.

## Related artefacts

- `experiments/diagnostics/*.json` — all raw experimental data (gitignored, regenerate via scripts)
- `mlruns/` — MLflow tracking for headline runs (Day 1, federated grids, HellaSwag) — gitignored
- `docs/figures/*.png` — see `figures_index.md` for full registry
