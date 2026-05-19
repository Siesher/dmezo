# Scripts index

Все 20 скриптов в `scripts/` сгруппированы здесь по функциональности. Файлы **не перемещаются** в подпапки чтобы не ломать ссылки в `notebooks/bootstrap_colab.ipynb` (24 cells с hardcoded paths) и в configs/. Этот README — единственная категоризация.

## 1. Baseline experiments (sanity + federated)

Эти скрипты запускают каноничные MeZO/D-MeZO-N runs, на которых построены paper §5 results.

| Script | Purpose | Section | Compute |
|---|---|---|---|
| `01_sanity_check_mezo.py` | Day 1 sanity: vanilla MeZO на Qwen3-4B / SST-2, проверка что pipeline работает | §5.1 | ~2.4 мин Blackwell |
| `03_dmezo_federated.py` | n-client federated D-MeZO simulator с consensus mixing (weight_avg / update_share). Поддерживает Nesterov + ρ-clip + β-decay. **Главный workhorse** для §5.2–5.6. | §5.2–5.6 | ~5-30 мин Blackwell |

Configs для этих runs — в `configs/`.

## 2. Diagnostic ablations (variance, ε, scheduling)

Скрипты для §6 paper investigations. Все produce JSON в `experiments/diagnostics/` + figure в `docs/figures/`.

| Script | Purpose | Section / Figure | Compute |
|---|---|---|---|
| `diagnose_batch_variance.py` | Эмпирическая проверка $1/\sqrt{B}$ CLT scaling — finding §6.4 что variance saturates | §6.4 / fig8 | ~3 мин local |
| `diagnose_eps_warmup.py` | Warmup-based ε autotuner: bias-proxy + variance proxy на grid из ε candidates. Сохраняет `eps_warmup_*.json`. | §6.7 / fig9 | ~3 мин per arch |
| `validate_eps_downstream.py` | Downstream-проверка autotuner-выбора ε vs Princeton: 100 шагов MeZO at multiple ε, eval@clean-θ. Showed autotuner ε* loses by 3-6×. | §6.7 / fig11 | ~3 мин per model |
| `compare_eps_warmup_cross_arch.py` | Compose fig10: cross-arch autotuner sweep summary. | §6.7 / fig10 | ~5 sec |

## 3. ε-schedule + finite-diff ablations (§6.7 follow-up)

| Script | Purpose | Section / Figure | Compute |
|---|---|---|---|
| `ablate_eps_schedule.py` | Per-step ε(t) injection in vanilla MeZO. 5 schedules × {Qwen3-0.6B, Qwen3.5-0.8B}. | §6.7 / fig13 | ~5 мин per model |
| `ablate_eps_schedule_dmezo.py` | Same but with `--variant {vanilla, dmezo_n}` — adds Nesterov + ρ-clip. Used on Qwen3-0.6B + Qwen3-1.7B. | §6.7 / fig15 | ~5-10 мин per model |
| `ablate_richardson_vs_2point.py` | 2-point vs 4-point Richardson vs 6-point Romberg-Richardson finite difference, equal-compute comparison. Negative result: higher-order doesn't rescue large ε. | §6.7 supplement / fig17 | ~5-10 мин |

## 4. Joint sweeps (§6.8) + validation

| Script | Purpose | Section / Figure | Compute |
|---|---|---|---|
| `sweep_lr_eps_hellaswag.py` | Joint lr × ε × variant grid (4×3×2 = 24 cells) на HellaSwag/Qwen3.5-4B-Base. **Colab-only** (4B model, 1000+ shards). Headline §6.8 sweep. | §6.8 / fig16/fig18 | ~3h Blackwell |
| `validate_dmezo_n_rescue_multiseed.py` | Multi-seed re-validation §5.5 D-MeZO-N rescue claim: 3 seeds × 2 variants × 1000 steps × 500-example eval. Outputs 95% bootstrap CI on Δacc. **Most important rigor-pass experiment**. | §5.5 follow-up / fig19 | ~2.5h Blackwell |

## 5. Compose / regen scripts (paper figures)

Только matplotlib — компонуют figures из существующих JSON-данных, не запускают модели.

| Script | Purpose | Output |
|---|---|---|
| `compose_fig12_eps_paradox.py` | Cross-arch autotuner vs downstream paradox figure | fig12 |
| `compose_fig14_eps_schedule.py` | Cross-arch ε-schedule ablation composite | fig14 |
| `compose_fig15_dmezo_schedule.py` | D-MeZO-N ε-schedule cross-scale composite | fig15 composite |
| `compose_fig18_joint_sweep_colab.py` | 24-cell joint sweep 2×4 grid summary | fig18 |

## 6. Paper-build utilities

| Script | Purpose |
|---|---|
| `99_build_paper_docx.py` | Convert paper_en.md → .docx (Pandoc-based, EN) |
| `99_build_paper_docx_ru.py` | Convert paper_ru.md → .docx (RU, with OMML math) |
| `99_generate_paper_figures.py` | Batch regen all main paper figures from `experiments/` JSONs |
| `99_generate_new_figures.py` | Aux: regen "new" figures added during revision passes |
| `99_render_equations.py` | LaTeX → OMML conversion helper for docx build |

## Запуск

Все скрипты используют `argparse`; вызывайте `python scripts/<name>.py --help` для list of options. Конвенции:

- Local Windows (5070 Ti / RTX 2080): `uv run --no-sync python scripts/<name>.py ...`
  - `--no-sync` чтобы не перетёрся CUDA torch на CPU build
- Colab Blackwell: см. `notebooks/bootstrap_colab.ipynb`

## Связанные docs

- `docs/experiments_summary.md` — master table всех experimental runs с findings
- `docs/figures_index.md` — registry of all paper figures
- `docs/robustness_matrix.md` — статистическая классификация findings (Robust / Tentative / Exploratory)
- `docs/hyperparameter_strategy.md` — Master's memo about lr, ε, B, K choices
