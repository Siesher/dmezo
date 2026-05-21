# CLAUDE.md — Decentralized Federated MeZO (D-MeZO)

Этот файл — рабочая память для Claude Code на этом проекте. Читай его в начале сессии.

## TL;DR проекта

Исследовательский проект на стыке zeroth-order оптимизации, federated learning, LLM fine-tuning и differential privacy. Главный артефакт — **D-MeZO-N v2 = combo (adaptive ρ-clip B1 + drift-reset B5)**, peer-to-peer decentralized federated ZO оптимизатор для дообучения LLM с формальными convergence theorems (T3 closes Princeton OP1) и DP-гарантией (T4).

**Ключевые отличия от FedKSeed (closest competitor):**
1. **Independent z_i per client** (не shared seed) → 1/n variance speedup по обеим компонентам шума (data + direction)
2. Peer-to-peer (any doubly-stochastic W), не star topology
3. Heavy-ball momentum stabilization с формальной Lyapunov-сходимостью
4. Dual-use ρ-clip как L2-sensitivity для Gaussian-mechanism DP

## Текущее состояние проекта (2026-05-21)

Этап: **paper-scale multi-seed validation FINALIZED, defense kit ready**. Защита 2026-05-23 (Bauman MSTU Калуга).

**Headline:** На Qwen3.5-4B-Base / MathLogicQA / 3 seeds paired D-MeZO-N v2 (combo B1+B5) robustly beats vanilla MeZO: Δ loss = −5.5% (3/3 same direction), Δ acc = +2.3pp mean, lowest std loss across семейства методов с моментом (0.010 vs vanilla 0.018). Первое paper-scale multi-seed validated empirical улучшение D-MeZO над vanilla MeZO. См. `docs/multiseed_analysis.md` §22.

**Что готово:**

- Core MeZO step (`src/dmezo/mezo/`) — adaptive clip + drift-reset + β-decay + DP noise. Princeton-style perturbation invariants сохранены.
- Federated simulator (`src/dmezo/federated/`) — independent per-client RNG (см. `client.py:62`). Поддерживает `weight_avg` (полный exchange) и `update_share` (16 байт/раунд) consensus modes. **Note:** `update_share + Nesterov` not yet integrated (NotImplementedError) — используем `weight_avg` для momentum runs.
- Tests: 128/128 pass, ~95% coverage critical paths.
- Configs: 28+ YAML files в `configs/`.
- Experiments: 28+ runs documented в `docs/experiments_summary.md`. Headline run — §22 paper-scale (3 seeds × 5 variants × 1000 rounds на Qwen3.5-4B-Base / MathLogicQA), ~12h Colab Blackwell.
- Defense kit: `docs/defense_*.md` (4 docs) — talking points, design brief для Claude Design, paper patches, FedKSeed Q&A. Plus `docs/math_intuition.md` для plain-language explanation.
- Theorems: T1 (convex+momentum+decentralized), T2 (PL no momentum), T3 (PL + heavy-ball + clip + β-decay, closes Princeton OP1), T4 (DP extension of T3). Полные доказательства в `docs/theory_rigorous.md`.
- Experiment tracking: **MLflow** (file backend, `./mlruns/`). НЕ предлагай wandb/Aim/TensorBoard.

**Локальная разработка на CUDA torch.** Текущее железо: **RTX 5070 Ti Blackwell (sm_120, 17 GB VRAM)** + Ryzen 9 9950X + 32 GB RAM, Windows 11 + PowerShell. Установлен `torch==2.12.0+cu130` через `uv pip install --index-url https://download.pytorch.org/whl/cu130 --reinstall-package torch`. Так как `pyproject.toml` не пинит CUDA-вариант, любой `uv run` (он же `uv sync`) перезатирает torch на CPU build. **Использовать `uv run --no-sync ...` для всех локальных команд**, иначе CUDA torch исчезнет.

**Capabilities новой машины** vs прежней RTX 2080 (история): Blackwell нативно поддерживает bf16 (`torch.cuda.is_bf16_supported() == True`), 17 GB VRAM позволяет локально загружать модели до Qwen3.5-4B-Base (~9.4 GB bf16); ранее на 2080 (8 GB Turing) максимум был Qwen3-0.6B. **`triton-windows` + `flash-linear-attention` локально установлены и работают** (2026-05-20, см. `docs/windows_fla_install.md`). Install: `pip install triton-windows && pip install flash-linear-attention`. Speedup подтверждён эмпирически: Qwen3.5-0.8B forward ~30-40 ms warmed up vs ~5 sec cold-start Triton compilation. **`causal-conv1d`** install BROKEN (не критично — gated DeltaNet kernels работают без него; Python-only fallback **ломает transformers Qwen3_5 import** — не ставить). **`flash-attn`** build broken на Windows + cu130 + Py3.13 — не критично, full-attention layers покрываются PyTorch SDPA. Несмотря на warning "fast path not available" при загрузке Qwen3.5, реальные fla-Triton kernels работают.

**Локальные модели, протестированные на 5070 Ti**: Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Qwen3.5-4B-Base (tight по памяти, но работает с batch=4 / seq_len=256).

## Главные инварианты, которые нельзя нарушать

**Seed-based in-place perturbation.** MeZO работает только если возмущение $z_t$ полностью определяется seed-ом и не хранится явно. См. `src/dmezo/mezo/perturbation.py`. Не вводи torch tensors для $z$ — это убьёт memory-efficiency, которая является главным selling point метода.

**Independent z_i per client (не shared seed).** Каждый клиент имеет свой `np.random.Generator` (`src/dmezo/federated/client.py:62`) и сэмплирует свой $s_i^t$ независимо. Это **критично** для $1/n$ variance speedup по direction noise (Theorem 2). НЕ менять на shared-seed broadcast (это FedKSeed-style, наш алгоритмический differentiator). См. § 3.3 в `docs/math_intuition.md` для обоснования.

**Между клиентами передаётся (seed, ρ) пара, не array.** В `consensus_via_updates` коде передавай между клиентами `(s_i: int, ρ_i: float)` — 16 байт/раунд/сосед. Каждый клиент локально регенерирует $z_j$ из полученного $s_j$ для apply update. Никаких array-обменов между клиентами (это FedAvg style).

**D-MeZO-N v2 = combo (B1 adaptive_clip + B5 drift-reset)**, не v1. v1 (fixed C=50) multi-seed falsified — 3/3 worse than vanilla на Qwen3.5-4B-Base / MathLogicQA. Recipe v2:
- Adaptive clip: `AdaptiveClipState(window=50, quantile=0.95, alpha=1.3)` в `src/dmezo/mezo/step.py`
- Drift-reset: в `src/dmezo/mezo/nesterov.py::NesterovState.check_drift_and_reset` (window=50, threshold=0.1)
- β-decay linear 0.9 → 0
- lr=3e-7, ε=1e-3 (Princeton defaults)

**Eval-mode и `inference_mode` во время MeZO forward.** Dropout должен быть выключен, autograd — выключен. См. `zo_forward` в принстонском коде.

**Параметры обновляются in-place через `.data`.** Не использовать `torch.no_grad()` присвоение или `param = param + ...` — нужно именно `param.data = param.data + ...` чтобы не сломать ссылки в оптимизаторе.

## Архитектурные решения

**Модель по умолчанию: Qwen3-4B** (standard transformer, Apache 2.0). HF: `Qwen/Qwen3-4B`. Размер FP16 ≈ 8 GB.

Альтернативы и upgrade path:

- Qwen3-8B (`Qwen/Qwen3-8B`) — стандартный трансформер, ~16 GB FP16. Upgrade для финальных экспериментов.
- Qwen3.5-4B-Base / Qwen3.5-4B (`Qwen/Qwen3.5-4B*`) — **hybrid linear-attention + full-attention vision-language модель** (подтверждено по config.json: layer_types = [linear, linear, linear, full] × 8 в text decoder, плюс 24-слойный ViT). Архитектура `Qwen3_5ForConditionalGeneration`, загружается через `AutoModelForImageTextToText`. Loader (`src/dmezo/models/loader.py::_load_vl_for_text_task`) автоматически замораживает vision tower; MeZO perturbает только text decoder. Config: `configs/qwen3_5_4b_base_sst2.yaml`. Это первый known test MeZO на linear-attention арх — Princeton paper только full-attention.

**Целевая платформа compute: Google Colab Pro+ с RTX PRO 6000 Blackwell (96 GB)**. Бюджет 600 compute units на месяц. Ноутбук `notebooks/bootstrap_colab.ipynb` готов к запуску в Colab.

**Стек:** Python 3.11, PyTorch 2.3+, Transformers 4.45+, datasets, accelerate, peft (для LoRA), tqdm, hydra-core (configs), wandb (опц.).

## Конвенции кода

- Type hints везде, docstrings в Google-style.
- Конфиги через Hydra (YAML в `configs/`). Никаких хардкодов гиперпараметров в скриптах.
- Логи и чекпойнты в `experiments/<run_name>/`. Чекпойнты — каждые 200 MeZO steps.
- Для Colab — обязательно сохранять каждые 30 минут в Google Drive (`/content/drive/MyDrive/dmezo_runs/`). Сессия может умереть.
- Тесты — `pytest`, в `tests/`. Минимум: тест на determinism perturbation, тест на mixing matrix properties.

## Что делать, когда пользователь просит реализовать новый компонент

1. **Сначала проверь** `docs/03-algorithm-spec.md` — там формальная спецификация D-MeZO-N. Не уверен в формуле — спроси, не угадывай.
2. **Сначала добавь тест** в `tests/`, потом имплементацию.
3. **Сравни с референсом**: если это компонент MeZO — открой соответствующее место в принстонском коде (linked в `docs/06-reading-list.md`) и не отклоняйся без причины.
4. **Если это про consensus/topology** — Koloskova et al. 2020 имеет канонические формулы; формулы из `docs/03-algorithm-spec.md` должны быть выводимы из её теоремы 2.

## Что делать, когда что-то ломается

**MeZO loss не падает.** Проверь: (a) `param.requires_grad=True` для всех параметров, (b) `zo_eps` в разумном диапазоне ($10^{-3}$ дефолт), (c) learning rate (для MeZO обычно $10^{-6}$–$10^{-7}$, существенно меньше чем для Adam), (d) что perturbation действительно in-place (manual_seed одинаковый для +/-).

**OOM на Colab.** Сначала: gradient_checkpointing=False (нам не нужно — нет backprop), убрать optimizer state (его не должно быть для MeZO). Если на Qwen3-4B всё равно OOM — баг, не feature.

**Несогласованность между клиентами в симуляторе.** Проверь, что counter PRNG один на всех клиентов (общая глобальная переменная или Lamport-style counter). Если клиенты получают разные seed на одном шаге — это баг.

**Nesterov + update_share падает с NotImplementedError.** Это сознательное ограничение — velocity-update внутри consensus не реализован (см. `docs/07-audit-harden.md` D1). Используй либо `consensus_mode="weight_avg"` (Nesterov работает локально), либо `nesterov_state=None`.

## Roadmap и приоритеты (на 2026-05-21, post-defense)

Краткосрочно (до защиты 2026-05-23):
- Финальный polish слайдов от Claude Design (см. `docs/defense_design_brief.md`)
- Dry-run защиты × 2
- Прочитать `docs/defense_talking_points.md` 2 раза

Post-defense follow-ups (в `docs/upgrade_roadmap.md`):
- HellaSwag rescue multi-seed (3 seeds × Qwen3-4B) — script готов
- Head-to-head FedKSeed (3 variants × 2-3 seeds) — script готов
- Scale-up Qwen3-8B / n=8 clients
- Generative tasks (SAMSum, GSM8K)
- Full decentralized Theorem 3 (Open Problem 2)
- Subsampling DP amplification

## Полезные команды

```bash
# Установка
uv pip install -e .

# Day 1 sanity check
uv run --no-sync python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml

# Multi-seed validation (paper headline reproduce)
uv run --no-sync python scripts/local_test_improvements.py \
    --model Qwen/Qwen3.5-4B-Base --task mathlogicqa --seeds 42 43 44 \
    --variants vanilla dmezo_n dmezo_n_drift dmezo_n_adaptive_clip dmezo_n_combo

# Тесты (128/128 pass)
uv run --no-sync pytest tests/ -v

# Rebuild paper docx после edits
uv run --no-sync python scripts/99_build_paper_docx_ru.py
uv run --no-sync python scripts/99_build_paper_docx.py
```

## Дополнительный контекст

Автор проекта (Максим) работает в MITS (готовится к ШАД), ведёт заметки в Obsidian (`C:\Users\Maksim\Yandex.Disk\Obsidian`), любит чистый markdown с LaTeX-формулами. Лекции по RL на уровне DAPO/SimPO/GRPO. Уровень математики высокий — не объяснять SGD/Adam/моменты, это база.

Когда пишешь docs — пиши по-русски (это рабочий язык), когда пишешь код — английский (стандарт).

## Ссылки на ключевые внешние ресурсы

- MeZO (Malladi et al. 2023): https://github.com/princeton-nlp/MeZO, arXiv:2305.17333
- FedKSeed (Qin et al. 2024, ICML): https://github.com/alibaba/FederatedScope/tree/FedKSeed, arXiv:2312.06353
- Ferret (Shu et al. 2024): https://github.com/allen4747/Ferret, arXiv:2409.06277
- FedZeN (Maritan et al. 2024): arXiv:2309.17241
- Koloskova et al. 2020 (Unified D-SGD): arXiv:2003.10422
- Nesterov-Spokoiny 2017: https://link.springer.com/article/10.1007/s10208-015-9296-2
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-4B
- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-4B
