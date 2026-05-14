# CLAUDE.md — Decentralized Federated MeZO (D-MeZO)

Этот файл — рабочая память для Claude Code на этом проекте. Читай его в начале сессии.

## TL;DR проекта

Исследовательский проект на стыке zeroth-order оптимизации, federated learning и LLM fine-tuning. Цель — построить и проанализировать **Decentralized Federated MeZO с Nesterov-ускорением** (рабочее имя: **D-MeZO-N**) и показать, что он эффективен на современных open-weight LLM (Qwen3-4B как основная модель).

Ключевая идея: MeZO передаёт между клиентами **один скаляр $\hat\rho$ + общий seed** вместо миллиардов весов. Это идеально подходит под decentralized peer-to-peer сценарий. Соединяем с consensus mixing (à la Koloskova et al. 2020) и Nesterov-look-ahead → получаем алгоритм, у которого нет прямого аналога в литературе.

## Текущее состояние проекта

Этап: **скелет создан, pre-flight локально пройден, Day 1 sanity готов к Colab**. Что есть:

- Core MeZO step (`src/dmezo/mezo/`) — основан на референсной имплементации Princeton, адаптирован под HF Transformers AutoModel.
- Federated simulator (`src/dmezo/federated/`) — in-process многоклиентный симулятор с настраиваемой topology. Покрыт интеграционными тестами (`tests/test_consensus.py`, `tests/test_simulator.py`); `consensus_via_updates` переписан под O(np), исправлен баг с двойным update в `update_share`. См. `docs/07-audit-harden.md`.
- Day 1 скрипт (`scripts/01_sanity_check_mezo.py`) — централизованный MeZO с MLflow tracking. Прогнан локально на Qwen3-0.6B / SST-2 (RTX 2080, fp16, 100 шагов): eval loss 1.55 → 0.33 (−78%), 12s wall-clock. Pipeline работает.
- Configs: `configs/qwen3_4b_sst2.yaml` (Colab/Blackwell), `configs/qwen3_06b_preflight.yaml` (локально на Turing — fp16, no flash-attn).
- Experiment tracking: **MLflow** (file backend, `./mlruns/`). См. `mlflow ui --backend-store-uri file:./mlruns`. НЕ предлагай wandb/aim/TensorBoard, выбор сделан осознанно.
- Документация в `docs/` — лит-обзор, спецификация алгоритма, шаблон теоремы, недельный план.

**Локальная разработка на CUDA torch.** Для RTX 2080 (Turing) поставлен `torch==2.12.0+cu126` через `uv pip install --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch`. Так как pyproject.toml не пинит CUDA-вариант, любой `uv run` (он же `uv sync`) перезатирает torch на CPU build. **Использовать `uv run --no-sync ...` для всех локальных команд**, иначе CUDA torch исчезнет.

## Главные инварианты, которые нельзя нарушать

**Seed-based in-place perturbation.** MeZO работает только если возмущение $z_t$ полностью определяется seed-ом и не хранится явно. См. `src/dmezo/mezo/perturbation.py`. Не вводи torch tensors для $z$ — это убьёт memory-efficiency, которая является главным selling point метода.

**Один скаляр на раунд коммуникации.** В federated коде передавай между клиентами `(seed: int, projected_grad: float)`. Никаких array-обменов между клиентами. Если соблазн появился — это значит, ушли в FedAvg-style, что противоречит идее проекта.

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

## Roadmap и приоритеты

См. `docs/05-week1-plan.md` для детального плана недели. Краткая последовательность:

1. **Day 1 (priority 0):** запуск `scripts/01_sanity_check_mezo.py` на Colab, подтверждение что MeZO сходится на Qwen3-4B / SST-2.
2. Day 2: лит-обзор (FedKSeed, FedZeN, Ferret), centralized baselines на BoolQ/COPA.
3. Day 3: теоретический шаблон, см. `docs/04-theory-template.md`.
4. Day 4: первая версия D-MeZO на 2 клиентах.
5. Day 5: 4 клиента + non-IID + topologies.
6. Day 6: stretch — Qwen3-8B или Qwen3.5-4B (gated-deltanet experiment).
7. Day 7: one-pager + ablations.

## Полезные команды

```bash
# Установка
pip install -e .

# Day 1 sanity check
python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml

# Тесты
pytest tests/ -v
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
