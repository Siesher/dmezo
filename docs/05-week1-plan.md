# 05. План первой недели

**Цель:** к концу воскресенья — обоснованное go/no-go решение и (если go) черновик figure 1 будущей статьи.

**Compute:** Colab Pro+ с RTX PRO 6000 Blackwell, бюджет 600 units. Расход за неделю: 400-550 units.

---

## День 1 — пн. Sanity check MeZO на Qwen3-4B

**Compute:** 30-50 units.

**Задачи:**
- Запустить `scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml` на Colab.
- Убедиться, что eval loss падает >10% за 1000 steps на SST-2.
- Параллельно прочитать Malladi 2023 целиком, включая appendix C про landscape analysis.

**Success criteria:**
- [ ] Eval loss падает (Pass)
- [ ] Нет NaN/Inf в projected gradient
- [ ] Wall-clock < 3 часов на Blackwell

**Возможные проблемы и диагностика:**
- Loss не падает → попробовать `lr=1e-5`, потом `lr=1e-7`. Если ни один не работает — bug в perturbation/restore логике.
- OOM → отключить flash attention, batch_size=4, max_length=128.

---

## День 2 — вт. Literature deep-dive + centralized baselines

**Compute:** ~20 units.

**Задачи:**
- Прочитать FedKSeed (Qin 2024) и FedZeN (Maritan 2024) целиком, включая код FedKSeed.
- Записать заметки в `docs/02-related-work.md` (расширить must-cite секцию деталями).
- На Colab: запустить FedLoRA baseline для Qwen3-4B на том же SST-2 — на L4 (free / cheap), не на Blackwell.
- Написать `docs/02-related-work.md` сравнение FedKSeed vs наш D-MeZO-N в виде diff-таблицы.

**Output к концу дня:** clear delta-statement: "our contribution over FedKSeed is X, Y, Z" в 3 буллетах. Если не получается — критический сигнал.

---

## День 3 — ср. Теоретический шаблон

**Compute:** ~50 units (фоновые прогоны).

**Задачи:**
- Углублённо прочитать Koloskova 2020 (Theorem 2 + proof в Appendix).
- Углублённо прочитать Nesterov-Spokoiny 2017 (Sections 2-4).
- В `docs/04-theory-template.md`: дописать **формулировку** теоремы детально (предположения, утверждение, константы). Не доказательство.
- Параллельно (фоном на Blackwell): запустить centralized MeZO на BoolQ и COPA — нужны как baseline numbers для Table 1.

**Output:** 1-2 страницы теоремы и Lyapunov-функции в Obsidian.

---

## День 4 — чт. D-MeZO на 2 клиентах

**Compute:** 50-80 units.

**Задачи:**
- Реализовать `scripts/03_dmezo_2clients.py` — простейший случай (используя `src/dmezo/federated/simulator.py`).
- Использовать `consensus_via_weights` для начала (проще debugged).
- Проверить, что 2-client federated сходится к примерно тому же loss, что centralized MeZO (разница <5%).
- На SST-2 IID разбиении.

**Success criteria:**
- Federated final loss ≤ centralized final loss × 1.1
- Total bytes communicated < 10 KB (с update-share consensus)

---

## День 5 — пт. 4 клиента + topologies + non-IID

**Compute:** 80-120 units.

**Задачи:**
- Расширить до 4 клиентов с `consensus_via_updates`.
- Три топологии: ring (rho≈0.6), random_regular d=3, complete.
- Три разбиения данных: IID, Dirichlet α=0.5, label-skew (extreme).
- Замерить три кривых: loss vs round, accuracy vs round, bytes per round.

**Output:** **главный график черновика paper** — D-MeZO converges на ring topology с трафиком < 100 bytes/round.

---

## День 6 — сб. Stretch: Qwen3-8B или Qwen3.5-4B

**Compute:** 100-150 units.

**Задача (выбрать одно):**

**Вариант A — Scaling на Qwen3-8B (safe, standard transformer):**
- 4 клиента, full-parameter MeZO + Nesterov, ring topology.
- Дает scaling-кривую (4B, 8B) — две точки.

**Вариант B — Novel architecture Qwen3.5-4B (риск, но потенциально сильный contribution):**
- Sanity check, что MeZO работает на Gated DeltaNet hybrid. Если да — это сам по себе novel result, который можно вынести в дополнительную секцию paper.

**Рекомендация:** Сначала A (часов 4), если успеется — B как bonus.

---

## День 7 — вс. One-pager + Nesterov ablation

**Compute:** 50 units.

**Задачи:**
- Запустить Nesterov vs no-Nesterov ablation на 4-client / ring / SST-2.
- Написать **one-pager** (2 страницы):
  - Algorithm pseudocode (из `docs/03-algorithm-spec.md`).
  - Шаблон теоремы (из `docs/04-theory-template.md`).
  - Три графика: loss vs round, comm-cost vs accuracy, topology ablation.
  - Дельта над FedKSeed/FedZeN (явно).
  - Realistic estimate на полный paper (3-6 месяцев).
- Решение: go / no-go / pivot.

---

## Compute-бюджет недельный

| День | Activity | Units |
|---|---|---|
| 1 | Sanity Qwen3-4B | 40 |
| 2 | Lit + FedLoRA baseline | 20 |
| 3 | Theory + BoolQ/COPA baselines | 50 |
| 4 | D-MeZO 2 clients | 70 |
| 5 | 4 clients + topologies + non-IID | 100 |
| 6 | Qwen3-8B scaling | 120 |
| 7 | Nesterov ablation + one-pager | 50 |
| **Total** | | **450** |

Остаётся 150 units на отладку / неудачные прогоны / неожиданные эксперименты.

---

## Что НЕ делать на этой неделе

- Llama-2 / Llama-3 (старый baseline, тратить compute не стоит).
- DPO / Constitutional AI fine-tuning (off-topic).
- Multi-task evaluation (focus на SST-2 для первой недели, потом расширим).
- DP-вариант (это вторая неделя как минимум).
- 70B модели (compute не хватит).
- Setup / vendoring FederatedScope, Flower — наш simulator достаточен для всей недели.

---

## Чек-лист готовности к понедельнику

- [x] Проект создан на `C:\Work\dmezo`
- [x] CLAUDE.md написан
- [x] Core MeZO + federated skeleton реализован
- [x] Configs готовы
- [x] Sanity check script готов
- [ ] **TODO:** Установить deps локально и запустить `pytest tests/` для проверки, что perturbation/topology тесты проходят
- [ ] **TODO:** Открыть `notebooks/bootstrap_colab.ipynb` в Colab и убедиться, что HF может скачать Qwen3-4B
