# 07. Audit & harden федерального ядра

**Дата spec:** 2026-05-14. **Статус:** реализовано 2026-05-14 (commit `a009976`). **План реализации:** [docs/08-audit-harden-plan.md](08-audit-harden-plan.md).

## Зачем

Перед тем как тратить Colab compute на Days 1-7, в текущем skeleton-коде есть два бага, которые сделают эксперименты Days 4-5 либо неработоспособными, либо **математически некорректными по сравнению со спекой** в `03-algorithm-spec.md`. Цель этого подхода — закрыть оба бага, добавить минимально-достаточные тесты, оставив остальное (Nesterov look-ahead, LoRA, FedKSeed baseline, теорему) на следующие итерации.

## Найденные проблемы

### B1. `consensus_via_updates` — O(n²·p²) regen z для каждого параметра

Файл: `src/dmezo/federated/consensus.py:80-152`.

Текущая структура цикла (упрощённо):

```
for i in clients:
    for name, param in client.model.named_parameters():
        for j in neighbors:
            torch.manual_seed(seeds[j])
            for n_, p_ in client.model.named_parameters():
                z = N(0,1, shape=p_.shape)
                if n_ == name:
                    update += W_ij * rho_j * z
                    break
```

Для каждого `name` функция:
1. Заново вызывает `torch.manual_seed(seeds[j])`.
2. Заново итерирует **всю** `model.named_parameters()` пока не дойдёт до `name`.

То есть для модели с `p` параметрами и `n` клиентами на каждом раунде делается `O(n · p · n · p_avg_position)` Gaussian sampling. На Qwen3-4B (`p` ≈ 600 параметров, средняя позиция ≈ 300) при 4 клиентах и ring topology (≈ 2 соседа): `4 · 600 · 2 · 300 = 1.4M` ненужных `torch.normal` вызовов за один раунд. Это минуты на раунд на GPU и часы на CPU.

### B2. Двойной update в `update_share` режиме

Файлы: `src/dmezo/federated/simulator.py:84-115`, `src/dmezo/federated/client.py:60-87`.

`ClientState.local_round` всегда применяет update через `mezo_update` или `nesterov_step`. Затем при `consensus_mode="update_share"` симулятор вызывает `consensus_via_updates`, которая делает **ещё один** weighted update.

В коде есть прямое признание проблемы (`simulator.py:104-114`): «callers should configure clients with local_steps=0-style behavior — see scripts/04 for usage». Но `scripts/04` не существует, а API `local_steps=0` тоже не реализован.

Эффект: формула из спеки

$$\theta_i^{t+1} = \theta_i^t - \eta \cdot \sum_j W_{ij} \cdot \rho_j \cdot z(s_t)$$

ломается на

$$\theta_i^{t+1} = \theta_i^t - \eta \cdot \rho_i \cdot z(s_t) - \eta \cdot \sum_j W_{ij} \cdot \rho_j \cdot z(s_t),$$

то есть локальный шаг применяется дважды (один раз сольно, второй — внутри суммы где $W_{ii} \neq 0$).

## Что чиним

### F1. Переписать `consensus_via_updates`

Целевая структура:

```python
def consensus_via_updates(clients, W, seeds, projected_grads, config):
    for i, client in enumerate(clients):
        named = [(n, p) for n, p in client.model.named_parameters() if p.requires_grad]
        # Аккумулятор для каждой подлежащей обновлению параметрической матрицы.
        accum = {name: torch.zeros_like(p.data) for name, p in named}

        # Один обход по соседям; для каждого соседа один manual_seed,
        # затем один проход по параметрам — никаких break-early и rebuild RNG.
        for j in range(len(clients)):
            wij = float(W[i, j])
            if wij == 0.0:
                continue
            coef = wij * float(projected_grads[j])
            torch.manual_seed(int(seeds[j]))
            for name, p in named:
                z = torch.normal(0.0, 1.0, size=p.data.size(),
                                  device=p.data.device, dtype=p.data.dtype)
                accum[name].add_(z, alpha=coef)

        # Один проход финального применения.
        for name, p in named:
            decay = config.weight_decay if _is_decay_param(name) else 0.0
            if decay != 0.0:
                accum[name].add_(p.data, alpha=decay)
            p.data.add_(accum[name], alpha=-config.lr)
```

Сложность: `O(|neighbors| · p)` тензорных операций (вместо `O(|neighbors| · p²)`).

Семантика сохраняется один-в-один:
- `z` зависит только от `seeds[j]`, генерируется в одном и том же порядке (named_parameters детерминирован) — итог тот же.
- Decay применяется ровно один раз к pre-update `θ_i` (как в текущей реализации).
- Доказательство эквивалентности: проверяется тестом `test_via_updates_matches_old_implementation_on_tiny_model` (см. ниже).

### F2. Контракт `local_round(apply: bool)`

Сигнатура:

```python
def local_round(
    self,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    *,
    apply: bool = True,
) -> List[Tuple[int, float, float]]:
    """Run local_steps MeZO computations.

    If apply=True (default), updates parameters in-place after each step.
    If apply=False, only returns (seed, rho, loss_plus); parameters untouched.

    Use apply=False with consensus_via_updates to avoid double-applying
    the local contribution.
    """
```

Реализация: внутри цикла `for _ in range(self.local_steps)` пропускаем блок `mezo_update`/`nesterov_step` если `apply=False`.

Симулятор выбирает `apply` по `consensus_mode`:

| `consensus_mode` | `apply` | Что делает consensus-шаг |
|---|---|---|
| `weight_avg` | `True` | Averaging параметров через `consensus_via_weights` |
| `update_share` | `False` | Единый update через `consensus_via_updates` |
| `none` | `True` | Ничего |

## Тесты (CPU, <30 сек суммарно)

### `tests/_fixtures.py` (новый)

- `tiny_causal_lm()` — `nn.Module` ~10K параметров с интерфейсом `model(input_ids=..., labels=...)` возвращающий объект с `.loss`. Реально 2-layer linear + cross-entropy на токенах. Используется как drop-in замена HF CausalLM в unit-тестах.
- `synthetic_token_loader(seq_len=8, vocab=32, batch_size=4, num_batches=16)` — `DataLoader` синтетических `(input_ids, labels)`.

### `tests/test_consensus.py` (новый)

1. **`test_via_weights_averages_params_on_complete_graph`** — 4 клиента с заранее заданными разными `θ`, complete graph, один вызов `consensus_via_weights`. Ассерт: все клиенты теперь равны центроиду.
2. **`test_via_updates_n1_equivalent_to_mezo_update`** — `n=1, W=[[1]]`. Сравнение `consensus_via_updates(model, W, [seed], [rho], cfg)` с прямым `mezo_update(model, seed, rho, cfg)`. Должно быть битово равно.
3. **`test_via_updates_deterministic_under_replay`** — повторный вызов с теми же `seeds, rhos` на той же модели даёт идентичный результат.
4. **`test_via_updates_complete_graph_consensus`** — 4 клиента с одинаковыми initial params, разными data, complete graph, 50 раундов `apply=False + consensus_via_updates`. Ассерт: `max ||θ_i - θ_j||_∞ < 1e-4` (clients не должны расходиться).
5. **`test_via_updates_matches_old_implementation_on_tiny_model`** — bridge-тест: запустить и старую, и новую реализации с одинаковыми входами, сравнить `θ` после одного шага. Толеранс `1e-6`. Этот тест **удалится** после первого успешного запуска, его цель — единократно подтвердить, что рефакторинг не сменил семантику.

   Поскольку проект **не git-репозиторий**, процедура такая:
   - **Шаг 1 (до правки consensus.py):** скопировать текущую функцию `consensus_via_updates` в `tests/_legacy_consensus.py` под именем `consensus_via_updates_legacy`, импорты разрешить вручную.
   - **Шаг 2:** написать bridge-тест и убедиться, что он работает с legacy-копией.
   - **Шаг 3:** переписать `src/dmezo/federated/consensus.py`.
   - **Шаг 4:** запустить bridge-тест — он должен пройти.
   - **Шаг 5:** удалить `tests/_legacy_consensus.py` и сам bridge-тест.

### `tests/test_simulator.py` (новый)

1. **`test_full_round_runs_on_tiny_model`** — 2 клиента, ring(2), `tiny_causal_lm`, `synthetic_token_loader`, 5 раундов `update_share`. Проверки:
   - `run_simulation` не падает.
   - `len(logs) == 5`.
   - Параметры клиентов изменились (`!= initial`).
   - Все `mean_local_loss` финитные.

2. **`test_no_double_update_in_update_share`** — 1 клиент с `W=[[1]]`. Манчально вычислить ожидаемый `θ` после одного раунда через формулу спеки. Сравнить с фактическим `θ` после `run_simulation(num_rounds=1, consensus_mode="update_share")`. Толеранс `1e-7`. Этот тест зафиксирует фикс B2: до фикса будет fail (двойной update), после — pass.

3. **`test_nesterov_with_update_share_raises`** — конструируем 2 клиента с `nesterov_state` заданным, `consensus_mode="update_share"`. Ожидаем `NotImplementedError` с понятным сообщением (см. D1 в Discovered-but-deferred).

## Файлы

Меняются:
- `src/dmezo/federated/consensus.py` (переписать `consensus_via_updates`).
- `src/dmezo/federated/client.py` (добавить `apply: bool` в `local_round`).
- `src/dmezo/federated/simulator.py` (выбор `apply` по mode + удаление apology-комментария).

Новые:
- `tests/_fixtures.py`
- `tests/test_consensus.py`
- `tests/test_simulator.py`
- `tests/_legacy_consensus.py` (временный, удалить после bridge-теста).

Не трогаются:
- `src/dmezo/mezo/*` (выглядит корректно — соответствует Princeton reference).
- `src/dmezo/federated/topology.py` (уже покрыто тестами, выглядит корректно).
- `src/dmezo/models/`, `src/dmezo/data/`, `src/dmezo/utils/` (вне scope).
- `scripts/01_sanity_check_mezo.py` (использует только `mezo/*`, не задет).

## Definition of done

- [ ] `pytest tests/ -v` — все тесты проходят (существующие + новые 5 в `test_consensus.py` + новые 3 в `test_simulator.py`).
- [ ] Bridge-тест `test_via_updates_matches_old_implementation_on_tiny_model` зелёный → удалён.
- [ ] Регресс-тест `test_no_double_update_in_update_share` зелёный (его фейл на старом коде — main свидетельство, что B2 был реальным багом).
- [ ] Документ `docs/07-audit-harden.md` (этот) обновлён с пометкой «реализовано» в шапке.
- [ ] `CLAUDE.md` обновлён: убрать упоминание `scripts/04` в TODO/блокерах, добавить «федеральный stack теперь покрыт интеграционными тестами».

## Discovered-but-deferred (нашли при написании spec, фиксим позже)

**D1. Nesterov-state + update_share не определён.** `consensus_via_updates` не трогает `nesterov_state.velocities`. Если клиент инициализирован с `nesterov_state != None` и `consensus_mode="update_share"` — velocity никогда не обновится, фактически Nesterov не работает. Спека (`docs/03-algorithm-spec.md`) предполагает velocity-update внутри consensus, но это требует расширения `consensus_via_updates`.

**Решение в текущем audit:** добавить early-fail при попытке этой комбинации в `simulator.run_simulation`:
```python
if config.consensus_mode == "update_share" and any(c.nesterov_state is not None for c in clients):
    raise NotImplementedError(
        "Nesterov + update_share не реализован. Используйте weight_avg, или "
        "consensus_mode=none + локальный Nesterov."
    )
```
Покрыть тестом `test_nesterov_with_update_share_raises`.

**Полное решение** (out of scope, отдельный spec) — расширить `consensus_via_updates` чтобы он обновлял velocity по формуле из спеки: `v = β·v + Σ_j W_ij·ρ_j·z_j + decay·θ; θ -= η·v`.

**D2. Per-client seeds vs shared seed.** Спека (`docs/03-algorithm-spec.md`) говорит: «Sample SHARED seed s_t (same on all clients)». Implementation (и `CLAUDE.md`) использует **per-client** seeds. Это меняет communication cost: spec обещает `|E|·8` байт за раунд, реальность — `|E|·(8+4)` байт (scalar + 4-byte seed integer).

**Решение в текущем audit:** не трогать (impl-вариант выразительнее). Зафиксировать как **известная divergence** в `docs/03-algorithm-spec.md` при следующем редактировании — отметить, что implementation использует per-client seeds, и это сознательное расширение спеки. Сейчас просто оставляем this note здесь.

## Out of scope (явно — следующие итерации)

| Задача | Когда |
|---|---|
| Nesterov look-ahead | После Day 1 sanity + Day 4 D-MeZO; нужен бенчмарк memory cost для Qwen3-4B |
| Полная интеграция Nesterov + update_share (D1) | Отдельный spec, после Day 4 baseline |
| Sync spec ↔ impl насчёт seeds (D2) | При следующем редактировании `docs/03` |
| LoRA-режим | Когда Day 6 (Qwen3-8B) станет blocker по памяти |
| CPU-offload velocity | Same |
| FedKSeed baseline implementation | Day 2 (lit-review день) |
| Скрипты Days 2-7 | Отдельные spec-документы, один на скрипт |
| Доказательство теоремы | Day 3, отдельный workstream |
| Test для `mezo_step` end-to-end | Желательно, но не блокер |

## Оценка времязатрат

| Стадия | Время |
|---|---|
| Дочитать оставшиеся файлы (`partition.py`, `models/loader.py`, `data/superglue.py`) | 20 мин |
| Написать `tests/_fixtures.py` и пустые скелеты тестов | 30 мин |
| Написать тесты (TDD: красные сначала) | 60 мин |
| Рефакторинг `consensus_via_updates` | 30 мин |
| Фикс `client.py` + `simulator.py` | 30 мин |
| Прогон, debug | 30 мин |
| Финальная зачистка (удалить `_legacy_consensus.py`, обновить `CLAUDE.md`) | 15 мин |
| **Итого** | **~3.5 часа** |

Compute: 0 Colab units (всё локально на CPU).
