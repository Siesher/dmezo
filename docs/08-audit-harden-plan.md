# Audit-Harden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть два бага в `src/dmezo/federated/` (consensus_via_updates O(n²·p²); двойной update при `consensus_mode="update_share"`); добавить 8 регрессионных/интеграционных тестов, выполняющихся локально на CPU за < 30 секунд.

**Architecture:** TDD с двумя стилями — для F1 (рефакторинг с сохранением семантики) сначала собираем «защитную сеть» equivalence-тестов и временную копию старой реализации (bridge-test), потом рефакторим; для F2 (исправление поведения) сначала пишем красный регрессионный тест, потом фиксим. После F2 добавляем оставшиеся behavioral тесты.

**Tech Stack:** Python 3.11, PyTorch 2.x (CPU), pytest. Никаких GPU и Colab compute — всё выполняется локально на Windows + .venv.

**Спека:** см. [docs/07-audit-harden.md](07-audit-harden.md).

---

## File Structure

| Файл | Действие | Назначение |
|---|---|---|
| `tests/_fixtures.py` | Создать | `TinyCausalLM` (~2K параметров, интерфейс `model(**batch) → .loss`), `synthetic_token_loader` (real `DataLoader`), `make_tiny_clients` helper |
| `tests/test_fixtures.py` | Создать | Sanity: модель + лоадер выдают финитный loss |
| `tests/test_consensus.py` | Создать | 5 тестов для `consensus_via_weights` и `consensus_via_updates` (один временный bridge-тест) |
| `tests/_legacy_consensus.py` | Создать → удалить | Verbatim-копия старой реализации `consensus_via_updates` для bridge-тестa |
| `tests/test_simulator.py` | Создать | 3 теста для федеративного раунда |
| `src/dmezo/federated/consensus.py` | Modify | Рефакторинг `consensus_via_updates` |
| `src/dmezo/federated/client.py` | Modify | Параметр `apply: bool = True` в `local_round` |
| `src/dmezo/federated/simulator.py` | Modify | Выбор `apply` по `consensus_mode` + ранний фейл для Nesterov+update_share |
| `docs/07-audit-harden.md` | Modify | Статус «реализовано» |
| `CLAUDE.md` | Modify | Пометка о покрытии федеративного стека тестами |

---

## Task 1: Test fixtures

**Files:**
- Create: `tests/_fixtures.py`
- Create: `tests/test_fixtures.py`

- [ ] **Step 1: Создать `tests/_fixtures.py`**

```python
"""Shared CPU-only fixtures for federated tests.

Not auto-collected by pytest (underscore prefix). Imported by test_*.py files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dmezo.federated.client import ClientState
from dmezo.mezo.step import MeZOConfig


@dataclass
class _TinyCausalOutput:
    """Minimal HF-style output container with a single ``.loss`` attribute."""

    loss: torch.Tensor


class TinyCausalLM(nn.Module):
    """Tiny causal-LM-like module compatible with ``causal_lm_loss``.

    Forward signature matches what ``causal_lm_loss`` in data/superglue.py
    expects: keyword args ``input_ids``, ``attention_mask``, ``labels``; output
    has a ``.loss`` attribute.

    Total params: ~vocab_size * hidden * 2 + linear bias ≈ 2K params with
    defaults. Small enough to fit on CPU for ~ms-fast forward.
    """

    def __init__(self, vocab_size: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002 (kept for API parity)
        labels: Optional[torch.Tensor] = None,
    ) -> _TinyCausalOutput:
        h = self.embed(input_ids)
        logits = self.proj(h)
        if labels is None:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return _TinyCausalOutput(loss=zero)
        # Standard HF causal-LM shift: predict token i+1 from position i.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return _TinyCausalOutput(loss=loss)


def make_tiny_causal_lm(seed: int = 0, vocab_size: int = 32, hidden: int = 16) -> TinyCausalLM:
    """Deterministically initialise a TinyCausalLM with a given seed."""
    torch.manual_seed(seed)
    return TinyCausalLM(vocab_size=vocab_size, hidden=hidden)


class _SyntheticDataset(Dataset):
    def __init__(self, num_examples: int, seq_len: int, vocab_size: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab_size, (num_examples, seq_len), generator=g)
        self.labels = self.input_ids.clone()
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def synthetic_token_loader(
    num_examples: int = 32,
    batch_size: int = 4,
    seq_len: int = 8,
    vocab_size: int = 32,
    seed: int = 0,
) -> DataLoader:
    """Synthetic (input_ids, labels) batches for tests. CPU, deterministic."""
    ds = _SyntheticDataset(num_examples=num_examples, seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def make_tiny_clients(
    n: int,
    *,
    mezo_lr: float = 1e-3,
    mezo_eps: float = 1e-3,
    weight_decay: float = 0.0,
    seed_offset: int = 0,
    same_init: bool = False,
) -> List[ClientState]:
    """Create ``n`` ClientState objects with TinyCausalLM models and synthetic data.

    Args:
        n: Number of clients.
        mezo_lr / mezo_eps / weight_decay: MeZO config passed to each client.
        seed_offset: Added to the per-client seed for model init.
        same_init: If True, all clients use seed 0 (parameters identical at start).

    Returns:
        List of ClientState, one per client.
    """
    clients: List[ClientState] = []
    cfg = MeZOConfig(lr=mezo_lr, eps=mezo_eps, weight_decay=weight_decay)
    for i in range(n):
        model_seed = seed_offset if same_init else seed_offset + i
        model = make_tiny_causal_lm(seed=model_seed)
        for p in model.parameters():
            p.requires_grad_(True)
        loader = synthetic_token_loader(seed=100 + i)
        clients.append(
            ClientState(
                client_id=i,
                model=model,
                dataloader=loader,
                mezo_config=cfg,
                rng=np.random.default_rng(1000 + i),
            )
        )
    return clients
```

- [ ] **Step 2: Создать `tests/test_fixtures.py`**

```python
"""Sanity tests for shared fixtures."""

from __future__ import annotations

import torch

from tests._fixtures import (
    make_tiny_causal_lm,
    make_tiny_clients,
    synthetic_token_loader,
)


def test_tiny_model_produces_finite_loss():
    model = make_tiny_causal_lm(seed=0)
    loader = synthetic_token_loader(num_examples=8, batch_size=4)
    batch = next(iter(loader))
    out = model(**batch)
    assert torch.isfinite(out.loss).item(), f"Expected finite loss, got {out.loss}"
    assert out.loss.ndim == 0, f"Loss must be a scalar tensor, got shape {out.loss.shape}"


def test_make_tiny_clients_returns_n_clients_with_distinct_params():
    clients = make_tiny_clients(n=3)
    assert len(clients) == 3
    p0 = list(clients[0].model.parameters())[0].data
    p1 = list(clients[1].model.parameters())[0].data
    assert not torch.allclose(p0, p1), "Clients with different seeds should have different params"


def test_make_tiny_clients_same_init_makes_identical_params():
    clients = make_tiny_clients(n=2, same_init=True)
    for (n0, p0), (n1, p1) in zip(
        clients[0].model.named_parameters(), clients[1].model.named_parameters()
    ):
        assert n0 == n1
        assert torch.allclose(p0.data, p1.data, atol=1e-12), f"Param {n0} differs"
```

- [ ] **Step 3: Запустить тесты**

Run: `python -m pytest tests/test_fixtures.py -v`
Expected: 3 tests pass.

Если падает с `ImportError: cannot import name '_fixtures'` — проверь, что в `tests/` нет `__init__.py` (pytest добавляет cwd в sys.path).Если падает с `ModuleNotFoundError: No module named 'dmezo'` — выполни `pip install -e .` из корня проекта.

- [ ] **Step 4: Commit**

```bash
git add tests/_fixtures.py tests/test_fixtures.py
git commit -m "test(fixtures): add tiny_causal_lm and synthetic_token_loader"
```

---

## Task 2: Equivalence test for `consensus_via_weights`

**Files:**
- Create: `tests/test_consensus.py`

- [ ] **Step 1: Создать `tests/test_consensus.py` с одним тестом**

```python
"""Tests for federated consensus mixing."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest
import torch
from torch import nn

from dmezo.federated.client import ClientState
from dmezo.federated.consensus import consensus_via_updates, consensus_via_weights
from dmezo.federated.topology import complete_graph, ring_graph
from dmezo.mezo.step import MeZOConfig, mezo_update

from tests._fixtures import make_tiny_clients


def _snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data.clone() for name, p in model.named_parameters()}


def test_via_weights_averages_params_on_complete_graph():
    """One step of consensus_via_weights with complete graph hits the centroid."""
    n = 4
    clients = make_tiny_clients(n=n)
    snapshots = [_snapshot(c.model) for c in clients]

    # Expected per-name centroid across clients.
    param_names = [name for name, _ in clients[0].model.named_parameters()]
    expected: Dict[str, torch.Tensor] = {
        name: torch.stack([snap[name] for snap in snapshots]).mean(dim=0)
        for name in param_names
    }

    W = complete_graph(n).W
    consensus_via_weights(clients, W)

    for i, c in enumerate(clients):
        for name, p in c.model.named_parameters():
            assert torch.allclose(p.data, expected[name], atol=1e-6), (
                f"Client {i} param {name!r} did not match centroid"
            )
```

- [ ] **Step 2: Запустить тест**

Run: `python -m pytest tests/test_consensus.py::test_via_weights_averages_params_on_complete_graph -v`
Expected: PASS (current `consensus_via_weights` is correct).

- [ ] **Step 3: Commit**

```bash
git add tests/test_consensus.py
git commit -m "test(consensus): add equivalence test for via_weights"
```

---

## Task 3: Equivalence tests for `consensus_via_updates` (current code)

**Files:**
- Modify: `tests/test_consensus.py`

- [ ] **Step 1: Добавить два теста после существующего**

В `tests/test_consensus.py` после `test_via_weights_averages_params_on_complete_graph` добавить:

```python
def test_via_updates_n1_equivalent_to_mezo_update():
    """For n=1, W=[[1]], consensus_via_updates should match a direct mezo_update."""
    clients_a = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)
    clients_b = make_tiny_clients(n=1, mezo_lr=1e-3, mezo_eps=1e-3)

    # Sanity: identical initial params.
    for (na, pa), (nb, pb) in zip(
        clients_a[0].model.named_parameters(), clients_b[0].model.named_parameters()
    ):
        assert na == nb
        assert torch.allclose(pa.data, pb.data, atol=1e-12)

    seed = 12345
    rho = 0.7
    cfg = clients_a[0].mezo_config

    # Path A: direct mezo_update.
    mezo_update(clients_a[0].model, seed=seed, projected_grad=rho, config=cfg)

    # Path B: consensus_via_updates with W=[[1]], same (seed, rho).
    W = np.array([[1.0]])
    consensus_via_updates(clients_b, W, seeds=[seed], projected_grads=[rho], config=cfg)

    for (na, pa), (nb, pb) in zip(
        clients_a[0].model.named_parameters(), clients_b[0].model.named_parameters()
    ):
        assert torch.allclose(pa.data, pb.data, atol=1e-6), (
            f"Param {na!r}: mezo_update and consensus_via_updates diverged"
        )


def test_via_updates_deterministic_under_replay():
    """Same (seeds, rhos) on the same initial params should produce the same outputs."""
    clients_a = make_tiny_clients(n=2)
    clients_b = make_tiny_clients(n=2)

    seeds = [111, 222]
    rhos = [0.3, -0.5]
    W = ring_graph(2).W
    cfg = clients_a[0].mezo_config

    consensus_via_updates(clients_a, W, seeds=seeds, projected_grads=rhos, config=cfg)
    consensus_via_updates(clients_b, W, seeds=seeds, projected_grads=rhos, config=cfg)

    for ca, cb in zip(clients_a, clients_b):
        for (na, pa), (nb, pb) in zip(
            ca.model.named_parameters(), cb.model.named_parameters()
        ):
            assert na == nb
            assert torch.allclose(pa.data, pb.data, atol=1e-9), (
                f"Replay diverged on {na!r}"
            )
```

- [ ] **Step 2: Запустить тесты**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: 3 tests pass (1 from Task 2 + 2 new). Если `test_via_updates_n1_equivalent_to_mezo_update` падает — это сигнал, что старая реализация уже расходится с `mezo_update`, и нужна более глубокая ревизия.

- [ ] **Step 3: Commit**

```bash
git add tests/test_consensus.py
git commit -m "test(consensus): add baseline equivalence tests for via_updates"
```

---

## Task 4: Bridge test — заморозить старую реализацию

**Files:**
- Create: `tests/_legacy_consensus.py`
- Modify: `tests/test_consensus.py`

- [ ] **Step 1: Перечитать `src/dmezo/federated/consensus.py:80-152`**

Это нужно, чтобы скопировать **текущую** функцию `consensus_via_updates` в legacy-файл до того, как мы её перепишем.

- [ ] **Step 2: Создать `tests/_legacy_consensus.py`** с verbatim-копией под новым именем

```python
"""Frozen snapshot of the pre-refactor ``consensus_via_updates``.

Used only by ``test_via_updates_matches_legacy_implementation_on_tiny_model``
in test_consensus.py for one-time semantic equivalence verification.

DELETE THIS FILE after the bridge test has passed against the refactored
implementation (see docs/07-audit-harden.md Task 6).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from dmezo.federated.client import ClientState
from dmezo.mezo.step import MeZOConfig


def consensus_via_updates_legacy(
    clients: List[ClientState],
    W: np.ndarray,
    seeds: List[int],
    projected_grads: List[float],
    config: MeZOConfig,
) -> None:
    n = len(clients)
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}")
    if len(seeds) != n or len(projected_grads) != n:
        raise ValueError("seeds and projected_grads must have len == n_clients")

    for i, client in enumerate(clients):
        for name, param in client.model.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            decay = (
                config.weight_decay
                if ("bias" not in lname) and ("layer_norm" not in lname) and ("layernorm" not in lname)
                else 0.0
            )
            update = torch.zeros_like(param.data)
            for j in range(n):
                wij = float(W[i, j])
                if wij == 0:
                    continue
                torch.manual_seed(int(seeds[j]))
                for n_, p_ in client.model.named_parameters():
                    if not p_.requires_grad:
                        continue
                    z = torch.normal(
                        mean=0.0, std=1.0,
                        size=p_.data.size(),
                        device=p_.data.device,
                        dtype=p_.data.dtype,
                    )
                    if n_ == name:
                        update.add_(z, alpha=wij * float(projected_grads[j]))
                        break
                else:
                    continue
            update.add_(param.data, alpha=decay)
            param.data.add_(update, alpha=-config.lr)
```

- [ ] **Step 3: Добавить bridge-тест в `tests/test_consensus.py`**

В конец файла:

```python
def test_via_updates_matches_legacy_implementation_on_tiny_model():
    """Bridge test: refactored consensus_via_updates must reproduce legacy results.

    This test is TEMPORARY — delete after the refactor (see docs/08 Task 6).
    """
    from tests._legacy_consensus import consensus_via_updates_legacy

    n = 3
    seeds = [11, 22, 33]
    rhos = [0.4, -0.2, 0.15]
    W = ring_graph(n).W

    clients_legacy = make_tiny_clients(n=n, mezo_lr=1e-3, weight_decay=1e-4)
    clients_new = make_tiny_clients(n=n, mezo_lr=1e-3, weight_decay=1e-4)
    cfg = clients_legacy[0].mezo_config

    consensus_via_updates_legacy(clients_legacy, W, seeds, rhos, cfg)
    consensus_via_updates(clients_new, W, seeds, rhos, cfg)

    for i, (cl, cn) in enumerate(zip(clients_legacy, clients_new)):
        for (nl, pl), (nn_, pn) in zip(
            cl.model.named_parameters(), cn.model.named_parameters()
        ):
            assert nl == nn_
            assert torch.allclose(pl.data, pn.data, atol=1e-6), (
                f"Client {i} param {nl!r}: legacy={pl.data.flatten()[:3]} "
                f"vs new={pn.data.flatten()[:3]}"
            )
```

- [ ] **Step 4: Запустить bridge-тест**

Run: `python -m pytest tests/test_consensus.py::test_via_updates_matches_legacy_implementation_on_tiny_model -v`
Expected: PASS (sanity — двое одинаковых функций должны давать одинаковый результат, потому что `consensus_via_updates` пока совпадает с `consensus_via_updates_legacy`).

- [ ] **Step 5: Commit**

```bash
git add tests/_legacy_consensus.py tests/test_consensus.py
git commit -m "test(consensus): save legacy via_updates as bridge baseline"
```

---

## Task 5: F1 — переписать `consensus_via_updates` под O(np)

**Files:**
- Modify: `src/dmezo/federated/consensus.py:80-152`

- [ ] **Step 1: Обновить импорт `_is_decay_param`**

В `src/dmezo/federated/consensus.py`, заменить строку
```python
from dmezo.mezo.step import MeZOConfig, _collect_optim_params
```
на
```python
from dmezo.mezo.step import MeZOConfig, _collect_optim_params, _is_decay_param
```

- [ ] **Step 2: Перезаписать функцию `consensus_via_updates`**

Заменить весь `def consensus_via_updates(...)` (строки 80-152) на:

```python
def consensus_via_updates(
    clients: List[ClientState],
    W: np.ndarray,
    seeds: List[int],
    projected_grads: List[float],
    config: MeZOConfig,
) -> None:
    """Apply consensus by exchanging (seed, rho) pairs and locally combining updates.

    For each client ``i``, computes::

        theta_i <- theta_i - lr * (sum_j W[i, j] * rho_j * z(seed_j) + decay * theta_i)

    where ``z(seed_j)`` is regenerated locally and ``decay`` is the
    weight_decay coefficient (applied only to non-bias / non-norm params).

    Complexity per round: O(n_clients * n_neighbors * p), where ``p`` is the
    number of trainable parameters. Each ``z`` is generated exactly once per
    (i, j, p_k) triple.

    Args:
        clients: All client states. Must share parameter order and shapes.
        W: ``n x n`` doubly-stochastic mixing matrix.
        seeds: Per-client seeds from the round's MeZO step (``len == n_clients``).
        projected_grads: Per-client projected gradients (``len == n_clients``).
        config: MeZO config (``lr``, ``weight_decay``).

    Note:
        Assumes ``local_steps == 1`` per round and that
        ``ClientState.local_round`` was called with ``apply=False`` so that no
        local update has been applied yet — this function is the single owner
        of parameter mutation in ``consensus_mode="update_share"``.
    """
    n = len(clients)
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}, got {W.shape}")
    if len(seeds) != n or len(projected_grads) != n:
        raise ValueError("seeds and projected_grads must have len == n_clients")

    for i, client in enumerate(clients):
        named = [(name, p) for name, p in client.model.named_parameters() if p.requires_grad]
        if not named:
            continue

        accum = {name: torch.zeros_like(p.data) for name, p in named}

        # Each neighbor contributes W[i, j] * rho_j * z(seed_j) in a single
        # deterministic pass over named parameters.
        for j in range(n):
            wij = float(W[i, j])
            if wij == 0.0:
                continue
            coef = wij * float(projected_grads[j])
            torch.manual_seed(int(seeds[j]))
            for name, p in named:
                z = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=p.data.size(),
                    device=p.data.device,
                    dtype=p.data.dtype,
                )
                accum[name].add_(z, alpha=coef)

        # Apply: theta -= lr * (accum + decay * theta).
        for name, p in named:
            decay = config.weight_decay if _is_decay_param(name) else 0.0
            if decay != 0.0:
                accum[name].add_(p.data, alpha=decay)
            p.data.add_(accum[name], alpha=-config.lr)
```

- [ ] **Step 3: Запустить все тесты consensus**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: все 4 теста pass, **в особенности `test_via_updates_matches_legacy_implementation_on_tiny_model`** — это даёт уверенность, что рефакторинг сохранил семантику.

Если bridge-тест падает на каком-то параметре с разницей > 1e-6 — диагностика:
1. Проверить, что `manual_seed(seeds[j])` вызывается **один раз** за `j`, а потом идёт **один** проход по named_parameters.
2. Проверить, что decay добавляется к `accum[name]` (а не дважды).
3. Проверить, что итерационный порядок `named` детерминирован (он должен быть — `named_parameters()` детерминирован для конкретной архитектуры).

- [ ] **Step 4: Commit**

```bash
git add src/dmezo/federated/consensus.py
git commit -m "refactor(federated): rewrite consensus_via_updates to O(np)"
```

---

## Task 6: Удалить bridge-артефакты

**Files:**
- Delete: `tests/_legacy_consensus.py`
- Modify: `tests/test_consensus.py` (удалить bridge-тест)

- [ ] **Step 1: Удалить файл**

```bash
git rm tests/_legacy_consensus.py
```

- [ ] **Step 2: Удалить bridge-тест из `tests/test_consensus.py`**

Удалить функцию `test_via_updates_matches_legacy_implementation_on_tiny_model` целиком.

- [ ] **Step 3: Запустить тесты consensus**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: 3 теста pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_consensus.py
git commit -m "chore(consensus): remove bridge-test artifacts after refactor"
```

---

## Task 7: F2 — failing regression test + apply flag + simulator wiring + nesterov guard

**Files:**
- Create: `tests/test_simulator.py`
- Modify: `src/dmezo/federated/client.py:60-87`
- Modify: `src/dmezo/federated/simulator.py:75-119`

- [ ] **Step 1: Написать падающий регрессионный тест в `tests/test_simulator.py`**

```python
"""Tests for the federated simulator (full-round behaviour)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dmezo.federated.simulator import SimulatorConfig, run_simulation
from dmezo.federated.topology import ring_graph
from dmezo.mezo.nesterov import NesterovState
from dmezo.mezo.step import MeZOConfig, mezo_step, mezo_update

from tests._fixtures import (
    make_tiny_causal_lm,
    make_tiny_clients,
    synthetic_token_loader,
)


def _causal_lm_loss(model, batch):
    """Inline loss fn that matches dmezo.data.superglue.causal_lm_loss interface."""
    out = model(**batch)
    return out.loss


def test_no_double_update_in_update_share():
    """One round with W=[[1]] under update_share must equal one mezo_update step.

    This is the regression test for B2 (double-update bug). Before the fix,
    local_round applies the update AND consensus_via_updates applies it again,
    so the final theta is theta - 2*lr*(rho*z + ...).
    """
    n = 1
    # Build two parallel setups starting from identical params.
    clients = make_tiny_clients(n=n, mezo_lr=1e-3, mezo_eps=1e-3, seed_offset=42)

    # Reference path: take the seed/rho that the simulator's first step would use,
    # apply ONE mezo_update directly on a clone of the model.
    reference_model = make_tiny_causal_lm(seed=42)
    for p in reference_model.parameters():
        p.requires_grad_(True)

    # Use a fixed rng so the simulator produces a deterministic seed/rho.
    clients[0].rng = np.random.default_rng(seed=7)
    # Pre-compute what mezo_step would produce on round 1's batch.
    batch = next(iter(synthetic_token_loader(seed=100)))  # matches client 0's loader seed
    rng_ref = np.random.default_rng(seed=7)
    seed, rho, _ = mezo_step(
        reference_model,
        batch,
        _causal_lm_loss,
        clients[0].mezo_config,
        rng=rng_ref,
    )
    mezo_update(reference_model, seed=seed, projected_grad=rho, config=clients[0].mezo_config)

    # Simulator path: one round in update_share mode.
    cfg = SimulatorConfig(
        num_rounds=1,
        consensus_mode="update_share",
        eval_every=0,
        log_every=0,
    )
    run_simulation(
        clients=clients,
        topology=ring_graph(n) if n > 1 else _self_loop_topology(),
        loss_fn=_causal_lm_loss,
        config=cfg,
    )

    # Compare params.
    for (nr, pr), (nc, pc) in zip(
        reference_model.named_parameters(), clients[0].model.named_parameters()
    ):
        assert nr == nc
        assert torch.allclose(pr.data, pc.data, atol=1e-6), (
            f"Param {nr!r}: simulator path diverged from single mezo_update. "
            f"Likely double-update bug (B2)."
        )


def _self_loop_topology():
    """A 1-node 'topology' with W=[[1]] for n=1 tests."""
    from dmezo.federated.topology import MixingMatrix

    return MixingMatrix(W=np.array([[1.0]]), name="self_loop")
```

- [ ] **Step 2: Запустить тест — он должен УПАСТЬ**

Run: `python -m pytest tests/test_simulator.py::test_no_double_update_in_update_share -v`
Expected: **FAIL**. Параметры simulator-пути будут смещены ровно вдвое — это и есть подтверждение бага B2.

Если тест неожиданно проходит — диагностика: проверить, что `local_round` действительно вызывается с дефолтным `apply=True` (он сейчас всегда применяет), и что `consensus_via_updates` не пропускает `j=i` (он не пропускает — `W[0,0]=1` участвует в сумме).

- [ ] **Step 3: Добавить `apply: bool` параметр в `local_round`**

Заменить метод `local_round` в `src/dmezo/federated/client.py` (строки 60-87) на:

```python
    def local_round(
        self,
        loss_fn: Callable[[nn.Module, dict], torch.Tensor],
        *,
        apply: bool = True,
    ) -> List[Tuple[int, float, float]]:
        """Execute ``local_steps`` MeZO steps locally.

        Args:
            loss_fn: Loss function ``(model, batch) -> scalar tensor``.
            apply: If True (default), apply each MeZO update in-place via
                ``mezo_update`` (or ``nesterov_step`` if ``nesterov_state`` is set).
                If False, return ``(seed, rho, loss_plus)`` triples without
                mutating parameters — use with ``consensus_via_updates``, which
                owns the eventual parameter mutation.

        Returns:
            List of ``(seed, projected_grad, loss_plus)`` from each local step.
        """
        history: List[Tuple[int, float, float]] = []
        for _ in range(self.local_steps):
            batch = self._next_batch()
            seed, rho, loss_plus = mezo_step(
                self.model, batch, loss_fn, self.mezo_config, rng=self.rng
            )
            if apply:
                if self.nesterov_state is not None:
                    nesterov_step(
                        self.model,
                        self.nesterov_state,
                        seed=seed,
                        projected_grad=rho,
                        lr=self.mezo_config.lr,
                        weight_decay=self.mezo_config.weight_decay,
                    )
                else:
                    mezo_update(self.model, seed=seed, projected_grad=rho, config=self.mezo_config)
            history.append((seed, rho, loss_plus))
        return history
```

- [ ] **Step 4: Обновить `simulator.run_simulation`**

Заменить тело `run_simulation` в `src/dmezo/federated/simulator.py` (строки 75-119) на:

```python
def run_simulation(
    clients: List[ClientState],
    topology: MixingMatrix,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    config: SimulatorConfig,
    eval_fn: Optional[Callable[[nn.Module, int], Dict[str, float]]] = None,
    logger: Optional[Callable[[Dict], None]] = None,
) -> List[Dict]:
    """Run the federated training loop.

    Args:
        clients: All client states. All clients must share parameter shapes.
        topology: Mixing matrix.
        loss_fn: Loss function applied to model + batch.
        config: Simulator config.
        eval_fn: Optional callable ``(model, round) -> dict`` for eval. Called
            on client 0's model.
        logger: Optional logger callable that receives per-round dicts.

    Returns:
        List of per-round logs.
    """
    if topology.n != len(clients):
        raise ValueError(
            f"Topology has {topology.n} nodes but {len(clients)} clients provided"
        )

    # Nesterov + update_share is not yet integrated (see docs/07-audit-harden.md D1).
    if config.consensus_mode == "update_share" and any(
        c.nesterov_state is not None for c in clients
    ):
        raise NotImplementedError(
            "Nesterov + update_share is not yet implemented. Use "
            "consensus_mode='weight_avg', or set nesterov_state=None on all clients."
        )

    # local_round must NOT apply the update when update_share owns the apply,
    # otherwise the update is applied twice (once locally, once in consensus).
    apply_local = config.consensus_mode != "update_share"

    logs: List[Dict] = []
    for r in range(config.num_rounds):
        round_log: Dict = {"round": r}

        # 1. Local MeZO steps.
        all_seeds: List[int] = []
        all_rhos: List[float] = []
        all_losses: List[float] = []
        for c in clients:
            history = c.local_round(loss_fn, apply=apply_local)
            last_seed, last_rho, last_loss = history[-1]
            all_seeds.append(last_seed)
            all_rhos.append(last_rho)
            all_losses.append(last_loss)

        round_log["mean_local_loss"] = float(np.mean(all_losses))
        round_log["mean_projected_grad"] = float(np.mean(all_rhos))

        # 2. Consensus.
        if config.consensus_mode == "weight_avg":
            consensus_via_weights(clients, topology.W)
        elif config.consensus_mode == "update_share":
            consensus_via_updates(
                clients, topology.W, all_seeds, all_rhos, clients[0].mezo_config
            )
        elif config.consensus_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown consensus_mode={config.consensus_mode!r}")

        # 3. Eval.
        if eval_fn is not None and config.eval_every > 0 and (r + 1) % config.eval_every == 0:
            eval_metrics = eval_fn(clients[0].model, r)
            round_log.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        # 4. Log.
        if config.log_every > 0 and (r + 1) % config.log_every == 0:
            if logger is not None:
                logger(round_log)

        logs.append(round_log)

    return logs
```

- [ ] **Step 5: Запустить регрессионный тест — должен пройти**

Run: `python -m pytest tests/test_simulator.py::test_no_double_update_in_update_share -v`
Expected: PASS.

- [ ] **Step 6: Прогнать весь набор тестов**

Run: `python -m pytest tests/ -v`
Expected: всё зелёное (test_perturbation + test_topology + test_fixtures + test_consensus + test_simulator).

- [ ] **Step 7: Commit**

```bash
git add src/dmezo/federated/client.py src/dmezo/federated/simulator.py tests/test_simulator.py
git commit -m "fix(federated): remove double-update in update_share via apply flag"
```

---

## Task 8: Дополнительные simulator-тесты

**Files:**
- Modify: `tests/test_simulator.py`

- [ ] **Step 1: Добавить `test_full_round_runs_on_tiny_model`**

В конец `tests/test_simulator.py`:

```python
def test_full_round_runs_on_tiny_model():
    """End-to-end smoke: 2 clients, ring(2), 5 rounds of update_share runs cleanly."""
    n = 2
    clients = make_tiny_clients(n=n, mezo_lr=1e-3, same_init=True)

    initial_params = [
        {name: p.data.clone() for name, p in c.model.named_parameters()} for c in clients
    ]

    cfg = SimulatorConfig(
        num_rounds=5, consensus_mode="update_share", eval_every=0, log_every=0
    )
    logs = run_simulation(
        clients=clients,
        topology=ring_graph(n),
        loss_fn=_causal_lm_loss,
        config=cfg,
    )

    assert len(logs) == 5
    for entry in logs:
        assert "mean_local_loss" in entry
        assert "mean_projected_grad" in entry
        assert np.isfinite(entry["mean_local_loss"])
        assert np.isfinite(entry["mean_projected_grad"])

    # At least one parameter changed for each client (params were not frozen).
    for i, c in enumerate(clients):
        changed = False
        for name, p in c.model.named_parameters():
            if not torch.allclose(p.data, initial_params[i][name]):
                changed = True
                break
        assert changed, f"Client {i} params unchanged after 5 rounds"
```

- [ ] **Step 2: Добавить `test_nesterov_with_update_share_raises`**

В конец `tests/test_simulator.py`:

```python
def test_nesterov_with_update_share_raises():
    """Nesterov + update_share is currently unsupported and must fail early."""
    n = 2
    clients = make_tiny_clients(n=n)
    for c in clients:
        c.nesterov_state = NesterovState(beta=0.9)

    cfg = SimulatorConfig(
        num_rounds=1, consensus_mode="update_share", eval_every=0, log_every=0
    )

    with pytest.raises(NotImplementedError, match="Nesterov.*update_share"):
        run_simulation(
            clients=clients,
            topology=ring_graph(n),
            loss_fn=_causal_lm_loss,
            config=cfg,
        )
```

- [ ] **Step 3: Запустить тесты**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: все 3 теста pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_simulator.py
git commit -m "test(simulator): add full-round and nesterov-guard tests"
```

---

## Task 9: Многоклиентский consensus тест (нужен F2)

**Files:**
- Modify: `tests/test_consensus.py`

- [ ] **Step 1: Добавить multi-client convergence тест**

В конец `tests/test_consensus.py`:

```python
def test_via_updates_complete_graph_consensus():
    """Same-init clients on a complete graph stay close after many update_share rounds.

    With identical initial params and a complete (rho=0) graph, all clients
    receive the same averaged update each round, so they should stay bit-identical.
    This validates that the F2 apply-flag plumbing is wired correctly via the
    public API: a single ``run_simulation`` call.
    """
    from dmezo.federated.simulator import SimulatorConfig, run_simulation

    n = 4
    clients = make_tiny_clients(n=n, mezo_lr=5e-4, same_init=True)
    cfg = SimulatorConfig(
        num_rounds=20, consensus_mode="update_share", eval_every=0, log_every=0
    )
    run_simulation(
        clients=clients,
        topology=complete_graph(n),
        loss_fn=_causal_lm_loss_local,
        config=cfg,
    )

    # All clients should have nearly identical parameters (rho=0 → one-step consensus).
    ref = {name: p.data for name, p in clients[0].model.named_parameters()}
    for i in range(1, n):
        for name, p in clients[i].model.named_parameters():
            diff = (p.data - ref[name]).abs().max().item()
            assert diff < 1e-4, (
                f"Client {i} param {name!r}: max abs diff {diff} > 1e-4 from client 0"
            )


def _causal_lm_loss_local(model, batch):
    """Local copy of the loss fn (avoids cross-test import coupling)."""
    out = model(**batch)
    return out.loss
```

Замечание: каждый клиент имеет свой `dataloader` с разным seed (`make_tiny_clients` назначает seed=100+i), значит `mezo_step` берёт разные батчи → разные `rho`. На complete graph эти разности усредняются, и **итоговый update одинаков для всех клиентов** — параметры остаются bit-close.

- [ ] **Step 2: Запустить**

Run: `python -m pytest tests/test_consensus.py -v`
Expected: 4 теста pass (3 предыдущих + новый).

- [ ] **Step 3: Commit**

```bash
git add tests/test_consensus.py
git commit -m "test(consensus): add complete-graph multi-client convergence test"
```

---

## Task 10: Финализация документации

**Files:**
- Modify: `docs/07-audit-harden.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Прогнать ВСЕ тесты ещё раз — финальный sanity-check**

Run: `python -m pytest tests/ -v`
Expected: все тесты pass (perturbation: 4 params × multiple = ~10; topology: ~11; fixtures: 3; consensus: 4; simulator: 3 — итого ~31).

Зафиксировать число прошедших тестов и время выполнения. Если что-то падает — стоп, фиксить.

- [ ] **Step 2: Обновить заголовок `docs/07-audit-harden.md`**

Заменить:
```markdown
**Дата:** 2026-05-14. **Статус:** spec (готов к implementation plan).
```
на:
```markdown
**Дата spec:** 2026-05-14. **Статус:** реализовано 2026-05-14 (commit hash: `<заполнить актуальным>`). **План реализации:** [docs/08-audit-harden-plan.md](08-audit-harden-plan.md).
```

(Hash взять из `git log -1 --format=%h`.)

- [ ] **Step 3: Обновить `CLAUDE.md`**

В секции «Текущее состояние проекта» заменить bullet про federated simulator на:

```markdown
- Federated simulator (`src/dmezo/federated/`) — in-process многоклиентный симулятор с настраиваемой topology. Покрыт интеграционными тестами (`tests/test_consensus.py`, `tests/test_simulator.py`); `consensus_via_updates` переписан под O(np), исправлен баг с двойным update в `update_share`. См. `docs/07-audit-harden.md`.
```

В секции «Что делать, когда что-то ломается» дописать пункт:

```markdown
**Nesterov + update_share падает с NotImplementedError.** Это сознательное ограничение — velocity-update внутри consensus не реализован (см. `docs/07-audit-harden.md` D1). Используй либо `consensus_mode="weight_avg"` (Nesterov работает локально), либо `nesterov_state=None`.
```

- [ ] **Step 4: Финальный commit**

```bash
git add docs/07-audit-harden.md CLAUDE.md
git commit -m "docs(audit): mark audit-harden as implemented and update CLAUDE.md"
```

- [ ] **Step 5: Verify git log**

Run: `git log --oneline`
Expected: ~12 коммитов начиная с `chore(repo): initialize git ...`:
1. chore(repo): initialize git with project baseline
2. docs(audit): add audit-harden spec for federated module (07)
3. test(fixtures): add tiny_causal_lm and synthetic_token_loader
4. test(consensus): add equivalence test for via_weights
5. test(consensus): add baseline equivalence tests for via_updates
6. test(consensus): save legacy via_updates as bridge baseline
7. refactor(federated): rewrite consensus_via_updates to O(np)
8. chore(consensus): remove bridge-test artifacts after refactor
9. fix(federated): remove double-update in update_share via apply flag
10. test(simulator): add full-round and nesterov-guard tests
11. test(consensus): add complete-graph multi-client convergence test
12. docs(audit): mark audit-harden as implemented and update CLAUDE.md

---

## Self-review notes (от автора плана)

**Spec coverage** — каждый пункт `docs/07-audit-harden.md` покрыт задачей:
- B1 (consensus_via_updates O(n²p²)) → Task 5 (refactor) под защитой Tasks 2-4 (regression-net + bridge).
- B2 (двойной update) → Task 7 (failing test → fix).
- D1 (Nesterov + update_share) → Task 7 step 4 (NotImplementedError), Task 8 (test_nesterov_with_update_share_raises).
- D2 (per-client vs shared seeds) — out-of-scope, остаётся как note.
- 5 тестов в test_consensus.py → Tasks 2, 3 (×2), 4 (bridge, удаляется в 6), 9.
- 3 теста в test_simulator.py → Tasks 7 (1), 8 (2).
- Update CLAUDE.md + docs/07 → Task 10.

**Placeholder scan** — нет TBD/TODO/«добавить обработку ошибок». Каждый шаг с кодом содержит полный код. ✓

**Type consistency** — `make_tiny_clients` всюду используется с одним и тем же сигнатурой; `_causal_lm_loss` определён локально в каждом тестовом файле, чтобы избежать кросс-импортов; `MixingMatrix` и `ClientState` импортируются из правильных модулей. ✓

**Ambiguity check** — единственное место, где требуется реальное измерение: в Task 7 регрессионный тест полагается на детерминированность `rng_ref` относительно `clients[0].rng` (оба seed=7). Если на старом коде тест ВНЕЗАПНО проходит — это может означать, что `mezo_step` использует свой rng не так, как ожидается; диагностика описана в Step 2. ✓

**Scope check** — единый coherent change set, ~3.5 часа работы, всё локально на CPU, 12 коммитов. Один plan, не нужно декомпозировать. ✓
