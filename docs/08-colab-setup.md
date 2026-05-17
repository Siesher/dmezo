# 08. Setup и запуск на Google Colab

Этот документ — пошаговая инструкция как доставить проект на Colab и запустить Day 1 sanity. Ориентир: Colab Pro+ с RTX PRO 6000 Blackwell.

## TL;DR

1. Доставить код на Drive ИЛИ GitHub (см. ниже).
2. Открыть `notebooks/bootstrap_colab.ipynb` в Colab.
3. Run all cells по порядку.
4. Ожидаемое: eval_loss падает с ~1.5-2 до <0.5 за 1000 шагов, `verdict=PASS`, wall-clock 10-20 мин.

## Опция A — GitHub (рекомендую)

**Плюсы:** легко обновлять (`git pull` в Colab), история коммитов, можно делиться с научруком.

**Минусы:** надо приватный repo (если не хочешь публиковать work-in-progress).

Локально:
```powershell
# 1. Создать приватный repo на GitHub через web UI или gh CLI:
gh repo create dmezo --private --source=. --remote=origin

# 2. Push:
git push -u origin main
```

В bootstrap notebook cell 4 заменить `cp -r` на:
```python
!git clone https://github.com/<your-username>/dmezo.git /content/dmezo
```

Если repo приватный — нужен PAT (Personal Access Token):
```python
import os
os.environ['GH_TOKEN'] = 'ghp_...'  # из Colab Secrets (key icon в sidebar)
!git clone https://${GH_TOKEN}@github.com/<your-username>/dmezo.git /content/dmezo
```

## Опция B — Google Drive

**Плюсы:** ноль setup кроме копирования файлов.

**Минусы:** sync вручную, нет истории.

Локально (PowerShell):
```powershell
# Замени путь на свой Drive Desktop mount.
# Обычно G:\Мой диск\ или G:\My Drive\ — смотри по тому что у тебя смонтировано.
robocopy . "G:\Мой диск\dmezo" /MIR /XD .venv .git mlruns experiments .pytest_cache __pycache__ .claude .ruff_cache .mypy_cache
```

Параметры:
- `/MIR` — зеркалирует (удаляет файлы которых нет в source). Безопасно для свежей синхронизации.
- `/XD <dirs>` — пропустить указанные директории (venv, history, кэши).

Bootstrap notebook cell 4 уже настроен под этот вариант: `cp -r /content/drive/MyDrive/dmezo /content/dmezo`.

## Что проверить перед Colab

**Локально перед доставкой:**
- [ ] `git status` чист (либо все uncommitted намерены)
- [ ] `uv run --no-sync pytest -q` зелёный (27/27)
- [ ] `configs/qwen3_4b_sst2.yaml` имеет lr=3e-7 (выверено локальной ablation, см. `docs/07-audit-harden.md` и MLflow experiment `dmezo_ablation_lr`)

**В Colab после запуска cell 8 (GPU check):**
- [ ] CUDA available: True
- [ ] GPU: NVIDIA RTX PRO 6000 (Blackwell) — или то что выдал Colab
- [ ] BF16 supported: True (нужно для основного конфига)
- [ ] Free memory > 50 GB (для Qwen3-4B FP16 + activations)

**В Colab после cell 10 (tests):**
- [ ] 27 passed in <30s

**В Colab во время cell 12 (sanity run):**
- [ ] step 100: eval_loss падает (типа 1.5 → 0.5-1.0)
- [ ] step 500: eval_loss < 0.5
- [ ] step 1000: eval_loss < 0.4, verdict=PASS

**Если что-то идёт не так:**
- *Loss не падает за 100 шагов* — баг в perturbation/restore. Скачать `console.log` из MLflow artifacts, проверить projected_grad (должен быть non-zero, oscillating).
- *Loss падает потом растёт после ~500 шагов* — divergence. Понизить lr (3e-7 → 2e-7 → 1e-7) в configs/qwen3_4b_sst2.yaml.
- *OOM* — снизить `data.batch_size` (8 → 4) или `data.max_length` (256 → 128).
- *Очень медленно (>1 час)* — Colab дал слабую карту (например L4 вместо Blackwell). Перезапустить runtime, проверить cell 8.

## После успешного Day 1

1. Скачать `mlruns/` обратно на локалку через Drive (если использовалась опция B) или просто оставить на Drive.
2. Локально открыть UI: `uv run --no-sync mlflow ui --backend-store-uri file:./mlruns` → сравнить локальные runs и Colab run.
3. Перейти к Day 2 (см. `docs/05-week1-plan.md`).

## Compute estimate

Из локальных runs (RTX 2080):
- Qwen3-0.6B / 1000 steps: 140s
- Qwen3-1.7B / 1000 steps: 329s

Blackwell ~ 3-5× быстрее на тех же моделях. Поэтому ожидание для Qwen3-4B:
- Если Blackwell справляется в линейной экстраполяции: 329 × (4/1.7) / 4 ≈ 200s ≈ 3-4 мин
- С учётом overhead на bf16, flash-attn, large activations: ~10-20 мин

Compute units на Pro+ Blackwell: 5-15 units на полный 1000-step run (учитывая что Colab берёт ~10-30 units/hour для топ карт). Это **намного меньше** чем 30-50 units из исходного плана.

---

## Section 14: HellaSwag run (D-MeZO-N validation on the hard task)

Section 14 в ноутбуке тестирует D-MeZO-N на **4-way commonsense reasoning** (HellaSwag) — это шаг от лексических tasks (SST-2/BoolQ) к настоящему world-knowledge reasoning.

### Preconditions

- Должны быть pushed коммиты `5474100` (theory) и `80ba96f` (HellaSwag pipeline) на `origin/main`. Локально перед запуском:
  ```powershell
  git push origin main
  ```
- В Colab session перед каждой ячейкой `!cd /content/dmezo && git pull` (уже встроено в cells 14a и 14b).
- Локально перед push: `uv run --no-sync pytest tests/test_hellaswag.py -v` должен быть зелёным (11/11).

### Cells

| Cell | Что делает | Compute |
|---|---|---|
| **14**  | Markdown intro: план 3 runs, success criterion | — |
| **14a** | Centralized HellaSwag baseline (`qwen3_4b_hellaswag.yaml`), 1000 steps | ~12-15 мин, 3-5 units |
| **14b** | D-MeZO-N v1 (4 clients complete IID, β-decay 0.9→0 + clip=50), 1000 rounds | ~15-20 мин, 5-7 units |
| **14c** | Comparison table: centralized vs federated vs random, verdict | <1 мин |

### Success criteria

1. **14a centralized:** `final_eval_acc ≥ 0.30` (≥ random + 5pp), loss падает на ≥10%.
   - Если ниже 0.30 — HellaSwag слишком сложна для 1000 steps на этом lr; попробуй lr=1e-6.
2. **14b federated D-MeZO-N:** `final_eval_acc ≥ centralized × 0.9` (partition tax ≤ 10%).
   - Monotonic descent (β-decay должна давать R1d-like поведение, без late drift).
3. **14c verdict:** PASS если оба критерия выполняются И `f_acc > random + 5pp = 0.30`.

### Что значит каждый исход

| Исход | Интерпретация для paper |
|---|---|
| **PASS (14b > 14a × 0.9)** | D-MeZO-N работает на нетривиальной reasoning task — закрывает раздел «Real-world validation» |
| **PARTIAL (14b > random + 5pp, но 14b < 14a × 0.9)** | Federated подход работает, но partition tax выше ожидаемого — нужен ablation alpha=2.0 или N=2 |
| **FAIL (14b ≤ random + 5pp)** | HellaSwag слишком сложна на этом compute budget — нужно либо больше steps, либо tune lr/eps, либо warm-start с SST-2 weights |

### Если что-то идёт не так

- **`final_acc == init_acc` (no learning):** lr слишком мал или ρ заклипилось в 0. Поднять lr до 1e-6, проверить `params.mezo.rho_clip` в logged MLflow.
- **NaN после ~200-300 раундов:** β-decay не помогла (наш Day 6 negative). Снизить начальный β до 0.7 или поднять clip до 30.
- **OOM на 14b:** Qwen3-4B × 4 clients = большая память. В config: `data.batch_size: 4` → `2`, или `federated.local_steps: 1` (уже стоит).

### Next ablations (если PASS)

После PASS — есть прямой путь к paper update:
- 14d: same config, но `partition_mode: dirichlet`, `partition_kwargs.alpha: 0.5` (non-IID HellaSwag — ни Malladi, ни FedKSeed это не делали)
- 14e: same config, но `topology: ring` (slow-mixing stress test на reasoning task)
- 14f: ablation `nesterov.enabled: false` — то же самое без D-MeZO-N rescue, для apples-to-apples comparison с paper Theorem 2 (vanilla MeZO PL bound)
