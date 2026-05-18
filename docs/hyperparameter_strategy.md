# D-MeZO-N: стратегия выбора гиперпараметров

Этот документ — практическое руководство по 4 ключевым гиперпараметрам
**D-MeZO-N** на основе наших эмпирических данных и теоретических границ:

1. **$K$** — число направлений multi-direction SPSA
2. **$\eta$ (lr)** — learning rate (с расписанием)
3. **$B$** — размер mini-batch
4. **$\varepsilon$** — магнитуда возмущения

Каждый раздел: (i) текущая практика, (ii) теория, (iii) эмпирические находки,
(iv) рекомендации, (v) открытые вопросы.

---

## 1. K — число направлений в multi-direction SPSA

### Текущая практика

`MeZOConfig.k_directions = 1` (default = стандартный MeZO). С `K > 1`
выполняется:

$$
\tilde g_t = \frac{1}{K} \sum_{k=1}^K \hat\rho_k z_{s_k}, \qquad
\hat\rho_k = \frac{L(\theta_t + \varepsilon z_{s_k}) - L(\theta_t - \varepsilon z_{s_k})}{2\varepsilon}.
$$

### Теория

- **Variance:** $\operatorname{Var}(\tilde g) = \operatorname{Var}(\hat g) / K$
  (независимые $z_k$). Эмпирически подтверждено в `tests/test_md_mezo.py`.
- **Bias:** не меняется с $K$ — bias приходит из $O(\varepsilon^2)$-членов
  Тейлора, не лечится averaging.
- **Compute:** $2K$ forward passes за шаг.
- **Communication (federated):** $K$ floats + $K$ ints за раунд на соседа.

### Эмпирические находки (2026-05-18, Day 5 worst cell ablation)

| Метрика | K=1 (R1d baseline) | **K=3 ablation** | Δ |
|---|---|---|---|
| Final eval loss | **0.1291** | **0.1828** | **+41.6%** ⚠️ |
| Final eval acc | 0.9563 | **0.9688** | **+1.25pp** ✅ |
| Wall-clock 1000 rounds | ~37 мин | ~68 мин | **+84%** |
| ρ-range post-clip | [-50, +50] | [-25, +33] | tighter |

**Ключевой вывод:** K=3 даёт **trade-off**, не Pareto improvement. K-direction
работает как **generalization regularizer** (acc up), не как pure speedup
(loss worse).

### Рекомендации по выбору K

| Цель | Рекомендация |
|---|---|
| **Минимизация loss** на fine-tuning | $K = 1$ (наш R1d рецепт) |
| **Максимизация accuracy** на downstream task | Попробовать $K = 3$ |
| **Compute-sensitive** runs | $K = 1$ |
| **Hard reasoning task** (где vanilla diverges) | $K = 1$ + clip=50 уже достаточно (HellaSwag rescue) |
| **High-noise task** (multi-token labels) | $K = 3$–$5$ может помочь |
| **Industrial deployment** | $K = 1$ (best loss + lowest cost) |

### Открытые вопросы

1. **Sweet spot $K$:** мы протестировали только $K=1$ и $K=3$. Возможно $K=5$
   даёт другой trade-off (test_md_mezo показывает variance reduction до $K=10$).
2. **$K$ + look-ahead Nesterov:** Day 6b показал что look-ahead diverges в 7×
   быстрее. Гипотеза reviewer'а: K=5 + look-ahead = OK. **Не протестировано**.
3. **Adaptive $K$:** start with $K=3$ для warmup (stable convergence), потом
   снизить до $K=1$ когда модель в noise neighborhood (cheap fine-tuning).
4. **$K$ × $B$ trade-off:** при фиксированном compute, что лучше —
   $K=3, B=4$ или $K=1, B=12$? §6.4 paper'а предсказывает $K$, но не
   протестировано напрямую.

---

## 2. lr (learning rate) — расписание

### Текущая практика

`MeZOConfig.lr_schedule = "constant"`, $\eta = 3 \cdot 10^{-7}$. Princeton MeZO
convention (Malladi 2023).

С commit `ec69fcf` доступны 3 schedules через `effective_lr(config, round_idx, num_rounds)`:

- `"constant"` — $\eta_t = \eta_0$ (текущий default)
- `"harmonic"` — $\eta_t = \eta_0 \cdot a / (t + 1)^{\alpha}$ (Spall 1992 §3.2, $\alpha = 0.602$ classical)
- `"cosine"` — $\eta_t = \eta_0 \cdot \tfrac{1}{2}(1 + \cos(\pi \cdot t / (T-1)))$

### Теория

**Constant lr** (наш):
- Сходимость только к **noise neighborhood** размера $4G^2/(3\mu)$ (Theorem 3).
- НЕ к истинному оптимуму $\theta^*$.
- Скорость на ранней стадии: $O(1/T)$.

**Harmonic decay** $\eta_t = a/(t+1)^{\alpha}$ (SPSA classical):
- Сходимость к $\theta^*$ **гарантирована** при $\alpha > 0.5$.
- Скорость: $O(1/T^{2\alpha - 1})$ для $\alpha < 1$.
- Для $\alpha = 0.602$: $O(1/T^{0.204})$ — медленнее чем constant initial descent.

**Cosine annealing:**
- Heuristic, без строгих SPSA-гарантий, но эмпирически популярная в DL.
- Достигает $\eta_T = 0$, что даёт сходимость к локальному минимуму noise neighborhood.

### Эмпирические наблюдения (LLM fine-tuning)

Princeton MeZO (Malladi 2023, Table 1) использовал **constant lr** на всех
задачах с success — empirical evidence что для fine-tuning LLM
"good-enough"-сходимость достаточна.

Наши runs (Day 1, Day 5, Day 8 R1d, HellaSwag, MathLogicQA) **все на constant
lr** — все PASS (acc > random + δ для соответствующих threshold'ов).

**Не тестировано:** harmonic vs cosine на одинаковом setup.

### Рекомендации по lr-schedule

| Сценарий | Рекомендация |
|---|---|
| **Short fine-tuning** (≤ 1000 шагов) | `constant` ($\eta = 3 \cdot 10^{-7}$) |
| **Long training** (≥ 10000 шагов) | `cosine` или `harmonic` — escape noise floor |
| **Theoretical guarantee** к $\theta^*$ | `harmonic` с $\alpha = 0.602$ |
| **Industrial deployment** | `constant` (proven, predictable) |
| **Hard task** где vanilla diverges | `cosine` (early warmup avoids divergence) |
| **Hyperparameter sweep новая модель** | Сравнить все три schedule на 200-step preview run |

### Открытые вопросы

1. **Empirical comparison** {constant, harmonic, cosine} на одном setup.
2. **Warmup + decay:** linear warmup 100 steps + cosine decay — стандартная
   schema для transformer pretraining, не тестировали на MeZO.
3. **Adaptive lr:** reduce-on-plateau стиля для federated case.
4. **Per-layer lr:** различные lr для embeddings vs attention vs FFN.

---

## 3. B — batch size

### Текущая практика

- SST-2: $B = 8$
- BoolQ: $B = 4$ (длинные passages)
- HellaSwag: $B = 4$
- MathLogicQA: $B = 4$

### Теория (CLT prediction)

Классический CLT: для эмпирической оценки градиента на batch size $B$,
стандартная ошибка scale'ируется как $1/\sqrt{B}$:

$$
\operatorname{std}[\nabla L_B(\theta)] = \operatorname{std}[\nabla L_1(\theta)] / \sqrt{B}.
$$

В MeZO эта oценка проецируется на $z$, давая:

$$
\operatorname{std}[\hat\rho(B)] \stackrel{\text{CLT}}{=} \operatorname{std}[\hat\rho(1)] / \sqrt{B}.
$$

### Эмпирические находки (2026-05-18, fig8_batch_variance.png)

**CLT-предсказание ФАЛЬСИФИЦИРОВАНО на Qwen3-0.6B / SST-2:**

| B | observed σ(ρ) | CLT-expected | ratio |
|---|---|---|---|
| 1 | 459.6 | 459.6 | 1.00× |
| 2 | 505.0 ⬆ | 325.0 | **1.55×** |
| 4 | 441.0 | 229.8 | **1.92×** |
| 8 | 261.8 | 162.5 | 1.61× |
| 16 | 261.1 | 114.9 | **2.27×** |
| 32 | 279.0 ⬆ | 81.3 | **3.43×** |

**std выходит на плато при $B \geq 8$**, увеличение $B$ дальше **не помогает**.

### Интерпретация

Доминантный источник шума в MeZO — **НЕ data sampling**, а:

1. **Direction noise** (выбор $z$). Главный вклад. Mitigation: multi-direction
   ($K > 1$), не batch.
2. **fp16/bf16 precision floor** при вычислении $L_+ - L_-$. Secondary.
3. **Тейлоровская нелинейность** при $\varepsilon \|z\| \gg 0$ (для нашего
   setup $\varepsilon \|z\| \approx 63$ на Qwen3-4B). Превышение линейного
   режима — variance имеет структуру не CLT-типа.

### Рекомендации по B

| Сценарий | Рекомендация |
|---|---|
| **Default** (most tasks) | $B = 4$–$8$ |
| **Long sequences** (max_length ≥ 512) | $B = 2$–$4$ (memory) |
| **Hard reasoning** (HellaSwag, MathLogicQA) | $B = 4$ |
| **Diminishing returns:** | $B > 16$ редко даёт пользу |
| **Optimization для variance:** | $K$ (multi-direction), а не $B$ |

### Открытые вопросы

1. Где именно лежит **noise floor** (precision vs direction)?
2. Влияет ли $B$ на bias (а не variance)? — теоретически нет.
3. Effective sample size в padded батчах: эффективное $B$ может быть меньше
   nominal $B$ из-за паддинга длинных sequences.

---

## 4. ε — магнитуда возмущения

### Текущая практика

`MeZOConfig.eps = 1e-3` (constant, Malladi 2023 convention).

### Теория

ε балансирует **bias** и **variance**:

$$
\hat\rho = \underbrace{z^\top \nabla L}_{\text{unbiased}} + \underbrace{O(\varepsilon^2 \|z\|^3 \|H\|_{\text{op}})}_{\text{Taylor bias}} + \underbrace{O(\sigma_{\text{numeric}} / \varepsilon)}_{\text{numerical noise}}
$$

- **Малый ε**: меньше bias, но больше численного шума ($L_+ - L_-$ ≈ 0)
- **Большой ε**: больше bias из higher-order Taylor terms, но меньше численного шума

**Optimum (Spall 1992):** $\varepsilon^* = O(\sigma_{\text{numeric}}^{1/3} / \|H\|^{1/3})$ —
зависит от модели и backend precision.

### Архитектурная sensitivity

Для $\varepsilon = 10^{-3}$ и $\|z\| = \sqrt{d}$:

| Модель | $d$ (params) | $\varepsilon \|z\|$ | Линейный режим? |
|---|---|---|---|
| Qwen3-0.6B | $6 \times 10^8$ | $\approx 25$ | Нет, далеко |
| Qwen3-4B | $4 \times 10^9$ | $\approx 63$ | Нет, далеко |
| Qwen3.5-4B-Base | $4.4 \times 10^9$ | $\approx 66$ | Нет |
| Hypothetical Llama-13B | $1.3 \times 10^{10}$ | $\approx 114$ | Сильно нелинейный |

**Все наши setup'ы — в нелинейном режиме.** Тем не менее MeZO работает —
значит loss surface достаточно "smooth in expectation" (Princeton MeZO
объясняет через $r(H) \ll d$).

### Эмпирические наблюдения

- Все наши успешные runs используют $\varepsilon = 10^{-3}$
- Не пробовали $\varepsilon = 10^{-2}$ или $\varepsilon = 10^{-4}$ системно
- ρ-magnitudes пропорциональны: на HellaSwag $|\rho| \sim 100$, на MathLogicQA
  $|\rho| \sim 300$ — разница архитектура+task, не ε

### Твоя интуиция: "оптимальный ε зависит от модели, OK иметь константным"

**Согласен с уточнением:** оптимальный ε зависит от:
- Архитектуры (spectrum $H$)
- Precision backend (fp16, bf16, fp32)
- Task (loss landscape)

Для **одной комбинации** (model, precision, task) ε можно сделать константным.
Но для cross-model/cross-task generality нужен **per-setup tuning**.

### Proposed warmup autotuner

Идея: первые $N \approx 50$ раундов — **probe phase**, где мы сэмплируем разные
$\varepsilon \in \{\varepsilon_1, \ldots, \varepsilon_M\}$ и измеряем
signal-to-noise ratio (SNR). Выбираем $\varepsilon^*$ с лучшим SNR и
продолжаем с ним.

**Algorithm:**

```
def autotune_eps(model, dataloader, candidates=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                 n_probes_per_eps=10):
    """
    Returns: optimal ε from candidate set based on signal-to-noise.
    """
    stats = {eps: {"rhos": []} for eps in candidates}
    for round_idx in range(n_probes_per_eps * len(candidates)):
        eps = candidates[round_idx % len(candidates)]
        batch = next(dataloader)
        seed = next_seed()
        # MeZO step at this ε (no update applied — just probing)
        rho = compute_rho(model, batch, eps, seed)
        stats[eps]["rhos"].append(rho)

    snr_table = {}
    for eps, st in stats.items():
        rhos = np.array(st["rhos"])
        # SNR = |mean ρ| / std ρ
        # High SNR = signal dominates noise = good ε
        snr = abs(rhos.mean()) / (rhos.std() + 1e-12)
        snr_table[eps] = snr

    eps_star = max(snr_table, key=snr_table.get)
    return eps_star, snr_table
```

**Critique нашего собственного дизайна:**

1. **SNR metric** не идеален — высокий SNR может означать просто что мы в
   деградированном режиме (ε слишком большой → детерминистический shift, low
   variance, но high bias). Лучше комбинировать с **bias estimate** (через
   $L(\theta+\varepsilon z) + L(\theta-\varepsilon z) - 2 L(\theta) \approx \varepsilon^2 z^T H z$).
2. **Probe phase overhead:** $5 \times 10 = 50$ rounds wasted на autotune.
3. **Same direction problem:** если мы используем один $z$ для всех ε, SNR
   статистика смешана с direction-noise. Нужно **rerolled** $z$ для каждого
   probe.
4. **Federated case:** автотюн нужно делать **глобально** (все клиенты
   соглашаются на $\varepsilon^*$), не локально. Один клиент-coordinator
   проводит warmup, broadcast'ит результат.

**Cleaner design (recommended):**

```
def warmup_eps_grid(model, dataloader, candidates, n_probes_per_eps,
                    target_metric="bias_var_ratio"):
    """
    Probe-and-pick: pick ε minimizing (bias² + var) on a held-out batch.

    bias² = (L(θ+εz) + L(θ-εz) - 2 L(θ))² / (2εζ²)²
            ~ (ε² z^T H z / 2)² — quadratic Taylor term

    var   = Var[ρ̂] over rolled z

    Trade-off score = bias²(ε) + var(ε)
    """
    L0 = forward(model, holdout_batch)  # baseline
    scores = {}
    for eps in candidates:
        rhos, biases = [], []
        for _ in range(n_probes_per_eps):
            seed = next_seed()
            apply_perturb_(model, seed, +eps)
            L_plus = forward(model, batch)
            apply_perturb_(model, seed, -2 * eps)
            L_minus = forward(model, batch)
            apply_perturb_(model, seed, +eps)  # restore
            rho = (L_plus - L_minus) / (2 * eps)
            bias_proxy = (L_plus + L_minus - 2 * L0) / (eps ** 2)  # Taylor 2nd
            rhos.append(rho); biases.append(bias_proxy)
        var = np.var(rhos)
        bias2 = np.mean(biases) ** 2  # squared mean Taylor proxy
        scores[eps] = bias2 + var
    return min(scores, key=scores.get), scores
```

**Compute cost autotune:** $|\text{candidates}| \cdot N_{\text{probes}} \cdot 2$
forwards. Для 5 candidates × 10 probes = 100 forwards. Сравнимо с 50 normal
MeZO steps. **Discounted из общего budget'а warmup.**

### Рекомендации по ε

| Сценарий | Рекомендация |
|---|---|
| **Default** (Qwen3-class) | $\varepsilon = 10^{-3}$ (Malladi) |
| **Новая модель** (другая arch) | Прогнать warmup autotuner (proposed) |
| **fp32 backend** | Можно меньшее $\varepsilon$ (меньше precision floor) |
| **fp16/bf16 backend** | $\varepsilon \geq 10^{-3}$ обязательно (precision) |
| **Long-context** (max_length ≥ 512) | $\varepsilon$ stays (зависит от model, не data) |

### Открытые вопросы

1. Ablation: $\varepsilon \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ на одинаковом setup.
2. ε-schedule: имеет ли смысл $\varepsilon_t \downarrow$ к концу обучения? (по SPSA — да, $\gamma \approx 0.101$)
3. Per-layer ε: возможно матрицы attention требуют другого ε чем LayerNorm
   (Hessian spectrum различен).

---

## 5. Сводная таблица рекомендаций

| HP | Default | When to change | Tool |
|---|---|---|---|
| **$K$** | 1 | Если acc более важно чем loss | `MeZOConfig.k_directions` |
| **$\eta$** | $3 \cdot 10^{-7}$ const | Long runs или новая arch | `MeZOConfig.lr_schedule` |
| **$B$** | 4-8 | Memory только | manual config |
| **$\varepsilon$** | $10^{-3}$ const | Новая arch или fp32 backend | (TODO) warmup autotuner |

## 6. Priority следующих экспериментов

1. **ε ablation** на one cell: $\varepsilon \in \{10^{-4}, 3 \cdot 10^{-4}, 10^{-3}, 3 \cdot 10^{-3}, 10^{-2}\}$ → паттерн bias-variance.
2. **K=5 + K=10 ablation** для проверки diminishing returns.
3. **lr schedule sweep:** constant vs harmonic vs cosine на R1d setup.
4. **eps warmup autotuner:** реализация + валидация на cross-arch (Qwen3 vs Qwen3.5).
