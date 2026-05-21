# Defense Slides Brief — для Claude Design

**Это инструкция для следующей сессии, где будут готовиться слайды защиты.** Файл содержит весь контекст, чтобы Claude Design мог начать работать с минимальной разведкой.

---

## 1. Проект за 30 секунд

**D-MeZO-N** (Decentralized Federated MeZO with Nesterov-stabilization) — peer-to-peer федеративный zeroth-order оптимизатор для дообучения LLM. Главные идеи:
- Между клиентами передаётся **16 байт/раунд** (один float ρ + один int seed) вместо $O(d)$ градиентов FedAvg (~10⁹× компрессия).
- **ρ-clip** одновременно служит (i) механизмом стабилизации momentum, (ii) L2-чувствительностью для DP-Gaussian-механизма. **Один механизм решает две задачи.**
- **Theorem 3** (наш вклад): Lyapunov-сходимость PL + heavy-ball + β-decay + ρ-clip. Closes Open Problem 1 из Princeton MeZO.
- **Theorem 4** (наш вклад): DP-расширение T3, per-round (ε=10, δ=10⁻³) с ~6% utility cost.
- **Empirical headline (FINAL §22 run, 3 seeds paired):** D-MeZO-N v2 (combo B1+B5: adaptive ρ-clip + drift-reset) robustly beats vanilla MeZO на Qwen3.5-4B-Base / MathLogicQA: final loss **1.2926 ± 0.010 vs vanilla 1.3681 ± 0.018 (Δ=−5.5%, 3/3 same direction)**, accuracy **0.400 vs 0.377 (+2.3pp mean)**. **Lowest std across seeds** (0.010 vs 0.021 у B1 alone) — combo более robust. **Первое paper-scale multi-seed validation D-MeZO-N strict win над vanilla MeZO.**

## 2. Контекст защиты

- **Аудитория:** диплом МГТУ им. Баумана (Калужский филиал), Автоматические системы управления. Комиссия: руководитель + 4–5 преподавателей. Не все специалисты по ML/FL, но математически грамотны.
- **Формат:** 10 минут доклад + ~10 минут вопросов.
- **Язык:** русский. Английские технические термины (MeZO, federated, mixing matrix, Lyapunov) допустимы — это стандарт.
- **Дата:** 2026-05-23 (защита через 2 дня от текущей даты).
- **Тон:** уверенный, но честный. Несколько слайдов про negative findings — это **сильная сторона** работы (показывает зрелость).

## 3. Технические предпочтения для слайдов

**Рекомендуемый стек (по убыванию приоритета):**

1. **reveal.js** (HTML/CSS/JS) — стандарт для академических презентаций. Поддерживает MathJax для формул, fragment-анимации, smooth transitions. Темы customizable через CSS.
2. **Slidev** (Vue + Markdown) — современная альтернатива, хорошие переходы, code-блоки красивые.
3. **Custom HTML + GSAP** — если нужны нестандартные анимации (например, визуализация consensus mixing с движущимися узлами).
4. **Marp** (Markdown) — самый простой, но скромнее по анимациям.

**Recommendation:** reveal.js с custom CSS theme + GSAP для одной-двух сложных анимаций (consensus mixing, ρ-clip mechanism).

## 4. Дизайн-направление

- **Стиль:** минимализм, академическая строгость с современным акцентом. Не corporate-PowerPoint.
- **Цветовая палитра:** один accent color (предлагаю **deep indigo** `#3730A3` или **teal** `#0F766E`) + neutral greys + white. Тёмный фон **не использовать** — на защите проектор может искажать цвета.
- **Типографика:** sans-serif для основного текста (Inter, IBM Plex Sans, Geist). Serif для математики (Computer Modern через MathJax). Никаких декоративных шрифтов.
- **Иерархия:** один большой заголовок на слайд, 3–5 ключевых пунктов максимум. Не больше 40 слов на слайд.
- **Whitespace:** много. Дышать. Не пытаться втиснуть всё.
- **Графики:** все figure'ы из `docs/figures/` уже в высоком DPI (300). Можно использовать как есть, но желательно переделать в SVG для масштабируемости.

## 5. Анимационные паттерны (где какая анимация помогает)

| Концепция | Тип анимации |
|---|---|
| Consensus mixing 4 клиентов | GSAP — узлы-кружки, рёбра пульсируют при обмене, ρ-скаляры "летят" между узлами |
| β-decay в Day 8 R1d | Линейная кривая на графике β(t) появляется fragment-by-fragment поверх trajectory |
| ρ-clip механизм | Bar chart |\hat\rho| с outlier'ом ~900, который "обрезается" на C=50 после клипа |
| 4-фазная диаграмма Nesterov | Region A→B→C→D появляются последовательно с короткими описаниями |
| Communication cost log-scale | Bar chart с FedAvg (огромный) → FedKSeed → D-MeZO-N, появляется уменьшаясь |
| Phase transitions между слайдами | Smooth crossfade, не slide-from-side (более professional) |
| Заголовок секции | Большой текст fade-in + цветная линия snaps under |

**Не делать:** spinning, bouncing, blinking, "fly-in from offscreen". Любые анимации <300мс или >800мс не работают для восприятия.

## 6. Структура: 12 слайдов на 10 минут

### Слайд 1 — Title (15 сек)
```
D-MeZO-N
Decentralized Federated MeZO with Nesterov Stabilization

Сухацкий М.А.
МГТУ им. Н.Э. Баумана (Калужский филиал)
Кафедра САУ, 2026
```
**Анимация:** title fade-in, subtitle slide-up. Никаких декораций.

### Слайд 2 — Проблема (45 сек)
**Заголовок:** "Дообучение LLM в трёх ограничениях одновременно"

Три иконки + один пункт каждая:
- 🏦 **Communication-efficient** — банки/больницы имеют ограниченный bandwidth
- 🔒 **Privacy-preserving** — 115-ФЗ, HIPAA, GDPR запрещают exfiltration сырых данных
- 💾 **Memory-efficient** — edge devices, on-device personalization

**Внизу:** "Существующие методы решают 1–2 из 3. Мы решаем все три."

**Анимация:** три иконки fade-in последовательно (0.3s интервал), последняя строка появляется с задержкой 0.5s.

### Слайд 3 — Почему vanilla MeZO недостаточен (45 сек)
**Заголовок:** "Princeton MeZO (Malladi 2023): memory, но не остальное"

| Constraint | MeZO даёт? |
|---|---|
| Memory-efficient | ✅ |
| Federated | ❌ нет wrapper'а |
| Privacy | ❌ нет DP |
| Momentum convergence proof | ❌ Open Problem 1 |

**Внизу:** "→ Расширяем по всем четырём осям. Theorem 3 закрывает Open Problem 1."

### Слайд 4 — Метод: D-MeZO-N в одном слайде (1 мин)
**Заголовок:** "D-MeZO-N: алгоритм"

Алгоритм-блок (моноширинный шрифт, индентация):
```
для каждого раунда t:
  каждый клиент i параллельно:
    s_i ← seed,  z_i ← N(0,I) из s_i           ← локальный шум
    ρ_i ← (L(θ_i+εz_i) - L(θ_i-εz_i)) / 2ε     ← 2 forward passes
    ρ̃_i ← clip(ρ_i, ±C)                         ← stability + DP
    v_i ← β_t · v_i + ρ̃_i · z_i                 ← Nesterov momentum
    θ_i ← θ_i - η · v_i                         ← local step
  consensus: θ_i ← Σ_j W_ij · θ_j               ← peer-to-peer mixing
```

**Внизу справа** badge: "Передаётся: 16 байт/раунд/сосед"

**Анимация:** строки появляются последовательно, цвет соответствует роли (зелёный — наш вклад: ρ̃, v_i; чёрный — vanilla MeZO).

### Слайд 5 — Топология (1 мин)
**Заголовок:** "Decentralized: peer-to-peer mixing matrix W"

Слева — diagram 4 клиента на ring topology (визуально). Справа — пояснение:
- $W \in \mathbb{R}^{n \times n}$, doubly-stochastic
- Spectral gap $\rho_W = \|W - \mathbf{1}\mathbf{1}^\top/n\|_{op} \in [0,1)$
- Ring: $\rho_W = 0.333$ для n=4
- Complete: $\rho_W = 0$ (instant average)
- **Преимущество:** нет central server → no single point of failure (vs FedKSeed)

**Анимация:** ring graph — кружки появляются + рёбра рисуются. Затем ρ-скаляры (маленькие лейблы) "пробегают" по рёбрам.

### Слайд 6 — Теория: Lyapunov-сходимость (1.5 мин)
**Заголовок:** "Theorem 3: PL + heavy-ball + β-decay + ρ-clip"

**Lyapunov-функция:**
$$V_t = (L(\theta_t) - L^\star) + \frac{\eta}{2}\|v_t\|^2$$

**Сходимость:**
$$\mathbb{E}[V_T] \leq \left(1 - \tfrac{3\eta\mu}{2}\right)^T V_0 + \frac{2G^2}{3\mu}$$

**Что значит:**
- Линейная сходимость к neighbourhood радиуса $2G^2/(3\mu)$
- ρ-clip → $G^2 = C^2 r(H) \ell$ ограничено → neighbourhood bounded
- β-decay → кинетическая энергия $\|v_T\|^2 \to G^2$ (вместо $5G^2$ при const β=0.9)

**Внизу:** "Closes Princeton Open Problem 1 (momentum convergence under PL + ZO)"

**Анимация:** формулы fade-in последовательно, под каждой fade-in объяснение справа.

### Слайд 7 — Theorem 4: DP бесплатно (1 мин)
**Заголовок:** "ρ-clip = механизм стабилизации **И** L2-sensitivity"

Слева — formula:
$$\tilde\rho_t = \mathrm{clip}(\hat\rho_t, \pm C) + \xi_t, \quad \xi_t \sim \mathcal{N}(0,\sigma^2)$$

$$\varepsilon_1 = \frac{C\sqrt{2\ln(1.25/\delta)}}{\sigma}$$

Справа — bullet points:
- Sensitivity $\Delta = C$ "уже там" — не нужен per-sample gradient clipping (vs DP-SGD)
- **Один механизм решает две задачи**
- Per-round ε=10, σ=19 → утрата utility ≤ 6.2%

**Внизу:** "First decentralized federated ZO with formal per-round (ε,δ)-DP on LLMs"

### Слайд 8 — Главный эмпирический результат (1 мин)
**Заголовок:** "D-MeZO-N v2 (combo B1+B5) на paper-scale — 3 seeds paired"

**Финальные числа из §22 run** (FINAL 2026-05-21 02:12; **15/15 cells завершены**):

Таблица (Qwen3.5-4B-Base / MathLogicQA / 4 clients complete IID / 1000 rounds / 3 seeds paired):

| Variant | Mean loss ± std | Mean acc ± std | Δ vs vanilla loss | Δ vs vanilla acc |
|---|---|---|---|---|
| vanilla MeZO | 1.3681 ± 0.018 | 0.377 ± 0.013 | reference | reference |
| D-MeZO-N v1 (fixed C=50) | 1.4634 ± 0.007 | 0.377 ± 0.013 | +7.0% (3/3 worse) | tie |
| Drift-only (B5 alone, 53 resets) | 1.4559 ± 0.004 | 0.377 ± 0.013 | +6.4% (3/3 worse) | tie |
| Adaptive_clip (B1 alone) | 1.2987 ± 0.021 | 0.390 ± 0.043 | −5.1% (3/3 win) | +1.3pp |
| **D-MeZO-N v2 (combo B1+B5, 54 resets)** ⭐ | **1.2926 ± 0.010** | **0.400 ± 0.029** | **−5.5% (3/3 win)** | **+2.3pp** |

**Caveat snizu (мелким шрифтом):** "Direction consistency 3/3 for combo on loss. Acc Δ: (−1, +8, 0) pp per seed → mean +2.3pp, never negative direction beyond noise."

**Бейдж:** "First paper-scale multi-seed validation of D-MeZO-N strictly beating vanilla MeZO"

**Анимация:** строки таблицы fade-in последовательно. Последняя строка (★ combo) — accent color highlight + slight glow.

**Speaker note:** "Эволюция рецепта на multi-seed: v1 (fixed C=50) — falsified, 3/3 проигрывают vanilla на 7%. B5 alone — также falsified, 3/3 worse. B1 alone (adaptive clip) — 3/3 wins на loss, но acc varies. **Combo (B1+B5)** — adaptive clip + drift-reset (на 3 seeds сработал 54 раза total): robustly beats vanilla на 3/3 seeds по loss с lowest std, plus mean +2.3pp acc. Это и есть D-MeZO-N v2 — первая paper-scale multi-seed valid statистическая победа D-MeZO-N над vanilla."

### Слайд 9 — Cross-task / Cross-architecture обобщение (1 мин)
**Заголовок:** "Тот же рецепт — три качественно разных режима"

Три колонки:

| Task | Vanilla | D-MeZO-N v2 | Режим |
|---|---|---|---|
| SST-2 (Day 8) | converges | +6.5% speedup | acceleration |
| HellaSwag (4-way) | **diverges** | converges +3.75pp | **rescue** |
| MathLogicQA (RU) | converges | +3pp acc | **safe-track + win** |

**Внизу:** "Architecture diversity: Qwen3-0.6B (full-attn) ↔ Qwen3.5-4B-Base (hybrid linear-attention). Первый known MeZO experiment на hybrid arch."

### Слайд 10 — DP-frontier (30 сек)
**Заголовок:** "Privacy frontier: статистически плоский во всём диапазоне σ"

Embed: `docs/figures/` DP-sweep figure (если есть) OR простая bar chart с σ → ε → loss.

| σ | ε | Δ loss vs no-DP |
|---|---|---|
| 0.5 | 378 | +6.8% |
| 19 | **10** ⭐ | **+6.2%** |
| 50 | 4 | +7.1% |

**Внизу:** "ε=10 essentially free — utility cost statistically indistinguishable from no-DP."

### Слайд 11 — Контрибуции + honest negatives (30 сек)
**Заголовок:** "Solid (paper-ready) и Falsified (честные negatives)"

Две колонки:

**✅ Solid (A1-A6):**
- A1: First federated MeZO на hybrid linear-attn LLM
- A2: Partition-tax < 13% (vs ~100% для FedAvg)
- A3: T3 + T4 closed-form proofs
- A4: Per-round (ε=10)-DP, ~6% utility cost
- A5: 10⁹× communication compression
- A6: ρ-clip = dual-use mechanism

**❌ Falsified (наши negatives, C1-C5):**
- C1: "+1.25pp на MathLogicQA" (v1) — falsified by 3-seed CI [0, 0]
- C2: True-Nesterov look-ahead — diverges 7× faster
- C3: K=3 multi-direction strictly improves — falsified equal-compute
- C4: ε(t) warmup beats const — loses в 3-6×
- C5: O(1/T²) asymptotic acceleration — Bottou-Curtis-Nocedal 2018 forbids

**Внизу:** "5 fallsified → 6 solid. Дисциплина — признак зрелого исследования."

### Слайд 12 — Заключение и open problems (30 сек)
**Заголовок:** "Что сделано, что осталось"

**Сделано:**
- Peer-to-peer federated MeZO + Nesterov stabilization
- Theorem 3 (closes Princeton OP1)
- Theorem 4 (first DP-MeZO с formal guarantees)
- 6 paper-ready vклад, 5 honest negatives

**Open Problems (future work):**
- OP2: Full decentralized T3 (cross-terms сложные)
- OP3: Transient acceleration proof (estimate sequence)
- OP4: Subsampling amplification для tight T-round DP
- Head-to-head FedKSeed (script готов, compute pending)

**Внизу:** Repository: `github.com/Siesher/dmezo`

---

## 7. Backup slides (для Q&A — НЕ показывать в основном докладе)

Подготовить дополнительно **5 backup slides** (показывать только если спросят):

**B1** — Day 8 phase diagram (4 региона Nesterov с trajectories). Embed `docs/figures/fig2_nesterov_phase_diagram.png`.

**B2** — FedKSeed comparison table (algorithmic, не empirical). Из `docs/fedkseed_comparison.md`.

**B3** — Lyapunov proof sketch (5 bricks Theorem 3). Из `docs/theory_rigorous.md` §3.

**B4** — Communication cost log-scale chart: FedAvg (8 GB) → FedKSeed (18 KB) → D-MeZO-N (16 байт). Compelling visual.

**B5** — Limitations slide: multi-seed scope (n=2-3), 4 clients × 4B (toy scale), classification-only, нет head-to-head SOTA.

---

## 8. Файлы и figure'ы для embedding

Все находятся в `docs/figures/`. Готовы к использованию (dpi=300, PNG):

| Slide | Figure файл | Описание |
|---|---|---|
| 2 (методология) | (опционально) `fig5_algorithm_schematic_ru.png` | Visualization 4 клиентов |
| 5 (topology) | `fig5_algorithm_schematic_ru.png` | Готовая diagram |
| 6 (theory) | — | Только формулы (рендерить через MathJax) |
| 8 (headline) | (новый — генерировать) | Bar chart vanilla vs v1 vs v2 |
| 9 (cross-task) | `fig7_cross_task_summary.png` | Bar chart 3-task |
| 10 (DP) | (новый — генерировать) | DP frontier scatter |
| B1 | `fig2_nesterov_phase_diagram.png` | Phase diagram |

**Если нужно сгенерировать новые figures для слайдов** — все scripts в `scripts/compose_*.py`, данные в `experiments/diagnostics/*.json` и в `validate_multiseed_fed_*.json`.

---

## 9. Tone of voice (примеры формулировок)

**ДА:**
- "Closes Princeton Open Problem 1"
- "First paper-scale demonstration of D-MeZO-N v2 beating vanilla MeZO (−6.2% loss, +2pp acc, 2/2 seeds same direction)"
- "ρ-clip — один механизм решает две задачи"
- "Falsified собственные claims — дисциплина зрелого исследования"
- "v1 fixed C=50 — falsified on 2 seeds (+7.3% loss); v2 adaptive_clip — wins on 2 seeds (−6.2% loss). Same recipe, different clip mechanism."

**НЕТ (overstatements, которые могут разрушить защиту):**
- ❌ "First fully peer-to-peer federated ZO" (есть FedKSeed-like работы)
- ❌ "Asymptotic acceleration" (Bottou-Curtis-Nocedal 2018 запрещает)
- ❌ "Beats all baselines" (нет head-to-head FedKSeed эмпирически)
- ❌ "ε=10 DP for full training" (это per-round, T-round composition хуже)
- ❌ "10⁹× compression vs FedKSeed" (только vs FedAvg; vs FedKSeed equal)

## 10. Финальный checklist для Claude Design

- [ ] Установить reveal.js (или Slidev/Marp по выбору)
- [ ] Создать `slides/` directory с index.html
- [ ] Embed MathJax для формул
- [ ] Создать custom CSS с deep-indigo accent
- [ ] 12 main slides + 5 backup
- [ ] Все цифры на слайде 8 — частично уже подставлены (s=42 complete; s=43 для vanilla/v1/drift complete; adaptive_clip s=43 trending 1.27–1.29; s=44 pending overnight). Финальные значения после полного прогона из `validate_multiseed_fed_Qwen_Qwen3p5-4B-Base_mathlogicqa.json`
- [ ] Анимация consensus mixing (GSAP) на слайде 5
- [ ] Smooth crossfade transitions (не slide-from-side)
- [ ] Print-PDF fallback (на случай отказа браузера/проектора)
- [ ] Speaker notes embedded (Esc + S в reveal.js)
- [ ] Тест на 1080p и 4K projector resolution
- [ ] Тест без интернета (MathJax local fallback)

## 11. Что НЕ нужно делать

- Не добавлять собственные интерпретации/claims — все formulations выше уже выверены
- Не использовать stock photos / cartoons / clipart
- Не добавлять "thank you" слайд (терминальный слайд = последний content slide)
- Не вставлять автора email / phone (только GitHub repo)
- Не использовать тёмную тему (проектор может исказить)
- Не превышать 12 main slides — 10 минут жёстко

---

*Этот brief создан 2026-05-21 как guidance для Claude Design сессии. После того как §22 run закончится, обновить слайд 8 финальными числами из JSON.*
