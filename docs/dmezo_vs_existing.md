# D-MeZO-N v2 vs существующие подходы — простыми словами

**Цель документа:** объяснить **каждую технику** проекта простым языком + показать **чем именно мы отличаемся** от всех известных конкурентов. Один артефакт для defense, для reviewer, для коллеги который заглянул в проект на 15 минут.

**Когда использовать:**
- Перед защитой — прочитать целиком (~30 мин).
- При вопросе "а в чём ваша новизна?" — открыть §3 (большая таблица) или §4 (8 уникальных сочетаний).
- При вопросе "а чем вы отличаетесь от FedKSeed/DPZero/SPSA?" — открыть соответствующую секцию §2.

**Связанные документы:**
- `docs/math_intuition.md` — **математика** простыми словами (зачем каждая формула)
- `docs/litreview_dmezo_n_2026-05-21.md` — **академический** literature review (arXiv IDs, точные tables)
- `docs/paper_ru.md` / `docs/paper_en.md` — статья
- `docs/defense_talking_points.md` — Q&A для защиты

---

## §0. TL;DR за 90 секунд

**Что мы строим:** алгоритм дообучения LLM, который одновременно:
1. **Federated peer-to-peer** (банки общаются напрямую, без центрального сервера)
2. **Forward-only** (без backprop → memory как у inference)
3. **16 байт на раунд на соседа** (вместо 8 ГБ у FedAvg)
4. **С формальной (ε,δ)-DP гарантией** (compliance ready)
5. **Со стабильным моментом** (heavy-ball + adaptive clip + drift-reset + β-decay)
6. **С доказанной сходимостью** под Polyak-Łojasiewicz (Theorem 3 closes Princeton OP1)

**Главный научный вклад:** ни одна работа в литературе не совмещает все 6 одновременно. Каждый компонент существует отдельно — наш вклад в их **совместной работе** + проверенных теоремах + multi-seed empirical validation.

**Empirical headline:** на Qwen3.5-4B-Base / MathLogicQA / 3 seeds paired:

| Метрика | vanilla MeZO | D-MeZO-N v2 (combo) | Δ |
|---|---|---|---|
| Loss (mean ± std) | 1.368 ± 0.018 | **1.293 ± 0.010** | **−5.5% (3/3 same direction)** |
| Accuracy (mean) | 37.7% | **40.0%** | **+2.3pp** |

Первое paper-scale multi-seed validated empirical improvement D-MeZO-N strictly over vanilla MeZO.

---

## §1. Дерево родословной — где мы в истории методов

```
1992  SPSA (Spall)                        ← Bernoulli ±1 perturbation, explicit z vector
       │
       │  + Gaussian z (Nesterov-Spokoiny 2017)
       │  + r(H)-trick для LLM
       │  + in-place seed-reconstruction
       ▼
2023  MeZO (Princeton, Malladi)            ← memory-efficient centralized ZO для LLM
       │                                     left Open Problem 1: momentum + PL convergence?
       │
       ├──────── 2024 ──────────────┬─────────────────┐
       │                            │                 │
       ▼                            ▼                 ▼
    FedKSeed                     DPZero          MeZO-SVRG / SubZero
    (Qin et al.)                 (Tang et al.)   (variance reduction)
    star topology,               centralized,
    shared K-seed pool,          no momentum,    centralized,
    no momentum,                 no federated    no DP,
    no DP                                        no federated
       │
       │  + independent z_i per client (gossip-friendly)
       │  + heavy-ball scalar momentum
       │  + adaptive ρ-clip (B1) + drift-reset (B5)
       │  + β-decay 0.9→0
       │  + dual-use ρ-clip → Gaussian DP
       │  + Theorem 3 (Lyapunov, closes Princeton OP1)
       │  + Theorem 4 (DP extension)
       ▼
2026  D-MeZO-N v2 (наш метод)
```

**Ключевые branches которые мы наследуем:**
- От **SPSA** (Spall 1992): идею two-point central diff + finite-difference gradient estimator.
- От **MeZO** (Princeton 2023): in-place seed-reconstruction + Gaussian z + r(H)-trick для LLM-scale.
- От **Koloskova 2020**: framework для decentralized SGD c doubly-stochastic mixing matrix W.
- От **Polyak 1964**: heavy-ball momentum (но переосмыслен как scalar для ZO).
- От **Dwork-Roth 2014**: Gaussian mechanism для (ε,δ)-DP.

**Что мы добавили сами:**
- 4 техники: independent z_i per client, scalar heavy-ball для ZO, adaptive+drift-reset clip stack, dual-use clip для DP.
- 4 теоремы: T1 (convex), T2 (PL no momentum), **T3 (PL+momentum, closes Princeton OP1)**, T4 (DP extension).

---

## §2. Каждая техника простыми словами

Формат каждой подсекции:
- **Что это** — plain-language explanation
- **Как было раньше** — что делали предыдущие работы
- **Что нового у нас** — наша конкретная модификация
- **Почему важно** — зачем эта модификация нужна

### 2.1 Zeroth-order optimization (общая идея)

**Что это.** Обновлять веса модели **без вычисления градиента**. Просто пробуем тыкнуть на $\varepsilon$ в случайном направлении, смотрим стало лучше или хуже, и в зависимости от знака идём туда или обратно.

**Как было раньше.**
- **SPSA (Spall 1992)** — оригинальная идея. Использовал Bernoulli $\pm 1$ perturbation $\Delta$, хранил весь вектор $\Delta$ в памяти. Применялся к малым моделям (control systems, $d \sim 10^2$–$10^4$).
- **Nesterov-Spokoiny 2017** — формализовал convex ZO с Gaussian perturbation, доказал rates.
- **MeZO (Princeton 2023)** — впервые на LLM ($d \sim 10^9$). Главная innovation: **in-place seed-reconstruction** — вместо хранить вектор $z$ из $10^9$ чисел, хранят только **seed** (4 байта) и регенерируют $z$ через PRNG два раза (для +ε и −ε), in-place в параметрах.

**Что нового у нас.** Мы **не меняем** этот core mechanism — он работает идеально. Мы строим **federated wrapper** поверх (см. §2.2) и **momentum stabilization** поверх (см. §2.3).

**Почему важно.** Без in-place seed-recon trick MeZO бы тратил $4 \times d$ байт памяти на $z$ — это $\sim 16$ ГБ для $d=4\text{B}$. С трюком — 4 байта. **Это единственная причина почему MeZO работает на LLM.**

---

### 2.2 Federated wrapper — independent z_i per client

**Что это.** Несколько клиентов (банков) дообучают модель совместно. На каждом раунде каждый клиент:
1. Локально делает MeZO-шаг (вычисляет свой $\hat\rho_i$ на своих данных).
2. Отправляет соседям по гusinэ только **(seed_i, $\hat\rho_i$)** = **16 байт**.
3. Получает от соседей такие же 16-байтные пакеты.
4. Локально регенерирует $z_j$ из полученных seed_j и применяет consensus update.

**Как было раньше.**

| Подход | Что передаётся | Размер | Topology |
|---|---|---|---|
| **FedAvg** (McMahan 2017) | Все веса модели | 8 ГБ/раунд | Star |
| **FedKSeed** (Qin 2024 ICML) | $K$ scalar ρ + indices в **shared seed pool** | 18 КБ/раунд | Star |
| **Ferret** (Shu 2024) | shared randomness + first-order gradients | $O(d)$ effectively (требует backprop!) | Star |
| **FedZeN** (Maritan 2024) | Hessian estimates | $O(d^2)$ или low-rank | Star, convex only |

**Что нового у нас.**

1. **Independent $z_i$ per client (не shared seed pool).** Каждый клиент имеет **свой** `np.random.Generator` и сэмплирует свой $s_i^t$ независимо. См. `src/dmezo/federated/client.py:62`.
2. **Peer-to-peer (gossip) topology.** Любая doubly-stochastic mixing matrix W (ring, complete, random regular graph). Нет central server.
3. **Передача (seed, scalar) пары** — 16 байт/раунд/сосед.

**Почему важно — два независимых эффекта.**

**(A) Variance reduction по обеим компонентам шума.** Sum of $n$ independent unbiased estimates: variance ÷ n по CLT. С shared seed (FedKSeed-style) variance reduction только по data noise (вариация per-client batches); direction noise остаётся одинаковым на всех клиентах. С independent $z_i$ — variance ÷ n **по обоим** источникам шума. Это и есть federated speedup $1/n$ в Theorem 2.

**(B) Real P2P / asynchronous compatibility.** Shared seed pool требует central coordination (кто-то должен знать какой seed на каком раунде). Independent generators — никакой координации не нужно. **Подходит для асинхронного gossip.**

> **Внимание на защите:** FedKSeed *тоже* делает 16 байт/раунд через shared seed pool. Compression vs FedAvg одинаковая. **Наша differentiation — НЕ в compression, а в (a) topology gossip vs star, (b) independent seeds, (c) momentum, (d) DP.** Не путать!

---

### 2.3 Heavy-ball scalar momentum

**Что это.** Накапливать "инерцию" между шагами, чтобы быстрее проходить узкие овраги loss-landscape.

**Как было раньше.**

| Контекст | Momentum | Storage |
|---|---|---|
| **Vanilla SGD** | $v_{t+1} = \beta v_t + \nabla L_t$ (vector momentum) | $O(d)$ |
| **MeZO (Princeton)** | **Нет momentum** — оставлено как Open Problem 1 | — |
| **SPSA + adaptive gain** (Spall 1997) | Harmonic decay learning rate $\eta_t = a/(t+A)^\alpha$ | $O(1)$ scalar |
| **AdaMeZO / Adam-style ZO** | Adam moments per-parameter | $O(d)$ |
| **FedKSeed** | **Нет momentum** | — |

**Что нового у нас. Scalar heavy-ball для ZO:**

$$v_{t+1} = \beta_t v_t + \hat\rho_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1} z_t$$

$v_t$ — **скаляр**, не вектор. Накапливает "среднюю крутизну в направлениях, которые мы видели". На каждом шаге умножается на **свежий** $z_t$ → moves в текущем направлении с накопленной "уверенностью".

**Почему важно.**

- **Storage:** $O(1)$ скаляр vs $O(d)$ vector у SGD-momentum или Adam-ZO. Сохраняет inference-level memory invariant.
- **Logical fit для ZO:** в MeZO направление $z_t$ меняется каждый шаг → накапливать vector momentum бесполезно (orthogonal directions cancel). Scalar momentum накапливает только **magnitude**, а directionность даёт свежий $z_t$.
- **Closes Princeton OP1:** Theorem 3 даёт closed-form Lyapunov convergence для heavy-ball ZO под PL — была открытая проблема в Malladi 2023.

> ⚠️ **Honest framing:** Theorem 3 даёт **тот же** rate как plain SGD под PL ($O((1-\eta\mu)^T)$). Bottou-Curtis-Nocedal 2018 T5.1 **запрещает** асимптотический speedup от momentum для stochastic non-convex с $\sigma > 0$. Наш вклад: **stability proven** (момент не ломает сходимость), не acceleration. Эмпирически наблюдаем transient speedup до R300 — это not contradicting BCN, это transient phenomenon.

---

### 2.4 ρ-clip — B1 adaptive clipping

**Что это.** Ограничивать (clip) величину наблюдаемого $\hat\rho$, чтобы выбросы не накачивали momentum в blow-up.

**Как было раньше.**

| Подход | Clipping |
|---|---|
| **Vanilla MeZO** | Нет clipping (нет и momentum, поэтому не нужно) |
| **HELENE** (arXiv:2411.10696) | Layer-wise gradient clipping + diagonal Hessian — orthogonal задача |
| **DP-SGD** (Abadi 2016) | Per-sample gradient clipping — для DP, не для stability |
| **DPZero** (Tang 2024) | Fixed clip C на ρ — но только для DP-sensitivity, не для momentum |

**Что нового у нас — adaptive clip:**

$$C_t = \alpha \cdot \text{quantile}_{0.95}\left(\{|\hat\rho_s|\}_{s=t-50}^{t-1}\right), \quad \alpha = 1.3$$

Threshold подстраивается под наблюдаемое распределение $\hat\rho$. 95-й перцентиль robust к outliers, × 1.3 для slack.

**Почему важно — диагностика истории.**

- **v1 (fixed C=50)** — мы изначально брали жёсткий clip 50. На SST-2/Qwen3-0.6B работало хорошо. На Qwen3.5-4B-Base/MathLogicQA median $|\hat\rho| \approx 180$ → fixed 50 обрезал **большую часть полезного сигнала** → momentum застопорен. Multi-seed §22 robustly falsified v1 (+7.0% loss vs vanilla, 3/3 worse).
- **B1 adaptive** — на 4B $C_t$ оседает в 165–270 (vs fixed 50 → в 3–5× tighter range). Multi-seed wins loss на 3/3 (−5.1%), но acc seed-varies.
- **B1 + B5 combo** (см. §2.5) — лучший вариант.

> Это **пример корректной научной коррекции**: false positive v1 (single-seed Day 8) → multi-seed falsified → диагностика "C=50 too tight" → adaptive B1 → robustly wins loss → плюс drift-reset → robust на обеих метриках.

---

### 2.5 Drift-reset — B5 surgical reset

**Что это.** Если eval_loss начинает расти (drift up), сбросить velocity в 0 и начать накапливать momentum заново.

**Как было раньше.** **Никто** этого не делал в ZO literature. Это наш ad-hoc fix, мотивированный наблюдением что adaptive clip alone позволяет late drift в 1/3 seeds.

**Что нового у нас.** В `src/dmezo/mezo/nesterov.py::NesterovState.check_drift_and_reset`:

```python
if eval_loss > rolling_min + 0.1:    # drift detected
    velocity = 0                      # reset
    rolling_min = current             # rebaseline
```

Параметры: window=50, threshold=0.1, fires в среднем 18 раз на seed (54 total на 3 seeds).

**Почему важно.**

- **B1 alone (adaptive clip)** wins loss на 3/3, но acc на s=43 colapsed (−3pp). Causal: после R700 trajectory drifts up, adaptive clip не остановил.
- **B1 + B5 combo** держит loss ниже (R600=1.286 → R1000=1.295 vs adaptive_clip R600=1.309 → R1000=1.314 на s=43). Mean acc gain +2.3pp, lowest std loss (0.010 vs 0.021).

**Аналогия с автомобилем.**
- B1 alone = круиз-контроль адаптируется к скорости трафика. Хорош на средней трассе, но без ABS может занести на скользкой.
- B5 alone = только ABS без круиз-контроля. Не помогает если изначальная скорость неверна (multi-seed: B5 alone +6.4% loss vs vanilla).
- **B1+B5 combo** = круиз-контроль + ABS. Работает в любых условиях.

---

### 2.6 β-decay schedule

**Что это.** Постепенно уменьшать коэффициент momentum $\beta_t$ от 0.9 до 0 в течение обучения.

$$\beta_t = \beta_0 \cdot (1 - t/T), \quad \beta_0 = 0.9$$

**Как было раньше.**

| Подход | β schedule |
|---|---|
| **Polyak heavy-ball** (1964) | Constant β |
| **SGD-momentum** (Sutskever 2013) | Constant β=0.9 |
| **SPSA + adaptive gain** (Spall 1997) | Decay learning rate $\eta_t$, не β |
| **MeZO** | No momentum |

**Что нового у нас.** Linear β-decay specifically для **ZO momentum stability**.

**Почему важно.**

- **Const β=0.9 на ZO** → late drift. Day 6 эксперимент: Qwen3.5-0.8B/SST-2/β=0.9 const → loss blows up на R140 (catastrophic).
- **Const β=0.9 + clip50** → не blow-up, но late drift после R300 (R1d на SST-2: best 0.119@R300, final 0.225).
- **β-decay 0.9→0** → monotonic descent до R1000 (R1d v1: 0.1291 final, beats vanilla 0.1381 by 6.5%).

Логика: в начале training мы далеко от оптимума → momentum полезен для acceleration через "плато". В конце мы рядом с оптимумом → momentum overshoot → нужно "доверять меньше" моменте, идти по чистому $\hat\rho_t z_t$ как vanilla MeZO. β-decay плавно переключает между этими режимами.

---

### 2.7 DP via dual-use ρ-clip

**Что это.** Добавлять Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2)$ к скаляру $\hat\rho_t$ перед передачей, чтобы получить формальную $(\varepsilon, \delta)$-DP гарантию.

**Как было раньше.**

| Подход | Mechanism | Sensitivity | Federated? |
|---|---|---|---|
| **DP-SGD** (Abadi 2016) | Per-sample gradient clipping + Gaussian on gradient | $O(d)$ vector | Centralized |
| **DP-FedAvg** (McMahan 2018) | Per-user clipping + Gaussian | $O(d)$ vector | Star FL |
| **DPZero** (Tang 2024) | **Scalar** Gaussian on $\hat\rho$ | Scalar clip $C$ | Centralized |
| **DPZV** (arXiv:2502.20565, 2025) | ZO + DP for **vertical** FL | Scalar | Vertical FL |

**Что нового у нас — dual-use ρ-clip:**

Клип $C$, который мы изначально ввели для **momentum stability** (B1 в §2.4), **одновременно служит** L2-чувствительностью для Gaussian mechanism:

$$\tilde\rho_t = \text{clip}(\hat\rho_t, \pm C) + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma^2)$$

$$\varepsilon_1 = \frac{C \sqrt{2 \ln(1.25/\delta)}}{\sigma}$$

**Тот же механизм решает две задачи** — stability + privacy. Никакого second clipping pass.

**Почему важно.**

1. **Elegant single-mechanism design.** DP-SGD per-sample clip — дорогой ($O(d)$ per-sample). У нас clip уже есть на скаляре по другой причине → DP практически free.
2. **Empirical flat frontier.** Sweep $\sigma \in \{0.5, 2, 5, 10, 19, 50\}$ показал loss в пределах $1.88 \pm 0.04$. **ε=10 with только +6.2% utility cost** vs no-DP baseline на Qwen3.5-4B-Base/MathLogicQA.
3. **First decentralized federated ZO с formal $(\varepsilon, \delta)$-DP для LLM.** Targeted search для "DP-SPSA" / "private decentralized ZO LLM" вернул нулевые результаты — мы заполняем эту нишу.

> **Honest limitation:** per-round ε=10 — стандарт для one-shot federated fine-tuning. T-round composition даёт $\varepsilon_T = O(\sqrt{T} \varepsilon_1)$ через RDP — пессимистично для $T=10^5$. Subsampling amplification — future work. Это explicit limitation в paper §7.

---

### 2.8 Decentralized peer-to-peer topology

**Что это.** Клиенты общаются друг с другом напрямую через произвольную mixing matrix W (doubly-stochastic). Нет центрального сервера.

**Как было раньше.**

| Метод | Topology | ZO? | LLM? |
|---|---|---|---|
| **FedAvg** (McMahan 2017) | Star | Нет | Нет |
| **FedKSeed** (Qin 2024) | Star | Да | Да |
| **Ferret** (Shu 2024) | Star | Нет (требует backprop) | Да |
| **FedZeN** (Maritan 2024) | Star, convex only | Да | Нет |
| **Koloskova 2020** | Gossip (any doubly-stoch W) | **Нет** | Нет |
| **Lian 2017** (D-PSGD) | Ring/torus | Нет | Нет |

**Что нового у нас.** Gossip topology + ZO + LLM = **first known combination**. Theorem 1 (convex) и T2 (PL) — Koloskova-style framework с ρ-clip variance bound + ZO-specific r(H) trick.

**Почему важно.**

1. **Real-world banks/hospitals не имеют central trusted server.** Cross-silo federated в финансах — peer-to-peer договор между банками. Star FL требует central aggregator (compliance issue).
2. **Async-friendly.** Gossip mixing работает с любой W; клиент A может отвечать B и C в разное время.
3. **Resilient.** Single point of failure отсутствует.

---

## §3. Сводная таблица — D-MeZO-N v2 vs all competitors

| Метод | ZO | LLM | Federated | Topology | Momentum | DP | Theory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **SPSA** (Spall 1992) | ✅ | ❌ | ❌ | — | ❌ | ❌ | Convex, asymptotic |
| **MeZO** (Princeton 2023) | ✅ | ✅ | ❌ | — | ❌ | ❌ | PL, no momentum (OP1) |
| **MeZO-SVRG** (2024) | ✅ | ✅ | ❌ | — | ❌ | ❌ | VR, +1 ref/period |
| **Sparse MeZO** (2024) | ✅ | ✅ | ❌ | — | ❌ | ❌ | Subset perturbation |
| **SubZero** (2024) | ✅ | ✅ | ❌ | — | ❌ | ❌ | Low-rank subspace |
| **HELENE** (2024) | ✅ | ✅ | ❌ | — | Annealing | ❌ | Diagonal Hessian |
| **AdaMeZO** | ✅ | ✅ | ❌ | — | Adam per-param | ❌ | — |
| **FedKSeed** (Qin 2024 ICML) | ✅ | ✅ | ✅ | **Star** | ❌ | ❌ | Convex, K-pool |
| **Ferret** (Shu 2024) | ❌ (FO!) | ✅ | ✅ | **Star** | ❌ | ❌ | — |
| **FedZeN** (Maritan 2024) | ✅ | ❌ | ✅ | Star | ❌ | ❌ | Convex Newton |
| **DPZero** (Tang 2024) | ✅ | ✅ | ❌ | — | ❌ | ✅ centralized | Gaussian on ρ |
| **DPZV** (2025) | ✅ | partial | Vertical | — | ❌ | ✅ | Vertical FL |
| **D-MeZO-N v2 (наш)** | ✅ | ✅ | ✅ | **Gossip/P2P** | ✅ **HB scalar + adaptive clip + drift-reset + β-decay** | ✅ **dual-use clip** | **T1+T2+T3 (closes OP1)+T4** |

**Читать как:** ни одна работа не имеет ✅ во всех колонках одновременно, кроме нас. **Это и есть наша научная ниша.**

---

## §4. Восемь уникальных сочетаний — наша новизна по пунктам

Следующие конкретные сочетания **отсутствуют** в найденных работах (см. `litreview_dmezo_n_2026-05-21.md` для arXiv IDs):

### 1. Decentralized (gossip/P2P) + ZO + LLM fine-tuning
- FedKSeed = star + ZO + LLM ✗ (star, не gossip)
- Ferret = star + first-order + LLM ✗ (не ZO)
- FedZeN = gossip + ZO ✗ (convex, не LLM)
- **D-MeZO-N v2 = first gossip ZO для LLM** ✓

### 2. Independent z_i per client (не shared seed pool)
- FedKSeed требует shared finite K-seed set + central coordination.
- У нас 1 float + 1 int = 16 байт, никакой shared state. **Подходит для асинхронного P2P.** ✓

### 3. Scalar heavy-ball + adaptive ρ-clip (B1) + drift-reset (B5) в ZO
- Adam-style ZO работы используют per-parameter Adam moments ($O(d)$ memory).
- Никто не использует heavy-ball **scalar** velocity с data-driven adaptive clip + surgical drift reset. ✓

### 4. Convergence proof для heavy-ball ZO + PL + β-decay (Theorem 3)
- arXiv:2303.16241 (2023) доказывает HB с biased approx gradient в general settings — не специфично ZO + PL + decentralized + clip + β-decay.
- Наш Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ — оригинальный contribution.
- **Closes Princeton Open Problem 1.** ✓

### 5. Dual-use ρ-clip
- Same threshold $C$ одновременно для (a) momentum stability + (b) L2-sensitivity Gaussian mechanism.
- DPZero использует noise на ρ без momentum (clip только для DP).
- Мы совмещаем clip для двух целей — elegant single-mechanism design. ✓

### 6. MeZO на hybrid linear-attention + full-attention арх (Qwen3.5)
- Princeton paper тестировал только full-attention transformers (OPT-13B, Llama-7B).
- Мы первые на hybrid linear-attention class (Qwen3.5-4B-Base с DeltaNet layers + ViT). ✓

### 7. Formal $(\varepsilon=10, \delta=10^{-3})$-DP в decentralized federated ZO
- DPZero (arXiv:2401.04343) — centralized only.
- DPZV (arXiv:2502.20565) — vertical FL, different task.
- Targeted search для "DP-SPSA" / "private decentralized ZO LLM" — нулевые результаты. ✓

### 8. Connection to classical SPSA (Spall 1992) — historical lineage
- D-MeZO-N v2 = SPSA + 4 innovations поверх MeZO: heavy-ball scalar momentum (vs SPSA's adaptive gain), adaptive clip (vs no SPSA stability mechanism), federated wrapper (no SPSA-LLM federated work), dual-use clip for DP (DP-SPSA не существовало под этим именем).
- **34 года stochastic approximation theory** за плечами — сильное positioning. ✓

---

## §5. Honest limitations — где конкуренты сильнее или мы не дотягиваем

### 5.1 Где конкуренты сильнее

| Aspect | Their lead | Our defense |
|---|---|---|
| **Variance reduction** | MeZO-SVRG +20% acc (centralized) | Federated SVRG требует extra communication round → conflict с "1 scalar/round" invariant |
| **Adam-style adaptivity** | AdaMeZO / Adam-ZO better centralized convergence | Adam moments в federated ZO = extra memory ($O(d)$) + non-trivial averaging semantics |
| **Non-convex generality** | T2/T3 доказан только под PL | PL is standard в modern deep learning rate-proofs; locally on trajectory plausible (Liu-Zhu-Belkin 2022) |
| **Empirical scale** | FedKSeed tested LLaMA-7B; мы Qwen3.5-4B | Hybrid linear-attention class is novel; не direct comparable |

### 5.2 Empirical weaknesses (честно)

- **Multi-seed paired falsifies original v1 single-seed claim** (+1.25pp на MathLogicQA → 3/3 worse). Мы признаём это в paper §6.11. v2 (combo B1+B5) — это **корректное исправление** через multi-seed validation.
- **Short-horizon SST-2 (200 rounds):** vanilla MeZO beats D-MeZO-N v2 в 3.4× loss. **Serious failure mode** при коротких runs. Combo нужен ≥500 rounds чтобы переиграть vanilla.
- **Нет head-to-head comparison с FedKSeed** на одинаковом dataset/model — reviewer обязательно попросит. Script готов (`scripts/compare_fedkseed.py`), нужно ~6.75h Colab compute. Запланировано post-defense.

### 5.3 Theoretical gaps

- **T3 noise floor $2G^2/(3\mu)$** — bound на $G^2$ pessimistic; tighter bound — future work.
- **T1 decentralized rate** — стандартная Koloskova-форма, без tight analysis ZO-heterogeneity.
- **β-decay schedule** выбрана эмпирически (linear 0.9→0); optimal decay не доказан.
- **Full decentralized T3** (combining momentum + consensus error in PL regime) — наш **Open Problem 2**, не закрыта.

---

## §6. Defense one-liners — готовые ответы на ключевые вопросы

### Q1: "В чём именно ваша новизна по сравнению с MeZO?"
> "MeZO — centralized без момента и DP. Мы добавили: (1) federated peer-to-peer wrapper с independent z_i per client → 1/n variance speedup по обеим компонентам шума, (2) scalar heavy-ball momentum со стабилизацией (adaptive clip + drift-reset + β-decay), (3) formal $(\varepsilon, \delta)$-DP через elegant dual-use clip mechanism, (4) Theorem 3 закрывает Princeton Open Problem 1 о momentum convergence под PL+ZO."

### Q2: "В чём отличие от FedKSeed?"
> "FedKSeed — star topology + shared finite K-seed pool + no momentum + no DP. Мы — peer-to-peer gossip topology (любая doubly-stochastic W) + **independent seeds per client** (gossip-friendly, async-compatible) + heavy-ball scalar momentum со стабилизацией + formal $(\varepsilon, \delta)$-DP. Compression vs FedAvg одинаковая (~16 байт/раунд), но это **не** наша differentiation — наша differentiation в topology, momentum, и DP."

### Q3: "Почему independent seeds, а не shared (FedKSeed-style)?"
> "Два независимых эффекта. (1) Variance reduction: $n$ independent estimates дают variance $\div n$ по обоим источникам шума — data + direction. Shared seed уменьшает только data noise. (2) Real P2P / async: shared seed pool требует central coordination, у нас никакой координации не нужно."

### Q4: "А не появится ли correlation между клиентами через consensus mixing?"
> "Consensus сближает **параметры** клиентов в одну точку — это convergence в trajectories, не correlation в источнике шума. На следующем раунде каждый клиент снова independently сэмплирует свой $z_i^{t+1}$. Lemma 4 (Koloskova-style consensus error) формализует это: $\rho_W^2/(1-\rho_W)^2$ amplifier ограничивает per-round drift."

### Q5: "Theorem 3 — это acceleration или нет?"
> "Не acceleration в asymptotic sense. Bottou-Curtis-Nocedal 2018 T5.1 запрещает momentum-based asymptotic speedup для stochastic non-convex с $\sigma > 0$. T3 даёт **тот же** rate как plain SGD под PL ($O((1-\eta\mu)^T)$). Наш вклад — **stability proven**: momentum + clip + β-decay не ломают сходимость, можно использовать без риска. Эмпирически наблюдаем transient speedup до R300, но это transient phenomenon, не доказан асимптотически."

### Q6: "DP free? Что это значит?"
> "Per-round free: clip $C$, который мы изначально ввели для momentum stability, одновременно служит L2-чувствительностью для Gaussian mechanism. Никакого second clipping pass. Empirically на Qwen3.5-4B-Base/MathLogicQA $\varepsilon=10$ стоит только +6.2% utility loss vs no-DP baseline. T-round composition — explicit limitation: $\varepsilon_T = O(\sqrt T \varepsilon_1)$ через RDP, бесполезно при $T=10^5$. Subsampling amplification — future work."

### Q7: "Почему v1 falsified, а v2 (combo) работает?"
> "v1 (fixed C=50) был tuned на Qwen3-0.6B/SST-2 single seed. Multi-seed на Qwen3.5-4B-Base/MathLogicQA: median $|\hat\rho| \approx 180$ → fixed 50 обрезал большую часть полезного сигнала. B1 adaptive clip ($C_t = 1.3 \cdot \text{quantile}_{0.95}$) оседает в 165–270 на 4B — robust к outliers, не обрезает сигнал. B5 drift-reset добавляет surgical reset когда eval_loss drifts up. v2 = combo: −5.5% loss (3/3), +2.3pp acc, lowest std (0.010 vs 0.018 у vanilla). Это **корректная научная коррекция** через multi-seed validation, не failure."

### Q8: "А что если scale up до Qwen3-8B / n=8 clients?"
> "Roadmap post-defense — yes. Scale-up Qwen3-8B + n=8 + Generative tasks (SAMSum, GSM8K) запланированы. Compute budget: ~30 units на Colab Pro+ Blackwell, готов запустить."

### Q9: "Почему PL, а не general non-convex?"
> "PL — стандарт в modern deep learning rate-proofs (Liu-Zhu-Belkin 2022 показывают плауsibly локально на trajectory для overparametrized neural nets). Под PL получаем линейную сходимость к neighbourhood. Без PL получили бы $O(1/\sqrt T)$ rate в general nonconvex — есть в T1 (convex variant). Если бы доказали под general non-convex с моментом — это был бы major theory paper сам по себе."

### Q10: "А SPSA? Вы вообще знаете о нём?"
> "Да, конечно — это foundational ancestor MeZO. Spall 1992 ввёл two-point central diff. MeZO добавил два LLM-specific innovation: Gaussian z (для r(H)-trick) и in-place seed-reconstruction (для memory). D-MeZO-N v2 добавляет ещё четыре поверх MeZO: federated wrapper, scalar heavy-ball, adaptive ρ-clip, dual-use clip для DP. Targeted search для 'DP-SPSA' вернул нулевые результаты — мы заполняем эту нишу. Theorem 3 — первое closed-form Lyapunov convergence для heavy-ball SPSA под PL за 34 года."

---

## §7. Cheat sheet — самое короткое summary

| Техника | Plain language | Главное отличие от всех существующих |
|---|---|---|
| **MeZO core** | Forward-only ZO, scalar $\hat\rho$ вместо gradient | Не наша — Princeton 2023, мы используем как baseline |
| **Federated wrapper** | 16 байт/раунд между соседями | **Independent $z_i$** + **gossip topology** (vs FedKSeed star+shared) |
| **Heavy-ball momentum** | Scalar $v_t$ накапливает крутизну | **Scalar** для ZO (vs vector Adam-ZO, vs no momentum vanilla MeZO) |
| **B1 adaptive clip** | Threshold = 1.3 × q_0.95 истории | Adaptive (vs fixed C=50 у v1 / DPZero), single mechanism для stability+DP |
| **B5 drift-reset** | Reset velocity если loss drifts up | **Unique** — никто в ZO literature не делает |
| **β-decay 0.9→0** | Постепенно "доверяем меньше" моменту | Linear decay specifically для ZO momentum (vs const β у Polyak) |
| **DP dual-use clip** | Same $C$ для stability + L2-sensitivity | **Elegant single-mechanism** (vs DP-SGD per-sample $O(d)$ clip) |
| **Gossip topology** | Любая doubly-stochastic W | **First** gossip + ZO + LLM combination |
| **Theorem 3** | Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ | **Closes Princeton OP1** — momentum convergence под PL+ZO |
| **Theorem 4** | DP extension of T3 | First $(\varepsilon, \delta)$-DP convergence для decentralized fed ZO LLM |

---

## §8. Что **НЕ** говорить (и почему)

| Фраза | Почему не говорить |
|---|---|
| "Мы получили $O(1/T^2)$ acceleration rate" | Bottou-Curtis-Nocedal 2018 T5.1 запрещает для stochastic non-convex |
| "Momentum ускоряет MeZO асимптотически" | Theoretical: same rate как T2 (plain SGD). Только transient empirical speedup |
| "K-direction strictly improves" | Equal-compute: K=3 проигрывает K=1 (+41.6% loss). Pareto trade-off, не win |
| "DP is fully free" | Free **per-round**; expensive **per-T-rounds** — explicit limitation |
| "D-MeZO-N v1 (fixed C=50) лучше vanilla" | Multi-seed falsified, paired Δ = +7.0% loss vs vanilla (3/3 worse) |
| "Все клиенты используют один seed" | Это про FedKSeed! У нас **independent z_i per client** — критическая differentiation |
| "10⁹× compression vs FedKSeed" | Compression одинаковая (~16 байт/раунд). Наша differentiation в topology + momentum + DP |
| "D-MeZO-N strictly beats vanilla на любом task" | На short-horizon SST-2 (200 rounds) vanilla beats нас 3.4×. Combo нужен ≥500 rounds |

---

## §9. Связи с другими документами

| Если хочешь | Открой |
|---|---|
| **Математику** (зачем каждая формула) | `docs/math_intuition.md` |
| **Точные arXiv IDs** для citation | `docs/litreview_dmezo_n_2026-05-21.md` |
| **Формальные proofs** теорем | `docs/theory_rigorous.md` |
| **Multi-seed validation analysis** | `docs/multiseed_analysis.md` §22 |
| **Defense Q&A** (21+ вопросов) | `docs/defense_talking_points.md` |
| **Algorithm pseudocode** | `docs/03-algorithm-spec.md` |
| **FedKSeed-specific defense** | `docs/defense_fedkseed_qa.md` |
| **Slides brief** для Claude Design | `docs/defense_design_brief.md` |
| **Honest limitations & negatives** | `docs/robustness_matrix.md` |

---

*Документ создан 2026-05-21. Synthesizes `math_intuition.md` (математика), `litreview_dmezo_n_2026-05-21.md` (академический ревью), `paper_ru.md` §2 (Related Work). Focused **специально** на differentiation от existing methods — для defense и для quick onboarding нового коллеги.*
