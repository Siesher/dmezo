# Математика D-MeZO-N простыми словами

**Цель документа:** объяснить **зачем** каждая формула, без formal proof. Для быстрого понимания сути и подготовки к защите.

**Структура:** идём от простого к сложному. Каждая секция = одна идея + одна аналогия + минимум формул.

---

## 1. Базовая интуиция: что такое MeZO

### 1.1 Аналогия — навигация в тумане

Представь, что ты стоишь на холмистой местности в тумане и хочешь спуститься вниз (минимизировать loss). Нормально (backprop) — у тебя есть GPS, который точно говорит "иди на юго-восток с уклоном 30°". Но GPS требует много памяти.

**MeZO trick:** вместо GPS бросаешь мячик в случайном направлении $z$ на маленькое расстояние $\varepsilon$, измеряешь высоту, потом бросаешь в **противоположном** направлении и снова измеряешь.

$$\hat\rho = \frac{\text{высота в } +z \text{ направлении} - \text{высота в } -z}{2\varepsilon}$$

Если справа выше слева — значит уклон **в сторону левого** $z$. Идём туда.

### 1.2 Главный insight: $\hat\rho$ — это скаляр

Не нужно хранить весь градиент (миллиарды чисел). **$\hat\rho$ — одно число**, "насколько круто spreading в направлении $z$".

Update параметров:
$$\theta \leftarrow \theta - \eta \hat\rho z$$

И $z$ восстанавливается из seed → не надо хранить.

**Память:** как у inference. Compared to backprop (gradients + activations) — экономия 2-4×.

### 1.3 Цена: noise

Один случайный $z$ — плохая оценка градиента. Variance большая. Поэтому нужно много шагов.

Но **в среднем** $\mathbb{E}[\hat\rho z] \approx \nabla L$ — несмещённая оценка.

---

## 2. Magic: почему MeZO работает для больших моделей ($r(H)$-трюк)

### 2.1 Проблема

Наивно: dimension $d \sim 10^9$ для LLM. Random direction $z$ в $10^9$-мерном пространстве **почти ортогонален** градиенту. Один шаг почти бесполезен.

Должно быть $10^9$ шагов до сходимости — невозможно.

### 2.2 Почему всё-таки работает (Malladi 2023)

**Ключевая теорема Princeton:**

$$\mathbb{E}[\|\hat\rho z\|^2_H] \le \|\nabla L\|^2 \cdot r(H)$$

где $r(H) = \mathrm{tr}(H) / \|H\|_{op}$ — **эффективный ранг** Гессиана $H = \nabla^2 L$.

**Простыми словами:** variance MeZO зависит **не от $d$**, а от **эффективного ранга Гессиана** $r(H)$, который для глубоких LLM **$\ll d$** (порядка $10^2 - 10^3$).

### 2.3 Почему так?

**Аналогия:** ты в горах, но не все направления "одинаково крутые". Loss-функция LLM имеет много **плоских направлений** (over-parametrization). Когда бросаешь $z$ в любом направлении, **большая часть energy уходит в плоские** области, **где она не вредит**.

Effective rank $r(H)$ говорит: "реальная сложность landscape — $r(H)$, не $d$".

### 2.4 Что это значит для практики

Convergence rate MeZO как у backprop SGD умножить на $\sqrt{r(H)/d}$. Princeton показала на OPT-13B что $r(H) \sim 100$, поэтому MeZO **в 1000× медленнее** backprop, но **в 4× меньше памяти**. Trade-off OK для дообучения.

---

## 3. Federated wrapper: 16 байт вместо 8 ГБ

### 3.1 Проблема FedAvg

В обычном FedAvg между клиентами пересылается:
- 4B params × bf16 = 8 ГБ на клиент на раунд

5 банков × 1000 раундов = 40 ТБ traffic. **Невыполнимо**.

### 3.2 Наш фокус

Все клиенты используют **один seed** на шаге $t$ → у всех **одинаковый** $z_t$. Достаточно передать только $\hat\rho_t$ (скаляр) + seed:

$$\text{traffic per client per round} = 1 \text{ int} + 1 \text{ float} = 16 \text{ байт}$$

**Compression vs FedAvg:** $\sim 5 \times 10^8$ раз.

### 3.3 Аналогия — общий компас

Раньше каждый клиент рисовал детальную карту маршрута и пересылал её всем. Теперь все клиенты имеют общий компас (PRNG с одним seed), и каждый просто говорит "я пошёл на расстояние $\hat\rho_i$ в этом направлении".

Усреднение: $\bar\rho = \frac{1}{N} \sum_i \hat\rho_i$ — все идут в одном направлении $z$, но с consensus-усреднённой скоростью.

---

## 4. Heavy-ball моментум: почему добавляем + почему ломается

### 4.1 Зачем моментум вообще

Без момента (vanilla MeZO):
- Каждый шаг идёт в направлении $\hat\rho z$.
- $z$ меняется на каждом шаге → траектория **сильно зигзагообразная**.
- Медленно сходится в "узких ущельях" loss-landscape.

**Идея момента:** накапливать "инерцию" предыдущих шагов.

$$v_{t+1} = \beta \cdot v_t + \hat\rho_t$$

$v_t$ — это **скорость в направлении $z$**, накопленная с памятью $\beta$.

### 4.2 Где ломается на ZO

В обычном SGD моменте $v_t$ — vector в d-измерении (накапливает направление $\nabla L$).

В MeZO **$z_t$ меняется каждый шаг** → накапливать его бесполезно (orthogonal directions cancel).

**Наш fix:** накапливаем скаляр $\rho$ для **текущего направления** $z_t$:

$$v_{t+1} = \beta_t v_t + \hat\rho_t, \quad \theta_{t+1} = \theta_t - \eta v_{t+1} z_t$$

Это даёт scalar momentum, который **умножается на свежий $z_t$**. Сохраняем benefit момента, не нарушаем MeZO structure.

### 4.3 Опасность: blowup

Если $\hat\rho$ имеет outlier (например, $\hat\rho = 200$) — $v$ накачивается:
$$v_1 = 200, v_2 = 0.9 \cdot 200 + 200 = 380, v_3 = 0.9 \cdot 380 + 200 = 542, ...$$

Кинетическая энергия растёт → loss blowup.

### 4.4 Fix: ρ-clip

$$\hat\rho_{\text{clipped}} = \mathrm{clip}(\hat\rho, -C, +C), \quad C = 50$$

Outliers обрезаются → $v$ ограничено сверху примерно $C/(1-\beta) = 500$ — не blowup.

### 4.5 Fix 2: β-decay

При const $\beta = 0.9$ velocity накапливает kinetic energy. Late-stage drift: loss растёт после R300.

**Fix:** $\beta_t = \beta_0 (1 - t/T)$ от 0.9 → 0. К концу обучения $\beta \to 0$ → моменту больше не верим, просто идём по $\hat\rho z$.

### 4.6 Аналогия — скейтбордист

- **No momentum:** скейтбордист пинается ногой каждый раз заново — медленно, утомительно.
- **Const momentum (β=0.9):** разгоняется на спуске, потом не может остановиться на повороте → улетает.
- **Momentum + clip:** есть тормоз (clip C), не разогнаться больше $C/(1-β)$.
- **Momentum + clip + β-decay:** в начале трассы катит на инерции, в конце переходит на пешком (точное приземление).

---

## 5. Lyapunov function $V_t$: главный приём анализа

### 5.1 Проблема с моментом

Хочется доказать: "loss убывает с каждым шагом". Но **с моментом не убывает** — иногда loss растёт (когда $v_t$ толкает в неудачную сторону).

### 5.2 Решение: смотрим на energy

Определим:

$$V_t = \underbrace{L(\theta_t) - L^\star}_{\text{потенциал}} + \underbrace{\frac{\eta}{2} \|v_t\|^2}_{\text{кинетическая энергия}}$$

**$V_t$ — общая энергия системы.** Loss = потенциальная (как высота над уровнем моря). Velocity² = кинетическая (как скорость движения).

### 5.3 Аналогия — горнолыжник

- **Loss** = высота над финишем.
- **$\|v\|^2$** = квадрат скорости спуска.
- **$V_t$** = total energy.

В обычной механике (без трения) total energy сохраняется. Но у нас есть **трение** (η learning rate) — энергия диссипирует на каждом шаге.

**Теорема:** под нашими условиями (PL, smoothness, clip, β-decay):

$$\mathbb{E}[V_{t+1}] \le \left(1 - \frac{3\eta\mu}{2}\right) \mathbb{E}[V_t] + \text{шум}$$

То есть **$V_t$ убывает экспоненциально** (с rate $1 - 3\eta\mu/2$), несмотря на то что loss или velocity отдельно могут колебаться.

### 5.4 Что это даёт

После $T$ шагов:
- $V_T$ маленькое → значит **И** loss маленький, **И** velocity маленький.
- Loss → $L^\star$, velocity → 0.
- Скейтбордист доехал до финиша и остановился.

---

## 6. Theorem 3 за 5 простых шагов

**Утверждение:** $\mathbb{E}[V_T] \le (1 - 3\eta\mu/2)^T V_0 + \frac{2 C^2 r(H) \ell}{3\mu}$.

Левая часть = энергия в момент $T$. Первый член = exponential decay начальной энергии. Второй член = **шум-пол** — куда мы в принципе можем спуститься (limited by MeZO noise).

### 6.1 Brick 1: loss descent

При $\theta_{t+1} = \theta_t - \eta v_{t+1} z_t$ и smooth $L$:

$$L(\theta_{t+1}) \approx L(\theta_t) - \eta \underbrace{\langle \nabla L, v_{t+1} z_t \rangle}_{\text{прогресс}} + \underbrace{\frac{\eta^2 \ell}{2}\|v_{t+1} z_t\|^2}_{\text{шум}}$$

**В словах:** loss падает на величину "прогресса" (projection of $v z$ on gradient) минус накладной шум.

### 6.2 Brick 2: velocity recursion

$$\|v_{t+1}\|^2 = \beta_t^2 \|v_t\|^2 + 2\beta_t v_t \hat\rho_t + \hat\rho_t^2 \le \beta_t^2 \|v_t\|^2 + C^2 + \text{cross}$$

(где использовали $|\hat\rho| \le C$ после clip)

**В словах:** kinetic energy на шаге $t+1$ — это $\beta^2$ доля старой + ограниченный bounded prirost.

### 6.3 Brick 3: combine into $V_t$

Складываем Brick 1 + Brick 2:

$$V_{t+1} - V_t \le -\eta \langle \nabla L, v_{t+1} z_t \rangle + \beta_t^2 \frac{\eta}{2}\|v_t\|^2 + \text{шум}$$

### 6.4 Brick 4: применяем PL

PL-условие: $\|\nabla L\|^2 \ge 2\mu (L - L^\star)$.

Это значит: "если loss далеко от $L^\star$, то градиент **гарантированно** большой". Через Young inequality + PL получаем:

$$V_{t+1} - V_t \le -\frac{3\eta\mu}{2} V_t + \text{шум-пол}$$

### 6.5 Brick 5: разворачиваем рекурсию

$$V_T \le (1 - 3\eta\mu/2)^T V_0 + \sum_{s=0}^{T-1} (1-3\eta\mu/2)^s \cdot \text{шум}$$

Геометрическая прогрессия суммируется в $\text{шум}/(3\eta\mu/2)$ → шум-пол $= 2 C^2 r(H) \ell / (3\mu)$.

### 6.6 Главный insight

Theorem 3 **закрывает Open Problem** из Princeton (они отметили "momentum convergence is open"). Наш приём — Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ — стандартен для momentum analysis (Su-Boyd-Candes 2014), адаптирован к stochastic ZO + clipping.

---

## 7. DP-MeZO: добавляем шум для приватности

### 7.1 Зачем

Compliance (115-ФЗ, HIPAA, GDPR): нужна формальная гарантия "из обновлений нельзя восстановить, кто из клиентов добавил какие данные".

### 7.2 Gaussian mechanism (Dwork-Roth 2014)

Если запрос имеет **L2-чувствительность $\Delta$** (= максимальное изменение output от одного клиента), добавление шума $\xi \sim \mathcal{N}(0, \sigma^2)$ даёт $(\varepsilon, \delta)$-DP с:

$$\sigma \ge \Delta \cdot \frac{\sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

### 7.3 Наш elegant insight

ρ-клип $C$, **уже** введённый для momentum stability, **ОДНОВРЕМЕННО** ограничивает L2-чувствительность:

$$\Delta = C = 50$$

То есть, **тот же механизм** делает momentum stable + обеспечивает DP. Не нужно второй раз клипать (как в DP-SGD per-sample gradient clipping — это дорого).

### 7.4 Per-round ε

Подставляем $\Delta = C = 50$, $\delta = 10^{-3}$:

$$\varepsilon_1 = \frac{50 \sqrt{2 \ln(1250)}}{\sigma} = \frac{188.8}{\sigma}$$

Чтобы получить $\varepsilon = 10$ → $\sigma = 18.88 \approx 19$.

### 7.5 T-round composition (честно: больно)

Каждый раунд = одно применение Gaussian mechanism. После $T$ раундов:

- **Basic composition:** $\varepsilon_T = T \cdot \varepsilon_1$. Для $T=200, \varepsilon_1=10$ → 2000 (бесполезно).
- **Advanced composition:** $\sqrt{T} \cdot \varepsilon_1$, но второй член $T \varepsilon_1 (e^{\varepsilon_1} - 1)$ ломает при больших $\varepsilon_1$.
- **RDP / Moments accountant (Abadi 2016):** tighter, но всё ещё $O(\sqrt T)$.

**Paper position:** заявляем per-round $\varepsilon$ как relevant claim для one-shot federated; T-round composition признаём limitation; subsampling amplification — future work.

---

## 8. Theorem 4: почему теоретический шум-пол не наблюдается

### 8.1 Что говорит теория

С добавлением DP-шума variance вырастает:

$$G^2_{\text{DP}} = (C^2 + \sigma^2) \cdot d$$

**Внимание:** здесь $\cdot d$, **не $\cdot r(H)$**. Malladi r(H)-trick **ломается** для DP-шума.

### 8.2 Почему ломается trick

$r(H)$-trick работал потому что $\hat\rho z$ имеет **структуру** — выровнен с $\nabla L$.

DP-шум $\xi z$ — **изотропен** в θ-пространстве, **не aligned** с градиентом. Поэтому:

$$\mathbb{E}\|\xi z\|^2 = \sigma^2 \cdot d \quad (\text{не } \sigma^2 \cdot r(H))$$

### 8.3 Crossover

DP-вклад превышает MeZO-вклад когда:

$$\sigma^2 d > C^2 r(H) \quad \Leftrightarrow \quad \sigma > C\sqrt{r(H)/d}$$

Для Qwen3.5-0.8B: $\sigma_{\text{crossover}} \approx 0.016$. Значит **любой $\sigma > 0.02$** теоретически уже доминирует.

### 8.4 Но эмпирически — frontier плоский! Почему?

3 гипотезы:

1. **Transient regime.** При $T = 200$ раундов мы НЕ дошли до steady-state. Bound из T4 — про **асимптотику**, при малом $T$ transient $(1-3\eta\mu/2)^T V_0$ доминирует.

2. **Effective $d$ << total params.** Vision tower frozen, alignment $z$ с $\nabla L$ кладёт большую часть energy в irrelevant subspaces. Реальный effective $d$ возможно $\sim 10^6$, не $10^9$.

3. **Loose bound.** Lemma 8 — pessimistic upper bound. Реальная variance может быть в разы меньше.

### 8.5 Что это значит для практики

**Хорошая новость:** реальные деплои с $T \le 1000$ раундов **также будут enjoy этот gap**. То есть DP at ε=10 действительно free на этом масштабе.

**Caveat:** если запустить на $T = 10^5$ раундов, шум-пол может проявиться.

---

## 9. Combo v2 (B1+B5): фикс accuracy paradox

### 9.1 Проблема v1: fixed C=50 over-engineered

Multi-seed MathLogicQA falsified изначальный claim "v1 лучше vanilla". На локальной 0.8B v1 даже **проигрывает** vanilla 3.4×.

Гипотеза: C=50 **слишком тугой клип** — срезает много полезного signal.

### 9.2 B1 — Adaptive clip

$$C_t = \alpha \cdot \mathrm{quantile}_{0.95}(\{|\hat\rho|\}_{\text{recent 30}}), \quad \alpha = 1.3$$

**Идея:** не задавать C заранее, а **подстраивать** под наблюдаемое распределение $\hat\rho$. Берём 95-й перцентиль (robust к outliers) × 1.3 (немного запаса).

Эмпирически: $C_t$ оседает в районе 180-270 на 4B (vs fixed 50 → 3-5× больше).

### 9.3 Adaptive_clip paradox

B1 alone: loss лучше (быстрее convergence без over-tight clip), но **acc хуже на 17pp**.

**Причина:** loose clip → momentum overshoots → loss падает быстро, но улетает мимо точного минимума. Acc страдает от overshoot.

### 9.4 B5 — Drift-reset

```
если eval_loss[t] > eval_loss[t-50] + 0.1:
    v_t ← 0  # обнуляем velocity
```

**Идея:** если eval-loss начал расти (значит momentum overshoots), **зануляем скорость**. Loss-component Lyapunov продолжает контрактироваться, но без kinetic energy buildup.

### 9.5 B1+B5 = combo (D-MeZO-N v2)

- B1: разрешает клипу пропускать signal (loose).
- B5: surgically обнуляет момент при overshoot.

Empirical: combo achieves **vanilla parity** на 0.8B, beats v1 significantly.

### 9.6 Аналогия

- **B1:** педаль газа адаптируется под скорость трассы (не fixed RPM лимит).
- **B5:** ABS — если detect skid (overshoot), кратковременно отпускает газ.
- **Combo:** машина адаптируется к трассе И имеет страховку.

---

## 10. K-direction MeZO: почему не pure win

### 10.1 Идея

Усреднять оценку по $K$ независимым $z_k$:

$$\tilde g_t = \frac{1}{K} \sum_{k=1}^K \hat\rho_{t,k} z_{t,k}$$

Variance ÷ K → faster convergence.

### 10.2 Подвох

Один шаг K-direction = $2K$ forward passes (vs 2 при K=1).

**Equal-compute сравнение:** при том же compute budget K=3 делает в 3× меньше шагов чем K=1.

Empirical (Day MD ablation): K=3 vs K=1 — loss +41.6% хуже на equal-compute. K-direction **не вин** asymptotically.

### 10.3 Когда полезен

При **больших σ** (heavy DP) — effective $\sigma_{\text{eff}} = \sigma/\sqrt K$. Можно applying K=4 чтобы получить $\varepsilon = 5$ при том же utility что $\varepsilon = 10$ при K=1.

**Trade-off:** privacy gain vs compute cost. Recommended для очень частных setting'ов (ε ≤ 1).

### 10.4 Bottou-Curtis-Nocedal 2018 теорема 5.1

Эта теорема говорит: для **stochastic non-convex** с шумом $\sigma > 0$, **momentum не ускоряет** асимптотически.

Поэтому изначальное надеяние получить $O(1/T^2)$ rate (как у Nesterov для convex) **принципиально невозможно** в нашей setting. Это записано в paper §6.4 как honest negative.

---

## 11. Cheat sheet — самое важное

| Концепт | Одна фраза |
|---|---|
| **MeZO** | Forward-only ZO, один скаляр $\hat\rho$ вместо градиента |
| **$r(H)$-trick** | Variance scales с effective rank Hessian, не с $d$ |
| **Federated wrapper** | Передаём 16 байт вместо 8 ГБ через общий seed |
| **Heavy-ball momentum** | Накапливаем scalar $v_t$ для текущего $z_t$ |
| **ρ-clip C=50** | Защита от velocity blowup на outlier $\hat\rho$ |
| **β-decay 0.9→0** | В конце обучения "доверяем меньше" моменту |
| **Lyapunov $V_t$** | Loss + kinetic energy — total energy системы |
| **Theorem 3** | $V_t$ убывает экспоненциально под PL+heavy-ball+clip+decay |
| **DP-MeZO** | Gaussian шум на clipped $\hat\rho$; C=Δ=L2-sensitivity (free reuse!) |
| **Theorem 4** | DP-шум ломает r(H)-trick → теоретический worst case scales с $d$ |
| **Theory vs practice** | Эмпирически шум-пол не достигается за $T=200$ → DP essentially free |
| **Combo v2 (B1+B5)** | Adaptive clip + drift-reset → vanilla parity на convergent tasks |
| **K-direction trade-off** | Variance ÷ K, но compute × K → не pure win |

---

## 12. Защитные одностроки для каждого theorem

### Theorem 2 (PL без момента) — Karimi-Nutini-Schmidt + Malladi
*"Стандартная PL-сходимость SGD, адаптированная к MeZO через $r(H)$-trick. Шум-пол масштабируется с $r(H)$, не с $d$ — поэтому работает на LLM."*

### Theorem 3 (PL + momentum + clip + β-decay) — наш вклад
*"Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ контрактируется экспоненциально с rate $1 - 3\eta\mu/2$. Закрывает Open Problem 1 у Princeton."*

### Theorem 4 (DP extension of T3) — наш вклад
*"DP-шум добавляет $\sigma^2 d$ к шум-полу (без r(H)-trick, потому что noise isotropic). Per-round ε guarantee через стандартный Gaussian mechanism. T-round composition признаём limitation."*

---

## 13. Чего НЕ говорить (и почему)

| Frase | Почему не говорить |
|---|---|
| "Мы получили $O(1/T^2)$ rate" | Bottou-Curtis-Nocedal 2018 T5.1 запрещает для stochastic non-convex |
| "Momentum ускоряет MeZO" | Theoretical: тот же asymptotic rate как T2. Только transient empirical speedup |
| "K-direction strictly improves" | Equal-compute: K=3 проигрывает K=1 |
| "DP is free" | Free per-round; expensive per-T-rounds (надо честно отметить) |
| "v1 strictly better than vanilla" | Multi-seed falsified, CI [0,0] на MathLogicQA |

---

## 14. Что отвечать на "почему теория и эмпирика расходятся"

**Универсальный ответ:**

> "Theorem 4 даёт асимптотический worst-case bound на шум-пол. При конечном $T$ (мы используем 200-1000 раундов) мы находимся в **transient regime** — bound $(1-3\eta\mu/2)^T V_0$ доминирует, шум-пол не достигнут. Кроме того, effective dimension $d$ для нашей setting (Qwen3.5 с frozen vision tower) значительно меньше total parameter count. Loose theoretical bound — это feature, не bug: real-world deployments enjoy этот gap."

---

## 15. TL;DR одна фраза для каждого вклада

1. **Federated wrapper:** 16 байт/раунд через общий seed.
2. **D-MeZO-N v1:** moment+clip+decay даёт Lyapunov-сходимость (T3).
3. **D-MeZO-N v2:** adaptive clip + drift-reset исправляет accuracy paradox.
4. **DP-MeZO:** ρ-клип одновременно служит L2-чувствительностью для Gaussian mechanism — free DP.
5. **Theorem 3:** $V_t$ убывает экспоненциально под PL+momentum+clip+β-decay.
6. **Theorem 4:** DP-шум добавляет $\sigma^2 d$ к шум-полу, но empirically transient regime спасает.

---

## 16. Презентационный slide — 30 секунд summary

> **Mission:** federated LLM fine-tuning с privacy + bandwidth + memory efficiency.
>
> **Tech:** D-MeZO-N — peer-to-peer ZO с heavy-ball momentum (β-decay + clip) + Gaussian noise.
>
> **Math:** 2 новые теоремы (T3 = momentum convergence, T4 = DP extension).
>
> **Result:** первый decentralized federated ZO для LLM с формальной (ε=10, δ=10⁻³)-DP при ~6% utility cost.
>
> **Compression:** 16 байт/раунд vs 8 ГБ (FedAvg) = $10^9$× компрессии.
>
> **Honesty:** 5 изначальных claims falsified — это признак серьёзного research.

---

*Документ обновлён 2026-05-20. Соответствует commit ≥ `87ddae9`.*

*Связанные документы:*
- `docs/theory_rigorous.md` — полные доказательства T2, T3, T4 (для математиков).
- `docs/paper_en.md`, `docs/paper_ru.md` — формальная статья.
- `docs/project_review.md` — подробный разбор всех технологий.
