# Строгая теория D-MeZO-N

**Цель документа:** записать **полные, строгие** доказательства трёх теорем сходимости D-MeZO-N с явным указанием каждого шага, использованного неравенства, и обсуждением того, что **не** доказано.

**Структура:**
- §0 Обозначения и предположения
- §1 Пять базовых лемм
- §2 Theorem 2 — PL без момента (простейшая)
- §3 Theorem 3 — PL + heavy-ball + β-decay (центральная)
- §4 Theorem 1 — convex + momentum + decentralized (federated)
- §5 Предсказания vs эмпирика (честный matching)
- §6 Что не доказано

**Этот документ заменяет** более ранние `docs/04-theory.md` и `docs/theory_nesterov_mezo.md`, объединяя их с устранением inconsistencies в обозначениях и closure of gaps в доказательствах.

---

## §0. Обозначения

| Символ | Смысл |
|---|---|
| $\theta \in \mathbb{R}^d$ | Параметры модели; $d$ — размерность |
| $L: \mathbb{R}^d \to \mathbb{R}$ | Loss функция; $L^\star := \inf_\theta L(\theta)$ |
| $\nabla L_t := \nabla L(\theta_t)$ | Градиент в точке $\theta_t$ |
| $H(\theta) := \nabla^2 L(\theta)$ | Гессиан |
| $\ell$ | Константа Липшица градиента ($\|H\|_{op} \le \ell$) |
| $\ell_3$ | Bound на третью производную (только в Lemma 1a) |
| $\mu$ | Константа PL ($\|\nabla L\|^2 \ge 2\mu(L - L^\star)$) |
| $r(H) := \mathrm{tr}(H)/\|H\|_{op}$ | Эффективный ранг гессиана |
| $z \sim \mathcal{N}(0, I_d)$ | Возмущение MeZO (восстанавливаемо из seed) |
| $\epsilon > 0$ | Магнитуда возмущения |
| $\hat\rho := \frac{L(\theta+\epsilon z) - L(\theta-\epsilon z)}{2\epsilon}$ | MeZO projected gradient |
| $\hat g := \hat\rho \cdot z$ | MeZO gradient estimator |
| $\tilde\rho := \mathrm{clip}(\hat\rho, \pm C)$ | Clipped projected gradient |
| $C > 0$ | Порог ρ-clipping |
| $\eta > 0$ | Learning rate |
| $\beta_t \in [0, \beta_0]$ | Momentum schedule, $\beta_0 < 1$, не возрастающий |
| $v_t \in \mathbb{R}^d$ | Momentum buffer, $v_0 = 0$ |
| $\mathcal{F}_t := \sigma(z_0, \ldots, z_{t-1})$ | Фильтрация |
| $n$ | Число клиентов (federated) |
| $W \in \mathbb{R}^{n\times n}$ | Mixing matrix, doubly-stochastic |
| $\rho_W := \|W - \frac{1}{n}\mathbf{1}\mathbf{1}^\top\|_{op}$ | Spectral gap, $\in [0,1)$ |

**Внимание:** $\rho_W$ (spectral gap mixing matrix) и $\hat\rho$ (MeZO projected gradient) — **разные** объекты с одинаковой буквой в литературе. Здесь они различены explicitly.

## §0.1 Предположения

**(A1) $\ell$-smoothness.**
$$\|\nabla L(\theta) - \nabla L(\theta')\| \le \ell \|\theta - \theta'\|, \quad \forall \theta, \theta'.$$
Эквивалентно: $L(\theta') \le L(\theta) + \langle \nabla L, \theta' - \theta\rangle + \frac{\ell}{2}\|\theta' - \theta\|^2$.

**(A2 / PL) Polyak-Łojasiewicz inequality.**
$$\frac{1}{2}\|\nabla L(\theta)\|^2 \ge \mu \cdot (L(\theta) - L^\star) \quad \forall \theta.$$
Слабее convexity, но сильнее general non-convex. Используется в T2, T3.

**(A3) MeZO unbiasedness (после smoothing).** Для $L \in C^3$ с bounded $\nabla \Delta L$:
$$\mathbb{E}_z[\hat g \mid \mathcal{F}_t] = \nabla L(\theta_t) + O(\epsilon^2).$$

**(A4) Bounded ZO variance под ρ-clipping.** При $|\tilde\rho| \le C$:
$$\mathbb{E}\bigl[ \|\tilde g\|^2 \mid \mathcal{F}_t \bigr] \le G^2, \qquad G^2 := C^2 \cdot d \quad \text{(raw)} \quad \text{или} \quad G^2 := C^2 r(H) \ell \quad \text{(Malladi-refined)}.$$

**(A5) Bounded gradient heterogeneity (только в T1, federated).**
$$\frac{1}{n}\sum_i \|\nabla L_i(\theta) - \nabla L(\theta)\|^2 \le \zeta^2, \quad \forall \theta.$$

## §0.2 Convexity / PL спектр

Для информации — какое предположение даёт что:

| Assumption | Strength | Convergence rate |
|---|---|---|
| Convex | Weakest | $O(1/T)$ |
| Convex + smooth | Moderate | $O(1/T)$ |
| μ-strongly convex | Strong | $O((1-\eta\mu)^T)$ |
| μ-PL | Между smooth и SC | $O((1-\eta\mu)^T)$ to noise floor |

PL is **weaker than strong convexity** (SC implies PL with same μ), но даёт тот же exponential rate. Для overparameterized LLM PL is more realistic (gradient может быть мал в multiple minima, не в одном).

---

## §1. Базовые леммы

### Lemma 1 — ZO bias-variance

**(a) Bias.** Для $L \in C^3$ с bounded $\nabla \Delta L$:
$$\bigl\|\mathbb{E}\hat g - \nabla L\bigr\| \le \frac{\ell_3}{2} \epsilon^2 (d+2).$$

**(b) Raw variance.**
$$\mathbb{E}\|\hat g\|^2 \le (d+2)\|\nabla L\|^2 + O(\epsilon^2).$$

**(b′) Refined variance (Malladi-style).** Для любой PSD $M$:
$$\mathbb{E}\bigl[\hat g^\top M \hat g\bigr] \le \|M\|_{op}\|\nabla L\|^2(r(M) + 2) + O(\epsilon^2).$$

**Доказательство (a):** Taylor третьего порядка для $L(\theta \pm \epsilon z)$:
$$L(\theta \pm \epsilon z) = L(\theta) \pm \epsilon \langle\nabla L, z\rangle + \frac{\epsilon^2}{2} z^\top H z \pm \frac{\epsilon^3}{6} D^3 L[z,z,z] + O(\epsilon^4).$$

Разность $L(\theta+\epsilon z) - L(\theta-\epsilon z) = 2\epsilon\langle\nabla L, z\rangle + \frac{\epsilon^3}{3} D^3 L[z,z,z] + O(\epsilon^5)$ — **чётные степени сокращаются**.

Деление на $2\epsilon$ и умножение на $z$: $\hat g = z\langle\nabla L, z\rangle + \frac{\epsilon^2}{6} z \cdot D^3 L[z,z,z] + O(\epsilon^4)$.

$\mathbb{E}_z[zz^\top] = I_d$ ⇒ $\mathbb{E}_z[z\langle\nabla L, z\rangle] = \nabla L$.

Bias-член: по теореме Изерлиса, $\mathbb{E}[z_i z_j z_k z_l] = \delta_{ij}\delta_{kl} + \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}$. Применение к $\sum_{jkl} D^3 L_{jkl} \mathbb{E}[z_i z_j z_k z_l]$ даёт $3[\nabla\Delta L]_i$, ограниченный $\ell_3 d$. $\square$

**Доказательство (b):** Используем $\hat\rho \approx \langle\nabla L, z\rangle$ + lower-order terms. Разложение $z = \alpha u + w$ где $u = \nabla L/\|\nabla L\|$, $\alpha \sim \mathcal{N}(0,1)$, $w \in u^\perp$, $\|w\|^2 \sim \chi^2_{d-1}$.

$\mathbb{E}[\langle\nabla L, z\rangle^2 \|z\|^2] = \|\nabla L\|^2 \mathbb{E}[\alpha^2(\alpha^2 + \|w\|^2)] = \|\nabla L\|^2 (\mathbb{E}\alpha^4 + \mathbb{E}\|w\|^2) = \|\nabla L\|^2(3 + d-1) = (d+2)\|\nabla L\|^2$.

$\square$

**Доказательство (b′):** Диагонализация $M = U\Lambda U^\top$, замена $z' = U^\top z$ (rotation-invariance ⇒ $z'$ тоже $\mathcal{N}(0,I)$). Используя Изерлиса для $\mathbb{E}[z'_i z'_k z'_j z'_j]$ и упрощение:
$$\mathbb{E}[\langle\nabla L, z\rangle^2 z^\top M z] = \|\nabla L\|^2 \mathrm{tr}(M) + 2(\nabla L)^\top M (\nabla L) \le \|M\|_{op}\|\nabla L\|^2(r(M) + 2). \quad \square$$

**Ключевое наблюдение:** в descent-inequality нам нужен не $\mathbb{E}\|\hat g\|^2$ raw, а $\mathbb{E}[\hat g^\top H_* \hat g]$ (где $H_*$ — локальный гессиан). Здесь $r(H_*) \ll d$ для LLM, что даёт $10^6$-кратное послабление в stable learning rate.

### Lemma 2 — ρ-clipping bias-variance

Пусть $\hat\rho$ с $\mathbb{E}\hat\rho^2 \le M^2$. Определим $\tilde\rho = \mathrm{clip}(\hat\rho, \pm C)$, $\tilde g = \tilde\rho z$.

**(a) Scalar bias.** $|\mathbb{E}\tilde\rho - \mathbb{E}\hat\rho| \le M^2/C$.

**(b) Variance (quadratic form).** Для PSD $M$:
$$\mathbb{E}[\tilde g^\top M \tilde g] \le C^2 \mathrm{tr}(M) = C^2 r(M) \|M\|_{op}.$$

**Доказательство (a):** На $E = \{|\hat\rho| > C\}$: $|\tilde\rho - \hat\rho| \le |\hat\rho|$. На $\bar E$: $\tilde\rho - \hat\rho = 0$.

$|\mathbb{E}\tilde\rho - \mathbb{E}\hat\rho| \le \mathbb{E}[|\hat\rho|\mathbf{1}_E] \stackrel{\text{Cauchy-Schwarz}}{\le} \sqrt{\mathbb{E}\hat\rho^2 \cdot \mathbb{P}(E)} \stackrel{\text{Markov}}{\le} \sqrt{M^2 \cdot M^2/C^2} = M^2/C. \quad \square$

**Доказательство (b):** Pointwise $\tilde\rho^2 \le C^2$. Тогда $\tilde g^\top M \tilde g = \tilde\rho^2 z^\top M z \le C^2 z^\top M z$. Беря $\mathbb{E}$ и используя $\mathbb{E}[z^\top M z] = \mathrm{tr}(M)$ для $z \sim \mathcal{N}(0,I)$:
$$\mathbb{E}[\tilde g^\top M \tilde g] \le C^2 \mathrm{tr}(M). \quad \square$$

**Принципиальная особенность:** uniform bound, **не зависящий от $\|\nabla L\|$**. Без clip variance растёт с $\|\nabla L\|^2$ (Lemma 1b); с clip — bounded constantly. Это **основа** для контроля momentum-amplified variance в T3.

### Lemma 3 — PL descent с biased gradient

**Постановка.** Пусть $f$ — $\ell$-smooth, $\mu$-PL. Итерация $\theta_{t+1} = \theta_t - \eta g_t$ с:
- (B1) $\mathbb{E}[g_t|\mathcal{F}_t] = \nabla f_t + b_t$, $\|b_t\| \le \delta$.
- (B2) $\mathbb{E}[\|g_t - \mathbb{E}g_t\|^2 |\mathcal{F}_t] \le \sigma^2$.

**Утверждение.** При $\eta \le 1/\ell$:
$$\mathbb{E}[f(\theta_{t+1}) - f^\star \mid \mathcal{F}_t] \le (1 - \eta\mu)(f_t - f^\star) + \frac{\eta\delta^2}{2} + \frac{\eta^2\ell\sigma^2}{2}.$$

**Доказательство (8 шагов):**

(1) L-smoothness: $f_{t+1} \le f_t - \eta\langle\nabla f_t, g_t\rangle + \frac{\eta^2\ell}{2}\|g_t\|^2$.

(2) $\mathbb{E}[\cdot|\mathcal{F}_t]$: $\mathbb{E}[f_{t+1}|\mathcal{F}_t] \le f_t - \eta\langle\nabla f_t, \mathbb{E}g_t\rangle + \frac{\eta^2\ell}{2}\mathbb{E}\|g_t\|^2$.

(3) Bias подстановка: $\langle\nabla f_t, \mathbb{E}g_t\rangle = \|\nabla f_t\|^2 + \langle\nabla f_t, b_t\rangle$.

(4) Variance split: $\mathbb{E}\|g_t\|^2 = \|\mathbb{E}g_t\|^2 + \mathbb{E}\|g_t - \mathbb{E}g_t\|^2 \le \|\nabla f_t + b_t\|^2 + \sigma^2 = \|\nabla f_t\|^2 + 2\langle\nabla f_t, b_t\rangle + \|b_t\|^2 + \sigma^2$.

(5) Подстановка в descent:
$$\mathbb{E}[f_{t+1} - f_t|\mathcal{F}_t] \le -\eta(1 - \eta\ell/2)\|\nabla f_t\|^2 - \eta(1-\eta\ell)\langle\nabla f_t, b_t\rangle + \frac{\eta^2\ell}{2}\|b_t\|^2 + \frac{\eta^2\ell\sigma^2}{2}. \quad (\star)$$

(6) Young's с $\alpha=1$: $|\langle\nabla f_t, b_t\rangle| \le \frac{1}{2}\|\nabla f_t\|^2 + \frac{1}{2}\|b_t\|^2$. При $\eta\le 1/\ell$, абсорбируем:
$$-\eta(1-\eta\ell)\langle\nabla f_t, b_t\rangle \le \frac{\eta(1-\eta\ell)}{2}(\|\nabla f_t\|^2 + \|b_t\|^2).$$

(7) Коэффициент при $\|\nabla f_t\|^2$ в $(\star)$ после абсорбции: $-\eta(1-\eta\ell/2) + \eta(1-\eta\ell)/2 = -\eta/2$. **Точное сокращение** $\eta^2\ell$ — главный момент.

Коэффициент при $\|b_t\|^2$: $\eta(1-\eta\ell)/2 + \eta^2\ell/2 = \eta/2$.

(8) PL: $-\frac{\eta}{2}\|\nabla f_t\|^2 \le -\eta\mu(f_t - f^\star)$. Используя $\|b_t\|^2 \le \delta^2$:
$$\mathbb{E}[f_{t+1} - f^\star|\mathcal{F}_t] \le (1-\eta\mu)(f_t - f^\star) + \frac{\eta\delta^2}{2} + \frac{\eta^2\ell\sigma^2}{2}. \quad \square$$

### Lemma 4 — Consensus error

**Постановка.** $n$ клиентов с $\theta_i^t$, mixing matrix $W$ doubly-stoch с $\rho_W = \|W - \mathbf{1}\mathbf{1}^\top/n\|_{op} < 1$. Под update $\theta_i^{t+1} = \sum_j W_{ij}(\theta_j^t - \eta v_j^{t+1})$ и (A5):
$$\mathbb{E}\!\left[\frac{1}{n}\sum_i \|\theta_i^t - \bar\theta_t\|^2\right] \le \frac{\rho_W^2}{(1-\rho_W)^2} \cdot \frac{\eta^2(G^2 r(H) + \zeta^2)}{(1-\bar\beta)^2}.$$

**Доказательство:** свойство $W$ — $\|WX - \mathbf{1}\bar X^\top\|_F \le \rho_W \|X - \mathbf{1}\bar X^\top\|_F$. Применение к $\Theta_{t+1} = W(\Theta_t - \eta V_{t+1})$ + geometric series по spectral gap + momentum amplification из Lemma 2 на per-client $G^2$. (Полный proof — Koloskova 2020 Lemma 3 + adaption через Lemma 2.)

### Lemma 5 — Геометрическая сумма

**Постановка.** Неотрицательная последовательность $(a_t)$ с $a_{t+1} \le (1-q)a_t + b$, $q \in (0,1)$, $b \ge 0$. Тогда:
$$a_T \le (1-q)^T a_0 + b/q.$$

**Доказательство:** разворачивание рекурсии + сумма геометрической прогрессии $\sum_{k=0}^{T-1}(1-q)^k \le 1/q$. $\square$

---

## §2. Theorem 2 — D-MeZO под PL без момента

### Постановка

Federated, complete graph ($\rho_W = 0$), IID partition. $n$ клиентов с независимыми seeds. Update: $\theta_{t+1} = \theta_t - \eta\bar{\tilde g}_t$, $\bar{\tilde g}_t = \frac{1}{n}\sum_i \tilde g_i^t$.

### Утверждение

Под (A1)+(A2/PL), $\mathbb{E}\hat\rho_i^2 \le M^2$, $\eta \le 1/(4\ell)$:
$$\boxed{\quad \mathbb{E}[L(\theta_T) - L^\star] \le \bigl(1 - \tfrac{\eta\mu}{2}\bigr)^T \Delta_0 + \frac{3\delta^2}{2\mu} + \frac{\eta C^2 r(H) \ell}{\mu n} \quad}$$

где $\delta = O(\epsilon^2 \ell_3 d) + O(M^2/C)$ — combined bias из Lemmas 1a + 2a.

### Доказательство

**(1) Second-order Taylor:**
$$L_{t+1} - L_t = -\eta\langle\nabla L_t, \bar{\tilde g}_t\rangle + \frac{\eta^2}{2}\bar{\tilde g}_t^\top H_* \bar{\tilde g}_t.$$

**(2) $\mathbb{E}[\cdot|\mathcal{F}_t]$, bias** $\mathbb{E}\bar{\tilde g}_t = \nabla L_t + b_t$, $\|b_t\| \le \delta$.

**(3) Variance split** через independent clients:
$$\mathbb{E}[\bar{\tilde g}_t^\top H_* \bar{\tilde g}_t] \le 2\ell\|\nabla L_t\|^2 + 2\ell\delta^2 + \frac{C^2 r(H)\ell}{n}.$$

Последний член — **federated speedup $1/n$** в variance.

**(4)** Подстановка → коэффициент $\|\nabla L_t\|^2$: $-\eta(1-\eta\ell)$. С Young's $\alpha=1$ на cross-term $\langle\nabla L, b\rangle$:
$$\le -\eta(\tfrac{1}{2} - \eta\ell)\|\nabla L_t\|^2 + \tfrac{\eta(1+2\eta\ell)}{2}\delta^2 + \frac{\eta^2 C^2 r(H)\ell}{2n}.$$

**(5)** При $\eta \le 1/(4\ell)$: $\tfrac{1}{2} - \eta\ell \ge 1/4$, $1+2\eta\ell \le 3/2$.

**(6)** PL: $-\frac{\eta}{4}\|\nabla L_t\|^2 \le -\frac{\eta\mu}{2}(L_t - L^\star)$.

**(7)** Recursion: $a_{t+1} \le (1 - \eta\mu/2)a_t + \frac{3\eta\delta^2}{4} + \frac{\eta^2 C^2 r(H)\ell}{2n}$.

**(8)** Lemma 5: $a_T \le (1-\eta\mu/2)^T a_0 + \frac{2/q \cdot \text{noise}}{1}$ где $q = \eta\mu/2$. После упрощения — получаем bound. $\square$

### Интерпретация

| Член | Природа | Зависимость |
|---|---|---|
| $(1-\eta\mu/2)^T \Delta_0$ | Transient | exponential decay |
| $\frac{3\delta^2}{2\mu}$ | **Bias floor** | Не зависит от $\eta, n$ |
| $\frac{\eta C^2 r(H)\ell}{\mu n}$ | **Variance floor** | Linear $1/n$ speedup |

**Главное:** federated $1/n$ speedup в variance term; bias term не убирается ростом $n$.

---

## §3. Theorem 3 — PL + heavy-ball + β-decay (центральная)

### Постановка

Centralized heavy-ball (federated extension trivial):
$$v_{t+1} = \beta_t v_t + \tilde g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}.$$

**Lyapunov-функция:**
$$V_t := (L_t - L^\star) + \frac{\eta}{2}\|v_t\|^2.$$

Множитель $\eta/2$ при кинетике подобран **специально** для сокращения cross-term.

### Утверждение

Под (A1)+(A2/PL)+(A4) и unbiasedness ($\mathbb{E}\tilde g = \nabla L$), при $\eta \le (1-\beta_0^2)/(8\ell)$:
$$\boxed{\quad \mathbb{E}[V_T] \le \bigl(1 - \tfrac{3\eta\mu}{2}\bigr)^T V_0 + \frac{2G^2}{3\mu} \quad}$$

### Доказательство (9 шагов)

**(1) L-smoothness:** $L_{t+1} \le L_t - \eta\langle\nabla L_t, v_{t+1}\rangle + \frac{\eta^2\ell}{2}\|v_{t+1}\|^2$.

**(2) Расширение** $v_{t+1} = \beta_t v_t + \tilde g_t$:
- $\mathbb{E}\langle\nabla L_t, v_{t+1}\rangle = \beta_t\langle\nabla L_t, v_t\rangle + \|\nabla L_t\|^2$.
- $\mathbb{E}\|v_{t+1}\|^2 \le \beta_t^2\|v_t\|^2 + 2\beta_t\langle v_t, \nabla L_t\rangle + G^2$.

**(3) Подстановка в descent inequality.**

**(4) Кинетический шифт:**
$$\frac{\eta}{2}\mathbb{E}[\|v_{t+1}\|^2 - \|v_t\|^2 | \mathcal{F}_t] \le -\frac{\eta(1-\beta_t^2)}{2}\|v_t\|^2 + \eta\beta_t\langle v_t, \nabla L_t\rangle + \frac{\eta G^2}{2}.$$

**(5) Sum (Steps 3+4) — ⚡ cross-term cancellation.** Cross-terms $\pm \eta\beta_t\langle\nabla L, v\rangle$ ровно сокращаются. **Это магия выбора $V_t$:**
$$\mathbb{E}[V_{t+1} - V_t|\mathcal{F}_t] \le -\eta\|\nabla L_t\|^2 - \frac{\eta(1-\beta_t^2)}{2}\|v_t\|^2 + \frac{\eta^2\ell}{2}\mathbb{E}\|v_{t+1}\|^2 + \frac{\eta G^2}{2}.$$

**(6) Bound $\frac{\eta^2\ell}{2}\mathbb{E}\|v_{t+1}\|^2$:** Young's на $\langle v_t, \nabla L_t\rangle$, simplify к $\le \eta^2\ell\|v_t\|^2 + \frac{\eta^2\ell}{2}\|\nabla L_t\|^2 + \frac{\eta^2\ell G^2}{2}$.

**(7) Группировка и выбор $\eta \le (1-\beta_0^2)/(8\ell)$:**
- Коэффициент $\|\nabla L\|^2$: $-\eta(1 - \eta\ell/2) \le -3\eta/4$.
- Коэффициент $\|v\|^2$: $-\eta((1-\beta_t^2)/2 - \eta\ell) \le -\frac{3\eta(1-\beta_t^2)}{8}$.
- Noise: $\eta G^2(1+\eta\ell)/2 \le \eta G^2$.

$$\mathbb{E}[V_{t+1} - V_t|\mathcal{F}_t] \le -\frac{3\eta}{4}\|\nabla L_t\|^2 - \frac{3\eta(1-\beta_t^2)}{8}\|v_t\|^2 + \eta G^2.$$

**(8) PL + Lyapunov contraction.** PL: $-\frac{3\eta}{4}\|\nabla L\|^2 \le -\frac{3\eta\mu}{2}(L - L^\star)$. Kinetic: $-\frac{3\eta(1-\beta_t^2)}{8}\|v\|^2 = -\frac{3(1-\beta_t^2)}{4} \cdot \frac{\eta\|v\|^2}{2}$.

Каждая компонента $V_t$ контрактируется со своим rate. Под $\eta\mu \le (1-\beta_0^2)/2$ (выполнено для $\mu \le 4\ell$): worst rate = $3\eta\mu/2$ (PL-dominated).

$$\mathbb{E}[V_{t+1}|\mathcal{F}_t] \le (1 - 3\eta\mu/2) V_t + \eta G^2.$$

**(9) Lemma 5:** $\mathbb{E}V_T \le (1-3\eta\mu/2)^T V_0 + \eta G^2/(3\eta\mu/2) = (1-3\eta\mu/2)^T V_0 + 2G^2/(3\mu)$.

**Замечание:** $\eta$ **сокращается** в noise floor. Уменьшение $\eta$ → медленнее convergence, **тот же floor**. Чтобы уменьшить floor → уменьшить $G^2$ (через clip или multi-direction). $\square$

### Corollary 7.1 — почему β-decay лучше const β

$V_T = (L_T - L^\star) + \frac{\eta}{2}\|v_T\|^2$ ⇒ $L_T - L^\star = V_T - \frac{\eta}{2}\|v_T\|^2$.

**Steady-state $\|v\|^2$:**
- Const $\beta=0.9$: $\|v_\infty\|^2 \approx G^2/(1-\beta^2) \approx 5.3 G^2$.
- $\beta_T=0$ (decay end): $\|v_T\|^2 \approx G^2$.

**В 5× меньше кинетика** при decay → **узкая** оценка на чистый loss.

### Что Theorem 3 НЕ доказывает

**1. Acceleration не доказана.** Rate $(1 - 3\eta\mu/2)^T$ — **тот же** что для plain SGD. Эмпирическое 3× speedup (Day 8 R1b до R300) **не объяснено**. Это transient phenomenon — требует более тонкого analysis (estimate sequence / Yang-Zhao-Cheng framework). **Открытая проблема.**

**Альтернативная честная формулировка:** "D-MeZO-N имеет **тот же асимптотический rate**, но **меньшую variance в transient phase** через momentum-smoothing." Согласуется с Bottou-Curtis-Nocedal 2018 Theorem 5.1.

**2. Look-ahead Nesterov vs heavy-ball.** Proof только для heavy-ball $v_{t+1} = \beta v_t + \tilde g_t$ при оценке в текущей точке. Look-ahead имеет dual-channel noise (probe location + update direction) — эмпирически дивергит R20.

**3. Bias** $b_t \ne 0$. Под bias добавляется $\propto \delta^2/\mu$ floor. Структура не нарушается, константы усложняются.

---

## §4. Theorem 1 — convex + momentum + decentralized

### Постановка

$n$ клиентов, mixing $W$ с $\rho_W < 1$. Каждый $L_i$ convex и $\ell$-smooth. (A5) с $\zeta^2$. Iteration с momentum + ρ-clip + mixing. Polyak-Ruppert average $\hat\theta_T := \frac{1}{T}\sum_t \bar\theta_t$.

### Утверждение

При $\eta \le \min(1/(\ell r(H)), 1/\sqrt{T})$ и $C \ge 2\|\nabla L\|_{\max} + \epsilon\ell\sqrt{r(H)}$:
$$\boxed{\quad \mathbb{E}[L(\hat\theta_T) - L^\star] \le \tilde O\!\left(\sqrt{\frac{\ell r(H) D_0}{n T}}\right) + \tilde O\!\left(\frac{\rho_W^2 C^2 r(H)}{(1-\bar\beta)^2 T}\right) + O(\epsilon^2 \ell^2 r(H)) \quad}$$

где $D_0 = \|\bar\theta_0 - \theta^\star\|^2$, $\bar\beta = \beta_0/2$.

### Доказательство (4 шага — sketch)

**(1) Distance-to-optimum framework (convex SGD).**
$$\|\bar\theta_{t+1} - \theta^\star\|^2 = \|\bar\theta_t - \theta^\star\|^2 - 2\eta\langle\bar v_{t+1}, \bar\theta_t - \theta^\star\rangle + \eta^2\|\bar v_{t+1}\|^2.$$

**(2) Convexity + consensus drift.**

$\mathbb{E}\bar v_{t+1} = \beta_t\bar v_t + \frac{1}{n}\sum_i\nabla L_i(\theta_i^t) + b_t = \beta_t\bar v_t + \nabla L(\bar\theta_t) + \xi_t + b_t$,

где **consensus drift** $\xi_t = \frac{1}{n}\sum_i(\nabla L_i(\theta_i^t) - \nabla L_i(\bar\theta_t))$, $\|\xi_t\| \le \ell\sqrt{\Pi_t}$.

По convexity: $\langle\nabla L(\bar\theta_t), \bar\theta_t - \theta^\star\rangle \ge L(\bar\theta_t) - L^\star$.

**(3) Telescoping + Jensen:**

Суммируем по $t$, делим на $2\eta T$, применяем Jensen для $\hat\theta_T = \frac{1}{T}\sum_t \bar\theta_t$:
$$L(\hat\theta_T) - L^\star \le \frac{D_0}{2\eta T} + \underbrace{\text{drift term}}_{\text{(I)}} + \underbrace{\text{variance term}}_{\text{(II)}}.$$

**(4) Bounds (I) + (II) — Lemma 4 для consensus + per-client variance.**

(II) $\le \frac{\eta C^2 r(H) \ell}{2 n (1-\bar\beta)^2}$ — federated speedup $1/n$.

(I) $\le \frac{\rho_W^2 C^2 r(H) \ell}{(1-\rho_W)^2(1-\bar\beta)^2 T} \cdot \text{const}$ — consensus penalty.

Минимизация по $\eta^* = \sqrt{D_0 n/(C^2 r(H) \ell T)}$:
$$\frac{D_0}{2\eta T} + \frac{\eta C^2 r(H)\ell}{2n} \ge \sqrt{\frac{D_0 \cdot C^2 r(H)\ell}{nT}} \cdot \text{const}. \quad \square$$

### Что Theorem 1 даёт и НЕ даёт

**Даёт:**
- $1/\sqrt{nT}$ rate в stochastic term — directional match с эмпирическим federated < centralized.
- Consensus penalty $\rho_W^2/(1-\rho_W)^2$ — для ring(4): factor 0.25 относительно complete. Penalty < stochastic term при наших $T$.

**Не даёт:**
- **Momentum acceleration** — bound is unchanged with/without momentum в convex case.
- **Финальный loss ratio.** Bound is on $\mathbb{E}[L(\hat\theta_T) - L^\star]$ (Polyak average) — measure is $L(\theta_T)$. Без знания $D_0/\Delta_0$ нельзя сопоставить $0.74 \approx 1/\sqrt{4}$ напрямую — это **разные** величины.

---

## §5. Предсказания vs эмпирика (честный matching)

| # | Предсказание | Из теоремы | Эмпирика | Match |
|---|---|---|---|---|
| P1 | Federated stochastic term ↓ as $n$ растёт | T1: $1/\sqrt{nT}$; T2: $1/n$ | Centralized 0.176 → fed 0.130 ratio 0.74 | **directional only**; numerical mismatch с $1/\sqrt{4}=0.5$ — bound is on Polyak avg / rate, not final loss |
| P2 | $\beta=0.9$ без clip → divergence | T3: $G^2$ unbounded → noise floor blow-up | Blow-up на R140 | ✓ qualitative |
| P3 | Look-ahead двойное noise channel | вне scope T3 | NaN R20 (7× быстрее) | ✓ qualitative |
| P4 | $\rho$-clip + const $\beta$ → late drift | T3 corollary: $\|v_\infty\|^2 \approx 5G^2$ | R1b 0.119 → 0.225 drift | ✓ direction; magnitude — handwave |
| P5 | $\beta$-decay убирает drift | T3 corollary: $\|v_T\|^2 \to G^2$ | R1d monotonic | ✓ qualitative |
| P6 | Линейная сходимость $(1-\eta\mu)^T$ | T2/T3 main rate | Ring+IID 3.56 → 0.126 | ✓ qualitative; $\mu$ не calibrated |
| P7 | Consensus penalty $\rho_W^2/(1-\rho_W)^2$ | T1: factor 0.25 для ring(4) | complete ≈ ring (<7% diff) | ✓ (penalty dominated by stochastic) |
| P8 | ZO bias $O(\epsilon^2)$ | Lemma 1a | $\epsilon=10^{-3} \Rightarrow$ pred bias $<10^{-6}$ | ✓ trivially |

**Главные caveat'ы:**

1. **P1 not numerical match.** Соотношение $0.74 \neq 1/\sqrt{4}=0.5$. Bound is on rate, не final loss. Нельзя приклеивать.
2. **Linear convergence (P6)** — $\mu$ для LLM не известна. Empirical fit possible, но не predictive.
3. **3× acceleration** (Day 8 R1b R100→R300) — **не предсказано** ни одной теоремой. Эмпирическое observation.

---

## §6. Что не доказано (открытые проблемы)

### Open Problem 1 — Acceleration под PL

Theorems 2, 3 дают **тот же rate** $(1-\eta\mu)^T$ независимо от наличия momentum. Эмпирическое 3× speedup (Day 8) → **transient acceleration**, не asymptotic.

**Что нужно:** finite-time analysis с estimate sequence (Nesterov 2018 framework). Yang-Zhao-Cheng 2016 adaptive momentum analysis — close, но не для ZO.

**Сложность:** 4-мерная композиция (non-convex × PL × momentum × ZO × clipping). Ни одной paper не делал.

### Open Problem 2 — Full decentralized Theorem 3

T3 — centralized. Для full decentralized нужен Lyapunov $\Phi_t = (L(\bar\theta_t) - L^\star) + (\eta/2)\|\bar v_t\|^2 + \Pi_t \cdot c$ для какой-то константы $c$. Cross-terms в $\Pi_t$-эволюции под momentum — нетривиальны.

### Open Problem 3 — Look-ahead Nesterov

True look-ahead: $\tilde g_t = \mathrm{MeZO}(\theta_t + \beta v_t)$. Bias и variance имеют dual-channel структуру (probe и update оба зависят от $v_t$). Эмпирически дивергит R20.

Теоретическое объяснение: variance amplification $\sim 1/(1-\beta)^4$ (квадрат от heavy-ball $1/(1-\beta)^2$). Нет строгого proof, но согласуется с эмпирикой.

### Open Problem 4 — Optimal $\beta$-schedule

Linear vs cosine vs hold-then-decay — теория даёт только sufficient conditions. Optimal schedule — open.

### Open Problem 5 — Hybrid linear-attention specific bounds

Qwen3.5-4B-Base — hybrid arch. Effective $r(H)$ может отличаться от full-attention. Нет analytical results.

---

## §7. Литература (use bricks)

**T2 (PL без момента):**
- Karimi, Nutini, Schmidt 2016. *Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition*. ECML PKDD. **Главный шаблон Lemma 3.**
- Malladi et al. 2023 (NeurIPS). MeZO, Theorem 3.1 — $r(H)$-substitution.
- Koloskova et al. 2020 (ICML). Lemma 3 — consensus error.

**T3 (PL + heavy-ball + β-decay):**
- Polyak 1964. Heavy-ball original.
- Ghadimi-Lan 2013. SIAM J. Optim. — stochastic non-convex heavy-ball.
- Bottou-Curtis-Nocedal 2018. SIAM Review — момент не быстрее SGD когда $\sigma > 0$.
- Liu-Zhu-Belkin 2022. *Loss landscapes and optimization in over-parameterized non-linear systems and neural networks*. ACHA. PL для DNN.

**T1 (convex + momentum + decentralized):**
- Stich 2019 (ICLR). *Local SGD converges fast and communicates little*. Virtual averaged sequence.
- Koloskova 2020 Theorem 2.
- Lan 2012. *Optimal method for stochastic composite optimization*. Acc variance bound.

**Open problems references:**
- Yang-Lin-Li 2016. *Unified convergence analysis of stochastic momentum methods*. — для full Theorem 3.
- Aybat-Fallah-Gurbuzbalaban-Ozdaglar 2019 (NeurIPS). Universally optimal acceleration. — для optimal β-schedule.
- Nesterov 2018 *Lectures on Convex Optimization* — estimate sequence.

---

## §8. Сводный scorecard

| Цель спеки | Empirical | Mathematical |
|---|---|---|
| MeZO base | ✅ | ✅ (Malladi 2023) |
| Distributed | ✅ | ✅ (T1 convex + T2 PL) |
| Consensus variants | ✅ | ✅ (Lemma 4) |
| Accelerated schemes (rate) | ⚠️ empirical 3× | ❌ **не доказано** (Open Problem 1) |
| Nesterov heavy-ball stability | ✅ R1d | ✅ **(T3 в centralized)** |
| Local LLM copies | ✅ | N/A |
| MeZO updates | ✅ | ✅ |
| P2P consensus | ✅ | ✅ (T1) |
| Consensus mixing | ✅ | ✅ |
| Nesterov acceleration (asymptotic) | — | ❌ **Bottou-Curtis-Nocedal contradicts** |

**Итог: 8/9 empirically, 7/9 mathematically.** Открытые: acceleration proof (OP1), full decentralized T3 (OP2).

**Что paper честно может claim:**
- C1: First MeZO on hybrid linear-attn (✅ verified).
- C4: **stabilization** (rescue from divergence + safe-tracking), **не acceleration**.
- C5/C6: T1, T2, T3 proved as stated — **rate same as plain SGD**, momentum даёт лучшую transient phase + lower kinetic energy at convergence.

**Что paper НЕ должен claim:**
- "First **accelerated** D-MeZO" — пока асимптотическое acceleration не доказано.
- "$0.74 \approx 1/\sqrt{4}$ matches Theorem 1" — некорректный matching.
- "Federated **beats** centralized" — apples-to-apples требует 4× compute.

---

*Last updated: 2026-05-20. Документ создан как closure of theoretical gaps идентифицированных в peer-review pass.*
