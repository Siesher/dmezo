# 04. Шаблон теоремы и план доказательства

Основная цель — доказать convergence rate D-MeZO-N в non-convex setting с заменой $d \to r(H)$ во variance term.

## Главное утверждение (target)

**Теорема (D-MeZO-N convergence, non-convex PL).**

Пусть выполнены предположения **(A1)**–**(A5)** из `03-algorithm-spec.md`. Тогда при выборе

- $\eta = c_1 \sqrt{\mu / (L \, r(H) \, T)}$
- $\beta \in [0, 1)$ такое что $1 - \beta = c_2 \eta L$
- $\epsilon \le c_3 / \sqrt{r(H) L T}$

алгоритм D-MeZO-N после $T$ раундов удовлетворяет:

$$\min_{0 \le t < T} \mathbb E \left\|\nabla \mathcal L\!\left(\bar\theta_t\right)\right\|^2 \le \tilde O\!\left(\sqrt{\frac{L \, r(H) \, \Delta}{n T}}\right) + \tilde O\!\left(\frac{\rho^2 \, \sigma^2}{(1-\beta)^2 T}\right) + O(\epsilon^2 L^2 r(H))$$

где $\Delta = \mathcal L(\bar\theta_0) - \mathcal L^*$ и константы $c_1, c_2, c_3$ зависят только от модели.

**Интерпретация.** Первый член — стохастическая часть с linear speedup $1/n$ от distributed. Второй — penalty за плохую topology, ослабляется моментом. Третий — bias ZO-оценки.

## Скелет доказательства

### Шаг 1. ZO unbiased estimator + variance

Из Nesterov-Spokoiny 2017:

$$\mathbb E_z [\hat\rho \cdot z] = \nabla \mathcal L_\epsilon(\theta) = \nabla \mathcal L(\theta) + O(\epsilon^2 L)$$

$$\mathbb E_z \|\hat\rho \cdot z\|^2 \le 2 \|\nabla \mathcal L(\theta)\|^2 + 2 \epsilon^2 L^2 d.$$

**Уточнение для MeZO landscape (Malladi 2023, Theorem 3.1 / Section 5):** на low-rank гессиане можно заменить $d$ на $r(H) = \text{tr}(H)/\|H\|_{op}$.

### Шаг 2. Consensus error bound

Из Koloskova 2020 (Lemma 3): после consensus mixing,

$$\frac{1}{n}\sum_i \|\theta_i - \bar\theta\|^2 \le \frac{\rho^2}{(1-\rho)^2} \cdot \text{(local update magnitude)}^2.$$

### Шаг 3. Lyapunov function

Стандартный приём: рассмотреть $\Phi_t = \mathcal L(\bar\theta_t) + c \|v_t\|^2 / (1-\beta)$ как Lyapunov.

Telescoping $\Phi_t$:

$$\Phi_{t+1} - \Phi_t \le -\eta_t \|\nabla \mathcal L(\bar\theta_t)\|^2 / 2 + \eta_t^2 L \cdot (\text{variance}) + \rho^2 \cdot (\text{consensus error}).$$

### Шаг 4. Сумма по $T$ и оптимизация $\eta, \beta$

Стандартный manipulation: суммируем по $t$, нормируем на $T$, подставляем оптимальный $\eta = O(\sqrt{1/T})$, получаем target rate.

## Технические трудности

1. **Bias-variance trade-off для ZO.** Variance бесконечно растёт при $\epsilon \to 0$ (численно), bias растёт при $\epsilon \to \infty$. Нужен careful balance в выборе $\epsilon = O(T^{-1/4})$.

2. **Momentum + non-convex + decentralized.** Не очевидно, что прямой Lyapunov-аргумент даёт нужный rate. Возможно потребуется использовать modified velocity $\tilde v$ как у Stich 2019 для local-SGD анализа.

3. **$r(H)$-аргумент Malladi требует определённой смягчения гипотез на гессиан.** Их Assumption 1: $\|H\| \le L_H$ и effective rank $r(H)$ ограничен. Нужно проверить, что аргумент проходит при наличии momentum.

## Альтернативный план (fallback)

Если полная теорема не получится за разумное время — можно публиковать в workshop с **эмпирическим** main result + **слабой** теоремой только для:

(a) Convex case (тривиально, как у Koloskova).
(b) PL-condition + ZO без момента (заметно проще).
(c) Asymptotic convergence (slope $\to 0$), без rate.

## Литература для доказательства (чек-лист чтения)

- [ ] Malladi 2023, Section 5 — где они выводят $r(H)$-bound. Расписать аккуратно.
- [ ] Koloskova 2020, Theorem 2 и её proof (Appendix). Использовать как шаблон.
- [ ] Nesterov-Spokoiny 2017, Sections 2-3 — variance bounds.
- [ ] Stich 2019, "Local SGD Converges Fast" — техника для local steps.
- [ ] Gadat & Panloup 2023 — momentum в ZO без anti-correlated noise.

## Промежуточные цели

- **Неделя 1:** только формулировка + Lyapunov ansatz (no proof).
- **Неделя 2-4:** доказательство convex case (для проверки техники).
- **Месяц 2:** non-convex PL без момента.
- **Месяц 3:** добавить момент.

## Sanity check для теоремы

Проверить, что в пределе $\rho = 0$ (complete graph) воспроизводится известный rate для централизованного MeZO. В пределе $\beta = 0$ — известный rate для D-ZO-SGD (Tang, Yuan, Yang 2020).
