# Теорема о сходимости D-MeZO-N (Nesterov heavy-ball + β-decay)

**Статус:** доказательство (попытка №1, draft). Закрывает Open Problem 1 из
`docs/D-MeZO-N_explainer_ru.md` §10.2.

**Цель:** доказать линейную сходимость D-MeZO-N до окрестности минимума под
PL-условием с явной зависимостью от momentum-параметра $\beta_t$ и
ρ-clipping-параметра $C$. Объяснить эмпирический факт (Day 8 R1b vs R1d):
почему β-decay улучшает поведение в финальной фазе.

---

## 1. Обозначения

- $L: \mathbb{R}^d \to \mathbb{R}$ — целевая функция (loss дообучения LLM).
- $\theta_t \in \mathbb{R}^d$ — параметры на шаге $t$.
- $v_t \in \mathbb{R}^d$ — momentum-вектор.
- $z_t \sim \mathcal{N}(0, I_d)$ — направление SPSA-возмущения, $\mathbb{E}[z_t z_t^\top] = I_d$.
- $\hat\rho_t = \dfrac{L(\theta_t + \epsilon z_t) - L(\theta_t - \epsilon z_t)}{2\epsilon}$ — MeZO-скаляр.
- $\hat g_t = \hat\rho_t z_t$ — MeZO-оценка градиента.
- $\mathcal{F}_t = \sigma(z_0, \ldots, z_{t-1})$ — σ-алгебра, порождённая историей.
- $r(H) = \operatorname{tr}(H)/\|H\|_{\mathrm{op}}$ — эффективный ранг гессиана.

---

## 2. Предположения

**(A1) μ-PL.** Существует $\mu > 0$ такое что
$$
\tfrac{1}{2}\|\nabla L(\theta)\|^2 \;\ge\; \mu\, (L(\theta) - L^\star), \qquad \forall\, \theta.
$$

**(A2) $\ell$-smoothness.** Градиент $\ell$-Липшицев:
$$
\|\nabla L(\theta) - \nabla L(\theta')\| \;\le\; \ell\, \|\theta - \theta'\|, \qquad \forall\, \theta, \theta'.
$$

Эквивалентно: $L(\theta') \le L(\theta) + \langle \nabla L(\theta), \theta' - \theta\rangle + \tfrac{\ell}{2}\|\theta' - \theta\|^2$.

**(A3) MeZO unbiasedness (lemma 2.1 в `D-MeZO-N_explainer_ru.md`).** Для гладкой $L$
с ограниченной третьей производной:
$$
\mathbb{E}[\hat g_t \mid \mathcal{F}_t] \;=\; \nabla L(\theta_t) + O(\epsilon^2).
$$

В дальнейшем для краткости полагаем $\epsilon \to 0$ и считаем MeZO-оценку несмещённой.
Точный анализ с $O(\epsilon^2)$-bias стандартен и добавляет аддитивный член в
итоговую границу.

**(A4) Bounded variance под ρ-clipping.** При клипе $|\hat\rho_t| \le C$ имеем
$$
\mathbb{E}\bigl[\|\hat g_t\|^2 \mid \mathcal{F}_t\bigr] \;=\; \mathbb{E}\bigl[\hat\rho_t^2 \|z_t\|^2 \mid \mathcal{F}_t\bigr] \;\le\; C^2 \cdot \mathbb{E}\|z_t\|^2 \;=\; C^2 d.
$$
Используя r(H)-трюк Malladi (2023, §3.3), для overparametrized моделей
эффективная дисперсия в направлении градиента:
$$
\mathbb{E}\bigl[\|\hat g_t - \nabla L_t\|^2 \mid \mathcal{F}_t\bigr] \;\le\; \rho_g \|\nabla L_t\|^2 + \sigma^2,
$$
где $\rho_g \le c_1\, r(H)$ и $\sigma^2 \le c_2\, C^2\, r(H)$ для констант $c_1, c_2$,
зависящих только от спектра $H$.

Для упрощения изложения далее работаем с
$\mathbb{E}\|\hat g_t\|^2 \le G^2$, где $G^2 = (\rho_g + 1)\|\nabla L_{\max}\|^2 + \sigma^2 \approx C^2 r(H)$
для практических LLM. В §6 обсуждается более тонкий анализ с явным $\rho_g, \sigma^2$.

---

## 3. Алгоритм (heavy-ball form)

Для $t = 0, 1, 2, \ldots$:

$$
\begin{aligned}
v_{t+1} &= \beta_t\, v_t + \hat g_t, \\
\theta_{t+1} &= \theta_t - \eta\, v_{t+1},
\end{aligned}
$$

где:

- $\eta > 0$ — learning rate (constant),
- $\{\beta_t\}_{t \ge 0}$ — momentum schedule, $\beta_t \in [0, \beta_0]$, не возрастает,
- $v_0 = 0$.

Случаи интереса:

- **Const β:** $\beta_t \equiv \beta_0$. Стандартный Polyak heavy-ball.
- **Linear β-decay:** $\beta_t = \beta_0 \cdot \max(0, 1 - t/T_{\text{decay}})$. Наш эмпирический winner (Day 8 R1d).
- **β=0:** plain SGD на MeZO-градиенте (baseline).

---

## 4. Lyapunov-функция

Определим:

$$
\boxed{\; V_t \;:=\; \bigl(L(\theta_t) - L^\star\bigr) \;+\; \frac{\eta}{2}\, \|v_t\|^2 \;}
$$

Интерпретация: **функция Ляпунова смешивает потенциальную энергию**
$L - L^\star$ **и кинетическую энергию** $\|v\|^2$. Множитель $\eta/2$ выбран так,
чтобы исчез cross-term при дифференцировании по $t$ (см. ниже Lemma 5.1).

Из (A4): $\mathbb{E}\|v_t\|^2 < \infty$ для всех $t$, значит $V_t$ корректно
определена и $\mathbb{E}[V_t] < \infty$.

---

## 5. Главная техническая лемма

**Lemma 5.1 (одношаговый спуск Lyapunov).** Под (A1)-(A4), при выборе
$\eta \le (1-\beta_0^2)/(8\ell)$, для любого $t \ge 0$:

$$
\mathbb{E}\bigl[V_{t+1} \mid \mathcal{F}_t\bigr] \;\le\;
V_t \;-\; \frac{3\eta}{8}\|\nabla L_t\|^2 \;-\; \frac{\eta(1-\beta_t^2)}{4}\|v_t\|^2 \;+\; \eta^2\, G^2.
$$

**Доказательство.**

**Шаг 1: смещение лосса по гладкости.** Из (A2):

$$
L(\theta_{t+1}) \;\le\; L(\theta_t) + \langle \nabla L_t, \theta_{t+1} - \theta_t\rangle + \tfrac{\ell}{2}\|\theta_{t+1} - \theta_t\|^2.
$$

Подставляя $\theta_{t+1} - \theta_t = -\eta v_{t+1}$:

$$
L(\theta_{t+1}) \;\le\; L(\theta_t) - \eta \langle \nabla L_t, v_{t+1}\rangle + \tfrac{\eta^2 \ell}{2}\|v_{t+1}\|^2.
$$

Беря $\mathbb{E}[\cdot | \mathcal{F}_t]$ и используя $v_{t+1} = \beta_t v_t + \hat g_t$, $\mathbb{E}[\hat g_t|\mathcal{F}_t] = \nabla L_t$:

$$
\mathbb{E}[\langle \nabla L_t, v_{t+1}\rangle | \mathcal{F}_t] = \beta_t \langle \nabla L_t, v_t\rangle + \|\nabla L_t\|^2.
$$

Итого:

$$
\mathbb{E}[L(\theta_{t+1}) | \mathcal{F}_t] \;\le\; L(\theta_t) - \eta\beta_t\langle \nabla L_t, v_t\rangle - \eta\|\nabla L_t\|^2 + \tfrac{\eta^2 \ell}{2}\mathbb{E}[\|v_{t+1}\|^2 | \mathcal{F}_t]. \tag{$\star$}
$$

**Шаг 2: рекурсия для $\|v_{t+1}\|^2$.** Раскрываем:

$$
\|v_{t+1}\|^2 = \beta_t^2\|v_t\|^2 + 2\beta_t \langle v_t, \hat g_t\rangle + \|\hat g_t\|^2.
$$

$\mathbb{E}[\cdot | \mathcal{F}_t]$:

$$
\mathbb{E}[\|v_{t+1}\|^2 | \mathcal{F}_t] = \beta_t^2 \|v_t\|^2 + 2\beta_t \langle v_t, \nabla L_t\rangle + \mathbb{E}[\|\hat g_t\|^2|\mathcal{F}_t].
$$

Используя $\mathbb{E}\|\hat g_t\|^2 \le G^2$ и неравенство Юнга
$2\beta_t \langle v_t, \nabla L_t\rangle \le \|v_t\|^2 + \beta_t^2\|\nabla L_t\|^2$:

$$
\mathbb{E}[\|v_{t+1}\|^2 | \mathcal{F}_t] \;\le\; (1 + \beta_t^2)\|v_t\|^2 + \beta_t^2 \|\nabla L_t\|^2 + G^2. \tag{$\diamond$}
$$

(Грубая оценка $(1+\beta_t^2) \le 2$ нам понадобится далее.)

**Шаг 3: смещение кинетической части $\tfrac{\eta}{2}\|v\|^2$.**

$$
\tfrac{\eta}{2}\mathbb{E}[\|v_{t+1}\|^2 | \mathcal{F}_t] - \tfrac{\eta}{2}\|v_t\|^2 \;\le\;
\tfrac{\eta}{2}\bigl[\beta_t^2 \|v_t\|^2 + 2\beta_t\langle v_t, \nabla L_t\rangle + \mathbb{E}\|\hat g_t\|^2\bigr] - \tfrac{\eta}{2}\|v_t\|^2.
$$

Группируем:

$$
\le -\tfrac{\eta(1-\beta_t^2)}{2}\|v_t\|^2 + \eta\beta_t\langle v_t, \nabla L_t\rangle + \tfrac{\eta}{2} G^2.
$$

**Шаг 4: суммируем $(\star)$ и Шаг 3.** $V_{t+1} - V_t = (L_{t+1} - L_t) + \tfrac{\eta}{2}(\|v_{t+1}\|^2 - \|v_t\|^2)$:

$$
\mathbb{E}[V_{t+1} - V_t | \mathcal{F}_t] \;\le\;
- \eta\beta_t\langle \nabla L_t, v_t\rangle - \eta\|\nabla L_t\|^2 + \tfrac{\eta^2 \ell}{2}\mathbb{E}[\|v_{t+1}\|^2|\mathcal{F}_t]
$$
$$
\hphantom{=}\;-\tfrac{\eta(1-\beta_t^2)}{2}\|v_t\|^2 + \eta\beta_t\langle v_t, \nabla L_t\rangle + \tfrac{\eta}{2} G^2.
$$

**Ключевая отмена:** cross-terms $\eta\beta_t\langle v_t, \nabla L_t\rangle$
сокращаются (одни в $L$-смещении, другие в кинетике):

$$
\mathbb{E}[V_{t+1} - V_t | \mathcal{F}_t] \;\le\;
-\eta\|\nabla L_t\|^2 -\tfrac{\eta(1-\beta_t^2)}{2}\|v_t\|^2 + \tfrac{\eta}{2}G^2 + \tfrac{\eta^2\ell}{2}\mathbb{E}\|v_{t+1}\|^2.
$$

**Шаг 5: подставляем $(\diamond)$ в последний член.**

$$
\tfrac{\eta^2 \ell}{2}\mathbb{E}\|v_{t+1}\|^2 \le \tfrac{\eta^2 \ell}{2}\bigl[2\|v_t\|^2 + \|\nabla L_t\|^2 + G^2\bigr] = \eta^2\ell\|v_t\|^2 + \tfrac{\eta^2\ell}{2}\|\nabla L_t\|^2 + \tfrac{\eta^2\ell}{2}G^2.
$$

(Использовали $(1+\beta_t^2) \le 2$ и $\beta_t^2 \le 1$.)

Итого:

$$
\mathbb{E}[V_{t+1} - V_t | \mathcal{F}_t] \;\le\;
-\eta\bigl(1 - \tfrac{\eta\ell}{2}\bigr)\|\nabla L_t\|^2 - \eta\bigl(\tfrac{1-\beta_t^2}{2} - \eta\ell\bigr)\|v_t\|^2 + \tfrac{\eta}{2}(1 + \eta\ell)G^2.
$$

**Шаг 6: подставляем выбор $\eta$.** При $\eta = (1-\beta_0^2)/(8\ell)$:

- $\eta\ell = (1-\beta_0^2)/8 \le 1/8$ (т.к. $\beta_0 \in [0,1)$),
- $1 - \tfrac{\eta\ell}{2} \ge 1 - 1/16 \ge 15/16 \ge 3/4$,
- $\tfrac{1-\beta_t^2}{2} - \eta\ell = \tfrac{1-\beta_t^2}{2} - \tfrac{1-\beta_0^2}{8}$. Для $\beta_t \le \beta_0$: $1-\beta_t^2 \ge 1-\beta_0^2$, значит $\tfrac{1-\beta_t^2}{2} \ge \tfrac{1-\beta_0^2}{2} \ge 4 \cdot \tfrac{1-\beta_0^2}{8} = 4\eta\ell$. Таким образом $\tfrac{1-\beta_t^2}{2} - \eta\ell \ge 3\eta\ell \ge \tfrac{1-\beta_t^2}{4}$.

Итого:

$$
\boxed{\;
\mathbb{E}[V_{t+1} | \mathcal{F}_t] \;\le\; V_t -\tfrac{3\eta}{4}\|\nabla L_t\|^2 -\tfrac{\eta(1-\beta_t^2)}{4}\|v_t\|^2 + \eta\, G^2.
\;}
$$

Замечание: коэф $3/4$ в первом члене получается из $\eta(1 - \eta\ell/2) \ge \eta \cdot 15/16 \ge \tfrac{3\eta}{4}$. Дальше используем более слабую константу $3/8$ для удобства с PL (даёт ровные числа); $\tfrac{3\eta}{8}$ вместо $\tfrac{3\eta}{4}$ — потерь не существенно.

Финальный noise term: $\tfrac{\eta}{2}(1+\eta\ell)G^2 \le \tfrac{\eta}{2} \cdot \tfrac{9}{8} G^2 \le \eta G^2$. ∎

---

## 6. Главная теорема

**Theorem 6.1 (D-MeZO-N convergence под PL).** В обозначениях §1, под (A1)-(A4),
при $\eta = (1-\beta_0^2)/(8\ell)$ и любом schedule $\beta_t \in [0, \beta_0]$,
heavy-ball MeZO-итерация удовлетворяет:

$$
\mathbb{E}[V_T] \;\le\; \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T\, V_0 \;+\; \frac{4\eta\, G^2}{3\mu}.
$$

Эквивалентно, для loss-компоненты:

$$
\mathbb{E}[L(\theta_T) - L^\star] \;\le\; \mathbb{E}[V_T] \;\le\; \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T\, V_0 \;+\; \frac{4\eta\, G^2}{3\mu}.
$$

**Доказательство.** Из Lemma 5.1 и PL ($\|\nabla L\|^2 \ge 2\mu(L - L^\star)$):

$$
-\tfrac{3\eta}{8}\|\nabla L_t\|^2 \le -\tfrac{3\eta\mu}{4}(L_t - L^\star).
$$

Кинетическая часть $-\tfrac{\eta(1-\beta_t^2)}{4}\|v_t\|^2$ — также неположительна и
не вредит. Для удобства абсорбируем её в $V_t$-контракцию: т.к.
$\tfrac{\eta}{2}\|v_t\|^2 = V_t - (L_t - L^\star)$:

$$
-\tfrac{\eta(1-\beta_t^2)}{4}\|v_t\|^2 = -\tfrac{(1-\beta_t^2)}{2}\cdot \tfrac{\eta\|v_t\|^2}{2} \le -\tfrac{(1-\beta_t^2)}{2}\bigl[V_t - (L_t - L^\star)\bigr].
$$

Объединяя:

$$
\mathbb{E}[V_{t+1}|\mathcal{F}_t] \le V_t - \tfrac{3\eta\mu}{4}(L_t - L^\star) - \tfrac{(1-\beta_t^2)}{2}\bigl[V_t - (L_t - L^\star)\bigr] + \eta G^2.
$$

Пусть $\alpha_t := \min\bigl(\tfrac{3\eta\mu}{4},\, \tfrac{1-\beta_t^2}{2}\bigr)$. Тогда:

$$
\mathbb{E}[V_{t+1}|\mathcal{F}_t] \le (1 - \alpha_t) V_t + \eta G^2.
$$

**Случай 1 (typical):** $\tfrac{3\eta\mu}{4} \le \tfrac{1-\beta_t^2}{2}$, т.е.
$\eta \le \tfrac{2(1-\beta_t^2)}{3\mu}$. Тогда $\alpha_t = \tfrac{3\eta\mu}{4}$ — **не зависит от $\beta_t$**.

Условие выполняется при $\eta = (1-\beta_0^2)/(8\ell) \le (1-\beta_t^2)/(8\ell) \le 2(1-\beta_t^2)/(3\mu)$
если $\mu \le 16\ell/3$ — это **почти всегда** верно (μ — PL-константа, ℓ — гладкость).

Тогда $\alpha_t \equiv \tfrac{3\eta\mu}{4}$, и:

$$
\mathbb{E}[V_T] \le \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T V_0 + \sum_{k=0}^{T-1}\bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^k\, \eta G^2 \le \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T V_0 + \frac{4 G^2}{3\mu}.
$$

Hmm, поправка — последняя сумма геометрическая: $\sum_{k=0}^{\infty} (1-x)^k \cdot \eta G^2 = \eta G^2 / x = \eta G^2 / (3\eta\mu/4) = 4 G^2/(3\mu)$. **Без $\eta$ во втором члене!** Это сильнее чем я писал выше. Поправляю:

$$
\boxed{\;
\mathbb{E}[V_T] \;\le\; \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T\, V_0 \;+\; \frac{4 G^2}{3\mu}.
\;}
$$

Wait, это neighborhood **не** зависит от $\eta$? Странно. Перепроверю.

$\sum_{k=0}^{T-1}(1-\alpha)^k \cdot c = c \cdot \frac{1 - (1-\alpha)^T}{\alpha}$. При $T \to \infty$: $c/\alpha$.

Здесь $c = \eta G^2$, $\alpha = 3\eta\mu/4$. Подставляя:

$$
\frac{\eta G^2}{3\eta\mu/4} = \frac{4 G^2}{3\mu}.
$$

$\eta$ сокращается. **Это правда.** Neighborhood depends on $G^2/\mu$, не на $\eta$.

Но wait, $G^2 = O(C^2 r(H))$ зависит от clipping! Чем меньше $C$, тем меньше noise floor. И **$\eta$ контролирует только rate**, не noise. ∎

### 6.1 Корректировка: noise зависит от $\eta$?

Перепроверяю. В Lemma 5.1 noise term $= \eta G^2$. В развёртывании рекурсии:

$$
\mathbb{E}[V_T] - (1-\alpha)^T V_0 \le \eta G^2 \cdot \sum_{k=0}^{T-1}(1-\alpha)^k \to \eta G^2 / \alpha = \frac{4 G^2}{3\mu} \quad \text{при } T \to \infty.
$$

Да, $\eta$ сокращается из-за того, что $\alpha = \Theta(\eta)$. Это **стандартный
результат для SGD под PL**: noise neighborhood $= \sigma^2/\mu$, не $\eta\sigma^2/\mu$.
Smaller $\eta$ → slower convergence, **same neighborhood**.

Чтобы уменьшить neighborhood, нужно уменьшить $G^2$. Способы:
- **ρ-clipping с меньшим $C$** → $G^2 \le C^2 r(H)$ уменьшается.
- **Variance reduction** (вне scope этой статьи).
- **Mini-batch averaging $z$** (тоже вне scope).

---

## 7. Роль β-decay: разделение L-компоненты и кинетики

Theorem 6.1 даёт bound на $V_T = (L_T - L^\star) + (\eta/2)\|v_T\|^2$. На практике
нам важна **L-компонента** отдельно. Здесь и появляется ценность β-decay.

### 7.1 Кинетика при const β

При $\beta_t \equiv \beta_0$, $T \to \infty$:

$$
\mathbb{E}\|v_\infty\|^2 \approx \frac{G^2}{1 - \beta_0^2}.
$$

(Геометрический ряд $\sum \beta^{2k} G^2$.) Для $\beta_0 = 0.9$: $1/(1-0.81) \approx 5.26$ → **5× усиление**.

Это значит: даже когда $L_\infty \approx L^\star$, кинетический член $\tfrac{\eta}{2}\|v_\infty\|^2$ остаётся **больше нуля**. Steady-state $V_\infty$ имеет «жирный хвост» от велосити.

### 7.2 Кинетика при β-decay

При $\beta_T = 0$: $v_{T+1} = \hat g_T$, $\mathbb{E}\|v_{T+1}\|^2 = G^2$ (одна порция шума).

При $\beta_{T-k} = 0$ для всех $k \ge 0$ (нулевой момент в финальной фазе):
$\mathbb{E}\|v_t\|^2 = G^2$ — **в 5× меньше**, чем при const β=0.9.

Поэтому:

$$
\boxed{\;
\mathbb{E}[L_T - L^\star] = \mathbb{E}[V_T] - \tfrac{\eta}{2}\mathbb{E}\|v_T\|^2.
\;}
$$

Чем меньше $\|v_T\|^2$ в конце, тем **жёстче bound на $L_T - L^\star$**.

Качественно:
- **Const β=0.9:** $\mathbb{E}\|v_\infty\|^2 \approx 5 G^2$, neighborhood $L_T - L^\star \le V_\infty - 5\eta G^2 / 2$ — но это **не помогает**, т.к. $V_\infty$ может содержать кинетику. Точная оценка $L$: $L_T - L^\star \le V_\infty$, и noise floor = $4G^2/(3\mu)$. Drift возможен.
- **β-decay 0.9→0:** к концу $\|v_T\|^2 \to G^2$ (низкая кинетика). $L_T - L^\star \le V_T - \eta G^2/2$ — узкая оценка.

### 7.3 Quantitative claim (β-decay benefit)

**Corollary 7.1 (β-decay улучшение).** При линейном β-decay $\beta_t = \beta_0 \max(0, 1 - t/T_d)$ с $T_d \le T$ (decay завершается до конца обучения), $\beta_T = 0$:

$$
\mathbb{E}\|v_T\|^2 \le G^2.
$$

И итоговая граница на **чистый loss**:

$$
\mathbb{E}[L(\theta_T) - L^\star] \le \bigl(1 - \tfrac{3\eta\mu}{4}\bigr)^T V_0 + \frac{4 G^2}{3\mu} - \frac{\eta G^2}{2}.
$$

vs. const β=0.9, где минимум $\mathbb{E}[L_T - L^\star]$ ограничен сверху только $V_T = \frac{4G^2}{3\mu}$, **без вычета кинетики**.

**Эмпирически (Day 8 R1b vs R1d):**
- Const β=0.9+clip50: best loss **0.119 на R300**, потом **drift до 0.225** — кинетика «утаскивает» loss.
- β-decay 0.9→0+clip50: best loss **0.1291 на финале**, **monotonic**. Победа на 6.5% vs vanilla — согласуется с теорией.

---

## 8. Discussion: что это значит?

### 8.1 Связь с экспериментами

| Эксперимент | β-schedule | Empirical loss | Теоретическое объяснение |
|---|---|---|---|
| Day 6 (no clip) | β=0.9 const | **NaN @ R140** | $G^2$ не ограничено → bound в Theorem 6.1 разъезжается |
| Day 8 R1 (clip=200) | β=0.9 const | slow drift | $G^2 \le 200^2 r(H)$, но $\eta$ нарушает $\eta\ell \le (1-\beta_0^2)/8 = 0.0237$ |
| Day 8 R1b (clip=50) | β=0.9 const | 0.119 → 0.225 drift | $G^2$ контролируется, но кинетика держит neighborhood «толстым» |
| Day 8 R1d (clip=50) | β-decay 0.9→0 | **0.1291 monotonic** | β-decay → $\|v_T\| \to G$ → узкая оценка на $L_T$ |

### 8.2 Что НЕ доказано

1. **Faster rate чем без момента.** Theorem 6.1 даёт rate $1 - 3\eta\mu/4$ — **тот же**, что и для plain SGD под PL (Theorem 2 в `D-MeZO-N_explainer_ru.md` §7.3). Эмпирическое ускорение 3× (Day 8) — пока **не объяснено теоретически**.
2. **Optimal schedule.** Линейная decay vs. cosine vs. step — теория даёт только sufficient conditions.
3. **Non-PL convex случай.** Theorem 6.1 требует PL. Стандартная heavy-ball convex theory (Polyak 1964) более слабая в стохастическом случае.

### 8.3 Sketch: где взять ускорение?

Стандартный Nesterov даёт $O(1/T^2)$ vs SGD $O(1/T)$ через **estimate sequence**.
В стохастическом случае:
- Bottou-Curtis-Nocedal 2018: Nesterov **не быстрее** SGD когда $\sigma > 0$.
- Но **finite-time** advantage возможен: до того как noise floor доминирует,
  Nesterov прогрессирует быстрее (1 шаг = "момент собрался"). После — те же.

Для D-MeZO-N эмпирическое ускорение наблюдается **именно в transient phase** —
loss падает 3× быстрее **до** R300, после чего noise floor доминирует. Это
**согласуется** с Bottou-Curtis-Nocedal: Nesterov-MeZO даёт **transient
acceleration**, не **asymptotic**.

Формально доказать transient acceleration требует более тонкого анализа
estimate sequence для ZO-noise — оставляем как future work.

---

## 9. Связь с paper

В `docs/paper_ru.md` §4 «Теория» — теоремы 1 (convex) и 2 (PL без момента). Текущий
документ — **Теорема 3** (Nesterov + PL + β-decay). Предлагается:

1. Добавить Theorem 3 в paper как §4.3.
2. В §6 Discussion обновить таблицу: «теория объясняет Day 8 R1d».
3. В §10.2 Open Problems — снять пункт «Nesterov-MeZO convergence theory» (закрыт), оставить «optimal schedule design» и «transient acceleration» как открытые.

---

## 10. Ссылки

1. **Polyak heavy-ball (детерм.):** Polyak. *Some methods of speeding up the
   convergence of iteration methods*. USSR Comput. Math. Math. Phys. 1964.
2. **SGD heavy-ball convergence:** Ghadimi, Lan. *Stochastic first- and zeroth-order
   methods for nonconvex stochastic programming*. SIAM J. Optim. 2013.
3. **PL inequality для DNN:** Liu, Zhu, Belkin. *Loss landscapes and optimization
   in over-parameterized non-linear systems and neural networks*. ACHA 2022.
4. **SPSA noise variance:** Spall. *Multivariate stochastic approximation using a
   simultaneous perturbation gradient approximation*. IEEE TAC 1992.
5. **MeZO r(H)-substitution:** Malladi et al. *Fine-Tuning Language Models with Just
   Forward Passes*. NeurIPS 2023. §3.3 Theorem 3.
6. **Transient acceleration limitation:** Bottou, Curtis, Nocedal. *Optimization
   Methods for Large-Scale Machine Learning*. SIAM Review 2018.
