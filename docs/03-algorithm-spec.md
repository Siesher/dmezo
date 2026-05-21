# 03. Алгоритм D-MeZO-N (спецификация)

## Обозначения

- $n$ — число клиентов
- $\theta_i \in \mathbb R^d$ — параметры локальной модели на клиенте $i$
- $\bar\theta = \frac{1}{n} \sum_i \theta_i$ — средняя по сети модель
- $\mathcal L_i(\theta) = \mathbb E_{\xi \sim \mathcal D_i}[\ell(\theta; \xi)]$ — локальная функция потерь
- $W \in \mathbb R^{n \times n}$ — doubly-stochastic симметричная mixing matrix
- $\rho = \|W - \frac{1}{n}\mathbf{1}\mathbf{1}^T\|_2$ — spectral gap
- $z_t \sim \mathcal N(0, I_d)$ — общее возмущение (один на раунд, генерируется из общего seed $s_t$)
- $\eta$ — learning rate, $\epsilon$ — magnitude возмущения, $\beta$ — momentum coefficient
- $\hat\rho_i^t = (\mathcal L_i(\theta_i^t + \epsilon z_t) - \mathcal L_i(\theta_i^t - \epsilon z_t)) / (2\epsilon)$ — projected gradient клиента $i$
- $v_i^t \in \mathbb R^d$ — velocity buffer клиента $i$

## Псевдокод D-MeZO-N (consensus_via_updates) — INDEPENDENT seeds per client

**Важно:** в отличие от FedKSeed (shared $z_t$ broadcast от центрального сервера), у нас **каждый клиент сэмплирует свой $s_i^t$ независимо**. Это критично для $1/n$ variance reduction в federated stochastic floor (Theorem 2 § 4.4 в paper и `docs/theory_rigorous.md` §2). Earlier draft этой spec ошибочно описывал shared seed — мы это исправили consistent с реальной имплементацией в `src/dmezo/federated/client.py::ClientState`.

```
INPUT: initial theta_0; lr eta; eps; momentum beta;
       mixing matrix W; rounds T; local data shards {D_i}
INIT:  for each client i: theta_i = theta_0, v_i = 0,
       rng_i = independent np.random.Generator(seed=base + i)

for t = 0 to T-1:
    # 1. Each client samples its OWN seed and computes projected gradient.
    for each client i in parallel:
        s_i = rng_i.integers(0, 2**31)        // INDEPENDENT per client
        z_i = generate(s_i)                    // unique direction per client
        sample local batch xi_i ~ D_i
        L_plus  = L_i(theta_i + eps * z_i; xi_i)
        L_minus = L_i(theta_i - eps * z_i; xi_i)
        rho_i = (L_plus - L_minus) / (2 * eps)
        rho_i_tilde = clip(rho_i, ±C)          // optional, для D-MeZO-N

    # 2. Decentralized consensus via update sharing.
    for each client i in parallel:
        # Receive (rho_j_tilde, s_j) pairs from all neighbors j with W[i, j] != 0.
        # Locally REGENERATE z_j from s_j (each client got its own seed).
        accum = 0
        for each j with W[i, j] != 0:
            z_j = generate(s_j)                // regenerated locally
            accum += W[i, j] * rho_j_tilde * z_j
        # Heavy-ball Nesterov:
        v_i = beta * v_i + accum + wd * theta_i
        theta_i = theta_i - eta * v_i
```

## Почему independent, а не shared

С shared $z_t$ (как в FedKSeed): $\bar g = \left(\frac{1}{n}\sum_i \hat\rho_i\right) z$ — variance reduction $1/n$ только по data noise; direction noise (вариация по $z$) **не усредняется**.

С independent $z_i$ (наш случай): $\bar g = \frac{1}{n}\sum_i \hat\rho_i z_i$, где каждое $\hat\rho_i z_i$ — независимая unbiased оценка $\nabla L$. По CLT: variance ÷ n **по обоим источникам шума** (data + direction).

Это и есть алгоритмический differentiator D-MeZO-N vs FedKSeed (см. `docs/fedkseed_comparison.md` для подробного теоретического сравнения).

## Что коммуницируется

За один раунд между парой соседей $(i, j)$ передаётся: **один float ($\rho_j$) + один int (seed $s_j$)** = ~12–16 байт (зависит от точности).

**Общий трафик** на раунд: $|E| \cdot 16$ байт для графа с $|E|$ рёбрами. Для ring topology с $n = 4$: 4 ребра × 2 направления × 16 байт = **128 байт на раунд**.

**Compression vs FedAvg:** для модели на 4B params в bf16 (8 ГБ/раунд/клиент) → ~$5 \times 10^7$× для ring(4), $10^9$× при усреднении по архитектуре с $d \to \infty$.

## Корреляции между клиентами — нет на уровне sampling

$z_i$ и $z_j$ — independent random vectors (разные seeds, разные numpy generators). После consensus mixing $\theta_i^{t+1} = \sum_j W_{ij} \theta_j^{t+1/2}$ параметры клиентов **сближаются** в одну точку, но это convergence в траекториях, не correlation в источнике шума. Lemma 4 (Koloskova-style consensus error) ограничивает per-round дрейф, $z_i$ остаются независимыми при каждом новом sampling.

## Варианты алгоритма

**A. Heavy-ball (по умолчанию):** velocity обновляется на основе градиента в текущей точке $\theta_i^t$.

**B. True Nesterov look-ahead:** проекторный градиент оценивается в точке $\theta_i^t + \beta v_i^t$. Требует дополнительной forward-pass на look-ahead перед основной MeZO step. Дороже, но теоретически лучше.

**C. Без Nesterov (baseline):** $\beta = 0$. Эквивалент plain D-MeZO.

**D. Параметр-обмен вместо update-обмена (для сравнения):** клиенты делают local step, потом обмениваются весами $\theta_i^{new} = \sum_j W[i, j] \theta_j$. Бандвидт растёт до $d$ float-ов на ребро. Используется как baseline.

## Инварианты, которые должна соблюдать любая имплементация

1. **Seed-определяемое возмущение.** $z_t$ полностью определяется $s_t$. Никогда не хранить $z_t$ как tensor.
2. **In-place perturbation.** Параметры обновляются через `param.data.add_(...)`. Никаких новых тензоров под $\theta$.
3. **Forward-only.** `torch.inference_mode()` обёртывает оба forward-pass MeZO step.
4. **Согласованность $z$ между клиентами.** Все клиенты, использующие $s_t$, должны итерировать параметры в одном и том же порядке (`model.named_parameters()` гарантирует это для одной и той же архитектуры).
5. **Velocity ≠ optimizer state в Adam-смысле.** Velocity — это $d$-мерный буфер; для full-MeZO он удвоит память. Для LoRA-MeZO это дёшево.

## Шаблон convergence statement

Под предположениями:

- **(A1)** Каждая $\mathcal L_i$ — $L$-smooth.
- **(A2)** $\mathcal L = \frac{1}{n}\sum_i \mathcal L_i$ удовлетворяет $\mu$-PL inequality.
- **(A3)** Bounded gradient diversity: $\frac{1}{n}\sum_i \|\nabla \mathcal L_i(\theta) - \nabla \mathcal L(\theta)\|^2 \le \sigma^2$.
- **(A4)** Bounded stochastic noise: $\mathbb E_\xi \|\nabla \ell(\theta;\xi) - \nabla \mathcal L_i(\theta)\|^2 \le \sigma_{\text{stoch}}^2$.
- **(A5)** Эффективный ранг гессиана: $r(H) = \text{tr}(H) / \|H\|_{op} \ll d$ (Malladi-style).

При выборе $\eta = O(\sqrt{\mu / (L T r(H))})$, $\beta = 1 - O(\eta)$:

$$\mathbb E\left\|\nabla \mathcal L(\bar\theta_T)\right\|^2 \le \tilde O\!\left(\sqrt{\frac{L \, r(H) \, \Delta}{nT}}\right) + \tilde O\!\left(\frac{\rho^2 \sigma^2}{(1-\beta)^2 T}\right) + O(\epsilon^2 L^2 r(H)),$$

где $\Delta = \mathcal L(\bar\theta_0) - \mathcal L^*$.

Главные особенности по сравнению с Koloskova 2020:

- $d \to r(H)$ во variance term (Malladi-style для ZO в low-rank landscape).
- $1/n$ — linear speedup от distributed.
- $\rho^2/(1-\beta)^2$ — penalty за плохую топологию, ослабляется моментом.

**Формальное доказательство — в работе.** См. `04-theory-template.md`.

## Hyperparameter ranges (стартовые точки)

| Параметр | Range | Default | Notes |
|---|---|---|---|
| `lr` | $10^{-7}$ – $10^{-5}$ | $10^{-6}$ | Меньше чем для Adam в $\sim 1000$ раз |
| `eps` | $10^{-4}$ – $10^{-2}$ | $10^{-3}$ | Princeton default |
| `beta` | 0.5 – 0.99 | 0.9 | $\beta=0$ — disable Nesterov |
| `weight_decay` | 0 – 0.01 | 0 | Чувствителен в MeZO, начать с 0 |
| `local_steps` | 1 – 5 | 1 | $K=1$ — самый чистый случай для теории |

## Эксперимент-чек-лист

Минимум для main paper:

- [ ] Sanity: централизованный MeZO на Qwen3-4B (baseline ceiling).
- [ ] D-MeZO без Nesterov на 4 клиентах, ring (baseline floor — показывает работоспособность decentralized).
- [ ] D-MeZO-N (с Nesterov) на 4 клиентах, ring — главный результат.
- [ ] Topology ablation: ring, random_regular, complete.
- [ ] Heterogeneity ablation: IID, Dirichlet $\alpha=0.5$, label-skew.
- [ ] Scaling: 7B/9B модель (Qwen3-8B или Qwen3.5-4B).
- [ ] Communication-cost график: accuracy vs. bytes.
- [ ] Сравнение с FedKSeed (центральный сервер, без момента) — на тех же данных.
