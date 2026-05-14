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

## Псевдокод D-MeZO-N (consensus_via_updates)

```
INPUT: initial theta_0; lr eta; eps; momentum beta;
       mixing matrix W; rounds T; local data shards {D_i}
INIT:  for each client i: theta_i = theta_0, v_i = 0

for t = 0 to T-1:
    # 1. Sample SHARED seed (e.g., from a synchronized counter).
    s_t = SharedPRNG.next()  // same on all clients

    # 2. Each client computes projected gradient.
    for each client i in parallel:
        sample local batch xi_i ~ D_i
        z_t = generate(s_t)  // identical across clients (regenerated, not stored)
        L_plus  = L_i(theta_i + eps * z_t; xi_i)
        L_minus = L_i(theta_i - eps * z_t; xi_i)
        rho_i = (L_plus - L_minus) / (2 * eps)

    # 3. Decentralized consensus via update sharing.
    for each client i in parallel:
        # Receive (rho_j) from all neighbors j with W[i, j] != 0.
        # Regenerate z_t locally from s_t.
        weighted_rho = sum_j W[i, j] * rho_j
        # Heavy-ball Nesterov:
        v_i = beta * v_i + weighted_rho * z_t + wd * theta_i
        theta_i = theta_i - eta * v_i
```

## Что коммуницируется

За один раунд между парой соседей $(i, j)$ передаётся: **один скаляр** ($\rho_j$). Общий seed $s_t$ синхронизируется через counter и хеш (т.е. фактически не передаётся, а вычисляется детерминированно из round number).

**Общий трафик** на раунд: $|E| \cdot 8$ байт (для FP64 скаляра) для графа с $|E|$ рёбрами. Для ring topology с $n = 4$: 4 ребра × 2 направления × 8 байт = **64 байта на раунд**.

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
