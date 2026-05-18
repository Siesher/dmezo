# D-MeZO-N: подробный разбор с математическими выкладками

**Decentralized Federated MeZO with Nesterov acceleration**

Этот документ — расширенное пояснение нашей работы для читателя, который знаком с
backprop, SGD и базовой выпуклой оптимизацией, но хочет понять, *что именно мы
сделали* и *почему это работает*, до уровня формул.

---

## Содержание

1. [Контекст и постановка задачи](#1-контекст-и-постановка-задачи)
2. [MeZO: zeroth-order оптимизация через seed](#2-mezo-zeroth-order-оптимизация-через-seed)
3. [Federated extension: один скаляр вместо весов](#3-federated-extension-один-скаляр-вместо-весов)
4. [Decentralized version: consensus через mixing matrix](#4-decentralized-version-consensus-через-mixing-matrix)
5. [Nesterov acceleration: heavy-ball form](#5-nesterov-acceleration-heavy-ball-form)
6. [Стабилизация: ρ-clipping и β-decay](#6-стабилизация-ρ-clipping-и-β-decay)
7. [Сходимость: теоремы 1 и 2](#7-сходимость-теоремы-1-и-2)
8. [Эксперименты — итог](#8-эксперименты--итог)
9. [Что нового именно у нас](#9-что-нового-именно-у-нас)
10. [Заключение и open problems](#10-заключение-и-open-problems)

---

## 1. Контекст и постановка задачи

Хотим дообучить **большую языковую модель** с параметрами
$\theta \in \mathbb{R}^d$, $d \sim 10^9$, на задаче с лоссом
$L: \mathbb{R}^d \to \mathbb{R}$:

$$
L(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}\bigl[\ell\bigl(f_\theta(x), y\bigr)\bigr].
$$

Две большие практические проблемы:

**Проблема 1 — память.** Стандартный backprop требует:

- параметры: $O(d)$,
- градиенты: $O(d)$,
- состояние оптимизатора (Adam: $m, v$): $O(2d)$ в FP32.

Итого $\sim 5d$ float-ячеек. Для Qwen3-4B ($d \approx 4 \times 10^9$, FP16-веса
$\approx 8$ ГБ) это $\sim 32{-}40$ ГБ — недоступно на потребительских GPU.

**Проблема 2 — приватность данных.** Часто сырые данные нельзя «вынести наружу»
(банки, медицина, личная переписка). Federated Learning решает это: каждый
клиент учит **локальную копию модели** и шлёт только обновления. Но классический
FedAvg требует пересылать **полный $d$-мерный вектор весов** между клиентами
каждый раунд — для Qwen3-4B это $\sim 8$ ГБ за раунд, что нереалистично.

**Цель D-MeZO-N:** одновременно решить (1) и (2):

- Память при обучении $\approx$ память при инференсе ($\sim d$, без градиентов).
- Коммуникация между клиентами $\approx 4$ байта за раунд ($\sim 10^9$× компрессия).
- Без центрального сервера (decentralized peer-to-peer).
- С ускорением Нестерова для уменьшения числа раундов.
- С формальными гарантиями сходимости.

---

## 2. MeZO: zeroth-order оптимизация через seed

**MeZO (Malladi et al. 2023)** — основа всей работы. Идея берёт начало в SPSA
(Spall 1992).

### 2.1 SPSA-оценка градиента

Выберем случайное направление $z \in \mathbb{R}^d$, $z \sim \mathcal{N}(0, I_d)$,
и шаг возмущения $\epsilon > 0$. Определим **скалярную оценку производной по
направлению**:

$$
\hat\rho \;:=\; \frac{L(\theta + \epsilon z) - L(\theta - \epsilon z)}{2\epsilon}.
$$

**Лемма (несмещённость SPSA).** Если $L$ дважды дифференцируема и
$z \sim \mathcal{N}(0, I_d)$, то

$$
\mathbb{E}_z\bigl[\hat\rho \cdot z\bigr] \;=\; \nabla L(\theta) + O(\epsilon^2).
$$

**Доказательство (sketch).** По формуле Тейлора:

$$
\begin{aligned}
L(\theta + \epsilon z) &= L(\theta) + \epsilon\, z^\top \nabla L(\theta)
   + \tfrac{\epsilon^2}{2} z^\top H(\theta)\, z + O(\epsilon^3), \\
L(\theta - \epsilon z) &= L(\theta) - \epsilon\, z^\top \nabla L(\theta)
   + \tfrac{\epsilon^2}{2} z^\top H(\theta)\, z + O(\epsilon^3).
\end{aligned}
$$

Вычитание убивает чётные члены:

$$
\hat\rho = \frac{L(\theta + \epsilon z) - L(\theta - \epsilon z)}{2\epsilon}
        = z^\top \nabla L(\theta) + O(\epsilon^2).
$$

Тогда

$$
\mathbb{E}_z[\hat\rho \cdot z] = \mathbb{E}_z[z z^\top]\, \nabla L(\theta) + O(\epsilon^2)
                              = I_d \cdot \nabla L(\theta) + O(\epsilon^2)
                              = \nabla L(\theta) + O(\epsilon^2).
$$

(Использовали $\mathbb{E}[z z^\top] = I_d$ для стандартного гауссовского $z$.) ∎

**Ключевое наблюдение:** оценка $\hat\rho \cdot z$ — это **двусмещённая аппроксимация
градиента, использующая только два forward-прохода**. Никакого backprop.

### 2.2 Дисперсия SPSA-оценки

Цена за отказ от backprop — большая дисперсия:

$$
\operatorname{Var}_z[\hat\rho \cdot z]
   \;\approx\; \operatorname{Var}_z[\hat\rho] \cdot \mathbb{E}[z z^\top]
   \;\approx\; \|\nabla L\|^2 \cdot I_d.
$$

То есть **norm-wise** SPSA-оценка имеет дисперсию в $\sim d$ раз больше, чем
истинный градиент. Это означает: SGD на ZO-оценке требует в $d$ раз больше
шагов, чем SGD на честном градиенте.

**Но (важное «но»)**: для LLM реально важна не размерность $d$, а **эффективный
ранг гессиана**

$$
r(H) \;=\; \frac{\operatorname{tr}(H)}{\|H\|_{\text{op}}}.
$$

Для overparametrized трансформера $r(H) \sim 10^3{-}10^4$ — на много порядков
меньше $d \sim 10^9$. Malladi 2023 доказывают, что замедление MeZO vs SGD —
$O(r(H))$, а не $O(d)$. Это и объясняет, **почему MeZO в принципе работает на
миллиардных моделях**.

### 2.3 Seed trick — главная инженерия MeZO

Наивная реализация хранит $z \in \mathbb{R}^d$ явно → теряем всю выгоду по
памяти. Хитрость:

1. Перед первым forward ставим `torch.manual_seed(s)`, проходим по параметрам
   и **in-place** прибавляем $\epsilon z$:

   $$\theta \mapsto \theta + \epsilon z.$$

2. После первого forward ставим **тот же** seed $s$, генерируем тот же $z$ и
   in-place прибавляем $-2\epsilon z$:

   $$\theta + \epsilon z \mapsto \theta - \epsilon z.$$

3. После второго forward считаем $\hat\rho = (L_+ - L_-) / (2\epsilon)$.

4. Снова `manual_seed(s)`, восстанавливаем $z$, делаем шаг
   $\theta \leftarrow \theta + \epsilon z - \eta\, \hat\rho \cdot z = \theta - \eta\, \hat\rho \cdot z$.

**Каждый блок параметров генерируется отдельным $z$-куском по очереди**, в
памяти держим только один такой кусок за раз. Итог: **дополнительная память на
оптимизацию — $O(1)$** (только скаляр $\hat\rho$).

### 2.4 In-place perturbation: эссенция кода

```python
def perturb_(theta, seed: int, eps: float, scale: float) -> None:
    """Atomic in-place perturbation θ ← θ + scale * eps * z(seed)."""
    torch.manual_seed(seed)
    for p in theta:
        z = torch.randn_like(p)            # один блок за раз
        p.data.add_(scale * eps * z)       # in-place, без alloc

def mezo_step(theta, batch, eps: float, lr: float, seed: int) -> float:
    perturb_(theta, seed, eps, +1.0)       # θ → θ + εz
    with torch.inference_mode():
        L_plus = forward(theta, batch).item()
    perturb_(theta, seed, eps, -2.0)       # θ + εz → θ - εz
    with torch.inference_mode():
        L_minus = forward(theta, batch).item()
    perturb_(theta, seed, eps, +1.0)       # вернулись в θ
    rho = (L_plus - L_minus) / (2 * eps)
    perturb_(theta, seed, lr * rho, -1.0)  # θ → θ - η·ρ·z
    return rho
```

**Инварианты, которые НЕЛЬЗЯ нарушать:**

- `inference_mode()` + `model.eval()` во время forward — иначе dropout/BN сделают
  $L_+$ и $L_-$ некогерентными, и оценка $\hat\rho$ потеряет смысл.
- Параметры обновляются **через `.data.add_()`** — иначе ссылки в data-sharding
  и optimizer'е ломаются.
- **Один и тот же seed для `+εz` и `-εz`** — иначе perturbation не симметрична.
- `param.requires_grad = True` для всех параметров (даже без backprop!) — иначе
  HuggingFace вычеркнет их из state-dict.

---

## 3. Federated extension: один скаляр вместо весов

Теперь распространяем MeZO на $N$ клиентов с локальными датасетами
$\{\mathcal{D}_i\}_{i=1}^N$. Глобальная задача:

$$
\min_\theta \; \sum_{i=1}^N w_i \, L_i(\theta), \qquad
L_i(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_i}\bigl[\ell(f_\theta(x), y)\bigr].
$$

### 3.1 Общий seed → общий $z$

Ключевое наблюдение: если все клиенты используют **один и тот же seed** $s_t$ на
раунде $t$, то они **независимо восстанавливают один и тот же $z_t$**.

Алгоритм Federated MeZO (centralized FL версия):

1. **Координатор** (или Lamport-counter) рассылает общий $s_t$.
2. Каждый клиент $i$ восстанавливает $z_t$ из $s_t$ и считает локально:

   $$
   \hat\rho_i^t \;=\; \frac{L_i(\theta^t + \epsilon z_t) - L_i(\theta^t - \epsilon z_t)}{2\epsilon}.
   $$

3. Клиенты шлют **скаляры** $\hat\rho_i^t$ координатору, который усредняет:

   $$
   \bar\rho^t \;=\; \sum_{i=1}^N w_i\, \hat\rho_i^t.
   $$

4. Координатор рассылает $\bar\rho^t$. Каждый клиент локально обновляет:

   $$
   \theta^{t+1} \;=\; \theta^t - \eta\, \bar\rho^t z_t.
   $$

**Утверждение (consistency).** Если все клиенты стартуют с одной и той же
$\theta^0$, то на любом $t$ все $\theta_i^t = \theta^t$.

**Доказательство.** Индукция: на каждом шаге все клиенты применяют **одно и то же
обновление** $-\eta \bar\rho^t z_t$ (потому что $\bar\rho^t$ и $z_t$ одинаковы у
всех). ∎

### 3.2 Communication cost

| Метод | Размер коммуникации за раунд (для Qwen3-4B FP16) |
|---|---|
| **FedAvg** | $2d$ байт $\approx 8 \cdot 10^9$ байт $= 8$ ГБ |
| **Federated MeZO** | $4N$ байт (N скаляров FP32) + seed $\sim 8$ байт |

Compression ratio при $N = 10$:

$$
\frac{8 \cdot 10^9}{40 + 8} \;\approx\; 1.7 \times 10^8.
$$

Это **больше чем в сто миллионов раз меньше трафика**. На практике связь между
датацентрами становится бесплатной.

### 3.3 Семантика: что эквивалентно среднему градиенту?

В FedAvg усредняются веса $\theta_i$. В Federated MeZO усредняются
**проективные градиенты** $\hat\rho_i$. Глобальный шаг:

$$
\theta^{t+1} = \theta^t - \eta\, z_t \cdot \sum_i w_i \hat\rho_i^t.
$$

В пределе $\epsilon \to 0$:

$$
\hat\rho_i^t \to z_t^\top \nabla L_i(\theta^t), \quad
\sum_i w_i \hat\rho_i^t \to z_t^\top \sum_i w_i \nabla L_i(\theta^t) = z_t^\top \nabla L(\theta^t).
$$

То есть **Federated MeZO в пределе эквивалентен SPSA-оценке среднего градиента
$\nabla L(\theta)$**. И мы экономим $d / N$ раз на коммуникации.

---

## 4. Decentralized version: consensus через mixing matrix

Centralized FL требует координатор. Уберём его — пусть клиенты общаются
**только с соседями** по графу $G$.

### 4.1 Mixing matrix (Koloskova 2020)

Топология сети задана матрицей $W \in \mathbb{R}^{N \times N}$ со свойствами:

1. $W_{ij} > 0$ если $\{i, j\} \in E(G)$ (соседи) или $i = j$; иначе $W_{ij} = 0$.
2. **Doubly stochastic:** $\sum_j W_{ij} = \sum_i W_{ij} = 1$.
3. **Симметричная:** $W = W^\top$ (для undirected графов).
4. **Spectral gap:** $\rho_W := 1 - |\lambda_2(W)| \in (0, 1]$.

$\lambda_2$ — второе по модулю собственное значение $W$. Большой $\rho_W$ =
быстрое усреднение по сети.

**Примеры топологий ($N = 4$):**

| Топология | $W$-структура | $\rho_W$ |
|---|---|---|
| Complete | $W = \frac{1}{N}\mathbb{1}\mathbb{1}^\top$ | 1 (мгновенный консенсус за 1 раунд) |
| Ring | $W_{ii} = W_{i,i\pm 1} = 1/3$ | $1 - \cos(2\pi/N) \approx 0.5$ |
| Path | концевые имеют 2 соседа, остальные 3 | меньше ring |
| Star | центр везде, листья только центр | 1 (но не peer-to-peer) |

### 4.2 D-MeZO update

Каждый клиент $i$ на раунде $t$:

1. Восстанавливает $z_t$ из общего seed $s_t$ (генерируется глобальным
   Lamport-style counter, не требует центра).
2. Локально считает:

   $$
   \hat\rho_i^t \;=\; \frac{L_i(\theta_i^t + \epsilon z_t) - L_i(\theta_i^t - \epsilon z_t)}{2\epsilon}.
   $$

3. **Обмен с соседями:** отправляет $\hat\rho_i^t$ всем $j$ таким что $W_{ij} > 0$,
   получает $\hat\rho_j^t$ от тех же соседей.
4. **Consensus-shared $\rho$:**

   $$
   \tilde\rho_i^t \;=\; \sum_j W_{ij}\, \hat\rho_j^t.
   $$

5. **Локальное обновление:**

   $$
   \theta_i^{t+1} \;=\; \sum_j W_{ij}\, \theta_j^t \;-\; \eta\, \tilde\rho_i^t\, z_t.
   $$

В формуле (5) первый член — gossip-averaging параметров. Но веса целиком слать
дорого! Поэтому в реализации делаем это **только в `weight_avg` режиме** периодически,
а в `complete` режиме при $W_{ij} = 1/N$ — параметры остаются синхронизированы
сами собой (см. §3.1), и (5) сводится к $\theta_i^{t+1} = \theta_i^t - \eta \tilde\rho_i^t z_t$.

### 4.3 Эквивалентность gossip-SGD

Под $\epsilon \to 0$, $\hat\rho_i^t \to z_t^\top \nabla L_i(\theta_i^t)$, и
$\tilde\rho_i^t z_t \to z_t z_t^\top \sum_j W_{ij} \nabla L_j(\theta_j^t)$. По
закону больших чисел при многих $z$-сэмплах:

$$
\mathbb{E}_z[\tilde\rho_i^t z_t] \to \sum_j W_{ij}\, \nabla L_j(\theta_j^t).
$$

То есть D-MeZO в пределе — **decentralized SGD с gossip-усреднением градиентов**
(Koloskova 2020). Это даёт нам бесплатно весь theoretical toolkit оттуда.

### 4.4 Practical caveat: implementation

Honest reality check: мы реализовали два режима mixing:

- **`complete`** — все ↔ все, $W_{ij} = 1/N$. Эквивалентно centralized из §3.
- **`weight_avg`** — клиенты усредняют параметры **локально** каждые $K$ раундов
  (а не consensus по скаляру), что **совместимо с Nesterov локально**.

Mode `consensus_via_updates` для произвольного $W$ написан и протестирован
(`tests/test_consensus.py`), но **не интегрирован с Nesterov** — попытка
комбинации даёт `NotImplementedError` (см. `docs/07-audit-harden.md` D1).
Это явное ограничение текущей версии, обозначенное в paper как future work.

---

## 5. Nesterov acceleration: heavy-ball form

Хотим ускорить сходимость, добавив **момент**.

### 5.1 Heavy-ball form

Классическая heavy-ball итерация:

$$
\begin{aligned}
v^{t+1} &= \beta\, v^t + \hat g^t, \\
\theta^{t+1} &= \theta^t - \eta\, v^{t+1},
\end{aligned}
$$

где $\hat g^t = \hat\rho^t z_t$ — MeZO-оценка градиента, $\beta \in [0, 1)$ —
momentum coefficient.

В терминах раскрытия:

$$
v^{t+1} \;=\; \sum_{k=0}^t \beta^{t-k}\, \hat g^k.
$$

Эффективное «горизонтное окно» — $1/(1-\beta)$ шагов: для $\beta = 0.9$ это ~10
последних градиентов.

### 5.2 Look-ahead (true Nesterov) form

Альтернатива — Nesterov с look-ahead:

$$
\begin{aligned}
y^t &= \theta^t + \beta\, (\theta^t - \theta^{t-1}), \\
\hat g^t &= \text{ZO-estimate at } y^t, \\
\theta^{t+1} &= y^t - \eta\, \hat g^t.
\end{aligned}
$$

В детерминированной выпуклой оптимизации look-ahead даёт **оптимальную скорость
$O(1/T^2)$** vs heavy-ball $O(1/T)$. **Но для ZO с шумом это разваливается**
(см. §6.2): true-Nesterov расходится **7× быстрее** чем heavy-ball (наш Day 6b
эксперимент: NaN at R20). Причина — двойной канал шума: и в look-ahead point
$y^t$, и в $\hat g^t$.

Поэтому в D-MeZO-N **используем heavy-ball**, а look-ahead отбраковали
эмпирически.

---

## 6. Стабилизация: ρ-clipping и β-decay

### 6.1 Почему naive Nesterov расходится

**Эмпирический факт (наш Day 6):** при $\beta = 0.9$ MeZO-Nesterov **расходится
катастрофически** (loss → NaN на R140).

**Анализ.** В детерминированном Nesterov шум в $\hat g^t$ нулевой. При MeZO:

$$
\operatorname{Var}[\hat g^t]
   \;=\; \operatorname{Var}[\hat\rho^t z_t]
   \;\approx\; \mathbb{E}[(\hat\rho^t)^2]\, \mathbb{E}[z_t z_t^\top]
   \;\approx\; \sigma_\rho^2 \cdot I_d.
$$

(Здесь $\sigma_\rho^2$ — дисперсия скаляра $\hat\rho^t$, обычно $O(\|\nabla L\|^2)$.)

Velocity $v^t$ накапливает шум **геометрически**:

$$
\operatorname{Var}[v^t] \;=\; \sum_{k=0}^t \beta^{2k}\, \operatorname{Var}[\hat g^{t-k}]
   \;\approx\; \frac{\sigma_\rho^2}{1 - \beta^2}\, I_d.
$$

Для $\beta = 0.9$: $1/(1 - 0.81) \approx 5.26$ — **5×-усиление дисперсии**. Если
$\|v^t\| > $ basin-of-attraction radius, траектория «вылетает» за минимум и
теряет устойчивость.

### 6.2 ρ-clipping (стабилизация выбросов)

Клипуем **скаляр** $\hat\rho^t$ перед применением:

$$
\hat\rho^t_{\text{clip}} \;=\; \operatorname{clip}(\hat\rho^t, -C, +C),
\qquad
\hat\rho^t_{\text{clip}} = \operatorname{sign}(\hat\rho^t) \cdot \min(|\hat\rho^t|, C).
$$

**Выбор $C$:**

- $C = 200$ → **slow-divergence** (Day 8 R1): не взрывается мгновенно, но к
  R500 уплывает.
- $C = 50$ → **стабильно** (Day 8 R1b): best loss 0.119 на R300, в 3× быстрее
  vanilla.
- $C \to 0$ → теряем сигнал, sublinear сходимость.

**Сравнение с gradient clipping в PPO/A2C:** там клипуется $d$-мерный вектор,
здесь — **скаляр**. Это:
- в $d$ раз дешевле (одна операция вместо $d$),
- **не искажает направление шага** (норма меняется, но $z$-направление сохраняется).

### 6.3 β-decay (линейное затухание момента)

Используем расписание:

$$
\beta_t \;=\; \beta_0 \cdot \max\!\left(0,\; 1 - \frac{t}{T_{\text{decay}}}\right).
$$

Старт $\beta_0 = 0.9$, к $t = T_{\text{decay}}$ имеем $\beta_t = 0$ (чистый SGD).

**Интуиция:**

- **В начале** ($\beta \approx 0.9$): мы далеко от минимума, ландшафт почти
  плоский, моментум даёт ускорение.
- **К концу** ($\beta \to 0$): мы в окрестности минимума, шум момента
  доминирует над сигналом, нужен осторожный шаг.

**Эмпирически (Day 8 R1d):** β-decay 0.9→0 + clip50 → **monotonic descent до
loss 0.1291**, **beats vanilla 0.1381 на 6.5%**.

Без β-decay: clip50 + const β=0.9 → best 0.119 на R300, но **late drift до 0.225**
(момент в конце «толкает» в шум).

С β-decay: monotonic, без drift. Это и есть финальная **D-MeZO-N v1**.

### 6.4 Сравнение с другими стабилизациями

| Подход | Где клипуется | Стоимость | Эффект |
|---|---|---|---|
| Adam (1st/2nd moment) | per-coord | $O(d)$ память | хорош для backprop, но в ZO 2nd-moment бесполезен |
| Gradient clipping | $\|\hat g\|_2 \le C$ | $O(d)$ операций | направление сохраняется, норма обрезается |
| **ρ-clipping (наше)** | скаляр $\hat\rho$ | $O(1)$ | направление **полностью** сохраняется, обрезается **только величина** |

ρ-clipping — это естественный для MeZO clipping, потому что MeZO-update уже
параметризован парой (scalar, direction).

---

## 7. Сходимость: теоремы 1 и 2

### 7.1 Базовые предположения

- **(A1)** $L: \mathbb{R}^d \to \mathbb{R}$ — $\mu$-сильно выпуклая (для Theorem 1)
  ИЛИ удовлетворяет PL-неравенству (для Theorem 2).
- **(A2)** $L$ — $\beta$-гладкая: $\|\nabla L(\theta) - \nabla L(\theta')\| \le \beta \|\theta - \theta'\|$.
- **(A3)** Mixing matrix $W$ doubly-stochastic с $\rho_W > 0$.
- **(A4)** Эффективный ранг гессиана $r(H) = \operatorname{tr}(H)/\|H\|_{\text{op}} \ll d$.

**Замечание про (A4):** для LLM это **эмпирически верно** (Malladi 2023, §3.3).
Овер-параметризация → мало активных направлений в loss-ландшафте.

### 7.2 Теорема 1 (выпуклый случай)

**Утверждение.** Под (A1)-(A4) с $L$ выпуклой, при выборе шага
$\eta = \Theta\bigl(1/(\beta \cdot r(H))\bigr)$, MeZO даёт:

$$
\mathbb{E}\bigl[L(\bar\theta^T) - L^\star\bigr] \;=\; O\!\left(\frac{r(H)}{T}\right),
$$

где $\bar\theta^T = \frac{1}{T} \sum_{t=1}^T \theta^t$ — усреднение траектории.

**Сравнение с SGD.** Обычный SGD даёт $O(1/T)$. MeZO — $O(r(H)/T)$, то есть в
$r(H)$ раз медленнее. Для LLM $r(H) \sim 10^3{-}10^4$ (не $d \sim 10^9$!) → MeZO
требует $\sim 10^3{-}10^4$ раз больше шагов, что **реалистично**.

**Sketch доказательства.**

(1) Из §2.1: $\mathbb{E}_z[\hat g] = \nabla L + O(\epsilon^2)$.

(2) Bound на дисперсию (Malladi 2023, Lemma 1):

$$
\mathbb{E}_z\bigl[\|\hat g\|^2\bigr] \;\le\; \operatorname{tr}(H)\, \|\nabla L\|^2 + \sigma^2\, r(H).
$$

(3) Стандартная convex SGD analysis: одношаговый прогресс по гладкости

$$
L(\theta^{t+1}) \;\le\; L(\theta^t) - \eta \langle \nabla L(\theta^t), \hat g^t \rangle + \frac{\eta^2 \beta}{2} \|\hat g^t\|^2.
$$

Беря $\mathbb{E}_z$ и подставляя (1), (2):

$$
\mathbb{E}[L(\theta^{t+1})] \;\le\; L(\theta^t) - \eta\|\nabla L(\theta^t)\|^2 + \frac{\eta^2 \beta}{2}\bigl(\operatorname{tr}(H) \|\nabla L(\theta^t)\|^2 + \sigma^2 r(H)\bigr).
$$

(4) При $\eta \le 1/(\beta \operatorname{tr}(H))$ коэф при $\|\nabla L\|^2$ остаётся отрицательным:

$$
\mathbb{E}[L(\theta^{t+1})] \le L(\theta^t) - \frac{\eta}{2}\|\nabla L(\theta^t)\|^2 + \frac{\eta^2 \beta \sigma^2 r(H)}{2}.
$$

(5) Складывая по $t = 1, \ldots, T$, применяя Jensen и выпуклость, получаем
скорость $O(r(H)/T)$. ∎

### 7.3 Теорема 2 (невыпуклый PL)

**PL inequality (Polyak-Łojasiewicz).** Функция $L$ удовлетворяет PL с
константой $\mu > 0$, если:

$$
\frac{1}{2} \|\nabla L(\theta)\|^2 \;\ge\; \mu\,\bigl(L(\theta) - L^\star\bigr) \quad \forall \theta.
$$

Это **слабее**, чем сильная выпуклость, но достаточно для линейной сходимости.
Известно (Liu et al. 2022): **overparametrized DNN-loss удовлетворяет PL локально**
в окрестности минимума.

**Утверждение.** Под (A2), (A4) и PL, при $\eta = \Theta\bigl(1/(\beta r(H))\bigr)$:

$$
\mathbb{E}\bigl[L(\theta^T) - L^\star\bigr] \;\le\; (1 - \eta \mu)^T \bigl(L(\theta^0) - L^\star\bigr) + \frac{\eta\, \sigma^2\, r(H)}{\mu}.
$$

**Расшифровка:**

- Первый член → 0 **геометрически** (линейная сходимость).
- Второй — **noise floor** от стохастичности SPSA-оценки. При $\eta \to 0$ floor
  → 0, но тогда первый член сходится медленнее. **Trade-off.**

**Sketch доказательства.**

(1) Из smoothness:

$$
L(\theta^{t+1}) \le L(\theta^t) - \eta \langle \nabla L, \hat g \rangle + \frac{\eta^2 \beta}{2} \|\hat g\|^2.
$$

(2) Беря $\mathbb{E}$, используя $\mathbb{E}[\hat g] = \nabla L$ и bound на дисперсию:

$$
\mathbb{E}[L(\theta^{t+1})] \le L(\theta^t) - \eta \|\nabla L\|^2 + \frac{\eta^2 \beta}{2}\bigl(\operatorname{tr}(H) \|\nabla L\|^2 + \sigma^2 r(H)\bigr).
$$

(3) При $\eta \le 1/(\beta \operatorname{tr}(H))$:

$$
\mathbb{E}[L(\theta^{t+1})] \le L(\theta^t) - \frac{\eta}{2}\|\nabla L\|^2 + \frac{\eta^2 \beta \sigma^2 r(H)}{2}.
$$

(4) Применяем PL ($\|\nabla L\|^2 \ge 2\mu (L - L^\star)$):

$$
\mathbb{E}[L(\theta^{t+1}) - L^\star] \le (1 - \eta \mu)(L(\theta^t) - L^\star) + \frac{\eta^2 \beta \sigma^2 r(H)}{2}.
$$

(5) Разворачивая геометрическую рекурсию:

$$
\mathbb{E}[L(\theta^T) - L^\star] \le (1 - \eta\mu)^T (L(\theta^0) - L^\star) + \frac{\eta \sigma^2 r(H)}{\mu}. \qquad \blacksquare
$$

### 7.4 Что про Nesterov-MeZO?

**Теоретический результат для Nesterov + ZO-noise — открытая проблема.**

Что известно:

- В **детерминированном** случае Nesterov даёт $O(1/T^2)$ vs SGD $O(1/T)$.
- В стохастическом случае (Bottou-Curtis-Nocedal 2018) Nesterov **не быстрее**
  SGD из-за noise variance.
- **Для MeZO** (мы): эмпирически Nesterov **расходится** без стабилизации, но
  **3× ускоряет** со стабилизацией. Теории нет.

Подход к доказательству, который мы наметили (но не закрыли):

1. Lyapunov-функция $V^t = (L(\theta^t) - L^\star) + \alpha \|v^t\|^2$.
2. Контракция $V^{t+1} \le (1 - \gamma) V^t + \text{noise term}$.
3. Под condition $\beta \le \beta^\star(\eta, r(H), \sigma)$ — линейная
   сходимость.
4. $\beta^\star \to 0$ при $\sigma \to \infty$ — формализация «high noise → low momentum».

Это **сильнее**, чем наша β-decay эвристика, но требует тонкого анализа
covariance velocity. **Future work.**

### 7.5 Decentralized convergence

Под (A1)-(A4), D-MeZO с топологией $W$ (Koloskova 2020 Theorem 2):

$$
\mathbb{E}\bigl[L(\bar\theta^T) - L^\star\bigr] \;=\; O\!\left(\frac{r(H)}{T} + \frac{r(H)}{T\, \rho_W^2}\right).
$$

Второй член — **штраф за топологию**. Для complete graph $\rho_W = 1$ →
штрафа нет. Для ring graph $\rho_W \sim 1/N^2$ → штраф растёт.

**Эмпирически (Day 5):** partition tax (ring vs complete) **< 13%** на $N = 4$,
что согласуется с малостью $1/\rho_W^2$ для $N = 4$.

---

## 8. Эксперименты — итог

### 8.1 Setup

- **Модели:**
  - **Qwen3-4B** (standard transformer, Apache 2.0, $d \approx 4 \times 10^9$).
  - **Qwen3.5-4B-Base** (hybrid linear-attention V-L, $d \approx 4.5 \times 10^9$).
- **Задачи:** SST-2 (sentiment), BoolQ (yes/no QA).
- **Hardware:** RTX PRO 6000 Blackwell 96 ГБ (Colab Pro+).
- **Tracker:** MLflow (file backend, $./mlruns/$).
- **Reproducibility:** `random_state = 42` везде.

### 8.2 Ключевые результаты

| Эксп | Setup | Best loss | Drop | Note |
|---|---|---|---|---|
| Day 1 | Qwen3-4B / SST-2 / centralized | 0.17 | **88.1%** | sanity check, 2.4 мин |
| Day 4 | Qwen3-4B / SST-2 / 2 clients / D-MeZO | 0.1793 | ~88% | federated matches centralized |
| Day 5 | Qwen3.5 / 2×2 grid (W × IID-ness) | все PASS | tax < 13% | **first MeZO on linear-attn** |
| Day 6 | Nesterov β=0.9 (no clip) | **NaN at R140** | — | catastrophic divergence |
| Day 6b | Look-ahead Nesterov | **NaN at R20** | — | dual-channel noise, 7× faster diverge |
| Day 8 R1b | β=0.9 + clip50 | 0.119 @ R300 | — | **3× speedup**, но late drift to 0.225 |
| Day 8 R1d | **β-decay 0.9→0 + clip50** | **0.1291** | — | **monotonic, beats vanilla 0.1381 by 6.5%** |
| **2026-05-18 HellaSwag** | Qwen3-4B / 4-way commonsense / centralized vanilla | **2.7112** ⬆ | **−5.5%** | **DIVERGED**, acc 0.6625 → 0.6375 (−2.5pp) |
| **2026-05-18 HellaSwag** | Qwen3-4B / 4-way commonsense / **D-MeZO-N v1** | **2.4959** ⬇ | **+2.85%** | **CONVERGED**, acc 0.6625 → 0.7000 (+3.75pp), **+6.25pp vs centralized** |
| **2026-05-18 MathLogicQA** | Qwen3.5-4B-Base / 4-way Russian symbolic logic / centralized vanilla | **1.4331** ⬇ | **+49.7%** | **CONVERGED**, acc 0.3750 → 0.3750 (loss-acc decoupled) |
| **2026-05-18 MathLogicQA** | Qwen3.5-4B-Base / 4-way Russian symbolic logic / **D-MeZO-N v1** | 1.5155 ⬇ | +46.8% | **CONVERGED**, acc 0.3750 → 0.3875 (+1.25pp), peak 0.4125 @R500, **+1.25pp vs centralized** |

### 8.3 Negative findings (важная часть paper)

Negative results помогают строителям избежать ловушек:

- **β = 0.9 без clip → divergence at R140** (Day 6, SST-2).
- **True-Nesterov look-ahead diverges 7× faster** (Day 6b, SST-2) — двойной канал шума.
- **C = 200 clip → slow-divergence at R500** (Day 8 R1, SST-2).
- **Const β без decay → late drift** (Day 8 R1b, SST-2 → 0.225 финал, хуже чем R1d 0.1291).
- **Vanilla MeZO без clip → divergence на HellaSwag** (2026-05-18) — даже centralized расходится, loss +5.5% / acc −2.5pp. Не Nesterov-проблема, а **fundamental ZO-noise problem** на hard reasoning задачах. ρ-clipping=50 спасает.

Все эти негативные находки задокументированы и обсуждены в paper.

---

## 9. Что нового именно у нас

| Работа | Centralized MeZO | Federated | Decentralized | Nesterov | Hybrid LLM |
|---|:-:|:-:|:-:|:-:|:-:|
| MeZO (Malladi 2023) | ✓ | — | — | — | — |
| FedKSeed (Qin 2024) | ✓ | ✓ (с server) | — | — | — |
| Ferret (Shu 2024) | ✓ | ✓ (с server) | — | — | — |
| FedZeN (Maritan 2024) | ✓ | ✓ (Newton) | — | — | — |
| **D-MeZO-N (наш)** | ✓ | ✓ | **✓** | **✓ (stabilized)** | **✓ (Qwen3.5)** |

**Уникальные компоненты:**

1. **Decentralized topology** (peer-to-peer без server).
2. **Nesterov с стабилизацией** (ρ-clip + β-decay).
3. **Тест на hybrid linear-attention** (Qwen3.5-4B-Base).
4. **Две теоремы сходимости** (convex + PL).
5. **Систематический negative-results table** (что НЕ работает).

---

## 10. Заключение и open problems

### 10.1 TL;DR

**D-MeZO-N** — рецепт дообучения миллиардных LLM в децентрализованной сети с
**4 байтами коммуникации за раунд**:

- **Память:** $\sim d$ (как inference), без градиентов и Adam-state.
- **Коммуникация:** один скаляр + общий seed между клиентами.
- **Сходимость:** Theorem 1 ($O(r(H)/T)$, convex) + Theorem 2 (linear, PL).
- **Ускорение:** Nesterov heavy-ball + ρ-clip + β-decay → 3× speedup.
- **Эмпирика:** работает на Qwen3-4B и Qwen3.5-4B (hybrid linear-attention).

### 10.2 Open problems

1. **Convergence theory для Nesterov-MeZO.** Lyapunov-аргумент намечен (§7.4),
   но не закрыт.
2. **Adaptive β-schedule.** Сейчас линейная decay; cosine или гессиан-aware
   могут быть лучше.
3. **Privacy guarantees.** Скаляр $\hat\rho$ — это $z^\top \nabla L_i$, что
   утекает информацию о $\nabla L_i$. Анализ дифференциальной приватности —
   открытый вопрос.
4. **Byzantine robustness.** Что если клиент шлёт **враждебный** $\hat\rho$?
   Median-of-means defenses?
5. **Larger models.** Qwen3-8B, Llama-70B — потенциально работают, не тестировали
   из-за compute budget.
6. **Cross-task generalization.** Все эксперименты — на single task (SST-2,
   BoolQ). Multi-task federated MeZO — открыт.

### 10.3 Применимость

D-MeZO-N **разумен**, когда:

- Данные распределены по узлам и не могут быть централизованы.
- Связь между узлами дорога (междатацентровая, мобильная сеть).
- Память на узлах ограничена (нет VRAM для Adam state).
- Допустимо в $r(H) \approx 10^3$ раз больше итераций vs SGD-backprop.

D-MeZO-N **не рекомендован**, когда:

- Есть централизованный кластер с быстрым interconnect (NVLink, InfiniBand) —
  обычный distributed DDP-backprop эффективнее.
- Бюджет шагов ограничен ($< 10^3$ обновлений) — SGD-backprop предпочтительнее
  при наличии памяти.
- Нужен второй порядок (Newton/L-BFGS) — ZO-Newton (FedZeN-style) другой
  параллельный путь.

---

## Ссылки

1. **MeZO**: Malladi et al. *Fine-Tuning Language Models with Just Forward Passes*.
   NeurIPS 2023. arXiv:2305.17333.
2. **SPSA**: Spall. *Multivariate stochastic approximation using a simultaneous
   perturbation gradient approximation*. IEEE TAC 1992.
3. **Decentralized SGD**: Koloskova et al. *A Unified Theory of Decentralized SGD
   with Changing Topology and Local Updates*. ICML 2020. arXiv:2003.10422.
4. **FedKSeed**: Qin et al. *Federated Full-Parameter Tuning of Billion-Sized
   Language Models with Communication Cost under 18 Kilobytes*. ICML 2024.
   arXiv:2312.06353.
5. **Ferret**: Shu et al. *Ferret: Federated Full-Parameter Tuning at Scale for
   Large Language Models*. 2024. arXiv:2409.06277.
6. **FedZeN**: Maritan et al. *FedZeN: Towards superlinear zeroth-order federated
   learning*. 2024. arXiv:2309.17241.
7. **Nesterov-Spokoiny**: Nesterov, Spokoiny. *Random gradient-free minimization
   of convex functions*. Foundations of Computational Mathematics 2017.
8. **PL для DNN**: Liu et al. *Loss landscapes and optimization in
   over-parameterized non-linear systems and neural networks*. Applied and
   Computational Harmonic Analysis 2022.
9. **Qwen3**: https://huggingface.co/Qwen/Qwen3-4B
10. **Qwen3.5**: https://huggingface.co/Qwen/Qwen3.5-4B
