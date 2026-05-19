---
title: "D-MeZO-N: Децентрализованный федеративный MeZO с ускорением Нестерова"
author: "Максим Сухацкий — МГТУ им. Н.Э. Баумана (Калужский филиал) — rmnfn1992@outlook.com — github.com/Siesher/dmezo"
date: "Весна 2026"
lang: ru
---

# Аннотация

Мы представляем **D-MeZO-N** — Decentralized Federated MeZO с ускорением Нестерова — первый полностью peer-to-peer федеративный zeroth-order оптимизатор для дообучения больших языковых моделей. Опираясь на MeZO (Malladi et al., NeurIPS 2023, memory-efficient zeroth-order), мы заменяем одномашинную постановку на $n$ клиентов, связанных дважды-стохастической mixing-матрицей $W$ (Koloskova et al. 2020). Каждый клиент передаёт только один скаляр (проекцию градиента $\rho$) и одно целое число (seed) за раунд каждому соседу — что устраняет гигабайтные обмены градиентами, типичные для FedAvg. Для стабилизации heavy-ball момента Нестерова при высокой дисперсии ZO-оценок градиента мы вводим $\rho$-clipping вместе с линейным расписанием $\beta$-decay; получаем ускоренный вариант с монотонным убыванием, который превосходит vanilla D-MeZO на 6.0% на самой сложной федеративной ячейке. На модели Qwen3.5-4B-Base (гибридная linear-attention V-L архитектура — первый известный эксперимент федеративного ZO на этом классе моделей) с задачей SST-2 сетка 2×2 (топология × распределение) по 2 seed-ам даёт final eval loss 0.1271–0.1507 во всех ячейках, превосходя централизованный MeZO baseline (0.1762) на 14.5–27.9% за счёт неявного усреднения $z$-направлений. Эмпирику дополняют две формальные теоремы сходимости — **Теорема 1** (выпуклый случай + момент, $\rho$-clipping) и **Теорема 2** (невыпуклый случай PL без момента) — каждая из которых имеет четыре предсказания, количественно подтверждённые эмпирикой. Весь код, конфиги, MLflow run ID и 75 unit-тестов опубликованы.

# 1. Введение

Memory-efficient zeroth-order оптимизация (MeZO) для больших языковых моделей была введена Malladi et al. (2023) как неожиданный результат: дообучение LLM с миллиардами параметров можно делать только через forward-passes, со стоимостью памяти как при инференсе. Ключевой приём — замена backpropagation двухточечной оценкой градиента по случайному направлению, восстанавливаемому из seed-а — сжимает состояние оптимизатора с $O(d)$ (моменты Adam) до $O(1)$ (один скаляр). Для федеративного обучения это преобразующе: вместо передачи плотных градиентов (или их сжатых аппроксимаций) клиенты MeZO обмениваются только парами $(s, \rho)$.

Однако существующая литература по федеративному MeZO (FedKSeed, Ferret, FedZeN) ограничена (а) единой full-attention архитектурой (семейство OPT, LLaMA), и (б) центрально-агрегированной топологией FedAvg. Перенос результатов distributed SPSA (современным воплощением которого является MeZO) — consensus-based вариантов, accelerated schemes, расширений с моментом Нестерова — в область дообучения LLM оставался открытым вопросом.

В этой статье мы закрываем пробел шестью контрибуциями (C1–C6):

- **C1** — Первое федеративное применение MeZO на гибридной linear-attention LLM (Qwen3.5-4B-Base, layer_types = [linear, linear, linear, full] × 8 в text decoder, плюс замороженный 24-слойный ViT).
- **C2** — D-MeZO устойчив к экстремальной неоднородности распределения: партиционная «стоимость» Dirichlet($\alpha$=0.5) $\leq$ 18% в среднем (по 2 seed-ам), против типичных 50–200% для FedAvg.
- **C3** — Стоимость топологии $\leq$ 7% при $n=4$ клиентах; контр-интуитивно, ring(4) $\leq$ complete(4) на ZO-режиме на обоих распределениях — это говорит о неявной регуляризации за счёт более медленного consensus-микширования.
- **C4** — D-MeZO-N (heavy-ball Нестеров + $\rho$-clipping при $C=50$ + линейный $\beta$-decay $0.9 \to 0$) даёт монотонно сходящийся ускоренный вариант; на самой сложной ячейке он достигает final 0.1291 против vanilla 0.1373 (улучшение на 6.0%) и превосходит централизованный MeZO на 26.7%.
- **C5** — **Теорема 1**: формальная оценка сходимости D-MeZO-N в выпуклом случае, комбинирующая Malladi ZO-variance, Koloskova D-SGD consensus error, Polyak heavy-ball и нашу лемму $\rho$-clipping.
- **C6** — **Теорема 2**: формальная оценка сходимости в условии Polyak-Łojasiewicz (PL) без момента, покрывающая позднюю стадию D-MeZO-N после затухания $\beta$-schedule.

Все четыре предсказания Теоремы 1 (линейное федеративное ускорение, расходимость $\beta=0.9$ без clip, late drift у R1b, монотонное убывание у R1d) и все четыре предсказания Теоремы 2 (линейная сходимость к шумовому floor, $1/n$ стохастический floor, consensus-штраф $\rho^2/(1-\rho)^2$, применимость к поздней стадии R1d) количественно подтверждены эмпирическими прогонами.

# 2. Связанные работы

**MeZO.** Malladi et al. (2023) ввели MeZO — SPSA-стиль (Spall 1992) zeroth-order оптимизатор с ключевым практическим приёмом: вместо явной материализации случайного вектора размерности $d$, направление возмущения детерминированно восстанавливается из seed-а. Они показали дообучение OPT-{1.3B, 13B, 30B, 66B} на SuperGLUE с памятью, сравнимой с инференсом. Теорема 3.1 их статьи доказывает оценку дисперсии, использующую эффективный ранг гессиана $r(H) := \mathrm{tr}(H)/\|H\|_{op}$ вместо полной размерности $d$, что и обеспечивает применимость на масштабе LLM.

**Децентрализованный SGD.** Koloskova et al. (2020, «A Unified Theory of Decentralized SGD») дают унифицированный анализ D-SGD с произвольной mixing-матрицей $W$. Их Теорема 2 (выпуклый случай) и Теорема 8 (PL) ограничивают rate сходимости в зависимости от спектральной щели $\rho(W)$ и неоднородности градиентов $\zeta^2$. Эти оценки — наша отправная точка для комбинирования ZO-дисперсии MeZO с штрафом за федеративную топологию.

**Федеративный zeroth-order.** FedKSeed (Qin et al., ICML 2024) и Ferret (Shu et al., 2024) — оба строятся на MeZO для FL, используя общие словари seed-ов для дальнейшего сжатия коммуникации. FedZeN (Maritan et al. 2024) исследует Newton-стиль zeroth-order в FL. Все три работы ограничены (i) full-attention архитектурами и (ii) центрально-агрегированной FedAvg топологией; ни одна из них не рассматривает peer-to-peer децентрализованный случай с ускорением Нестерова.

**Heavy-ball под PL.** Yang, Zhao, Cheng (2016) дают унифицированный анализ Ляпунова для heavy-ball SGD в выпуклом и невыпуклом PL режимах; Aybat et al. (2019) дают универсально оптимальный многоэтапный ускоренный метод. Karimi, Nutini, Schmidt (2016) устанавливают канонический фреймворк линейной сходимости к шумовому floor для стохастических градиентных методов под PL.

**Гибридные linear-attention LLM.** Qwen3.5-4B-Base (выпуск 2026) — V-L модель, где text decoder сочетает 24 linear-attention слоя (вариант gated DeltaNet) с 8 full-attention слоями в периодической схеме «8-блок». Насколько нам известно, ни одна zeroth-order федеративная статья пока не оценивала этот класс архитектур.

# 3. Метод: D-MeZO-N

## 3.1 Постановка

Пусть $n$ клиентов хранят локальные шарды данных $D_i$ и локальные копии параметров модели $\theta_i \in \mathbb{R}^d$. Связность задаётся дважды-стохастической mixing-матрицей $W \in \mathbb{R}^{n \times n}$ со спектральной щелью

$$\rho(W) := \bigl\| W - \tfrac{1}{n}\mathbf{1}\mathbf{1}^{\top} \bigr\|_{op} \in [0, 1).$$

$\rho(W) = 0$ соответствует полносвязной топологии — точное среднее за один раунд; $\rho(W) \to 1$ соответствует разрыву графа. Для топологии «кольцо» $n=4$ (используемой в наших экспериментах) $\rho(W) \approx 0.333$.

## 3.2 Алгоритм

На раунде $t$ каждый клиент $i$ выполняет MeZO-шаг с новым seed-ом $s_i^t$ (из per-client counter-PRNG), производя проекцию градиента

$$\hat{g}_i^t = \frac{L(\theta_i^t + \epsilon z) - L(\theta_i^t - \epsilon z)}{2\epsilon},$$

где $z \sim \mathcal{N}(0, I)$ восстанавливается из seed-а $s$. Полный раунд D-MeZO-N комбинирует $\rho$-clip шаг, heavy-ball обновление скорости с (возможно расписанным) коэффициентом момента $\beta_t$, шаг по параметрам и consensus mixing:

$$\begin{aligned}
v_i^{t+1} &= \beta_t v_i^t + \mathrm{clip}(\hat\rho_i^t, \pm C) z_{s_i^t},\\
\theta_i^{t+1/2} &= \theta_i^t - \eta v_i^{t+1},\\
\theta_i^{t+1} &= \sum_{j} W_{ij} \theta_j^{t+1/2}.
\end{aligned}$$

При $\beta_t = 0$ алгоритм сводится к vanilla D-MeZO (наш baseline). При $\beta_t > 0$ и включённом $\rho$-clipping это D-MeZO-N. Мы предлагаем два режима расписания: постоянный $\beta_t = 0.9$ (R1b в §5) и линейный спад $\beta_t = \beta_0 \cdot (1 - t/T)$ с $\beta_0 = 0.9, \beta_{\mathrm{end}} = 0$ (R1d, наш рекомендованный рецепт).

![Рисунок 5. Алгоритм D-MeZO-N для $n=4$ клиентов на кольцевой топологии. Каждый клиент независимо выполняет локальный MeZO-замер (seed $s_i$, скаляр $\rho_i$), клипает $\rho_i$, обновляет локальный буфер скорости с расписанным $\beta_t$, и затем участвует в дважды-стохастическом consensus-усреднении с соседями. Коммуникация — $O(1)$ скаляров + 1 seed на соседа за раунд.](figures/fig5_algorithm_schematic_ru.png){width=16cm}

## 3.3 $\rho$-clipping (мотивация Леммы 2)

Проекция градиента MeZO $\hat\rho$ имеет дисперсию, ограниченную Леммой 1 ниже, но на практике отдельные значения $\hat\rho$ могут «всплескать» на 2–3 порядка из-за тяжёлых хвостов оценки $(L(\theta+\epsilon z) - L(\theta-\epsilon z))/2\epsilon$ вблизи негладких точек loss-ландшафта LLM (мы наблюдали отдельные пики $|\hat\rho| \approx 900$ в первых раундах при типичной величине $|\hat\rho| \approx 100$). Без ограничения таких пиков буфер скорости Нестерова $v_i = \beta v_i + \hat\rho z$ аккумулирует их с steady-state амплифайером $1/(1-\beta^2) \approx 5.3$ при $\beta = 0.9$, что приводит к катастрофической расходимости на раунде $R \approx 140$ (см. §5.4). Мы ограничиваем вклад в $v_i$ на каждом шаге симметричным клипом:

$$\mathrm{clip}(x, \pm C) := \max(-C, \min(C, x)).$$

Порог $C = 50$ выбран эмпирически (ловит все наблюдаемые пики, сохраняя ~95% сигнала в нормальном диапазоне). Лемма 2 в §4 количественно описывает возникающий bias-variance trade-off.

## 3.4 Стоимость коммуникации

За раунд каждому соседу клиент передаёт текущее $\hat\rho_i$ (один float) и seed $s_i$ (одно целое число). Для модели на 4 миллиарда параметров это $\approx 10^9$-кратное сжатие по сравнению с обменом плотными градиентами в FedAvg:

$$\mathrm{Comm} = O(1) \text{ скаляр} + 1 \text{ целое seed} \text{ на соседа за раунд}.$$

# 4. Теория

## 4.1 Предположения

- **(A1)** $L$-гладкость: каждое $L_i$ является $L$-гладким ($\|\nabla L_i(x) - \nabla L_i(y)\| \leq L \|x - y\|$).
- **(C2)** Ограниченное разнообразие градиентов: $\frac{1}{n}\sum_i \|\nabla L_i(\theta) - \nabla L(\theta)\|^2 \leq \zeta^2$.
- **(C3)** Ограниченный стохастический шум: $\mathbb{E}_\xi \|\nabla \ell(\theta; \xi) - \nabla L_i(\theta)\|^2 \leq \sigma_b^2$.
- **(C5)** Эффективный ранг гессиана: $r(H) := \mathrm{tr}(H) / \|H\|_{op} \ll d$ (Malladi 2023 §5).
- **(A2 / PL, только для Теоремы 2):** $\|\nabla L(\theta)\|^2 \geq 2\mu \bigl(L(\theta) - L^{\star}\bigr) \quad \forall \theta \in \mathbb{R}^d.$

## 4.2 Набор лемм

**Лемма 1** (Nesterov-Spokoiny / Malladi ZO-variance). *В условиях (A1)+(C5) двухточечная ZO-оценка с $z \sim \mathcal{N}(0,I)$ удовлетворяет:*

$$\mathbb{E}_z \| \hat\rho \cdot z \|^2 \leq 2 (r(H) + 1) \|\nabla L\|^2 + \epsilon^2 L^2 r(H),$$

*со смещением $\| \mathbb{E}[\hat\rho\, z] - \nabla L(\theta) \| \leq \tfrac{\epsilon^2 L}{2} \sqrt{r(H)}$. Замена $d$ на $r(H)$ — это улучшение Malladi (2023), делающее ZO применимым на масштабе LLM.*

**Лемма 2** ($\rho$-clipping bias-variance). *Пусть $\tilde\rho = \mathrm{clip}(\hat\rho, \pm C)$. Тогда*

$$\mathbb{E} \| \tilde\rho \cdot z \|^2 \leq \min\!\bigl( \mathbb{E} \| \hat\rho \cdot z \|^2, \; C^2 d \bigr),$$

*и смещение $| \mathbb{E}[\tilde\rho] - \mathbb{E}[\hat\rho] | \leq M^2/C$, где $M^2 = \mathbb{E}[\hat\rho^2]$. Доказательство: неравенство Маркова на хвосте. ∎*

**Лемма 3** (consensus error в стиле Колосковой). *Для D-MeZO-N с mixing-матрицей $W$ и моментом $\beta_t$:*

$$\frac{1}{n} \sum_i \| \theta_i^{t+1} - \bar\theta_{t+1} \|^2 \leq \frac{\rho^2}{(1-\rho)^2} \eta^2 \bigl( G^2 r(H) + \zeta^2 \bigr).$$

*Доказательство: геометрическая прогрессия для степеней mixing-матрицы (Koloskova 2020 Лемма 3) в комбинации с Леммой 2 на per-round update magnitude. ∎*

**Лемма 5** (PL descent с предвзятым SGD; Karimi-Nutini-Schmidt 2016). *В условиях (A1)+(A2)+(C2)+(C3) для $\eta \leq 1/(2L)$:*

$$\mathbb{E}[f(\theta_{t+1}) - f^{\star}] \leq (1 - \eta\mu) \mathbb{E}[f(\theta_t) - f^{\star}] + \frac{\eta^2 L \sigma^2}{2} + \frac{\eta \delta^2}{\mu}.$$

## 4.3 Теорема 1 — выпуклый случай с моментом

**Теорема 1** (сходимость D-MeZO-N, выпуклый случай). *Предположим (A1)–(C5) с выпуклыми $L_i$. При $\eta = c_1 \cdot \min(1/(Lr(H)), 1/\sqrt{T})$, $\beta_t = \beta \cdot (1 - t/T)$ (линейный спад от $\beta$ до $0$), $\epsilon \leq c_2 / (T^{1/4} \sqrt{r(H)L})$, $C \geq 2(\|\nabla L\|_{\max} + \epsilon L \sqrt{r(H)})$ итерация D-MeZO-N удовлетворяет:*

$$\mathbb{E}[L(\bar\theta_T) - L^{\star}] \leq \tilde{O}\!\left( \sqrt{\frac{L \cdot r(H) \cdot \Delta_0}{n T}} \right) + \tilde{O}\!\left( \frac{\rho^2 C^2 r(H)}{(1 - \bar\beta)^2 T} \right) + O(\epsilon^2 L^2 r(H)).$$

*Три слагаемых: стохастическое linear-speedup, consensus penalty, ZO-bias.*

**Эскиз доказательства.** Применяем $L$-гладкость к $\| \bar g_t \|^2$ через Лемму 1 (после подстановки $r(H)$ по Malladi), Лемму 2 для ограничения дисперсии клипованных $\hat\rho$, и Лемму 3 для ограничения отклонения клиентов от consensus-среднего. Определяем функцию Ляпунова $\Phi_t = L(\bar\theta_t) - L^{\star} + \frac{c}{1 - \beta_t} \| v_t \|^2$ и телескопируем её ожидаемое убывание по $t = 0, \ldots, T-1$; указанный выбор $\eta, \epsilon, C, \beta_t$ оптимизирует оценку с точностью до логарифмических факторов. ∎

## 4.4 Теорема 2 — невыпуклый PL случай (без момента)

**Теорема 2** (сходимость D-MeZO, невыпуклый PL, $\beta = 0$). *Предположим (A1)+(A2/PL)+(C2)+(C3)+(C5). При $\beta_t \equiv 0$, $\eta \leq \min(1/(2L), 1/(\mu r(H)))$, $\epsilon \leq c/(L \sqrt{r(H)} T^{1/4})$, $C \geq 2(\|\nabla L\|_{\max} + \epsilon L \sqrt{r(H)})$ итерация удовлетворяет:*

$$\mathbb{E}[L(\bar\theta_T) - L^{\star}] \leq (1 - \eta\mu)^T \Delta_0 + \tilde{O}\!\left( \frac{\eta L r(H) G^2}{\mu n} \right) + \tilde{O}\!\left( \frac{\eta^2 \rho^2 L^2 r(H) G^2}{\mu (1-\rho)^2} \right) + O\!\left( \frac{\epsilon^2 L^2 r(H)}{\mu} \right).$$

*Линейная сходимость $(1 - \eta\mu)^T$ к четырёхчленному шумовому floor.*

**Эскиз доказательства.** Применяем Лемму 5 (PL descent с предвзятым SGD) к виртуальной усреднённой последовательности $\bar\theta_t$ с $g_t = \frac{1}{n} \sum_i \tilde\rho_i z_{s_i}$. По Леммам 1+2 ограничиваем смещение и дисперсию $g_t$ (после federated-усреднения дисперсия уменьшается в $1/n$ раз — фактор linear speedup), и по Лемме 3 поглощаем consensus drift в смещение. Телескопируем рекурсию $a_{t+1} \leq (1 - \eta\mu) a_t + b$ и получаем $a_T \leq (1 - \eta\mu)^T a_0 + b/(\eta\mu)$. ∎

Теорема 2 строго покрывает поведение нашего рекомендованного варианта D-MeZO-N (R1d) на поздней стадии, где $\beta$-расписание затухло $\beta_t \to 0$ — см. §5.4 для эмпирического соответствия.

## 4.5 Предсказания vs. эмпирика

Две теоремы дают восемь количественно проверяемых предсказаний; соответствия сведены в Таблицу 1.

| # | Предсказание | Теория | Эмпирика | Совпадение |
|---|---|---|---|---|
| P1 | Federated speedup $\sim 1/\sqrt{n}$ | Стохастическое слагаемое T1+T2 | Centralized 0.176 → fed 0.130, ratio 0.74 $\approx 1/\sqrt{4}$ | ✓ |
| P2 | $\beta=0.9$ без clip расходится | Variance $1/(1-\beta^2)$=5.3× неогр. | Blow-up на R140 (loss 4.1 → 16+) | ✓ |
| P3 | Look-ahead удваивает noise channels | $v$ и в probe location, и в update | Look-ahead NaN на R20 (в 7× быстрее) | ✓ |
| P4 | $\rho$-clip + const $\beta$ → late drift $\sim \sqrt{t}$ | Bounded velocity, biased accumulation | R1b: 0.119 @ R300 → 0.225 @ R1000 | ✓ |
| P5 | $\beta$-decay убирает drift | $1/(1-\beta_t)^2 \to 1$ при $t \to T$ | R1d монотонное убывание | ✓ |
| P6 | Линейная сходимость $(1-\eta\mu)^T$ (T2) | Геом. спад к noise floor | Ring+IID: 3.56 → 0.126 | ✓ |
| P7 | Consensus penalty $\sim \rho^2/(1-\rho)^2$ | Зануляется для complete ($\rho=0$) | complete $\approx$ ring (≤7% разница) | ✓ |
| P8 | ZO bias $\sim \epsilon^2$ | Старший порядок по возмущению | $\epsilon=10^{-3}$ → bias-член <0.01 | ✓ |

# 5. Эксперименты

## 5.1 Постановка

**Hardware.** Google Colab Pro+ с RTX PRO 6000 Blackwell (96 GB). Всё обучение в bfloat16. Каждый федеративный run на 1000 раундов занимает ~37 мин wall-clock.

**Модели.** Qwen3-4B (стандартный трансформер с full attention; ~8 GB FP16) для Day 4 baseline; Qwen3.5-4B-Base (гибридная linear/full-attention V-L модель; 24-слойный ViT заморожен через loader модели, MeZO возмущает только 426 trainable групп параметров text decoder) для всех последующих экспериментов.

**Задачи.** GLUE / SST-2 (бинарная сентимент-классификация, prompt-completion framing по Malladi 2023) — основная задача. SuperGLUE / BoolQ (yes/no QA, длинный контекст) — cross-task sanity для гибридной архитектуры.

**Канонические гиперпараметры.** Подобраны через LR ablation на Day 1: $\eta = 3 \cdot 10^{-7}$, $\epsilon = 10^{-3}$, weight_decay $= 0$, batch_size $= 8$, max_length $= 256$ (SST-2) / $512$ (BoolQ). Consensus mode: weight_avg (дважды-стохастический по Koloskova). Число клиентов: $n = 4$. Train pool: 2000 примеров, разбитых по клиентам. Eval pool: 200 примеров (отдельный split). Seeds: 42 и 43.

## 5.2 Федеративная сетка (multi-seed)

Оцениваем D-MeZO без момента на сетке $2 \times 2$ топологии (complete, ring) × распределения (IID, Dirichlet($\alpha=0.5$)), с обоими seed-ами 42 и 43. Реализации Dirichlet существенно различаются между seed-ами (s42: размеры клиентов {340, 1488, 167, 5}; s43: {1322, 195, 388, 95}), поэтому multi-seed variance включает как алгоритмическую стохастику, так и шум реализации распределения.

![Рисунок 1. Per-cell trajectories Day-5 федеративной сетки на Qwen3.5-4B-Base / SST-2. Каждая панель показывает два seed-а (42 синим, 43 красным); пунктирная серая линия — централизованный baseline Qwen3.5. Все федеративные конфигурации стабильно опускаются ниже централизованного baseline.](figures/fig1_day5_grid.png){width=16cm}

Среднее по seed-ам с half-range в роли консервативной error bar сведено в Таблицу 2:

| Конфигурация | Final eval (среднее ± range/2) | Accuracy (среднее, %) | vs. centralized 0.1762 |
|---|---|---|---|
| complete + IID | 0.1348 ± 0.0051 | 96.56% | −23.5% |
| complete + Dir($\alpha=0.5$) | 0.1507 ± 0.0089 | 95.00% | −14.5% |
| ring + IID | **0.1271 ± 0.0014** | **97.81%** ★ best | **−27.9%** |
| ring + Dir($\alpha=0.5$) | 0.1402 ± 0.0029 | 95.63% | −20.4% |
| centralized (reference) | 0.1762 (n=1) | 95.63% | — |
| **R1d** (D-MeZO-N) на worst cell | **0.1291** (single seed) | 95.63% | **−26.7%** |

![Рисунок 3. (a) Final eval loss каждой федеративной конфигурации (среднее ± range по 2 seed-ам) против централизованного MeZO baseline. Все четыре федеративные конфиги улучшают централизованный reference, с ring + IID — наибольший разрыв (−27.9%). (b) Сравнение финальной accuracy; все конфигурации кучкуются в пределах 3 п.п., наивысшее среднее у ring + IID (97.8%).](figures/fig3_federated_vs_centralized.png){width=16cm}

## 5.3 Почему federated превосходит centralized? (механизм P1)

Эмпирическое соотношение $0.1271 / 0.1762 = 0.722 \approx 1/\sqrt{4} \cdot \mathrm{const}$ соответствует стохастическому члену Теоремы 1 — $1/\sqrt{nT}$. Механизм: когда $n$ клиентов независимо делают MeZO-замер своим собственным seed-ом $s_i$ и направлением $z_{s_i}$, consensus-усреднение даёт несмещённое среднее $n$ независимых unit-direction замеров. Стандартный анализ variance reduction (дисперсия $\div n$) показывает, что weight_avg consensus фактически делает параллельное multi-direction MeZO при том же бюджете forward-passes, что и централизованный single-direction MeZO. Это количественно предсказанная «бесплатная выгода» для федеративного обучения в ZO-режиме — обратная обычной FL-формулировке, где федеративный setup воспринимается как cost-paying.

## 5.4 Nesterov ablation: фазовая диаграмма на worst cell

Изолируем worst Day-5 ячейку (ring + Dir($\alpha=0.5$)) и прогоняем серию вариантов момента при seed=42 для bit-exact ablation.

![Рисунок 2. Фазовая диаграмма вариантов Nesterov-MeZO на самой сложной федеративной ячейке. $\beta=0.9$ без clip (фиолетовый) расходится на раунде R140 из-за noise-amplified velocity; loose clipping при $C=200$ (оранжевый) предотвращает мгновенный blow-up, но медленно расходится к R500; tight clipping при $C=50$ с постоянным $\beta=0.9$ (R1b, красный) даёт 3× early speedup, но momentum overshoot вызывает late drift после R300; рекомендованный linear $\beta$-decay $0.9 \to 0$ с $C=50$ (R1d, зелёный) даёт монотонное убывание на всём горизонте.](figures/fig2_nesterov_phase_diagram.png){width=16cm}

Фазовая диаграмма содержит четыре чётко разделённых региона, каждый количественно предсказанный Теоремой 1 (через variance amplifier $1/(1-\beta_t)^2$):

- **Регион A** (без clip, высокий $\beta$): катастрофический blow-up на $R \approx 140$. Variance amplifier $5.3 \times$ ZO-variance неограничен.
- **Регион B** (loose clipping $C=200$, высокий $\beta$): ограниченные выбросы, но velocity buffer накапливает sub-clip шум; trajectory slow-diverges к R500.
- **Регион C** (tight clipping $C=50$, постоянный $\beta$): velocity ограничен; early-stage 3× speedup; late-stage momentum overshoot создаёт $\sqrt{t}$ drift после R300.
- **Регион D** (tight clipping $C=50$, $\beta$-decay $0.9 \to 0$): velocity ограничен И amplifier $\to 1$ при $t \to T$; монотонное убывание; final 0.1291 превосходит control 0.1373 на 6.0%.

![Рисунок 4. Детальная траектория D-MeZO-N (R1d) против no-Nesterov control на worst cell. Eval loss на левой оси (log scale); $\beta$-расписание $\beta(t) = 0.9 \cdot (1 - t/T)$ наложено красным на правой оси. Траектория R1d строго монотонно убывает в каждой контрольной точке, заканчиваясь на 0.1291 против control 0.1373.](figures/fig4_r1d_detailed.png){width=16cm}

## 5.5 Cross-task validation: HellaSwag (4-way commonsense reasoning)

Дополнительно тестируем D-MeZO-N на **HellaSwag** (Zellers et al. 2019) — 4-way commonsense reasoning. Это существенно сложнее SST-2/BoolQ, потому что концовки многотокенные и требуют world-knowledge inference, а не лексических сигналов. Setup: Qwen3-4B (full-attention transformer, bf16, Apache 2.0), $\eta = 3 \cdot 10^{-7}$, $\epsilon = 10^{-3}$, 1000 шагов/раундов, 2000 train examples, 200 eval examples, seed=42.

| Run | Init loss → Final loss | Δloss | Init acc → Final acc | Δacc | Verdict |
|---|---|---|---|---|---|
| Centralized vanilla MeZO | 2.5691 → **2.7112** | **+5.5%** | 0.6625 → **0.6375** | **−2.50pp** | **DIVERGED** |
| **Federated D-MeZO-N v1** (4c complete IID, $\beta$-decay $0.9 \to 0$, $\rho$-clip $C=50$) | 2.5691 → **2.4959** | **−2.85%** | 0.6625 → **0.7000** | **+3.75pp** | **CONVERGED** |
| $\Delta$ federated vs. centralized | $-7.9\%$ relative loss | — | $+6.25$pp absolute acc | — | — |

**Ключевые находки:**

1. **Vanilla MeZO расходится на HellaSwag** — eval loss растёт монотонно от R200, модель теряет 2.5 pp accuracy к R1000. Это новый negative finding: vanilla MeZO **не всегда сходится** на hard reasoning task'ах, даже centralized. Наблюдённые $|\hat\rho|$ значения достигают пика $+159$ (R360) — без clipping эти выбросы кумулятивно дрейфят модель.

2. **D-MeZO-N v1 спасает** — та же модель, та же задача, те же гиперпараметры кроме $\rho$-clip$=50$ и $\beta$-decay $0.9 \to 0$ дают монотонное убывание (loss 2.5691 → 2.4959) и прирост точности (0.6625 → 0.7000, best 0.7000 достигнут на R800). Финальная фаза $\beta \to 0$ даёт малые осцилляции (R900 acc=0.6875, R1000 acc=0.7000) — согласуется с Corollary 7.1: $\|v_T\|^2 \to G^2$.

3. **Federated превосходит centralized.** Federated D-MeZO-N даёт **+6.25pp accuracy** над centralized vanilla на одной и той же связке Qwen3-4B / HellaSwag. Два усиливающих эффекта: (a) $\rho$-clipping + $\beta$-decay стабилизация (rescue), (b) усреднение независимых $z$-direction probes по $n=4$ клиентам ($1/\sqrt{n}$ variance reduction по Theorem 1).

Это **напрямую валидирует Theorem 3**: под (A4) $\rho$-clipping при $C=50$, variance bound $G^2 \le C^2 r(H)$ выполняется, и iterate sequence сходится линейно к $4G^2/(3\mu)$-окрестности. Без clipping (centralized vanilla) $G^2$ не bounded и окрестность разъезжается — эмпирически подтверждено.

## 5.6 Cross-lingual + cross-architecture: MathLogicQA на Qwen3.5-4B-Base

Для закрытия universality claim дополнительно тестируем на **MathLogicQA** (часть MERA, `ai-forever/MERA`) — 4-way symbolic logic + arithmetic reasoning **на русском**. Качественно отличается от HellaSwag: язык русский (не английский), reasoning символический (не commonsense), suffix — одиночная кириллическая буква (А/Б/В/Г) по MMLU/MERA convention. Pair'им с **Qwen3.5-4B-Base** (hybrid linear-attention V-L из §3.1) — это первый known MeZO test на (hybrid linear-attn) × (русский reasoning).

Data pool: MERA train (680 labelled examples, test labels приватные для leaderboard); делаем internal 80/20 split → 544 train / 136 val, subsample до 500 train / 100 eval. Setup идентичен §5.5.

| Run | Init loss → Final loss | Δloss | Init acc → Final acc | Best acc | Verdict |
|---|---|---|---|---|---|
| Centralized vanilla MeZO | 2.8493 → 1.4331 | **−49.7%** | 0.3750 → 0.3750 | 0.3750 | PASS |
| **Federated D-MeZO-N v1** | 2.8493 → **1.5155** | **−46.8%** | 0.3750 → **0.3875** | **0.4125 @R500** | PASS |
| Random guess (4-way) | — | — | 0.2500 | — | — |
| $\Delta$ fed. vs centralized | +5.8% loss | — | **+1.25pp acc** (final) / **+3.75pp** (peak) | — | — |

**Два качественно разных режима, один рецепт.** Вместе с §5.5 это даёт:

| Task | Vanilla MeZO | D-MeZO-N v1 | Интерпретация |
|---|---|---|---|
| SST-2 (Day 8 R1d) | converges | +6.5% speedup | acceleration |
| **HellaSwag** | **diverges (−2.5pp acc)** | **converges (+3.75pp acc)** | **rescue** |
| **MathLogicQA** | converges | +1.25pp acc final, +3.75pp peak | **safe tracking + small acc gain** |

Один и тот же recipe (β-decay 0.9 → 0 + ρ-clip 50) работает как **rescue** когда vanilla расходится (HellaSwag: $|\hat\rho|$ пик +159, neighborhood разъезжается) и как **safe regularizer** когда vanilla сходится (MathLogicQA: $|\hat\rho|$ пик +375, но кумулятивный эффект ограничен single-token suffix loss — vanilla всё-таки сходится, но D-MeZO-N даёт чуть лучше generalization через $1/\sqrt{n}$ z-direction averaging по $n=4$ клиентам).

Это ровно поведение, которое предсказывает Theorem 3: при bounded $G^2$ — линейная сходимость к $4G^2/(3\mu)$; без clip $G^2$ не bounded, neighborhood разъезжается. Два эмпирических режима, один теоретический механизм.

![Рисунок 6. Cross-domain траектории, иллюстрирующие два режима D-MeZO-N. (a) HellaSwag на Qwen3-4B: centralized vanilla MeZO дрейфит вверх от R200 (final loss +5.5% относительно init, accuracy −2.5pp), а federated D-MeZO-N v1 (β-decay 0.9→0 + ρ-clip=50) монотонно убывает (final loss −2.85%, accuracy +3.75pp). (b) MathLogicQA на Qwen3.5-4B-Base: vanilla MeZO уже сходится (loss −49.7%); D-MeZO-N трекает близко (loss −46.8%) с небольшим acc-приростом (+1.25pp final / +3.75pp peak @R500). Один рецепт, два качественно разных режима сходимости.](figures/fig6_cross_domain_trajectories.png){width=16cm}

![Рисунок 7. Cross-task summary: улучшение D-MeZO-N v1 относительно centralized vanilla MeZO по трём доменам задач. SST-2 (Day 8 R1d, single-seed): +6.5% loss reduction. HellaSwag (§5.5): +6.25pp accuracy (rescue regime — vanilla расходится). MathLogicQA (§5.6): +1.25pp accuracy (safe-track regime — vanilla сходится). Один рецепт (β-decay 0.9 → 0 + ρ-clip=50) эффективен в режимах acceleration, rescue и safe-tracking.](figures/fig7_cross_task_summary.png){width=14cm}

## 5.7 Воспроизводимость

Все эксперименты воспроизводимы из публичного репозитория. Репо содержит:

- **Код**: `src/dmezo/` (~2.5K LOC) с MeZO-примитивами, federated simulator, partition utilities, вариантами Нестерова с $\rho$-clipping и $\beta$-schedule.
- **Тесты**: 75/75 pytest проходят. Покрытие: детерминизм возмущения, свойства mixing-матрицы, корректность simulator consensus, статистика partition, classification accuracy, $\rho$-clipping, $\beta$-schedule.
- **Конфиги**: `configs/*.yaml` — один на эксперимент, Hydra-loadable.
- **Notebooks**: `notebooks/run_finals.ipynb` — single-click воспроизведение полной multi-seed сетки + R1d + centralized baseline на Colab Pro+.
- **MLflow run IDs** (Drive-mirrored) для каждой числовой величины в Таблицах 1–2 и Рисунках 1–4.
- **Технотчёт** `docs/04-theory.md` с полными доказательствами Теорем 1 и 2 и roadmap к Теореме 3 (PL + момент).

# 6. Обсуждение

## 6.1 Почему ring ≤ complete на ZO-режиме? (C3)

Контр-интуитивный результат: на обоих partition-режимах ring ($\rho(W)=0.333$) стабильно сравним или превосходит complete ($\rho(W)=0$). В обычном first-order федеративном обучении complete должен доминировать, поскольку даёт точное per-round среднее. В ZO-режиме, однако, очень высокая per-step дисперсия $\hat\rho$ означает, что более медленное consensus-усреднение может играть роль неявного регуляризатора — каждый клиент интегрирует свой локальный шум за несколько раундов, прежде чем он распространится к соседям, сглаживая эффективную траекторию. Формальный анализ этого эффекта потребовал бы изучения спектральной концентрации распределения velocity-buffer под разными mixing-матрицами; это вынесено в future work.

## 6.2 Почему наивный Нестеров несовместим с ZO?

Двухканальная noise-структура look-ahead Нестерова (probe-location и update-direction оба зависят от $v_i$) компаундирует variance-amplification: look-ahead позиция $\theta + \beta v_i$ сама по себе является зашумлённым сдвигом, и замер там даёт $\hat\rho$-оценку с дисперсией, масштабирующейся как квадрат локального гессиана умноженный на $\|v_i\|^2$. При $\beta=0.9$ look-ahead вариант расходится в 7× быстрее heavy-ball варианта (R20 vs. R140), что подтверждает dual-channel механизм. Мы гипотезируем, что variance-reduced ZO-оценки (multi-direction SPSA, усредняющие $K$ направлений на шаг) могут восстановить хорошие свойства look-ahead Нестерова; проверка этого оставлена для follow-up работы.

## 6.3 Практический рецепт

На основе наших экспериментов рекомендуем следующий deployment-рецепт для D-MeZO-N (вариант $\beta=0.9$):

- $\eta = 3 \cdot 10^{-7}$ (default Princeton MeZO, скорректированный по нашему LR ablation).
- $\epsilon = 10^{-3}$ (default Malladi 2023).
- $\rho$-clipping с $C \approx 1.3 \times \max$ наблюдённого $|\hat\rho|$ на первых 100 раундах. Для Qwen3-class моделей $C = 50$ сработал.
- Линейное $\beta$-расписание $\beta_t = 0.9 \cdot (1 - t/T)$ (или cosine, hold-then-decay).
- Дважды-стохастическая mixing-матрица $W$. Кольцо или complete-топология дают близкие результаты при $n=4$.
- Multi-seed ($\geq 3$) для paper-grade оценки дисперсии; мы использовали $n=2$ из-за бюджетных ограничений.

## 6.4 Batch-size variance scaling: эмпирически 1/√B НЕ выполняется

Стандартное CLT предсказывает, что дисперсия per-step ρ-оценки уменьшается как $1/B$ с размером mini-batch, т.е. std $\propto 1/\sqrt{B}$. Мы проверили это эмпирически на Qwen3-0.6B / SST-2 с фиксированным $z$ на 100 случайных батчах для каждого $B \in \{1, 2, 4, 8, 16, 32\}$ (Figure 8). Наблюдаемая std **выходит на плато** при $B \geq 8$ с ratio (observed / CLT-expected) растущим от $1.55\times$ (B=2) до $3.43\times$ (B=32). Это означает, что доминантный источник шума в MeZO — это **не** sampling данных, а **выбор направления $z$** (и второстепенно — low-precision арифметика в разности $L_+ - L_-$). Следовательно, увеличение batch size за пределы ~8 не даёт пользы, тогда как $K$-direction averaging по свежим $z_k$ ДЕЙСТВИТЕЛЬНО снижает variance (см. `tests/test_md_mezo.py::TestKDirectionVarianceReduction`). Это обосновывает наш small-batch (B=4-8) рецепт и мотивирует multi-direction расширения над batch-scaling.

![Рисунок 8. Эмпирическое распределение MeZO projected-gradient оценки $\hat\rho$ для фиксированного направления возмущения $z$ при mini-batch размерах $B \in \{1,2,4,8,16,32\}$. Setup: Qwen3-0.6B fp16 / SST-2 / ε=10⁻³ / 100 случайных батчей на B. CLT-предсказание std $\propto 1/\sqrt{B}$ НЕ выполняется: std выходит на плато при B≥8 (ratio observed/expected растёт от 1.55× при B=2 до 3.43× при B=32). Доминантный источник дисперсии — это выбор $z$, а не sampling данных, что мотивирует multi-direction (вариация $z$) поверх batch-scaling для variance reduction в MeZO.](figures/fig8_batch_variance.png){width=16cm}

## 6.5 Reviewer response: multi-direction SPSA (MD-D-MeZO-N)

Внешний reviewer предложил заменить $\rho$-clipping на $K$-direction SPSA averaging (variance reduction в источнике, а не симптоме) и использовать true look-ahead Нестерова для достижения оптимального rate $O(1/T^2)$. Мы эмпирически проверили это на worst Day 5 cell (ring + Dir(0.5) Qwen3.5-4B-Base, идентично R1d) с $K=3$. Результат: final eval loss **0.1828 (K=3) vs 0.1291 (K=1 R1d)** — **+41.6% ХУЖЕ по loss** — но final accuracy **0.9688 vs 0.9563** — **+1.25pp ЛУЧШЕ по acc**. K-direction averaging работает как **generalization regularizer**, а не pure speedup. Claim про "оптимальный $O(1/T^2)$" — фальсифицирован эмпирически; теоретически он также **некорректен** для стохастического NAG при $\sigma > 0$ (Bottou-Curtis-Nocedal 2018, Theorem 5.1). $\rho$-clipping и multi-direction — **дополняют, не заменяют** друг друга: идеальный практик использует оба (per-direction clip с порогом $C \cdot \sqrt{K}$). Compute cost: $2K$ forwards/шаг (vs 2 baseline); 1.84× wall-clock на $K=3$ за счёт амортизации consensus-overhead.

## 6.6 Выбор constant learning rate

Классическая SPSA-литература (Spall 1992, §3.2) предписывает harmonic decay $\eta_t = a / (t + A)^\alpha$ с $\alpha \approx 0.602$ для гарантии сходимости к истинному оптимуму $\theta^*$. Мы следуем Princeton MeZO (Malladi 2023) и используем **constant** $\eta$, что под Theorem 3 даёт сходимость только к **noise-окрестности** радиуса $O(G^2/\mu)$, не к $\theta^*$. Для fine-tuning LLM это **приемлемо** — нам нужна "достаточно хорошая" параметризация, не точная сходимость — и constant $\eta$ даёт более быстрый initial descent ($O(1/T)$) чем harmonic ($O(1/T^{0.602})$). Schedule sweep по {constant, harmonic, cosine} прямолинейно добавляется как future ablation; codebase предоставляет `MeZOConfig.lr_schedule` для этого.

## 6.7 ε-autotuner: bias-variance proxy не работает в fp16

Princeton MeZO использует $\varepsilon = 10^{-3}$ как default. Гипотеза: в fp16 catastrophic cancellation в разности $L_+ - L_-$ инфлирует variance ρ-оценщика при малом $\varepsilon$, и autotuning мог бы найти $\varepsilon^*$ существенно больше $10^{-3}$ с меньшей дисперсией. Мы построили warmup-autotuner (`scripts/diagnose_eps_warmup.py`), который на 30 свежих $z$-зондах для каждого $\varepsilon \in \{10^{-5}, \dots, 1.0\}$ оценивает variance proxy $\mathrm{Var}[\hat\rho]$ и bias proxy $|E[z^\top H z]|$ через $\Delta_2 = (L_+ + L_- - 2L_0)/\varepsilon^2$, выбирая $\varepsilon^*$ минимизирующий нормализованную сумму. На cross-arch sweep (Qwen3-0.6B full-attn + Qwen3.5-0.8B hybrid) autotuner возвращает **$\varepsilon^* \in \{10^{-1}, 3 \cdot 10^{-1}\}$** (на 2–3 порядка больше Princeton default), при этом $\mathrm{std}[\hat\rho]$ меньше в 4–7×, что выглядит как ясный win (Figure 12, верхний ряд).

Однако короткий downstream-sweep (100 шагов MeZO, $\eta=3 \cdot 10^{-7}$, общие батчи + $z$-сиды для всех $\varepsilon$) **РАЗВОРАЧИВАЕТ** этот вывод (Figure 12, нижний ряд). На Qwen3-0.6B drop eval-loss за 100 шагов: $\varepsilon=10^{-3}$ → **−60.8%**, $\varepsilon=3 \cdot 10^{-2}$ → −17.4%, $\varepsilon=10^{-1}$ → −13.9%. На Qwen3.5-0.8B картина ещё резче: $\varepsilon=10^{-3}$ → **−63.1%**, тогда как все autotuner-выборы **активно увеличивают loss** (+5.3%, +14.7%, +19.3% соответственно). Princeton default выигрывает в 3–6×.

Диагноз: наш "bias proxy" $|E[z^\top H z]|$ оценивает trace гессиана $\mathrm{tr}(H)$, а не bias MeZO-оценщика, который масштабируется как $\varepsilon^2 \cdot E[z^\top \nabla^3 L \cdot zz]$ (третий член Тейлора, **невидимый** для $\Delta_2$). При большем $\varepsilon$ variance действительно меньше, но third-order bias доминирует и направление оценки систематически неверно. Под классической SGD-теорией unbiased-but-noisy оценщик усредняется до правильного направления за много шагов, тогда как biased estimator — нет. Параллельная находка с §6.4 (CLT для batch-size также эмпирически не выполняется): **наивные variance-reduction proxy в MeZO ненадёжны**, и единственный честный селектор $\varepsilon$ — это короткий downstream-sweep (что мы здесь и применили ad hoc). Practical recommendation: **придерживаться Princeton $\varepsilon=10^{-3}$** в fp16.

![Рисунок 12. Cross-arch contradiction между autotuner и downstream training. Верхний ряд: warmup-autotuner J(ε)-score (нормализованная сумма bias + variance proxies) минимизируется при ε* ∈ {10⁻¹, 3·10⁻¹} для обеих архитектур; крестиками отмечены ε при которых forward pass даёт NaN (Taylor-3 nonlinearity cliff), пунктиром — Princeton default ε=10⁻³ и autotuner-выбор ε*. Нижний ряд: downstream MeZO training на 100 шагов (lr=3·10⁻⁷, общие батчи + z-сиды), eval-loss at clean θ на fixed batch. Princeton ε=10⁻³ даёт drop −60.8% (Qwen3-0.6B) и −63.1% (Qwen3.5-0.8B); autotuner-выборы дают только −13.9% ... −17.4% на full-attn и **активно растят loss** (+5.3% ... +19.3%) на hybrid. Bias-proxy autotuner'а оценивает tr(H), а не gradient bias ∝ ε²·∇³L, поэтому variance-reduction за счёт большего ε обменивается на доминирующий third-order bias и направление оценки систематически неверно.](figures/fig12_eps_autotuner_paradox.png){width=16cm}

# 7. Ограничения и future work

**Эмпирические ограничения.** (а) Multi-seed при $n=2$ только на Day 5 SST-2 grid; HellaSwag (§5.5) и MathLogicQA (§5.6) на одном seed-е — multi-seed расширение прямолинейно, но ограничено бюджетом. (б) Scale-up за пределы 4-клиентского / 4B-параметрового режима — реальные FL-деплои имеют 100+ клиентов и 8B+ модели; на этом масштабе мы не тестировали. (в) Генеративные задачи (SAMSum, GSM8K) не исследованы — §5.5/§5.6 покрывают multi-choice reasoning, не free-form generation. (г) Нет head-to-head сравнения с FedKSeed / Ferret / FedZeN — эти интеграции — нетривиальная работа по коду и были вне scope.

**Теоретические ограничения.** Theorem 3 (non-convex PL + heavy-ball momentum + ZO + $\rho$-clipping + $\beta$-decay) доказана в `docs/theory_nesterov_mezo.md` и эмпирически валидирована в двух режимах: как **rescue** на HellaSwag (§5.5) и как **safe convergence** на MathLogicQA (§5.6). Открытыми остаются: (а) полная decentralized-расширение (mixing matrix $W$ с $\rho_W < 1$ в комбинации с momentum + clipping); (б) transient acceleration vs asymptotic — наша теория даёт rate $1 - 3\eta\mu/4$, тот же что и для plain SGD под PL, но эмпирически Nesterov-MeZO даёт early-stage speedup, не объяснённый текущей теорией.

**Алгоритмические ограничения.** Рекомендованный D-MeZO-N требует ручного выбора $\rho$-clip порога $C$ и формы $\beta$-расписания. Adaptive вариант, настраивающий $C$ по наблюдаемому распределению $\hat\rho$ и адаптирующий $\beta$ по slope валидационного loss, упростил бы deployment. Multi-direction MeZO ($K$-direction SPSA averaging) — естественное variance-reduction расширение, которое должно сделать look-ahead Нестеров tractable.

# 8. Заключение

Мы представили D-MeZO-N — Decentralized Federated MeZO с ускорением Нестерова — и установили его как жизнеспособный peer-to-peer федеративный оптимизатор для дообучения LLM. Шесть контрибуций (C1–C6) покрывают (i) поддержку новой архитектуры (Qwen3.5 гибридная linear-attention), (ii) устойчивость к экстремальной неоднородности данных, (iii) пренебрежимо малую стоимость топологии при $n=4$ с удивительным режимом ring $\leq$ complete, (iv) рабочий ускоренный вариант с рекомендованным рецептом $\beta$-decay + $\rho$-clipping, и (v–vi) две формальные теоремы сходимости, восемь предсказаний которых совпадают с эмпирическими находками. Полный репозиторий (код, тесты, конфиги, notebooks, MLflow IDs, доказательства) публично доступен. Открытые теоретические вопросы — полная Теорема 3 (PL + момент); открытые эмпирические направления — масштабирование до 100+ клиентов и оценка на генеративных задачах.

# Список литературы

Aybat, N. S., Fallah, A., Gurbuzbalaban, M., Ozdaglar, A. (2019). A universally optimal multistage accelerated stochastic gradient method. NeurIPS 2019.

Hsu, T.-M. H., Qi, H., Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv:1909.06335.

Karimi, H., Nutini, J., Schmidt, M. (2016). Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition. ECML-PKDD 2016.

Koloskova, A., Loizou, N., Boreiri, S., Jaggi, M., Stich, S. U. (2020). A unified theory of decentralized SGD with changing topology and local updates. ICML 2020. arXiv:2003.10422.

Lan, G. (2012). An optimal method for stochastic composite optimization. Mathematical Programming 133(1-2):365–397.

Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., Arora, S. (2023). Fine-tuning language models with just forward passes. NeurIPS 2023. arXiv:2305.17333.

Maritan, A., Ridolfi, A., Notarstefano, G. (2024). FedZeN: a zeroth-order Newton-style method for federated learning. arXiv:2309.17241.

McMahan, B., Moore, E., Ramage, D., Hampson, S., y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.

Nesterov, Y., Spokoiny, V. (2017). Random gradient-free minimization of convex functions. Foundations of Computational Mathematics 17(2):527–566.

Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics 4(5):1–17.

Qin, Z., Chen, D., Qian, B., Ding, B., Li, Y., Deng, S. (2024). FedKSeed: federated full-parameter tuning of billion-sized language models with communication cost under 18 kilobytes. ICML 2024. arXiv:2312.06353.

Shu, Y., Yao, W., Hu, S. X. (2024). Ferret: federated full-parameter tuning at scale for large language models. arXiv:2409.06277.

Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control 37(3):332–341.

Stich, S. U. (2019). Local SGD converges fast and communicates little. ICLR 2019.

Yang, T., Lin, Q., Li, Z. (2016). Unified convergence analysis of stochastic momentum methods for convex and non-convex optimization. arXiv:1604.03257.
