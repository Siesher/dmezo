# 01. Контекст проекта

## Проблема

Современный federated fine-tuning LLM упирается в три ограничения:

1. **Коммуникация.** В наивном FedAvg клиенты гоняют обновления размера модели (~14 GB для Llama-7B FP16) за раунд. Это нереально для miscellaneous-edge / cross-device сценариев.
2. **Память.** Backprop требует optimizer state + activations (~3x размер модели для Adam). На edge-устройствах full fine-tuning невозможен.
3. **Приватность.** DP-SGD на полных градиентах — известная боль (per-sample clipping миллиардов параметров).

## Идея

**MeZO** (Malladi et al. NeurIPS 2023) обходит проблемы памяти и backprop, заменяя градиент SPSA-style оценкой через две forward-pass. Память равна inference, обновление параметра $\theta$ имеет вид

$$\theta_{t+1} = \theta_t - \eta \cdot \hat\rho_t \cdot z_t,$$

где $z_t$ — Гауссово возмущение, регенерируемое из общего seed $s_t$, а $\hat\rho_t = (\mathcal L(\theta + \epsilon z_t) - \mathcal L(\theta - \epsilon z_t))/(2\epsilon)$ — projected gradient (один скаляр).

**Наблюдение.** Если все клиенты используют один и тот же seed $s_t$, для коммуникации федеративного MeZO нужно передавать всего $(\hat\rho_t^{(i)}, s_t)$ — несколько десятков байт. Это решает проблему коммуникации радикально.

## Что уже сделано в литературе

- **FedKSeed** (Qin et al., ICML 2024, arXiv:2312.06353) — federated MeZO в star-топологии (с центральным сервером), без момента. Передача: $k$ seeds + $k$ scalars.
- **FedZeN** (Maritan et al. 2024, arXiv:2309.17241) — federated ZO с инкрементальной оценкой гессиана, ускорение, но **centralized**.
- **Ferret** (Shu et al. 2024, arXiv:2409.06277) — first-order federated full-parameter tuning with shared randomness. Не ZO.

## Что мы добавляем

**D-MeZO-N** = **D**ecentralized + **MeZO** + **N**esterov:

- **Peer-to-peer** (не star), с произвольной mixing matrix $W$.
- **Nesterov-style** ускорение через explicit velocity buffer.
- **Convergence теорема** в non-convex setting с $r(H)$ вместо $d$ (адаптация Koloskova 2020 + Malladi 2023).

Этого сочетания трёх свойств в литературе нет.

## Целевая площадка публикации

Первый цикл: workshop NeurIPS 2026 (FedFM, OPT, или Federated Learning) — низкий риск, быстрая обратная связь.

Второй цикл (если результаты сильные): main track ICML 2027 или NeurIPS 2026 spotlight.

## Целевая модель

**Qwen3-4B** (Apache 2.0, стандартный трансформер, 8 GB FP16) как основной target.

Upgrade: Qwen3-8B для scaling-кривой. Опционально: Qwen3.5-4B для проверки MeZO на gated-deltanet архитектуре — это **дополнительный потенциальный научный вклад**.

## Compute

Google Colab Pro+ с RTX PRO 6000 Blackwell (96 GB, 5-е поколение Tensor cores, GDDR7). Бюджет 600 compute units/месяц.
