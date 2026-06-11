# Defense speech — D-MeZO-N v2 · 2026-05-23

**Назначение:** дословный текст для 10-минутного доклада. 13 слайдов главного deck'a.
**Бюджет:** 600 сек, 110–130 слов/мин в спокойном темпе → ~1400 слов всего.
**Стиль:** разговорный, в первом лице («мы»), с явными переходами между слайдами.
**Pause markers:** "(пауза)" — короткий вдох; "(пауза-смысл)" — 1–2 сек перед ключевой фразой.

**Совет:** прочитать дома × 2 вслух с таймером. На защите смотреть на reviewer'а,
а не на слайд — слайды дублируют, а не нести основной смысл.

---

## Slide 1 · Title — 15 сек

> Здравствуйте. Меня зовут Максим Сухацкий, и сегодня я представляю **D-MeZO-N v2** —
> распределённый gradient-free оптимизатор для дообучения больших языковых моделей.
> Это совместная работа в рамках Трека 3 программы «Мультиагентные технологии и
> роевой интеллект» Сириуса 2026. Со мной — Владислав Щеглов, ответственный за перенос
> на Vision Transformers, и Олег Лысенко, делавший пробный side-эксперимент Swarm-MeZO.

> **(пауза-смысл)** На слайде четыре числа — это вся работа в одном кадре.
> К ним вернёмся в конце.

> *(переход)* Начну с того, **почему классический backprop с AdamW упирается в стену.*

---

## Slide 2 · Backprop wall — 50 сек

> Когда мы хотим дообучить, скажем, Qwen3.5 на четыре миллиарда параметров —
> память состоит из **трёх источников**. **Веса** — восемь гигабайт в bf16, это
> неизбежно. **Активации** — все промежуточные тензоры для backward-прохода,
> ещё около шестнадцати гигабайт. И **optimizer state** — у AdamW это первый и второй
> момент в fp32, ещё **тридцать два гигабайта**. (пауза)

> Итого fine-tune AdamW требует около **пятидесяти шести гигабайт** — H100 на восемьдесят
> тянет впритык, RTX 4090 на двадцать четыре — не тянет совсем. Edge-устройства —
> только инференс.

> **(пауза-смысл)** Принстонский MeZO 2023 года решил эту проблему: forward-only
> оптимизатор, два прохода вместо backward, оценка градиента — один скаляр $\hat\rho$.
> Активации не хранятся. Optimizer state — один скаляр. Память fine-tune становится
> **равной памяти inference**.

> *(переход)* Но MeZO решает только **один из трёх** federated constraint'ов.

---

## Slide 3 · Three constraints — 55 сек

> В **распределённом** дообучении мы упираемся не в один, а в **три** ограничения
> **одновременно**. Первое — **communication**: FedAvg на 4B модели отправляет
> восемь гигабайт на каждом раунде на каждого агента, **петабайты трафика** на
> сотнях раундов. Второе — **privacy**: regulators 152-ФЗ, GDPR требуют формальной
> $(\varepsilon, \delta)$-DP гарантии, raw градиенты этого не дают. Третье — **memory**,
> уже обсудили.

> На слайде таблица critic'ов. **FedAvg** проваливает все три. **Princeton MeZO**
> закрыл память, но он single-node, без communication и privacy. **FedKSeed** —
> ближайший конкурент 2024 года, ICML — закрыл communication, но без момента и
> без DP. И это его, между прочим, **Open Problem номер 1** — сходимость момента
> для ZO.

> **(пауза-смысл)** **D-MeZO-N v2 закрывает все три constraint'а в одном методе** —
> шестнадцать байт на раунд, формальная $(\varepsilon, \delta)$-DP и Theorem 1
> для момента.

> *(переход)* И теперь **почему** именно эта работа актуальна **именно сейчас**.

---

## Slide 4 · Актуальность — 45 сек

> Сходятся три тренда. **Edge-LLM**: модели 2–8 миллиардов параметров теперь
> запускаются на ноутбуках и смартфонах — Qwen3, Phi-3, Llama-3.2. Inference работает,
> fine-tune через backprop — нет. **Регулирование**: privacy перестал быть опцией.
> GDPR, 152-ФЗ, медицина, банкинг требуют формальной DP. **Bandwidth**: compute
> дешевеет быстрее, чем трафик, поэтому 8 ГБ на раунд — реальное узкое место.

> **(пауза-смысл)** До 2026 года **ни одна** работа не закрывала все три constraint'а
> одновременно — с моментом и формальной сходимостью.
> Это окно, в котором мы работаем.

> *(переход)* На что мы опираемся — три **фундаментальные опоры**.

---

## Slide 5 · Foundation — 45 сек

> **MeZO** Малладти 2023 — даёт нам forward-only оценку градиента и память $O(d)$.
> **Distributed SPSA** Ерофеевой, Граничина, Сергеенко из IEEE TAC 2026 года — даёт
> $N$-агентный consensus mixing через матрицу $W$ и инструмент spectral gap для
> анализа сходимости. **Russian ZO school** — Гасников, Безносиков, обзор 2023 года —
> даёт $\ell_2$-smoothing two-point estimator и bias/variance bounds, которые лежат
> в основе доказательства нашей Theorem 1.

> **(пауза-смысл)** Справа — таблица **differentiation от FedKSeed**, нашего closest
> competitor. Compression у обоих ~16 байт — но это **не** differentiation.
> Differentiation в **топологии** (peer-to-peer vs star), **independent z_i**
> (vs shared seed), **моменте** с формальной сходимостью, и **DP-гарантии**.

> *(переход)* Теперь сам **алгоритм**.

---

## Slide 6 · v2 Algorithm — 60 сек

> Семь шагов, два новых механизма, один скаляр в обмене. На каждом раунде клиенты
> сэмплируют общий seed $s_t$, генерируют $z$, делают два forward-прохода с возмущениями
> $\theta \pm \varepsilon z$, считают $\rho_i$ — оценку проекции градиента на $z$.
> (пауза)

> Дальше — **два механизма v2**. **B1, adaptive clip**: вместо fixed $C=50$ из v1 мы
> ведём экспоненциальное скользящее среднее модуля $\rho$ и обрезаем по $C_t = 1.5\,\hat c_t$.
> **B5, drift reset**: если клиент уходит дальше $10\|\theta_0\|$ от consensus —
> обнуляем velocity. В наших экспериментах сработало 54 раза за 1000 раундов.

> Затем — heavy-ball момент с **$\beta$-decay** линейно от 0.9 до 0, обмен **парами
> $(s_t, \tilde\rho_i)$** между соседями — это 16 байт на ребро, — и **consensus
> mixing** через $W$.

> **(пауза-смысл)** Главное: между клиентами **никогда** не передаются веса
> или массивы. Только seed плюс скаляр. Это и есть наша 16-байтная коммуникация.

> *(переход)* Ключевой механизм — $\rho$-clip — заслуживает отдельного слайда.

---

## Slide 7 · ρ-clip dual-use — 55 сек

> $\rho$-clip играет **две роли одновременно**.

> **Первая** — стабилизатор момента. Без клипа heavy-ball при $\beta=0.9$
> усиливает дисперсию в $1/(1-\beta^2) \approx 5.3$ раза. На раунде 140 он
> взрывается — мы это видели в Day 6 эксперименте. **Клип ограничивает
> $\|v\|$ → Lyapunov $V_t$ корректно убывает** (Theorem 1).

> **Вторая** — **L₂-sensitivity** для DP. Тот же порог $C$, который ограничивает
> момент, **автоматически** даёт ограниченную чувствительность к данным:
> $\Delta_2 \le 2C$. Один Gaussian noise $\mathcal{N}(0, \sigma^2)$ — и получаем
> $(\varepsilon, \delta)$-DP по Theorem 2.

> **(пауза-смысл)** Это **не два независимых трюка** — это **одно архитектурное
> решение**, которое **бесплатно** закрывает privacy-constraint.

> *(переход)* Формальные результаты — две теоремы на одной странице.

---

## Slide 8 · Theorems T1 + T2 — 60 сек

> **Theorem 1 — сходимость**. Мы строим **Lyapunov** $V_t = (\mathcal{L}_t - \mathcal{L}^*)
> + \frac{\eta}{2}\|v_t\|^2$ — gap по loss плюс kinetic energy от момента. Под условием
> Polyak-Łojasiewicz доказываем **линейную сходимость** к noise floor $\sim G^2/\mu$.
> (пауза) **Что нового vs Малладти 2023:** это closes **Open Problem 1** —
> heavy-ball + клип + $\beta$-decay не ломают сходимость.

> **(пауза-смысл)** **Важная честная рамка**: Bottou-Curtis-Nocedal 2018 запрещают
> асимптотическое ускорение от моментов для stochastic non-convex. Поэтому наша
> формулировка — **stabilizes, не accelerates**. Эмпирически наблюдаем transient
> speedup до раунда 300.

> **Theorem 2 — privacy**. Та же $C_t$, что в Theorem 1, даёт $L_2$-sensitivity
> $\Delta_2 \le 2 C_t$. Добавляем Gaussian noise с $\sigma = 2C \sqrt{2 \ln(1.25/\delta)}/\varepsilon$ —
> и получаем формальное $(\varepsilon, \delta)$-DP per round.

> Внизу — **Open Problem 2, наша**: полное decentralized $\rho(W) < 1$ + момент.
> Лyapunov для consensus error плюс heavy-ball ещё не выведен.

> *(переход)* А теперь — **главный эмпирический результат**.

---

## Slide 9 · Main results — 75 сек

> **Qwen3.5-4B-Base**, **MathLogicQA** на русском, четыре клиента, complete topology,
> IID partition. Три seed'а, paired comparison.

> **Слева** — наш ablation. Vanilla MeZO даёт loss 1.368. **v1 с fixed clip 50** —
> 1.463, на 7 процентов хуже vanilla на всех трёх seed'ах. (пауза) Почему? Потому что
> median $|\rho|$ на 4B-Base — около **180**, а не 35 как на маленьких моделях.
> Fixed clip = 50 обрезает полезный сигнал. **Single-seed «+1.25 pp acc» от v1
> мы falsified** на multi-seed.

> **Adaptive clip один** — 1.299, минус 5.1 процента, **3 из 3 seeds**. **Drift reset
> один** — не помогает. **Combo B1+B5** — 1.293, **минус 5.5 процента**, lowest
> standard deviation. **(пауза-смысл)** Это **первое paper-scale multi-seed
> validated** улучшение D-MeZO-N strictly над vanilla MeZO.

> **Справа** — head-to-head с FedKSeed. **3 метода × 3 seed'а × 500 раундов**.
> v2 бьёт **vanilla** на **−8.8 процентов** loss, **3 из 3**. v2 бьёт **FedKSeed**
> на **−9 процентов** loss и **+5.3 pp accuracy**. FedKSeed на этой cross-arch
> задаче даже **проигрывает vanilla** на 2 из 3 seeds.

> **Честная оговорка по accuracy**: CI на $\Delta$ — directional, нижняя граница на
> нуле. Loss-claim strict, accuracy-claim directional.

> *(переход)* Когда применять и когда **не** применять.

---

## Slide 10 · Applicability — 60 сек

> Три части слайда. **Слева** — DP frontier. Sweep по $\sigma$ от 0.5 до 50, что
> соответствует $\varepsilon$ от 4 до 378. **Frontier практически плоский** —
> не cliff. На **$\varepsilon=10$, типичный banking compliance threshold,
> мы платим всего +6.2 процента utility cost**. (пауза)

> **Центр** — три режима одного recipe на разных задачах. **SPEEDUP** на SST-2:
> минус 6.5 процентов loss. **RESCUE** на HellaSwag: vanilla diverges на минус 2.5 pp,
> v2 converges на плюс 3.75 pp — net plus 6.25 pp. **SAFE-TRACK** на MathLogicQA — наш
> headline. **Один recipe** даёт **три разных поведения** в зависимости от задачи.

> **Справа** — **honest negatives**. Пять штук. v1 falsified, Nesterov $\beta=0.9$
> const blow-up R140, short-horizon vanilla wins, K=3 Pareto trade-off, $1/\sqrt{B}$
> CLT не выполняется. **(пауза-смысл)** Negatives **усиливают** работу, а не
> ослабляют — это признак зрелости.

> Внизу — синтез **когда использовать**: federated 4B+, long-horizon, memory или
> privacy-bound. **Когда не использовать**: short-horizon, cold-start vision, K=3
> equal-compute.

> *(переход)* Перенос на vision.

---

## Slide 11 · ViT vision transfer — 45 сек

> ViT-Large, head-only refinement, CUB-200-2011 — 200 видов птиц fine-grained.
> Warm-start от sklearn linear probe на 86.23 процента, пять тысяч MeZO шагов.

> **Overall — parity**: vanilla 86.09, v2 86.16. Дельта плюс 0.07 pp **в noise band**.
> (пауза) **Но** — есть **тридцать шесть случаев** на тестовом наборе из 5794, где
> vanilla noise-driven step **перевернул** правильную метку на похожий вид, а v2
> stabilization **сохранила** правильное предсказание. Показываю три самых ярких.

> **Indigo Bunting vs Blue Grosbeak** — два sapphire-blue songbirds. **Ringed vs
> Belted Kingfisher**. **Red-cockaded vs American Three-toed Woodpecker**. Все случаи —
> closely-related species, confidence 45–57 процентов, **borderline**.

> **(пауза-смысл)** **Честная рамка**: это **не aggregate win**. Это empirical
> confirmation **Theorem 1**: combo **stabilizes**, не accelerates. На borderline
> решениях это и делает разницу между правильным и неправильным предсказанием.

> Справа — Flowers102 sanity. Parity ✓, stability под 10× LR ✓. Cold start — known
> MeZO limit Princeton paper, не v2-specific.

> *(переход)* Что **ещё** пробовали — Swarm-MeZO.

---

## Slide 12 · Swarm-MeZO — 40 сек

> Параллельная ветка — **репутационный consensus** поверх MeZO. Веса агрегации
> $W_{ij} \propto e^{\beta r_j}$ — softmax по fitness. $\beta=0$ — FedAvg,
> $\beta \to \infty$ — global best из PSO.

> **Что подтвердилось**: $1/N$-закон с slope минус 0.996, seed-bank ломает $1/N$ —
> это **прямое обоснование** independent-$z_i$ design choice в D-MeZO-N v2.
> Spectral gap матрицы $W$ становится Lyapunov-инструментом для Theorem 1.

> **Что не взлетело**: **$\beta$-окно не переносится на LLM**. На синтетике
> окно открывается при $\beta=0.1$. На RoBERTa плюс SST-2 — **ничья при низких
> $\beta$ и каскадный отказ минус 8 пунктов при $\beta=10$**. Перепроверено
> по четырём осям.

> **(пауза-смысл)** Honest negative. Поэтому в **D-MeZO-N v2** — doubly-stochastic
> $W$, $\beta=0$, ускорение приходит **не** из репутации, а из **Nesterov плюс
> adaptive clip**.

> *(переход)* Что вошло в работу — финальный слайд.

---

## Slide 13 · Contributions + Q&A — 35 сек

> **Шесть contributions D-MeZO-N v2.** C1 — federated wrapper с independent $z_i$.
> C2 — peer-to-peer gossip. C3 — heavy-ball с dual-use $\rho$-clip. **C4 — Theorem 1,
> closes Princeton Open Problem 1.** **C5 — Theorem 2, $(\varepsilon, \delta)$-DP.**
> C6 — первый MeZO test на hybrid linear-attention LLM.

> Плюс **§22 multi-seed headline** и **head-to-head FedKSeed** — оба со звёздами.
> Vision Transformer — parity nuanced. Swarm-MeZO дал **сильный positive** S1 и
> **honest negative** S2.

> **(пауза-смысл)** **Три числа, которые остаются с вами:**
> **минус 5.5 процентов loss на трёх seed'ах. Шестнадцать байт на раунд.
> Плюс 6.2 процента — цена $\varepsilon=10$ DP.**

> *(финал)* Спасибо. Готов отвечать на вопросы.

---

## Q&A — топ-10 anticipated вопросов

### 1. «Почему именно MeZO, а не LoRA / QLoRA?»

> Memory-эквивалентность с inference. LoRA убирает optimizer state, но активации
> для backward всё равно остаются. MeZO убирает и то, и другое — fine-tune memory =
> inference memory. Плюс LoRA не federated, и DP-гарантий не даёт.

### 2. «Чем отличаетесь от FedKSeed?»

> FedKSeed — star topology, shared K-seed pool, без момента, без DP.
> Мы — peer-to-peer gossip, **independent $z_i$ per client**, heavy-ball с
> формальной сходимостью, $(\varepsilon, \delta)$-DP. Compression у обоих ~16 байт —
> это **не** наша differentiation, наша differentiation в topology, momentum, DP.

### 3. «Independent $z_i$ — почему это важно?»

> FedKSeed shares one seed → все клиенты возмущают в одном направлении →
> variance reduction только по data noise. Мы даём каждому клиенту независимый
> $z_i$ → variance делится на $n$ **по обоим источникам** — data **и** direction.
> Это $1/n$ federated speedup в Theorem 1.

### 4. «Почему Δ accuracy не strict significant?»

> CI на $\Delta$ accuracy — [+0.000, +0.040] по бутстрапу, нижняя граница касается
> нуля. Это **directional**, не strict statistical. **Loss-claim strict** — 3/3
> same direction на минус 5.5 процентов. Honest framing — обе границы заявляем
> точно как есть.

### 5. «$O(1/T^2)$ ускорение от момента?»

> Нет. Bottou-Curtis-Nocedal 2018 Theorem 5.1 — асимптотическое ускорение от моментов
> запрещено для stochastic non-convex. Наш T3 даёт тот же rate, что plain SGD под PL.
> **Stabilizes, не accelerates.** Эмпирически — transient speedup до R300.

### 6. «Russian school — кто такие Гасников и Безносиков?»

> Гасников — научный руководитель направления distributed/randomized optimization
> в России, h-index больше 25. Безносиков — его ученик, IMP Сколтех, ICML/NeurIPS papers.
> Их обзор 2023 года — arXiv:2211.13566 — даёт $\ell_2$-smoothing bounds, которые
> мы используем в proof T1. Цитировано в §A0.5 нашего литобзора.

### 7. «Open Problem 2 — что именно открыто?»

> Полное decentralized: только $\rho(W) < 1$ без assumption complete topology, плюс
> heavy-ball момент. Lyapunov для consensus error плюс kinetic energy ещё не выведен.
> Это **наша open problem**, не Princeton'овская.

### 8. «Multi-seed на vision не делали?»

> Нет, на vision только single-seed (seed=42). Multi-seed на vision — это **post-defense
> follow-up**, в roadmap. На vision наш claim — **parity overall** плюс **specific
> stabilization mechanism** на borderline cases. Это и не нуждается в multi-seed
> для defensible claim.

### 9. «Cold start на vision — это поражение?»

> Это **known MeZO inherent limitation**, не v2-specific. Princeton MeZO paper §6
> прямо документирует: MeZO designed для refinement existing models, не для training
> from scratch. На vision мы это reproduce'ли — это **confirmation** of expected
> behavior, не provoking new failure mode.

### 10. «Когда практически применять D-MeZO-N v2?»

> Sweet spot: federated 4B+ модель, long-horizon $T \ge 500$ rounds, memory или
> privacy-bound сценарий. **Banking compliance** — типичный сценарий: $\varepsilon=10$
> threshold, distributed training через несколько банков, нельзя обмениваться сырыми
> весами. Edge-LLM federated — другой сценарий.

---

## Подсказки для произнесения

**Темп:** 110–130 слов в минуту. Не торопиться, дать аудитории успеть переварить цифры.

**Ключевые акценты — выделять голосом:**
- «**Open Problem 1**» (slide 8)
- «**первое paper-scale multi-seed validated**» (slide 9)
- «**три числа, которые остаются с вами**» (slide 13)

**Где сделать паузу 2–3 сек:**
- После headline числа на каждом слайде
- Перед **«stabilizes, не accelerates»** (slide 8)
- Перед «**это не aggregate win**» (slide 11)

**Что НЕ говорить (red flags):**
- «v2 wins on vision» (parity не win)
- «strict acc improvement» (acc directional)
- «$O(1/T^2)$ acceleration» (forbidden)
- «FedKSeed broken» (underperforms on this task)
- «10⁹× compression vs FedKSeed» (compression одинаковая)

---

## Что взять с собой на защиту (физически или в файле)

- Печатный текст этого doc (1 копия в кармане, 1 на трибуне)
- `presentation/standalone.html` на ноутбуке (single file, открыть → F11 fullscreen)
- Запасной флешка с standalone.html (если ноут умрёт)
- Воду
- Часы / телефон с секундомером

**Перед заходом:** прочитать вслух разделы Slide 1 + Slide 13 (открытие и закрытие — самые
важные). Замерить общее время — должно быть **9:30–10:30 минут**.

---

*Speech written 2026-05-22 для защиты 2026-05-23 на кафедре САУ МГТУ им. Баумана
(Калужский филиал). Структура соответствует `presentation/index.html` (13 main слайдов).*
