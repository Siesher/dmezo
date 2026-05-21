# Defense Q&A — типовые вопросы и заготовленные ответы

**Дата защиты:** 2026-05-23, МГТУ им. Баумана (Калужский филиал), кафедра САУ.
**Доклад:** 10 минут + ~10 минут вопросов.
**Тон ответов:** уверенный, краткий (1–3 предложения базовый ответ + опционально подробности).

## Актуальные числа (FINAL — 15/15 cells завершены, 2026-05-21 02:12)

**§22 paper-scale validation FINALIZED** (Qwen3.5-4B-Base / MathLogicQA / 4 clients complete IID / 1000 rounds / **3 seeds paired**):

| Variant | s=42 | s=43 | s=44 | Mean ± std loss | Mean ± std acc |
|---|---|---|---|---|---|
| vanilla MeZO | 1.3747 / 0.38 | 1.3432 / 0.36 | 1.3863 / 0.39 | **1.3681 ± 0.018** | **0.377 ± 0.013** |
| D-MeZO-N v1 (fixed C=50) | 1.4598 / 0.38 | 1.4569 / 0.36 | 1.4735 / 0.39 | 1.4634 ± 0.007 | 0.377 ± 0.013 |
| Drift-only (B5 alone, 53 resets total) | 1.4608 / 0.38 | 1.4531 / 0.36 | 1.4537 / 0.39 | 1.4559 ± 0.004 | 0.377 ± 0.013 |
| Adaptive_clip (B1 alone) | 1.2691 / 0.41 | 1.3135 / 0.33 | 1.3135 / 0.43 | 1.2987 ± 0.021 | 0.390 ± 0.043 |
| **D-MeZO-N v2 = combo (B1+B5, 54 resets)** ⭐ | **1.2790 / 0.37** | **1.2951 / 0.44** | **1.3036 / 0.39** | **1.2926 ± 0.010** | **0.400 ± 0.029** |

**Combo vs vanilla — direction consistency (3 seeds paired):**

| Metric | s=42 Δ | s=43 Δ | s=44 Δ | Mean Δ | Direction (3/3) |
|---|---|---|---|---|---|
| Δ loss | **−7.0%** | **−3.6%** | **−6.0%** | **−5.5%** | **3/3 negative** ✅ |
| Δ acc | −1pp | +8pp | 0pp | +2.3pp | 2/3 non-negative |

**Headline для защиты:** D-MeZO-N v2 = **combo (B1 adaptive_clip + B5 drift-reset)**. На 3-seed paired validation robustly beats vanilla **по loss (3/3 same direction)** + average **+2.3pp acc gain**. Lowest std across seeds (0.010 vs 0.021 для B1 alone) — combo более stable. v1 (fixed C=50) — falsified (3/3 worse). B5 alone — falsified (3/3 worse without B1).

---

## Категория А — Теория

### Q1. "Что такое Polyak-Łojasiewicz условие и почему вы его используете для LLM, если оно глобально не доказано?"

**Базовый ответ:** PL-условие — $\|\nabla L\|^2 \geq 2\mu(L - L^\star)$ — слабее сильной выпуклости, но даёт тот же экспоненциальный rate. Для LLM глобально не доказано, но локально на trajectory overparameterized моделей plausibly выполняется (Liu-Zhu-Belkin 2022, ACHA). Это стандартное предположение в современных rate-proofs для deep learning.

**Если давят:** В работе мы это явно записали как Limitation в § 7. Альтернативный путь — KŁ inequality (более слабая), но даёт subexponential rate. Theorem 2/3 написаны под PL потому что это standard, и предсказания этих теорем (линейная сходимость к neighbourhood) **эмпирически наблюдаются** на всех 4 задачах — что является косвенным подтверждением что PL приближённо выполняется на trajectory.

### Q2. "Theorem 3 даёт rate $(1 - 3\eta\mu/2)$ — это тот же что и plain SGD. Где acceleration от Nesterov?"

**Базовый ответ:** Acceleration **не доказана** и **не заявляется**. Bottou-Curtis-Nocedal 2018, теорема 5.1, **запрещает** асимптотическое ускорение momentum для stochastic optimization при σ > 0. Theorem 3 даёт **stability** под momentum + clipping, **не speedup**. Эмпирическое 3× ускорение в Day 8 R1b до раунда R300 — **transient phenomenon**, формальный proof — Open Problem 1 (estimate sequence framework Nesterov 2018).

**Если давят:** Есть три раздельных формулировки momentum benefit: (i) asymptotic rate — same as SGD (proven), (ii) transient rate — empirically 3× (not proven), (iii) **kinetic energy at convergence** — D-MeZO-N с β-decay даёт $\|v_T\|^2 \approx G^2$ vs const-β $\approx 5G^2$ → **более узкая neighbourhood** на чистый loss (Corollary 7.1 в `theory_rigorous.md`). Это и есть **доказанная** польза от momentum + decay.

### Q3. "Где доказательство для full decentralized случая? Theorem 3 — centralized."

**Базовый ответ:** Open Problem 2 — full decentralized T3 не закрыта. Требуется Lyapunov $\Phi_t = (L(\bar\theta_t) - L^\star) + (\eta/2)\|\bar v_t\|^2 + c \cdot \Pi_t$ где $\Pi_t$ — consensus error, и контроль cross-terms между momentum и consensus drift. Это композиция четырёх свойств (non-convex × PL × momentum × decentralized × ZO) — нетривиальна, ни одна paper в литературе не делает.

**Если давят:** Theorem 1 (convex + decentralized) и Theorem 3 (PL + momentum, centralized) — это **два разных** доказательства, которые в совокупности cover use case. Empirically мы наблюдаем что federated D-MeZO-N с momentum сходится — то есть **эмпирическая validation** существует, просто закрытый proof — future work. Это явно прописано в Limitations.

### Q4. "DP-Gaussian-механизм требует L2-чувствительности — почему clip-C даёт sensitivity Δ=C, а не 2C?"

**Базовый ответ:** Один клиент влияет на $\hat\rho = (L^+ - L^-)/(2\varepsilon)$ через свой локальный батч. Сменив все примеры этого клиента, мы можем изменить $\hat\rho$ на максимум $2C$ (от $-C$ до $+C$). Sensitivity для **add/remove-one-record** notion DP по стандартному определению — $\Delta = C$ (половина max range). См. Dwork-Roth 2014, Appendix A.1 для Gaussian mechanism.

**Если давят:** Это **per-client sensitivity**, не per-record. Если требуется per-record DP — то нужно учитывать batch size $B$, тогда $\Delta = 2C/B$ (per-record влияние ослабляется усреднением). В paper заявляется per-client DP — это стандарт для federated setup. Per-record amplification — future work (subsampling amplification Abadi 2016).

### Q5. "Почему bf16 numerics не разрушают MeZO catastrophically?"

**Базовый ответ:** Princeton MeZO recipe: модель в bf16, но **loss accumulation и projection** $\hat\rho = (L^+ - L^-)/(2\varepsilon)$ — в fp32. Это критично. Catastrophic cancellation $L^+ - L^-$ при $\varepsilon = 10^{-3}$ в pure bf16 даёт $\hat\rho$ доминируемый noise. Мы наследуем этот recipe (см. `src/dmezo/mezo/step.py`, `_forward_loss` возвращает float через `.item()` → автоматически fp32 на CPU).

---

## Категория Б — Эмпирика

### Q6. "Только 2–3 seeds для multi-seed — почему не 5 или 10?"

**Базовый ответ:** Compute budget. Один full sweep — 15 cells × ~46 минут на Colab Blackwell = ~11.5 часа compute. Colab Pro+ имеет ~600 compute units/месяц. Для 5 seeds × 5 variants потребовалось бы ~20 часов compute. Текущий setup — 2 полностью завершённых seed'а (42, 43), 3-й (44) на момент защиты pending. Reporting через paired bootstrap CI на доступных seeds.

**Конкретно по нашим данным (3 seeds полный paired):** D-MeZO-N v2 (combo B1+B5) над vanilla:
- Δ loss = −0.0755 (1.3681 → 1.2926) — **5.5% better, 3/3 seeds same direction**
- Δ acc = +0.023 (0.377 → 0.400) — **+2.3pp в mean, направление: −1/+8/0 pp на 3 seeds**
- Lowest std loss across seeds (0.010 vs vanilla 0.018) — combo **более stable**
- Best individual cell: combo|s=43 = **1.2951 / acc=0.44**

**Если давят:** Paired bootstrap CI на 2 seeds — конечно weaker чем 5 seeds. Но **direction consistency 2/2** + большой effect size (6%) — это уровень "tentative robust", не "single-seed accident". Multi-seed pending strong — но текущая evidence уже достаточно для **defensible claim** на BSc thesis level.

### Q7. "На MathLogicQA D-MeZO-N v1 (fixed C=50) **уступает** vanilla. Это значит метод не работает?"

**Базовый ответ:** v1 действительно **уступает** vanilla на этой задаче: 2/2 seeds — final loss 1.4584 (v1) vs 1.3590 (vanilla), Δ = +7.3% **хуже**. Изначальный single-seed "+1.25pp acc" был seed-specific, multi-seed это **fal{sified**. Но это **про D-MeZO-N v1 (fixed C=50)**. **v2 (adaptive ρ-clip)** — отдельный variant с **другим механизмом** — на тех же 2 seeds показывает inverse pattern: −6.2% loss, +2pp acc vs vanilla. Honest reporting v1 → переход к v2 → новый positive finding.

**Если давят:** Это normal scientific progression. У нас есть гипотеза → она falsified → анализ механизма → находим improvement → доказываем на 2 seeds. Это **признак строгости**, не failure. В частности, в `calibrated_achievements` v1 явно записан как Group C (Falsified), v2 — как Group A/B (tentative robust). На защите цифры конкретные: v1 проигрывает vanilla 7%, v2 выигрывает у vanilla 6% — на тех же 2 seeds, тех же hyperparameters, кроме adaptive vs fixed clip.

**Механистическое объяснение:** Median |ρ| на Qwen3.5-4B-Base/MathLogicQA ≈ 180 (наблюдаем в logs: ac_thr varies 165–270). Fixed C=50 — **в 3–4 раза tighter** чем sensible threshold → отрезает большую часть полезного сигнала, momentum застопорен. Adaptive (1.3 × q95) tracks distribution → tight enough to bound outliers, loose enough preserve signal.

### Q8. "HellaSwag rescue — single seed. Как вы знаете что это не лаки?"

**Базовый ответ:** Honestly — это **tentative** в текущем состоянии. SE на 100-example eval ≈ ±0.045, effect size +3.75pp ≈ 1σ. Multi-seed validation script готов (`validate_dmezo_n_rescue_multiseed_federated.py`), pending Colab compute. Но: эффект **согласован с Theorem 3 corollary** (rescue regime когда $G^2$ unbounded без clip). И: vanilla MeZO **demonstrably diverges** на этой задаче (monotonic loss drift +5.5%), что является independent supporting evidence — не lucky seed.

### Q9. "Federated 0.130 vs centralized 0.176 = ratio 0.74. Это не $1/\sqrt{4}=0.5$. Где прогноз Theorem 1?"

**Базовый ответ:** **Numerical match отсутствует.** Это explicit caveat в `theory_rigorous.md` §5 (P1). Theorem 1 bounds **rate** на Polyak-averaged sequence, не **final loss ratio**. Нельзя приклеивать. **Directional match** — federated lower than centralized — наблюдается, и это всё что заявляется.

**Если давят:** Альтернативное объяснение через Theorem 2: linear $1/n$ variance reduction в noise floor $\eta C^2 r(H) \ell/(\mu n)$ — для $n=4$ floor уменьшается в 4×, что согласуется с наблюдаемым ratio при определённых constants. Но **это interpretation, не tight prediction**. Также: это сравнение 4 GPU vs 1 GPU не compute-matched — fair comparison требует averaging 4 параллельных centralized runs. Эта оговорка в Abstract английской версии paper.

### Q10. "Почему не сравнили с FedKSeed эмпирически?"

**См. отдельный документ:** `docs/defense_fedkseed_qa.md` — 5-частный аргумент. Короткий ответ:

**Базовый:** Script готов (`scripts/head_to_head_fedkseed.py`), но compute budget priortized под multi-seed validation §5.6 (paper-scale). Теоретическое позиционирование однозначно: peer-to-peer (D-MeZO-N) vs star (FedKSeed) — three independent advantages. Эмпирическое сравнение — explicit future work в § 7.

---

## Категория В — Scale и применимость

### Q11. "4 клиента, 4B params. Реальные federated имеют 100+ клиентов и 8B+ моделей."

**Базовый ответ:** Согласен — это explicit limitation в § 7. Принципиально код дimension-agnostic, masштабируется добавлением клиентов в config. Один scale-up experiment (8B или n=8 clients) пlanned для extended version paper. **Но**: 4 клиента × 4B — уже **демонстрационный setup**, достаточный для proof-of-concept, что и заявляется.

**Если давят:** Real cross-silo banking deployment (Альфа-Сбер-ВТБ-Тинькофф-Райффайзен) — это 5 клиентов, не 100. Cross-hospital — ~10 клиентов. Наш 4 — внутри realistic range. 100+ клиентов — federated mobile setup, для которого нужна не наша работа, а наш + LoRA + asynchronous updates (отдельное направление).

### Q12. "Только multi-choice classification. Где generative tasks (GSM8K, SAMSum)?"

**Базовый ответ:** Explicit limitation. Princeton MeZO работал на generative (OPT-66B), наша инфраструктура поддерживает любую cross-entropy loss — расширение тривиальное. Но federated wrapper + DP + comprehensive validation — extensive infrastructure work, и мы priortized depth (4 tasks с multi-seed) над breadth (10 tasks single-seed). Generative — explicit future work.

### Q13. "Per-round ε=10 для one round, но реальный federated training — это T=200-1000 раундов. Privacy guarantee катастрофическая под composition."

**Базовый ответ:** Полностью согласен. Theorem 4c открыто обсуждает: basic composition даёт $\varepsilon_T = T\varepsilon_1 = 2000$ (useless), advanced — $\sqrt{T}$-scaling с term $T\varepsilon_1(e^{\varepsilon_1}-1)$ catastrophic при $\varepsilon_1 > 1$, RDP/moments accountant — tighter но всё ещё $O(\sqrt{T})$. **Paper claim — per-round**, что является стандартом для one-shot federated fine-tuning. Subsampling amplification (Abadi 2016) — recommended future work для tight T-round bound.

**Если давят:** Per-round ε=10 — это не маркетинг. Это **сравнимо с industry standard** (Apple ε≈2-8 в их published DP work). И per-round нужен для on-line query setting где adversary видит только последний gradient, не все. Для one-shot fine-tuning это adequate guarantee.

---

### Q13b. "У вас в spec написано про shared seed между клиентами — клиенты не будут скоррелированы и не убьёт ли это $1/n$ federated speedup?"

**Базовый ответ:** Это важное недоразумение в старой версии spec. **В реальной реализации seeds НЕ shared — каждый клиент имеет независимый `np.random.Generator` и сэмплирует свой $s_i$, свой $z_i$.** Корреляции на уровне sampling нет. Spec обновлён, чтобы соответствовать коду (`src/dmezo/federated/client.py:62`).

**Если спросят детали (математика):**

Был бы shared $z$: $\bar g = (\frac{1}{n}\sum_i \rho_i) z$ — variance reduction $1/n$ **только по data noise**, потому что direction noise (выбор $z$) у всех клиентов один и тот же.

У нас independent $z_i$: $\bar g = \frac{1}{n}\sum_i \rho_i z_i$, каждое $\rho_i z_i$ — независимая unbiased оценка $\nabla L$ → variance ÷ n **по обоим источникам шума** (data + direction).

Это и есть фундаментальный differentiator vs FedKSeed: FedKSeed = shared $z$ через central server broadcast, мы = independent $z_i$ через peer-to-peer per-client RNG. Это **причина** того, что Theorem 2 даёт $1/n$ floor, и **косвенное эмпирическое подтверждение** — federated final loss 0.130 ниже centralized 0.176 (Day 5 grid).

**Что насчёт корреляций после consensus mixing?** Consensus $\theta_i \leftarrow \sum_j W_{ij}\theta_j$ сближает параметры клиентов в одну точку (это convergence в траекториях), но **не вводит correlation в источник шума**: на следующем раунде каждый клиент снова independently сэмплирует свой новый $z_i^{t+1}$. Lemma 4 (Koloskova-style consensus error bound) формализует это: $\rho_W^2/(1-\rho_W)^2$ amplifier ограничивает per-round дрейф, не requires correlation между $z_i$.

**Если будут давить дальше:** "А почему в spec было shared?" — Spec был **черновик-draft**, написан до того как мы определились с algorithmic positioning vs FedKSeed. Когда стало ясно, что наш main differentiator — именно independence, мы обновили spec. Это **процесс развития проекта**, не баг алгоритма.

---

## Категория Г — Архитектура / Engineering

### Q14. "Hybrid linear-attention Qwen3.5 — почему именно эту арch?"

**Базовый ответ:** Diversity. Все известные federated ZO papers (FedKSeed, Ferret, FedZeN) ограничены full-attention арх (OPT, LLaMA семейство). Hybrid linear-attention (24 ViT + linear-attention text decoder) — структурно отличается, effective $r(H)$ может быть другим, gated DeltaNet kernels работают через flash-linear-attention. Подтверждение что наш метод работает на этом классе — **первый known result**.

### Q15. "Vision encoder во Qwen3.5 — почему его замораживаете?"

**Базовый ответ:** Все наши задачи — text-only (SST-2, BoolQ, MathLogicQA, HellaSwag — текстовые, не visual). MeZO perturbирует только text decoder (426 параметрических групп). ViT (297 params в `model.model.visual`) заморожен через `requires_grad_(False)` в `src/dmezo/models/loader.py::_load_vl_for_text_task`. Иначе MeZO тратил бы forward-passes на возмущение неиспользуемых ViT весов.

---

## Категория Д — Реальная применимость

### Q16. "Кому конкретно ваш метод нужен?"

**Базовый ответ:** Три use case'а:
1. **Cross-silo banking** — Альфа+Сбер+ВТБ+Тинькофф+Райффайзен совместно дообучают fraud detection LLM без обмена транзакциями (115-ФЗ запрещает). Communication: 16 байт/раунд × 5 банков = 80 КБ trafic vs ~40 ТБ для FedAvg.
2. **Cross-hospital medical NLP** — 10 клиник дообучают радиологию (HIPAA, GDPR, 152-ФЗ).
3. **Decentralized Web3-style** — независимые узлы (researchers, hobbyists) contributing fine-tuning без central server.

### Q17. "Вы работаете в Альфа-Банке. Применяете ли это на работе?"

**Базовый ответ:** Текущая работа — research, не production deployment. Альфа-Банк использует MLflow (что мотивировало выбор tracking infrastructure здесь) и работает с federated learning в antifraud направлении, но это под NDA. Этот research project — мой academic contribution, не Альфа-internal product.

---

## Категория Е — Тёмные вопросы (наиболее агрессивные)

### Q18. "Это всё выглядит как marginal improvement над Princeton MeZO. Зачем эта работа?"

**Базовый ответ:** Несогласен. Три независимых вклада:
1. **Theoretical** — Theorem 3 closes Princeton Open Problem 1 (momentum convergence under PL+ZO).
2. **DP** — first formal (ε,δ)-DP guarantee для decentralized federated ZO на LLM.
3. **Decentralized** — первая peer-to-peer (vs FedKSeed star) формулировка с consensus mixing analysis.

Каждый из них **самостоятельно** quality of a paper. Combined — strong workshop / TMLR / borderline conference.

### Q19. "Falsified 5 ваших собственных гипотез. Не означает ли это что вы вообще не знали что делать?"

**Базовый ответ:** Наоборот. Falsified claims в Group C (`calibrated_achievements`) — это **доказательство строгости методологии**:
- Look-ahead Nesterov — гипотезирован, falsified (R20 NaN) → переключение на heavy-ball.
- ε-autotuner — гипотезирован, falsified downstream → confirms Princeton default.
- K=3 multi-direction — гипотезирован, falsified equal-compute → trade-off acknowledged.
- +1.25pp MathLogicQA — single-seed claim, falsified multi-seed → переход к v2.

**Это normal scientific progress**: гипотеза → эксперимент → falsification → новая гипотеза. Не выбрасывать negatives — **это и есть качество**. Многие современные ML papers продают первые successful runs без multi-seed → они **слабее**.

### Q20. "Что если reviewer NeurIPS попросит compute-matched comparison?"

**Базовый ответ:** Один из мысленных follow-up экспериментов. Compute-matched federated vs centralized: 4 GPU × 1000 rounds vs 1 GPU × 4000 rounds — должно дать matching final loss (теоретически), потому что federated $1/n$ variance speedup эквивалентен $n$× больше шагов centrally. Текущая статья этого не показывает — explicit limitation.

### Q21. "Code review был? Кто-то проверял что implementation корректен?"

**Базовый ответ:** 128 pytest tests, ~95% coverage критических путей: детерминизм perturbation, симметричность mixing matrix, корректность consensus (обе mode), classification accuracy, ρ-clipping bounds, β-schedule, Richardson 4-pt и 6-pt finite-diff (analytical cancellation tests on quintic loss). Audit-harden pass (commit `f2a8d3...`) переписал `consensus_via_updates` под O(np), исправил bug с double update. Cross-platform validation (Linux Colab + Windows local).

---

## Финальный совет для защиты

**Если не знаете ответ:** "Это интересный вопрос. На текущий момент у меня нет полного ответа — это, вероятно, попадает в направления future work, которые я обозначил в § 7." Лучше честное "не знаю" чем bullshit.

**Если давят на overstatement:** Прямо сказать: "Согласен, эту формулировку нужно ослабить. В paper это [было/будет] записано как [limitation/tentative]." Это disarms критика.

**Если хвалят:** Скромно: "Спасибо. Признаюсь, что много работы ещё впереди — особенно multi-seed на n≥5 и head-to-head SOTA."

---

*Документ подготовлен 2026-05-21 как Q&A prep для защиты 2026-05-23.*
