# Upgrade Roadmap: D-MeZO-N → Publishable

**Контекст:** Текущая работа находится в состоянии "strong workshop / TMLR / undergraduate-thesis quality". Чтобы продвинуть до **main-track conference** (NeurIPS / ICML / ICLR), требуется закрыть 4 категории gaps:

- **A**. Теоретические разрывы (acceleration proof, full decentralized T3)
- **B**. Эмпирические разрывы (multi-seed, head-to-head, scale)
- **C**. Framing & overstatement removal (paper-writing)
- **D**. Engineering (Optional — reproducibility hardening)

Документ структурирован по **ROI** (impact / effort).

---

## A. Теоретические upgrades

### A.1 [HIGH ROI] Снять claim "acceleration" — переписать framing

**Проблема:** в `paper_en.md` / `paper_ru.md` C4 формулируется как "accelerated variant". Theorem 3 проверена в `docs/theory_rigorous.md` — даёт **тот же** asymptotic rate как plain SGD под PL. Acceleration **не доказана**.

**Что делать:**
1. В `paper_en.md` § Method: заменить "Nesterov-style acceleration" на "Nesterov-style stabilization with bounded variance".
2. В § Theorem 3 statement: explicit note: "Rate matches plain SGD; momentum reduces transient variance through smoothing, not asymptotic rate."
3. В § Experiments: Day 8 R1b "3× speedup" → "**transient** speedup in first 300 rounds; asymptotic rate matches plain D-MeZO."
4. В Discussion: cite Bottou-Curtis-Nocedal 2018 Theorem 5.1 (momentum doesn't accelerate SGD when σ > 0).

**Effort:** 2 часа. **Impact:** убирает major reviewer concern, делает paper защищаемым.

### A.2 [HIGH ROI] Прокальник "federated beats centralized" — пересмотр C2/P1

**Проблема:** "$0.74 \approx 1/\sqrt{4}$" — incorrect matching. $1/\sqrt{4}=0.5 \neq 0.74$. Это "ratio of final losses", не "rate", не сопоставимо с теоретическим $1/\sqrt{nT}$.

**Что делать:**
1. В Theorem 1 § Predictions: убрать "matches $1/\sqrt{n}$ const" — оставить только "directional match: federated lower than centralized."
2. Объяснить альтернативно через T2: linear $1/n$ variance reduction in noise floor. $0.130/0.176 = 0.74$ объясняется через $\eta C^2 r(H) \ell/(\mu n)$ floor reduction with $n=4$.
3. В § Discussion: явный disclaimer — "fair compute-matched comparison: 4 GPUs (federated) vs 1 GPU (centralized) — not the same compute budget. Effect could be reproduced as 4 independent centralized runs (averaged)."

**Effort:** 1 час. **Impact:** убирает potential reviewer "math doesn't match" critique.

### A.3 [MEDIUM ROI] Доказать full Theorem 3 для decentralized

**Цель:** объединить T3 (centralized PL + momentum) с T1 (decentralized convex). Требуется Lyapunov $\Phi_t = (L(\bar\theta_t) - L^\star) + (\eta/2)\|\bar v_t\|^2 + c \cdot \Pi_t$ для какой-то константы $c$.

**Что делать:**
1. Применить consensus-error bound (Lemma 4) к momentum-aware update.
2. Вывести $\mathbb{E}[\Pi_{t+1} | \mathcal{F}_t] \le \rho_W^2 \cdot \Pi_t + \text{velocity-induced source}$ с учётом momentum amplification.
3. Combine с T3 Lyapunov machinery.

**Сложность:** 4-d composition (non-convex × PL × momentum × decentralized × ZO). Не делалось в литературе. **2-4 weeks careful work.**

**Effort:** ~80 hours. **Impact:** **главная теоретическая новизна** работы → может тянуть на NeurIPS theory track. Если получится — значительно повышает venue.

### A.4 [MEDIUM ROI] Transient acceleration argument

**Цель:** объяснить empirical 3× speedup в R1b R100 → R300.

**Подход:** finite-time analysis с estimate sequence. Для $T \le T^*(\epsilon)$ при low noise regime momentum даёт acceleration before noise floor доминирует. После $T^*$ — matched rate.

**Что делать:**
1. Найти $T^*$ как функцию $\sigma^2$ и $\mu$. Эмпирически $T^* \approx 300$ для R1b.
2. Показать что в transient phase ($t < T^*$) momentum даёт $O(1/T^2)$ vs $O(1/T)$ из plain SGD.
3. После $T^*$ оба переходят в noise-floor-bounded regime.

**Сложность:** medium. Bottou-Curtis-Nocedal 2018 + Aybat 2019 → есть шаблон.

**Effort:** 20-30 hours. **Impact:** closes empirical gap.

### A.5 [LOW ROI] Tight constants для T3

**Цель:** улучшить factor $2G^2/(3\mu)$ до $G^2/\mu$ через tighter Young's. Marginal improvement, скорее cosmetic.

**Effort:** 5 hours. **Impact:** low.

---

## B. Эмпирические upgrades

### B.1 [CRITICAL ROI] Multi-seed validation для headline experiments

**Проблема:** R1d (C4), HellaSwag rescue (§5.5), MathLogicQA (§5.6) — все single seed. SE на acc ~ ±0.045. Effects 1-1.5σ.

**Что делать:**

**B.1.1** R1d ablation (worst Day 5 cell, β-decay + clip50): 3 seeds. Уже есть infrastructure (`scripts/03_dmezo_federated.py`). Compute: ~2 hours Colab.

**B.1.2** HellaSwag rescue (vanilla diverges vs D-MeZO-N converges): 3 seeds. **Уже частично сделано** (`validate_dmezo_n_rescue_multiseed.py` — Section 19 в notebook). Нужно finish + 500-example eval pool. Compute: ~2.5 hours Colab.

**B.1.3** MathLogicQA safe-tracking: 3 seeds × 2 variants. **Currently running** (sweep идёт). Завершится ~5 hours.

**Total compute:** ~10 hours Colab Blackwell.
**Effort:** 1 working day (mostly waiting).
**Impact:** **превращает 3 "tentative" в 3 "robust"** в `robustness_matrix.md`. Critical для submission.

**Готовность инфраструктуры:** ✅ (scripts exist).

### B.2 [CRITICAL ROI] Head-to-head vs FedKSeed

**Проблема:** FedKSeed (Qin et al. 2024 ICML) — ближайший конкурент. Нет direct comparison → reviewer ставит major weakness.

**Что делать:**
1. Clone FedKSeed (https://github.com/alibaba/FederatedScope/tree/FedKSeed).
2. Адаптировать к нашему setup: Qwen3-4B, SST-2, 4 clients, same data partition.
3. Прогнать 1000 rounds, same hyperparameters.
4. Plot side-by-side: convergence curve, final accuracy, communication cost.
5. Expected outcome: D-MeZO-N в peer-to-peer mode даёт **same loss** + **lower communication** (один скаляр на ребро vs star topology с central server).

**Подвох:** FedKSeed использует star topology + multiple seeds per round + LoRA. Нужно: либо адаптировать наш setup под их LoRA mode, либо вытащить только их seed-update механизм и применить к full-parameter MeZO.

**Effort:** 2 working days + 5 hours Colab.
**Impact:** **закрывает крупнейший weakness**. Без этого — workshop only.

### B.3 [HIGH ROI] Scale-up: один эксперимент на 8B model или n=8 clients

**Проблема:** 4 clients × 4B params — small scale. Reviewer: "does this scale?"

**Что делать (выбрать одно):**

**B.3.1** Qwen3-8B на SST-2 federated: 4 clients × 1000 rounds. Compute: ~6 hours Colab. Modify: только loader, остальной код dimension-agnostic.

**B.3.2** Qwen3-4B с n=8 clients: ring topology, ~1.5× train data partition. Compute: ~3 hours Colab. Modify: только partition + topology config.

**Effort:** 1 day total (config + run + analysis).
**Impact:** **один data point "scale OK"** убирает reviewer concern.

### B.4 [MEDIUM ROI] Generative task: SAMSum или GSM8K

**Проблема:** все эксперименты — multi-choice classification. Generative tasks не покрыты.

**Что делать:**
1. SAMSum (dialogue summarization, ~16k train) или GSM8K (math word problems, ~7k train).
2. Loss: cross-entropy на token generation. MeZO machinery same.
3. 500 train / 100 eval, 1000 rounds, 4 clients.

**Effort:** 1 working day (data pipeline + eval) + 5 hours Colab.
**Impact:** "**не только multi-choice**" closure.

### B.5 [MEDIUM ROI] Communication cost plot

**Что делать:**
1. Plot bytes-per-round vs accuracy: D-MeZO-N (8 bytes × edges), FedKSeed (~18 KB), FedAvg (gigabytes).
2. Log-scale plot — dramatic visual.

**Effort:** 2 hours.
**Impact:** **strongest visual** для paper — single plot, conclusive story.

### B.6 [LOW ROI] Larger batch / longer training

3× more steps (3000) — diminishing returns. Skip unless reviewer requests.

---

## C. Paper-writing upgrades

### C.1 [CRITICAL] Удалить overstatements

**Что:**
- "first fully peer-to-peer federated zeroth-order optimizer for LLM" → "first federated decentralized MeZO with formal analysis"
- "novel architecture support (hybrid linear-attention)" → keep, but specify "first known federated ZO test on this architecture class"
- "$0.74 \approx 1/\sqrt{4}$ matches Theorem 1" → remove
- "accelerated variant" → "stabilized variant" (см. A.1)

**Effort:** 3 hours.
**Impact:** убирает 3-4 potential reviewer "overstatement" complaints.

### C.2 [HIGH] Pre-register cross-task hypothesis

**Проблема:** D-MeZO-N v1 был tuned на SST-2 (Day 8), потом validated post-hoc на HellaSwag/MathLogicQA. Это нестрого с точки зрения научного метода.

**Что делать:**
1. В § Method: явно указать "we tune (β-decay schedule, clip C=50) on SST-2; subsequent experiments use **same** hyperparameters across all tasks without re-tuning."
2. Эмфаз: "no hyperparameter tuning per task — universality test."
3. В § Limitations: "post-hoc validation; ideal would be pre-registered hypothesis with separate dev set."

**Effort:** 1 hour.
**Impact:** addresses "cherry-picking" concern proactively.

### C.3 [HIGH] § Limitations должен быть длиннее и честнее

Текущий § Limitations — 1 paragraph. Должен быть 1 page.

**Что добавить:**
- Multi-seed limitation explicit + planned fix.
- Compute scale (4 clients, 4B) — real FL has 100+ clients, 8B+.
- No comparison vs FedKSeed/FedZeN (planned).
- T3 не доказывает acceleration; emperical 3× speedup is transient.
- PL assumption for LLM unproven.
- HellaSwag rescue single seed.

**Effort:** 2 hours.
**Impact:** **показывает научную зрелость**. Reviewers preferred honest limitations over fake claims.

### C.4 [MEDIUM] Restructure § Theory

Текущий § Theory выбирает Theorem 3 как "central" finding. После A.1 (acceleration removal): T3 = "stability theorem", не "acceleration theorem". Restructure:

- T1 (convex): federated speedup mechanism.
- T2 (PL no momentum): rate + linear $1/n$ floor.
- T3 (PL + heavy-ball): **stability** under variance amplification. Lyapunov technique.

**Effort:** 3 hours.
**Impact:** chiarer narrative.

### C.5 [LOW] Add diagram explaining D-MeZO-N flow

Algorithm box visually. Already in fig5_algorithm_schematic.png. Improve caption.

**Effort:** 1 hour.

---

## D. Engineering (Optional)

### D.1 [MEDIUM] CI for tests

GitHub Actions: `pytest tests/` on push. Уже 128 tests pass. Только runner setup.

**Effort:** 2 hours.

### D.2 [LOW] Docker image для reproducibility

Single command to reproduce all figures. Already Colab-based.

**Effort:** 4 hours.
**Impact:** low for now; high if paper accepted.

---

## Timeline (рекомендованный)

### Sprint 1 (Week 1): Critical fixes

**Day 1-2:** A.1 + A.2 + C.1 + C.2 — все framing changes. **6 hours work.**

**Day 3-4:** B.1 — multi-seed validation. Mostly waiting on Colab; 1 hour work + 10 hours compute.

**Day 5:** B.5 — communication cost plot. **2 hours work.**

**Result:** paper goes from "single-seed claims with overstatements" to "multi-seed CI with calibrated claims." Defensible.

### Sprint 2 (Week 2): Major upgrades

**Day 1-2:** B.2 — head-to-head FedKSeed. **2 working days.**

**Day 3:** B.3 — scale-up (one experiment). **1 working day.**

**Day 4-5:** C.3 + C.4 — paper restructure. **2 days writing.**

**Result:** paper has competitive comparison + scale evidence + restructured narrative.

### Sprint 3 (Weeks 3-4, optional): Theoretical depth

**Days 1-10:** A.3 (full decentralized T3) — **2 weeks careful theoretical work**.

**Days 11-14:** A.4 (transient acceleration) — **1 week if A.3 unlocks it**.

**Result:** main-track conference-worthy theory contribution.

### Final assessment after Sprint 1+2

| Venue tier | Without A.3/A.4 | With A.3/A.4 |
|---|---|---|
| NeurIPS/ICML/ICLR main | Borderline reject (single theory contribution thin) | Borderline accept (strong theory + empirics) |
| AAAI/IJCAI | Borderline accept | Accept |
| TMLR | Accept | Accept (highly regarded) |
| FL workshop | Strong accept | Strong accept |
| Russian top venues (АИСТ) | Strong accept | Top paper |
| Bauman diploma | Excellent | Outstanding |

**Recommended path:** Sprint 1+2 minimum для quality. Sprint 3 — если есть compute budget и время до защиты.

---

## Что НЕ делать

1. **Не запускать новые ablations без plan.** Текущая работа имеет 5+ negative findings (§6.4, §6.7) — этого достаточно. Каждый new ablation = 5 hours compute + ризк null result.

2. **Не пытаться доказать acceleration без careful framework.** Bottou-Curtis-Nocedal 2018 показывает что момент **не** ускоряет SGD asymptotically с $\sigma > 0$. Любой "acceleration proof" должен быть **transient only** — иначе противоречит established literature.

3. **Не комбинировать B.3 и B.4 в один эксперимент.** Scale-up + generative task — confounded experiment. Separate.

4. **Не масштабировать compute наугад.** Один scale-up experiment (B.3.1 или B.3.2) — достаточно для "scale OK" claim. Не нужно 8 различных scales.

5. **Не убирать negative findings из paper.** §6.4 (batch variance not 1/√B), §6.7 (ε autotuner fails), §6.8 (joint sweep ambiguous) — **это сильные стороны** paper. Показывают научную честность.

---

## Метрики прогресса

После Sprint 1+2 ожидаются:

| Метрика | Сейчас | После Sprint 1+2 | Target |
|---|---|---|---|
| Robust findings (`robustness_matrix.md`) | 4 | 7-8 | ≥ 7 |
| Tentative → robust upgrades | — | 3 (B.1) | 3 |
| Comparison vs SOTA | 0 | 1 (FedKSeed) | ≥ 1 |
| Scale data points | 1 (4B/4 clients) | 2 (+ scale-up) | 2-3 |
| Overstatements removed | 0 | 4 | All |
| Theorem 3 framed as "stabilization" | "acceleration" | "stabilization" | ✓ |

---

*Last updated: 2026-05-20. Документ создан как actionable plan после theoretical pass + peer-review.*
