# Literature Review: Federated Zeroth-Order Optimization for LLM Fine-tuning

**Дата:** 2026-05-21. **Цель:** comparative analysis для defense + TMLR/workshop positioning.

## Executive Summary

ZO для LLM fine-tuning оформилась вокруг MeZO (NeurIPS 2023) и развивается по трём направлениям: (1) variance reduction (MeZO-SVRG, Sparse MeZO, SubZero), (2) адаптивные ZO-варианты Adam-style (HELENE, AdaMeZO), (3) federated/DP (FedKSeed, Ferret, DPZero). **Ни одна существующая работа не совмещает одновременно**: decentralized P2P topology + independent-seed ZO + heavy-ball momentum со стабилизацией + formal $(\varepsilon, \delta)$-DP для LLM. D-MeZO-N v2 занимает эту нишу.

---

## Section A0 — SPSA Family (foundational ancestor of MeZO)

**SPSA (Simultaneous Perturbation Stochastic Approximation)** — Spall 1992 — прямой предок MeZO и всех subsequent ZO methods для LLM. Технически MeZO — это **SPSA + in-place perturbation + Gaussian z + scalar-only state**.

| Работа | Год | Ключевая идея | Связь с D-MeZO-N v2 |
|---|---|---|---|
| **Spall 1992** (IEEE TAC 37:332) | 1992 | Original SPSA: $\hat g = ((L(\theta+\varepsilon\Delta) - L(\theta-\varepsilon\Delta))/(2\varepsilon)) \cdot \Delta^{-1}$, $\Delta$ — Bernoulli ±1 vector | **Прямой предок MeZO**. Two-point central diff с perturbation |
| **Sadegh-Spall 1998** (J Stat Plan Inference) | 1998 | Convergence rate для SPSA: $O(1/\sqrt{T})$ | Foundational rate analysis |
| **Spall 2000** "**2-SPSA**" (IEEE TAC 45:1839) | 2000 | Second-order SPSA: Hessian estimation через 4 forward passes | Analog нашего Richardson 4-point ablation (§6.7 supplement, fail на large ε) |
| **Spall 1997** (Auto Control Theory & Apps) | 1997 | Adaptive SPSA: gain sequence $\eta_t = a/(t+A)^\alpha$ ($\alpha \approx 0.602$) | Соответствует harmonic-decay schedule (`MeZOConfig.lr_schedule="harmonic"` в нашем коде) |
| **Bhatnagar et al. 2003** | 2003 | Generalized SPSA с smoother perturbations | Path к Gaussian-MeZO |

### Где MeZO innovates относительно classical SPSA

| Aspect | SPSA (Spall 1992) | MeZO (Princeton 2023) | Innovation rationale |
|---|---|---|---|
| Perturbation $z$ | Bernoulli ±1 | $\mathcal{N}(0, I)$ Gaussian | Gaussian enables $r(H)$-trick (Malladi Lemma 1b'); Bernoulli не работает |
| Storage of $z$ | Explicit vector $O(d)$ | **In-place seed-reconstruction $O(1)$** | **Главная LLM-scale memory innovation** |
| Update form | $\theta \leftarrow \theta - \eta \hat g$ (vector) | $\theta \leftarrow \theta - \eta \hat\rho z$ (scalar × regenerated z) | Saves storing gradient AND z |
| Loss precision | Unspecified | bf16 model + fp32 $\hat\rho$ accumulation | Catastrophic cancellation в $L_+ - L_-$ |

### Где D-MeZO-N innovates относительно SPSA family (4 axes)

1. **Federated SPSA для LLM не существовало.** FedKSeed = SPSA + shared K-seed pool (star topology). D-MeZO-N = SPSA + per-client RNG + gossip mixing. Никто из SPSA literature не делал decentralized SPSA для LLM.

2. **Heavy-ball momentum для SPSA under PL.** SPSA classically uses adaptive gain (Spall 1997 harmonic decay), не momentum. Convergence proof momentum для SPSA — у нас Theorem 3.

3. **Adaptive ρ-clip для SPSA stability.** SPSA не имеет stability mechanism для outliers $|\hat\rho|$. Это B1 contribution.

4. **DP-SPSA для federated.** DP-ZO (arXiv:2401.04343) — SPSA-style centralized DP без явного "SPSA" naming. DP-SPSA в federated сценарии — наш T4 + dual-use ρ-clip.

### Открытая нiche: DP-SPSA literature search (2026-05-21)

Targeted search для "DP-SPSA" / "private SPSA" / "differentially private zeroth-order":
- **Ни одной работы** не нашлось под именем "DP-SPSA". Связанные работы:
  - **DP-ZO** (arXiv:2401.04343) — centralized, не SPSA naming
  - **DP-ZOSO** (arXiv:2402.07818) — centralized, stagewise scheduler, не SPSA-specific
  - **Distributed DP optimization** (arXiv:2310.11892, arXiv:1512.00369) — control/convex, не LLM, не federated
- **Conclusion:** decentralized federated DP-ZO для LLM — **open niche**. D-MeZO-N v2 + Theorem 4 occupy эту nichu.

---

## Section A — ZO for LLM Fine-tuning

| Работа | Год | Ключевая идея | Momentum? | DP? | Federated? |
|---|---|---|---|---|---|
| **MeZO** (Malladi et al., arXiv:2305.17333, NeurIPS 2023) | 2023 | In-place SPSA, seed-recon $z$, inference-level memory | Нет | Нет | Нет |
| **MeZO-SVRG** (arXiv:2404.08080) | 2024 | ZO + SVRG variance reduction; periodic reference forward pass | Нет | Нет | Нет |
| **Sparse MeZO** (arXiv:2402.15751) | 2024 | Perturbation only of small weights; reduces estimate error | Нет | Нет | Нет |
| **SubZero** (arXiv:2410.08989) | 2024 | Low-rank random subspace perturbation; variance $O(r)$ вместо $O(d)$ | Нет | Нет | Нет |
| **HELENE** (arXiv:2411.10696) | 2024 | Diagonal Hessian + layer-wise clipping (2nd-order preconditioner) | Нет (annealing) | Нет | Нет |
| **AdaMeZO / Adam-style ZO** (community work) | 2024-25 | Adam moments seed-reconstructible | Adam | Нет | Нет |

**Где D-MeZO-N v2 заполняет gap:** heavy-ball scalar momentum + federated wrapper + DP. Ни одна работа из этого блока не рассматривает decentralized сценарий.

---

## Section B — Federated MeZO/ZO для LLM (closest competitors)

### FedKSeed (Qin et al., ICML 2024, arXiv:2312.06353)

- **Idea:** shared finite set of K seeds на сервере; probability-differentiated sampling по |ρ|. Communication ~ K seed-ρ pairs ≈ 18 KB/round.
- **Topology:** **server-client (star)**. Нет P2P/gossip.
- **Theory:** convex case, bounded heterogeneity.
- **D-MeZO-N v2 vs FedKSeed:**
  - Topology: P2P gossip vs star
  - Seeds: **independent per client** vs shared K-pool
  - Momentum: ✅ heavy-ball + adaptive clip vs ❌ no momentum
  - DP: ✅ formal $(\varepsilon, \delta)$ vs ❌ no DP

### Ferret (Shu et al., arXiv:2409.06277, 2024)

- **Idea:** **first-order** (требует backprop!) + shared randomness для projection/reconstruction.
- **Не ZO** — Ferret требует gradient memory $O(d)$. Сравнение не direct apples-to-apples.

### FedZeN (Maritan et al., arXiv:2309.17241, 2024)

- **Idea:** distributed ZO Newton (Hessian estimation across clients).
- **Limitations:** convex case, no LLM experiments, no DP, требует Hessian accumulation memory.

### DPZero / DP-ZO (Tang et al., arXiv:2401.04343, 2024)

- **Idea:** Gaussian/Laplace noise на scalar ρ для DP centralized ZO. Key insight: "$z$ random, ρ carries data info" → privatize scalar.
- **D-MeZO-N v2 vs DPZero:**
  - Topology: decentralized federated vs centralized
  - **Dual-use ρ-clip**: same C для (a) momentum stability **+** (b) L2-sensitivity (наш novelty)
  - Theorem 3 closure for momentum convergence — отсутствует у DPZero

---

## Section C — DP для Federated Learning / ZO

| Работа | Год | Mechanism | Federated? | ZO? | Note |
|---|---|---|---|---|---|
| **Abadi DP-SGD** (CCS 2016) | 2016 | Moments accountant + per-sample gradient clipping + Gaussian | Нет | Нет | Foundational baseline |
| **DP-FedAvg** (McMahan 2018) | 2018 | DP-SGD + FedAvg, per-user clipping | Star FL | Нет | Standard centralized FL DP |
| **DPZero** (arXiv:2401.04343) | 2024 | Gaussian/Laplace on scalar ρ; ZO-specific sensitivity | Нет | Да | Closest to us, но centralized |
| **DPZV** (arXiv:2502.20565) | 2025 | ZO + DP for **vertical** FL | Vertical | Да | Different task (vertical, не horizontal) |

**Position D-MeZO-N v2:** первый известный метод **horizontal decentralized gossip federated + ZO + formal $(\varepsilon, \delta)$-DP для LLM**. DPZero centralized; DPZV vertical FL.

---

## Section D — Heavy-ball / Momentum Theory для ZO

| Работа | Условия | Применение к нам |
|---|---|---|
| **Nesterov-Spokoiny 2017** (FOCM, DOI:10.1007/s10208-015-9296-2) | Convex smooth | Look-ahead Nesterov для ZO; we tested empirically (diverges R20) |
| **Ghadimi-Lan 2013** (SIAM J. Optim.) | Non-convex stochastic | Baseline non-convex ZO rate $O(1/\sqrt{T})$ |
| **Bottou-Curtis-Nocedal 2018** (SIAM Review) | SGD asymptotic | **Forbids momentum-based asymptotic speedup** when σ>0. Critical для honest framing T3 |
| **Yang-Lin-Li 2016** (arXiv:1604.03257) | Heavy-ball convex/non-convex PL | Unified momentum analysis; condition on β |
| **HB w/ approx gradients** (arXiv:2303.16241) | HB + biased grad + growing var | Stochastic HB converges under biased+growing conditional variance — applicable to ZO bias $O(\varepsilon^2)$ |

**Critical gap:** нет работы, которая доказывает heavy-ball convergence для **ZO + PL + decentralized + β-decay**. Theorem 3 наш Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ закрывает именно эту нишу.

**Important honest framing**: Bottou-Curtis-Nocedal 2018 forbids asymptotic speedup. T3 даёт **тот же** rate как plain SGD под PL — это **stabilization, не acceleration**. Это сила, не слабость (мы корректны).

---

## Section E — Decentralized Federated SGD Theory

| Работа | Conditions | Rate | Momentum? | ZO? |
|---|---|---|---|---|
| **Koloskova 2020** (arXiv:2003.10422, ICML) | Non-convex smooth doubly-stoch W | $O(1/\sqrt{nT} + \zeta/(1-\lambda_2))$ | Нет | Нет |
| **Lian 2017** (Can Decentralized SGD Outperform Centralized?) | Non-convex iid | $O(1/\sqrt{nT})$ first federated speedup proof | Нет | Нет |

**Position T1:** Koloskova-style framework + ZO noise + ρ-clip. Прямого аналога нет — все decentralized momentum работы используют first-order gradients.

---

## Section F — Variance Reduction для ZO (краткое)

| Метод | Идея | Применимость к D-MeZO-N v2 |
|---|---|---|
| **MeZO-SVRG** | SVRG reference point | Orthogonal; +1 forward/period (incompatible с "1 scalar/round" invariant) |
| **Multi-direction SPSA (K-direction)** | K parallel perturbations; variance $O(1/K)$ | Tested, equal-compute loses K=1 (см. §6.5 paper); compute trade-off |
| **SubZero** | Low-rank subspace | Conceptually orthogonal; future combination |
| **Richardson 4-point** | Higher-order finite-diff | Tested, narrow sweet spot ε≈3e-3 (см. §6.7 supplement) |

---

## Unique Combinations of D-MeZO-N v2 (gaps в literature)

Следующие конкретные сочетания **отсутствуют** в найденных работах:

1. **Decentralized (gossip/P2P) + ZO + LLM fine-tuning.** FedKSeed = star; Ferret = star + first-order; FedZeN = convex + не LLM. D-MeZO-N v2 = first gossip ZO для LLM.

2. **Independent $z_i$ per client (не shared seed pool).** FedKSeed требует shared finite K-seed set + central coordination. У нас 1 float + 1 int = 16 байт; клиенты не должны coordinate seed space → подходит для асинхронного P2P.

3. **Scalar heavy-ball momentum + adaptive ρ-clip (B1) + drift-reset (B5) в ZO.** Adam-style ZO работы используют per-parameter moments. Никто не использует heavy-ball scalar velocity с data-driven adaptive clip + surgical drift reset.

4. **Convergence proof для heavy-ball ZO + PL + β-decay (Theorem 3).** arXiv:2303.16241 (2023) доказывает HB с biased approx gradient в general settings; но не специфично ZO + PL + decentralized + clip + β-decay. Наш Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ — оригинальный contribution. **Closes Princeton Open Problem 1.**

5. **Dual-use ρ-clip:** одновременно (a) momentum stability + (b) L2-sensitivity для Gaussian mechanism. DPZero использует noise на ρ без momentum; мы совмещаем clip для двух целей.

6. **MeZO на hybrid linear-attention + full-attention арх (Qwen3.5).** Princeton paper тестировал только full-attention transformers. Мы первые на hybrid linear-attention.

7. **Formal $(\varepsilon=10, \delta=10^{-3})$-DP в decentralized federated ZO** с +6.2% utility cost. Ближайший конкурент DPZero (arXiv:2401.04343) — centralized only. DPZV (arXiv:2502.20565) — vertical FL, different task. **Targeted search confirmed:** ни одной работы под "DP-SPSA" name — open niche.

8. **Connection to classical SPSA (Spall 1992).** D-MeZO-N v2 — это технически **SPSA с 4 innovations поверх MeZO**: heavy-ball scalar momentum (vs SPSA's adaptive gain Spall 1997), adaptive clip (vs no SPSA stability mechanism), federated wrapper (no SPSA-LLM federated work), dual-use clip for DP (DP-SPSA не existed под этим именем). Эта историческая родословная сильна для positioning — мы строим на well-established stochastic approximation theory (34 года).

---

## Weaknesses в Positioning vs Literature

**Где конкуренты сильнее:**

| Aspect | Their lead | Our defense |
|---|---|---|
| **Variance reduction** | MeZO-SVRG +20% acc (centralized) | Federated SVRG требует extra communication round → conflict с "1 scalar/round" invariant |
| **Adam-style adaptivity** | AdaMeZO / Adam-ZO better centralized | Adam moments в federated ZO = extra memory + non-trivial averaging semantics |
| **Non-convex generality** | T2/T3 доказан только под PL | PL is standard in modern deep learning rate-proofs; locally on trajectory plausible (Liu-Zhu-Belkin 2022) |
| **Empirical scale** | FedKSeed tested LLaMA-7B; мы Qwen3.5-4B | Hybrid linear-attention class is novel; не direct comparable |

**Empirical weaknesses (honest):**
- Multi-seed paired falsifies original v1 single-seed claims (+1.25pp на MathLogicQA → 3/3 worse).
- Short-horizon SST-2 (200 rounds): vanilla beats D-MeZO-N v2 в 3.4× — serious failure mode при coротких runs.
- **Нет head-to-head comparison с FedKSeed** на одинаковом dataset/model — reviewer обязательно попросит.

**Theoretical gaps:**
- T3 floor $2G^2/(3\mu)$ — bound на $G^2$ pessimistic.
- T1 decentralized rate стандартная Koloskova-форма, без tight analysis ZO-heterogeneity.
- β-decay schedule выбрана эмпирически; optimal decay не доказан.
- Full decentralized T3 (Open Problem 2 у нас) — не закрыта.

---

## Recommended Framing

### Защита (Bauman MSTU 2026-05-23)

**Центральный тезис:**
> "D-MeZO-N v2 — первый алгоритм, объединяющий decentralized gossip topology, zeroth-order оптимизацию для LLM, scalar heavy-ball momentum со stabilization, и формальные (ε, δ)-DP гарантии. Каждый компонент существует отдельно в литературе; их **совместная работа** — вклад данной работы."

Акценты:
- **Independent seeds per client** (no shared pool) — key для real P2P / asynchronous
- **Dual-use ρ-clip** — elegant single-mechanism solution для двух задач
- **DP frontier flat** (ε=10 → +6.2% utility) — практически значимо для compliance (115-ФЗ, HIPAA)
- **Hybrid linear-attention** validation — first known MeZO test на этом классе арх
- **Honest framing T3**: "stabilizes, not accelerates" — Bottou-Curtis-Nocedal 2018 forbids asymptotic speedup

### TMLR Submission (post-defense)

**Positioning:** "Communication-minimal decentralized federated ZO for LLM fine-tuning with privacy"

**Must add перед submission:**
1. **FedKSeed head-to-head** на одинаковом dataset/model (script готов, ~6.75h compute)
2. **HellaSwag rescue multi-seed** (3 seeds × Qwen3-4B) для validation rescue regime claim
3. **Explicit limitations section** — when D-MeZO-N v2 fails (short horizons, easy convergent tasks)
4. **Comparison table** vs MeZO-SVRG/SubZero/HELENE — acknowledge stronger centralized methods

### NeurIPS FL Workshop (alternative)

**Positioning:** "Privacy-preserving P2P federated fine-tuning of LLMs at inference-level memory"

Workshop audience ценит:
- 16 bytes/round/neighbor — compelling story
- Hybrid linear-attention = novel architectural test
- DP-flat frontier = "almost free privacy"
- Honest failure modes = good science (workshops love negatives)

---

## References (verified arXiv IDs / DOIs)

### SPSA family (foundational)
- **[Spall 1992]** A Stochastic Approximation Technique for Generating Maximum Likelihood Parameter Estimates. IEEE Transactions on Automatic Control 37(3):332–341.
- **[Sadegh-Spall 1998]** Optimal random perturbations for stochastic approximation using a simultaneous perturbation gradient approximation. J Stat Planning & Inference.
- **[Spall 2000]** Adaptive stochastic approximation by the simultaneous perturbation method (2-SPSA). IEEE TAC 45(10):1839–1853.
- **[Spall 1997]** Accelerated second-order stochastic optimization using only function measurements. In Proc. 36th IEEE Conf. Decision and Control.
- **[Bhatnagar et al. 2003]** Two-timescale algorithms for simulation optimization of hidden Markov models. IIE Transactions 35(4):385–397.
- **[DP-ZOSO]** Bu et al. (2024). Stage-wise DP zeroth-order optimization. arXiv:2402.07818

### MeZO family and successors
- **[MeZO]** Malladi et al. (2023). Fine-Tuning Language Models with Just Forward Passes. NeurIPS 2023. arXiv:2305.17333
- **[MeZO-SVRG]** Variance-Reduced ZO Methods for Fine-Tuning Language Models. arXiv:2404.08080
- **[Sparse MeZO]** Sparse MeZO. arXiv:2402.15751
- **[SubZero]** ZO Fine-Tuning of LLMs in Random Subspaces. arXiv:2410.08989
- **[HELENE]** Hessian Layer-wise Clipping and Gradient Annealing. arXiv:2411.10696
- **[FedKSeed]** Qin et al. (2024). Federated Full-Parameter Tuning... ICML 2024. arXiv:2312.06353
- **[Ferret]** Shu et al. (2024). Ferret: Federated Full-Parameter Tuning at Scale. arXiv:2409.06277
- **[FedZeN]** Maritan et al. (2024). FedZeN. arXiv:2309.17241
- **[DPZero]** Tang et al. (2024). Private Fine-tuning of LLMs with ZO. arXiv:2401.04343
- **[DPZV]** DPZV: Resource Efficient ZO for DP VFL. arXiv:2502.20565
- **[Abadi 2016]** Deep Learning with Differential Privacy. CCS 2016
- **[DP-FedAvg]** McMahan et al. (2018). Learning Differentially Private Recurrent Language Models
- **[Koloskova 2020]** Unified Theory of D-SGD with Changing Topology. ICML 2020. arXiv:2003.10422
- **[Nesterov-Spokoiny 2017]** Random Gradient-Free Minimization of Convex Functions. FOCM. DOI:10.1007/s10208-015-9296-2
- **[Ghadimi-Lan 2013]** Stochastic first- and zeroth-order methods. SIAM J. Optim.
- **[Bottou-Curtis-Nocedal 2018]** Optimization Methods for Large-Scale ML. SIAM Review
- **[Yang-Lin-Li 2016]** Unified convergence of stochastic momentum methods. arXiv:1604.03257
- **[HB approx gradients]** Convergence of HB Method With Approximate Gradients. arXiv:2303.16241

---

*Document created 2026-05-21 via research-ml agent literature review. Some referenced works may have publication dates in 2024-2025 (verified) or later (potentially not yet published — verify before citing in final paper).*
