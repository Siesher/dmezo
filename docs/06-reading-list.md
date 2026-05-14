# 06. Reading list (в порядке приоритета)

## P0 — обязательно к Day 4

1. **Malladi et al. 2023.** *Fine-Tuning Language Models with Just Forward Passes.* NeurIPS. [arXiv:2305.17333](https://arxiv.org/abs/2305.17333). Особенно: Section 3 (algorithm), Section 5 (theory), Appendix C (landscape analysis).

2. **Qin et al. 2024.** *FedKSeed: Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes.* ICML. [arXiv:2312.06353](https://arxiv.org/abs/2312.06353). + Code: [github.com/alibaba/FederatedScope/tree/FedKSeed](https://github.com/alibaba/FederatedScope/tree/FedKSeed).

3. **Koloskova et al. 2020.** *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates.* ICML. [arXiv:2003.10422](https://arxiv.org/abs/2003.10422). Особенно: Theorem 2 + Appendix proof.

## P1 — обязательно к Day 6

4. **Nesterov & Spokoiny 2017.** *Random Gradient-Free Minimization of Convex Functions.* Foundations of Computational Mathematics. Sections 1-4.

5. **Maritan et al. 2024.** *FedZeN: Quadratic Convergence in Zeroth-Order Federated Learning via Incremental Hessian Estimation.* [arXiv:2309.17241](https://arxiv.org/abs/2309.17241).

6. **Shu et al. 2024.** *Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models.* [arXiv:2409.06277](https://arxiv.org/abs/2409.06277). + [Code](https://github.com/allen4747/Ferret).

## P2 — обязательно для проработки теоремы

7. **Stich 2019.** *Local SGD Converges Fast and Communicates Little.* ICLR. — техника для local steps в decentralized.

8. **Nedić et al. 2017.** *Achieving Geometric Convergence for Distributed Optimization over Time-Varying Graphs (DIGing).* SIAM J. Optim.

9. **Spall 1992.** *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation.* IEEE TAC.

## P3 — расширенный фон (читать когда будет время)

10. **Chen, Zhang, Koyejo 2024.** *DeepZero: Scaling Up Zeroth-Order Optimization for Deep Neural Networks.* ICLR. [arXiv:2310.02025](https://arxiv.org/abs/2310.02025).

11. **Zhao et al. 2024.** *Second-Order Fine-Tuning without Pain for LLMs (HiZOO).* [arXiv:2402.15173](https://arxiv.org/abs/2402.15173).

12. **Tang, Yuan, Yang 2020.** *Distributed Zeroth-Order Algorithms for Nonconvex Multi-Agent Optimization.*

13. **Akhavan, Pontil, Tsybakov 2022.** Нижние оценки для distributed ZO.

14. **Gadat & Panloup 2023.** Momentum и Nesterov в ZO с anti-correlated noise.

15. **Lian et al. 2017.** *Can Decentralized Algorithms Outperform Centralized?* NeurIPS.

## P4 — мониторинг (не для глубокого чтения)

16. **OpenFedLLM** — benchmark репозиторий для federated LLM tuning. Полезно для эмпирических baselines.

17. **FederatedScope LLM module** — Alibaba framework, имеет FedKSeed implementation.

18. **Flower** — federated learning framework, гибкая поддержка кастомных топологий.

---

## Поисковые алерты для регулярного обновления

```
Google Scholar Alerts на:
  - "MeZO"
  - "federated" AND "zeroth-order" AND "LLM"
  - citing Malladi 2023 (arXiv 2305.17333)
  - citing FedKSeed (arXiv 2312.06353)
```

Проверять раз в неделю — горячая область, может быть новый препринт прямо в нашу нишу.

## Конференции к подаче

- **NeurIPS 2026 workshop (FL/OPT/FedFM)** — submission в сентябре 2026.
- **ICML 2027 main track** — submission в феврале 2027.
- **Spotlight на ICLR 2027** — submission в октябре 2026.

Domestic:
- **AI Journey 2026** (Москва, осень) — если делаем русскоязычный multilingual angle.
- **Dialog 2027** — лингвистический конференция, можно подать federated-MeZO-multilingual статью.
