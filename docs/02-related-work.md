# 02. Связанные работы

Шаблон записи:

> **<Название>** (Авторы, год, venue). <arxiv link>. <github link, если есть>.
>
> *Core:* 1-2 предложения о вкладе.
>
> *Delta от D-MeZO-N:* где пересекаются, где остаётся наш вклад.
>
> *Verdict:* must-cite / nice-to-cite / orthogonal.

---

## Прямые предшественники / конкуренты (must-cite)

### MeZO (Malladi et al., NeurIPS 2023)

[arXiv:2305.17333](https://arxiv.org/abs/2305.17333) | [github.com/princeton-nlp/MeZO](https://github.com/princeton-nlp/MeZO)

*Core:* Memory-efficient zeroth-order optimizer для fine-tuning LLM. Показано, что предобученные LLM имеют низкий эффективный ранг гессиана, что делает ZO-обучение работоспособным несмотря на проклятие размерности.

*Delta:* Centralized only. Мы расширяем на decentralized federated сетап + добавляем Nesterov.

*Verdict:* must-cite, базовая работа.

### FedKSeed (Qin et al., ICML 2024)

[arXiv:2312.06353](https://arxiv.org/abs/2312.06353) | [github.com/alibaba/FederatedScope/tree/FedKSeed](https://github.com/alibaba/FederatedScope/tree/FedKSeed)

*Core:* Federated full-parameter fine-tuning LLM через MeZO с обменом seeds + scalars (<18 KB на раунд). Star-топология, без момента.

*Delta:* Мы (а) decentralized peer-to-peer вместо star, (б) добавляем Nesterov-ускорение, (в) даём convergence-теорему с $\rho$ (spectral gap) — у них её нет.

*Verdict:* must-cite, ближайший конкурент.

### FedZeN (Maritan et al. 2024)

[arXiv:2309.17241](https://arxiv.org/abs/2309.17241)

*Core:* Federated ZO с инкрементальной оценкой гессиана + Newton-style ускорение. Quadratic convergence rate.

*Delta:* Они используют инкрементальный гессиан (более информативный, но дороже по памяти). Мы — first-order momentum (Nesterov), легче. Они centralized, мы decentralized.

*Verdict:* must-cite, ближайший по идее ускорения.

### Ferret (Shu et al. 2024)

[arXiv:2409.06277](https://arxiv.org/abs/2409.06277) | [github.com/allen4747/Ferret](https://github.com/allen4747/Ferret)

*Core:* First-order federated full-parameter tuning с shared randomness и low-dim projection. НЕ ZO.

*Delta:* Они first-order (нужен backprop, OOM на edge). Мы ZO (inference-level память). Прямого пересечения нет, но они должны быть в related work как state-of-the-art для federated full-parameter.

*Verdict:* must-cite, для positioning.

---

## Теоретический фундамент (must-cite)

### Koloskova et al. 2020

*"A Unified Theory of Decentralized SGD with Changing Topology and Local Updates"* — [arXiv:2003.10422](https://arxiv.org/abs/2003.10422)

*Core:* Унифицированный convergence-фреймворк для decentralized SGD с spectral gap $\rho$, local steps, time-varying topology.

*Delta:* Их теорема 2 — наш шаблон. Заменяем first-order оценку на ZO + добавляем $r(H)$-тонкость из MeZO.

### Nesterov & Spokoiny 2017

*"Random Gradient-Free Minimization of Convex Functions"* — Foundations of Computational Mathematics.

*Core:* Базовая теория ZO с Гауссовым сглаживанием. Convergence rates для convex и non-convex.

*Delta:* Используем их bound на variance ZO-оценки.

### Spall 1992

*"Multivariate stochastic approximation using a simultaneous perturbation gradient approximation"* — IEEE TAC.

*Core:* Оригинальный SPSA, asymptotic normality, rate $T^{-1/3}$ для constant step.

*Delta:* Корни всего метода.

---

## Расширенные эталоны (nice-to-cite)

### DeepZero (Chen et al. 2024)

[arXiv:2310.02025](https://arxiv.org/abs/2310.02025) — Sparse ZO для DNN с моментом. Полезно для design choices Nesterov-velocity.

### HiZOO (Zhao et al. 2024)

[arXiv:2402.15173](https://arxiv.org/abs/2402.15173) — Hessian-informed ZO для LLM с диагональным precondition. Альтернатива Nesterov через preconditioner.

### Lian et al. 2017

*"Can Decentralized Algorithms Outperform Centralized?"* — NeurIPS — для positioning decentralized vs. centralized.

### Sahu, Yuan, Yang 2018

*"Distributed Zeroth-Order Optimization over Random Networks"* — базовая D-ZO-SGD теория.

### Akhavan, Pontil, Tsybakov 2022

Нижние оценки для distributed ZO. Полезно для оценки оптимальности.

---

## Orthogonal (можно не цитировать, если место ограничено)

- DPO/PPO/GRPO семья — это RL fine-tuning, ортогонально supervised MeZO.
- Адаптеры (LoRA, QLoRA, prefix tuning) — мы их используем как option, не сравниваемся с ними как с methods.
- Communication compression (sparse SGD, sign-SGD) — другой угол атаки на ту же боль.

---

## Поисковые запросы для регулярного апдейта

```
"federated" + "zeroth-order" + "language model"
"MeZO" + "federated"
"decentralized" + "forward-only" + "LLM"
"federated full-parameter" + "fine-tuning"
Citing: Malladi 2023, Qin 2024 (FedKSeed)
```

Поставить Google Scholar Alert.
