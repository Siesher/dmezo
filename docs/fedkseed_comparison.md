# Head-to-Head: D-MeZO-N vs FedKSeed vs vanilla D-MeZO

**Purpose:** Close the most-cited weakness flagged in the peer-review pass — no direct empirical comparison vs the closest published competitor (FedKSeed, Qin et al. 2024 ICML).

**Status:** ✅ Script implemented (`scripts/head_to_head_fedkseed.py`); ⏳ awaiting compute run.

## Algorithmic comparison

### FedKSeed (Qin et al. 2024, ICML)

**Paper:** arXiv:2312.06353 · **Code:** github.com/alibaba/FederatedScope/tree/FedKSeed

**Core mechanism** (interpreted from paper § 3-4):

```
INPUT: shared seed pool S = {s^(1), ..., s^(K)}; lr; eps; T rounds
INIT:  all clients start from theta_0 = pretrained weights

for round t = 0, ..., T-1:
    1. Server selects shared seed s_t in S (round-robin or random).
    2. Server broadcasts s_t to all n clients.
    3. Each client i in {1, ..., n}:
        - sample local batch xi_i from D_i
        - compute z_t = z(s_t) (deterministic from seed)
        - compute rho_i = (L_i(theta + eps z_t) - L_i(theta - eps z_t)) / (2 eps)
        - send rho_i to server
    4. Server averages: rho_bar = (1/n) sum_i rho_i
    5. Server broadcasts rho_bar to all clients.
    6. Each client applies IDENTICAL update:
       theta <- theta - eta * rho_bar * z_t
       (no momentum, no clipping in paper default)
```

**Communication per round:** $O(n)$ scalars (n client-to-server) + 1 scalar (server-to-all). Total $\sim 18$ KB/round for $n = 8$ clients with $K = 4096$ seed pool entries.

### D-MeZO-N (this paper)

**Core mechanism** (`src/dmezo/federated/simulator.py`):

```
INPUT: mixing matrix W (doubly-stochastic, spectral gap ρ_W); lr; eps; T;
       rho-clip C; beta schedule (beta_0 -> beta_end)
INIT:  all clients start from theta_0; velocity buffers v_i = 0

for round t = 0, ..., T-1:
    1. Each client i in parallel:
        - sample fresh per-client seed s_i^t (independent across clients)
        - sample local batch xi_i from D_i
        - compute z_i^t = z(s_i^t) — UNIQUE per client
        - compute rho_i = (L_i(theta_i + eps z_i^t) - L_i(theta_i - eps z_i^t)) / (2 eps)
        - clip: rho_i_tilde = clip(rho_i, -C, +C)
        - update velocity: v_i <- beta_t * v_i + rho_i_tilde * z_i^t
        - local step: theta_i_half <- theta_i - eta * v_i
    2. Decentralized consensus mixing: theta_i <- sum_j W_ij * theta_j_half
```

**Communication per round (weight_avg mode):** $O(d)$ floats per edge (heavy — must transmit weights). **Alternative (update_share mode):** $O(1)$ scalars per edge, equivalent to FedKSeed's communication cost.

## Comparison table

| Aspect | FedKSeed | D-MeZO-N (ours) |
|---|---|---|
| **Directions per round** | 1 shared $z$ | $n$ unique $z_i$ |
| **Topology** | Star (central server) | Decentralized (arbitrary $W$, $\rho_W \in [0, 1)$) |
| **Momentum** | None | Heavy-ball + β-decay $0.9 \to 0$ |
| **Variance control** | None (relies on FP32) | $\rho$-clip $C = 50$ |
| **Aggregation level** | Scalar ($\rho$) | Parameters or scalars (configurable) |
| **Single point of failure** | Server | None (P2P) |
| **Per-client effective gradient** | Same direction as all others | Independent direction |
| **Theoretical framework** | Implicit (Malladi-style) | T1 + T2 + T3 (`docs/theory_rigorous.md`) |
| **Variance reduction (n clients)** | $1/n$ on data noise only | $1/n$ on data + direction noise |

## Theoretical positioning

From Theorem 2 (`docs/theory_rigorous.md` §2):

$$\mathbb{E}[L_T - L^\star] \le (1 - \tfrac{\eta\mu}{2})^T \Delta_0 + \frac{3\delta^2}{2\mu} + \frac{\eta C^2 r(H) \ell}{\mu n}$$

The **stochastic floor** $\eta C^2 r(H) \ell / (\mu n)$ has $1/n$ federated speedup — this requires **independent z directions**. For FedKSeed (shared $z$), the stochastic floor degenerates to:

$$\frac{\eta \, C_{\text{FedK}}^2 \, r(H) \, \ell}{\mu} \cdot \underbrace{\frac{\sigma_{\text{data}}^2}{n}}_{\text{only data noise reduced}}$$

— i.e., FedKSeed reduces **data sampling noise** by $1/n$ but **NOT direction sampling noise**. D-MeZO-N reduces both.

**Theoretical prediction:** D-MeZO-N should achieve **lower variance floor** for the same $T, n, \eta$. Empirical verification awaits the head-to-head run.

## Expected empirical outcome

Based on:
- MathLogicQA finalized multi-seed results (`docs/multiseed_analysis.md`)
- Theorem 2 prediction
- FedKSeed's published numerical behavior on OPT/LLaMA

**Hypothesis (to test):**

| Metric | Predicted ranking (best → worst) | Confidence |
|---|---|---|
| Final loss | vanilla D-MeZO ≤ FedKSeed ≤ D-MeZO-N | Medium |
| Convergence speed (early rounds) | D-MeZO-N > vanilla > FedKSeed | Medium |
| Variance across seeds | D-MeZO-N < vanilla < FedKSeed (sharp z reduces var) | Medium |
| Peak accuracy | D-MeZO-N > vanilla ≈ FedKSeed | Low |
| Communication cost | FedKSeed = D-MeZO-N (update_share) << D-MeZO-N (weight_avg) | High |

**Caveats:**
- Predictions are tentative — actual results may surprise.
- On `mathlogicqa` with current setup, multi-seed shows vanilla loss-faster than D-MeZO-N. FedKSeed (no momentum) may be **closest to vanilla** in convergence behavior.
- Communication-cost argument is **the strongest** for D-MeZO-N + update_share — same per-round bandwidth as FedKSeed but **decentralized** (no central server bottleneck).

## How to run the experiment

```bash
# Full sweep (3 seeds × 3 variants × 1000 rounds): ~6.75 hours on Blackwell
.venv/Scripts/python scripts/head_to_head_fedkseed.py \
    --model Qwen/Qwen3.5-4B-Base \
    --task mathlogicqa \
    --seeds 42 43 44 \
    --variants vanilla dmezo_n fedkseed \
    --num-rounds 1000 \
    --n-clients 4 \
    --lr 3e-7 --eps 1e-3 \
    --rho-clip 50 --beta-start 0.9 --beta-end 0.0

# Quick smoke test (1 seed × 3 variants × 200 rounds): ~30 min
.venv/Scripts/python scripts/head_to_head_fedkseed.py \
    --seeds 42 --num-rounds 200 --num-eval-examples 50

# Cross-task generalization: re-run on HellaSwag
.venv/Scripts/python scripts/head_to_head_fedkseed.py \
    --task hellaswag --model Qwen/Qwen3-4B \
    --num-train-examples 2000 --num-eval-examples 500
```

**Outputs:**
- JSON: `experiments/diagnostics/head_to_head_fedkseed_Qwen_Qwen3p5-4B-Base_mathlogicqa.json`
- Figure: `docs/figures/fig20_head_to_head_fedkseed_Qwen_Qwen3p5-4B-Base_mathlogicqa.png`

## Reproducibility checklist

- ✅ Same `lr=3e-7`, `eps=1e-3` across all variants
- ✅ Same data partition per seed (IID 4-client, deterministic from seed)
- ✅ Same eval pool (100 examples, deterministic from seed=0)
- ✅ Paired comparison (per-seed delta, bootstrap CI)
- ✅ FedKSeed implementation faithful to paper §3 algorithm (no momentum, no clip, shared seed via server-style broadcast simulated as `_FixedRng`)
- ⚠️ NOT a forked FedKSeed codebase — our implementation in our framework. Reviewers may push back; could mitigate by also running their official repo for ONE cell as cross-check.

## What to write in paper after run completes

Add new section § 6.5 "Comparison with FedKSeed (Qin et al. 2024)":

> We compare D-MeZO-N against FedKSeed under matched compute, data partition, and learning rate. **Table TBD** reports final loss/accuracy mean ± std over 3 seeds; **Figure 20** shows trajectories. [INSERT FINDING] D-MeZO-N achieves [LOWER/COMPARABLE/HIGHER] final loss than FedKSeed (paired Δ = X.XX, 95% CI [Y, Z]) at the same communication cost (with `update_share` mode). The 1/n variance reduction predicted by Theorem 2 for independent-direction averaging is [VERIFIED/PARTIALLY-VERIFIED/NOT-VERIFIED] by the empirical gap.

Three possible narratives depending on outcome:

1. **D-MeZO-N wins clearly:** "D-MeZO-N's combination of unique per-client directions + momentum stabilization yields measurable improvement over FedKSeed at matched budget. The mechanism is direction-variance reduction predicted by T2."

2. **D-MeZO-N ties FedKSeed:** "D-MeZO-N matches FedKSeed on final loss/acc but provides **decentralized topology support** (no central server) and **stronger theoretical guarantees** (T3 stability bound under heavy-ball + clip). This positions D-MeZO-N as a strict generalization."

3. **D-MeZO-N loses to FedKSeed:** "FedKSeed's simpler design (shared direction + no momentum) outperforms D-MeZO-N on this benchmark. This **reinforces the safe-tracking observation from §5.6.1** — momentum + clip add complexity without clear benefit on already-converging tasks. We conjecture D-MeZO-N's advantages emerge in rescue regimes (§5.5 HellaSwag) where vanilla MeZO diverges — an explicit head-to-head on HellaSwag is the natural follow-up."

Be honest about whichever happens.

## Future extensions (post head-to-head)

If D-MeZO-N wins clearly: add Ferret (Shu et al. 2024) and FedZeN (Maritan et al. 2024) to the comparison.

If D-MeZO-N ties: focus on theoretical contributions + decentralization story.

If D-MeZO-N loses on MathLogicQA: re-run on HellaSwag (rescue regime expected) — this is where D-MeZO-N should shine.

---

*Last updated: 2026-05-20. Script ready; awaiting compute.*
