# DP-MeZO σ-sweep plan

**Goal:** Map the **privacy/utility frontier** for D-MeZO-N + Gaussian noise (D2 improvement). Determine whether D-MeZO-N can achieve **publishable DP** ($\varepsilon \le 10$) with acceptable utility loss — this would position the paper as **"first privacy-preserving decentralized federated ZO optimizer."**

## Theoretical setup

Gaussian mechanism (Dwork-Roth 2014, Theorem A.1):

$$\sigma \ge \frac{\Delta \sqrt{2 \ln(1.25 / \delta)}}{\varepsilon}, \quad \Leftrightarrow \quad \varepsilon = \frac{\Delta \sqrt{2 \ln(1.25 / \delta)}}{\sigma}$$

For our setup:
- **Sensitivity** $\Delta = C = 50$ (ρ-clip threshold — natural L2 sensitivity bound)
- **δ** $= 10^{-3}$ (standard small failure probability)
- $\sqrt{2 \ln(1.25 / 10^{-3})} \approx 3.776$

So:
$$\varepsilon = \frac{50 \times 3.776}{\sigma} \approx \frac{188.8}{\sigma}.$$

## Planned sweep

| σ | ε | Privacy class | Expected utility |
|---|---|---|---|
| 0.5 | 378 | **No privacy** | ≈ D-MeZO-N no-DP (reference) |
| 2.0 | 94 | Trivial | ≈ D-MeZO-N no-DP |
| 5.0 | 38 | Trivial | Slight degradation expected |
| 10.0 | 19 | Weak | Moderate degradation |
| **19.0** | **10** | **Medium (paper threshold)** | **Key data point** |
| 50.0 | 4 | Medium-strong | Significant degradation |

**Reference baselines included:**
- vanilla MeZO (no DP, no momentum)
- D-MeZO-N v1 (no DP, full momentum recipe)

**Total cells:** (1 vanilla + 1 dmezo_n + 6 σ) × 2 seeds = **16 cells**.

## Predicted outcomes

Three scenarios:

### Scenario A — DP-MeZO succeeds (publishable!)

**If at σ=19 (ε=10) the utility loss is within ~10% of D-MeZO-N no-DP:**
- Major paper-changing result
- Headline claim: "First DP federated ZO optimizer for LLMs, ε ≤ 10 with X% utility retention"
- DP works because ρ-clip provides natural sensitivity bound — no extra mechanism needed

### Scenario B — DP-MeZO degrades smoothly

**If utility drops linearly with log(σ):**
- Standard DP/utility trade-off observed
- Paper section §6.6 can show the curve and discuss frontier
- Honest negative result for "strong privacy on these tasks"

### Scenario C — DP-MeZO catastrophic at σ ≥ 10

**If σ ≥ 10 makes loss diverge or stuck:**
- DP mechanism not directly viable — need K-direction averaging to reduce per-step noise
- Fallback: combine with multi-direction MD-D-MeZO-N (already implemented) to get effective $\sigma_{\text{eff}} = \sigma / \sqrt{K}$

## Compute estimate

| Resource | Per-cell | Total |
|---|---|---|
| Wall-clock (Qwen3.5-0.8B w/ fla, 200 rounds) | ~3-4 min | **~55-65 min** (16 cells) |
| GPU memory | ~16 GB | (sequential, not parallel) |
| Disk | ~50 KB | (JSON + figures) |

Same model + task + hyperparams as `local_test_improvements.py` for direct comparison.

## How to launch

```powershell
.venv\Scripts\python scripts\sweep_dp_sigma.py `
    --model Qwen/Qwen3.5-0.8B `
    --task mathlogicqa `
    --seeds 42 43 `
    --sigmas 0.5 2 5 10 19 50 `
    --num-rounds 200 `
    --rho-clip 50 `
    --dp-delta 1e-3
```

Outputs:
- `experiments/diagnostics/sweep_dp_sigma_Qwen_Qwen3p5-0p8B_mathlogicqa.json`
- `docs/figures/fig_sweep_dp_sigma_trajectories_Qwen_Qwen3p5-0p8B_mathlogicqa.png`
- `docs/figures/fig_sweep_dp_sigma_frontier_Qwen_Qwen3p5-0p8B_mathlogicqa.png`

## Faster preview (single seed × 100 rounds)

For initial sanity check (~15 min instead of 55):

```powershell
.venv\Scripts\python scripts\sweep_dp_sigma.py `
    --model Qwen/Qwen3.5-0.8B `
    --task mathlogicqa `
    --seeds 42 `
    --sigmas 0.5 5 19 50 `
    --num-rounds 100 `
    --num-train-examples 200 `
    --num-eval-examples 50
```

## Analysis script

(Same `scripts/analyze_local_test.py` works on the output JSON, since the cell format is identical. For ε vs utility, see the auto-generated frontier figure.)

## Paper integration plan

If results are favourable (Scenario A or B):
- New section **§7 in paper: "Privacy-preserving D-MeZO-N (DP-MeZO)"**
- Table: σ, ε, final loss, final acc, gap vs no-DP D-MeZO-N
- Figure: privacy/utility frontier (already auto-generated)
- Theorem-level discussion: ρ-clip provides natural Δ=C sensitivity, momentum doesn't break DP composition
- Reference comparison: DP-SGD (Abadi 2016), DP-MeZO (Tang 2024 if any)

If results are unfavourable (Scenario C):
- Honest negative in §6.6.3 follow-up
- Frame as: "naive DP-MeZO degrades at ε ≤ 10; multi-direction (MD-D-MeZO-N + K-fold variance reduction) is the natural fix to be tested"

---

*Last updated: 2026-05-20. Script ready; awaiting GPU availability after combo test.*
