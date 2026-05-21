# D-MeZO-N: Decentralized Federated MeZO with Nesterov Stabilization

Research project: peer-to-peer federated fine-tuning of LLMs without backpropagation, with momentum stability under bounded variance.

**Status (2026-05-21):** Paper-scale multi-seed validation complete. Defense kit ready.

## Headline result

On **Qwen3.5-4B-Base / MathLogicQA / 3 seeds paired**, D-MeZO-N v2 = combo (adaptive ρ-clip + drift-reset) **robustly beats vanilla MeZO**:

| Metric | vanilla | D-MeZO-N v2 | Δ | Direction |
|---|---|---|---|---|
| Final loss (mean ± std) | 1.368 ± 0.018 | **1.293 ± 0.010** | **−5.5%** | **3/3 same direction** |
| Final accuracy | 0.377 | **0.400** | **+2.3pp** | 2/3 positive, 1 tie |

First paper-scale multi-seed validated empirical improvement of decentralized federated MeZO over vanilla MeZO. See `docs/multiseed_analysis.md` §22 for raw data.

## What this is

We combine four ideas:

1. **MeZO** (Malladi et al., NeurIPS 2023) — fine-tuning LLMs using only forward passes via SPSA-style gradient estimates.
2. **Decentralized stochastic optimization** with consensus mixing (Koloskova et al. 2020) — peer-to-peer training without a central server.
3. **Heavy-ball momentum stabilization** for ZO regime: adaptive ρ-clip + drift-reset + linear β-decay. Closes Princeton Open Problem 1 (T3 in `docs/theory_rigorous.md`).
4. **Differential privacy via dual-use ρ-clip** — same clip threshold used for momentum stability **simultaneously serves as L2-sensitivity** for Gaussian mechanism. Per-round (ε=10, δ=10⁻³)-DP at ~6% utility cost (T4).

Key algorithmic differentiator vs FedKSeed: **independent z_i per client** (not shared seed) → 1/n variance reduction across **both** data and direction noise components.

## Quick start

```bash
# Install (Python 3.11, uv recommended)
uv pip install -e .

# Day 1 sanity-check: MeZO converges on Qwen3-4B / SST-2
uv run --no-sync python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml

# Federated D-MeZO-N v2 on Qwen3.5-4B-Base / MathLogicQA (multi-seed)
uv run --no-sync python scripts/local_test_improvements.py \
    --model Qwen/Qwen3.5-4B-Base --task mathlogicqa --seeds 42 43 44

# Run tests (128/128 pass)
uv run --no-sync pytest tests/ -v
```

For Colab Pro+: open `notebooks/bootstrap_colab.ipynb`.

## Project layout

```
dmezo/
├── CLAUDE.md                       # Claude Code project memory
├── README.md                       # this file
├── pyproject.toml                  # dependencies and packaging
├── src/dmezo/
│   ├── mezo/                       # MeZO step, perturbation, Nesterov, adaptive clip
│   ├── federated/                  # client (per-client RNG), topology, consensus, simulator
│   ├── models/                     # HF model loading (incl. Qwen3.5 V-L hybrid)
│   ├── data/                       # SuperGLUE + HellaSwag + MathLogicQA + IID/non-IID partitioning
│   └── utils/                      # logging, checkpoint, config
├── scripts/                        # entry-point scripts for each experiment
├── configs/                        # Hydra YAML configs
├── docs/
│   ├── paper_ru.md, paper_en.md    # main paper
│   ├── theory_rigorous.md          # full proofs T1–T4
│   ├── math_intuition.md           # plain-language explanation
│   ├── 03-algorithm-spec.md        # formal pseudocode
│   ├── multiseed_analysis.md       # §22 paper-scale 3-seed validation
│   ├── robustness_matrix.md        # findings classified by statistical rigor
│   ├── defense_*.md (4 files)      # defense kit (Bauman MSTU 2026-05-23)
│   └── figures/                    # 41 PNG figures
├── notebooks/                      # Colab-ready notebook
├── tests/                          # 128 unit tests
└── experiments/                    # outputs (gitignored)
```

## Models tested

| Model | Architecture | Tasks | Local? |
|---|---|---|---|
| Qwen3-0.6B | Full attention | SST-2, batch variance ablation | ✅ RTX 5070 Ti |
| Qwen3-1.7B | Full attention | SST-2 | ✅ Local |
| Qwen3.5-0.8B | Hybrid linear-attention | SST-2, MathLogicQA | ✅ Local |
| Qwen3-4B | Full attention | SST-2, HellaSwag (rescue regime) | Colab Blackwell |
| **Qwen3.5-4B-Base** | Hybrid linear-attention + frozen ViT | **MathLogicQA (paper headline)**, SST-2 grid | **Colab Blackwell** |

First known federated ZO experiments on hybrid linear-attention architecture (Qwen3.5 family).

## Communication cost

| Method | Per-round/client | For 4B model × 1000 rounds × 4 clients |
|---|---|---|
| FedAvg | 8 GB (bf16 weights) | ~32 TB |
| FedKSeed (K=4096 seeds + ρ) | ~18 KB | ~72 MB |
| **D-MeZO-N (`update_share`)** | **16 bytes (1 float + 1 int)** | **~64 KB** |

D-MeZO-N achieves 10⁹× compression vs FedAvg. Same order as FedKSeed; differentiator is **peer-to-peer topology** + momentum convergence proof + DP.

## Theorems

| # | Name | Setting | Key result |
|---|---|---|---|
| T1 | Convex + momentum + decentralized | Each $L_i$ convex, ρ-clip, momentum, mixing matrix W | $\tilde{O}(\sqrt{Lr(H)\Delta_0/(nT)})$ + consensus penalty |
| T2 | PL + ZO (no momentum) | μ-PL, ρ-clip | Linear $(1-\eta\mu/2)^T$ to noise floor with $1/n$ federated speedup |
| **T3** | **PL + heavy-ball + β-decay + clip** | μ-PL, momentum, β-decay 0.9→0 | Lyapunov $V_t = (L-L^\star) + (\eta/2)\|v\|^2$ contracts at rate $(1-3\eta\mu/2)$ to neighbourhood $2G^2/(3\mu)$. **Closes Princeton Open Problem 1.** |
| T4 | DP extension of T3 | T3 + Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2)$ | Per-round $(\varepsilon_1, \delta)$-DP with $\varepsilon_1 = C\sqrt{2\ln(1.25/\delta)}/\sigma$ |

**Important:** T3 rate matches plain SGD under PL — no asymptotic acceleration claimed (consistent with Bottou-Curtis-Nocedal 2018 T5.1). Transient empirical 3× speedup observed but remains open problem.

Full proofs: `docs/theory_rigorous.md`.

## Honest negatives (multi-seed validated)

- v1 (fixed ρ-clip C=50): 3/3 seeds worse than vanilla (+7.0% loss). Adaptive clip needed.
- Drift-reset alone (B5 without adaptive clip): 3/3 worse (+6.4% loss). Needs B1 partner.
- Look-ahead Nesterov: diverges 7× faster than heavy-ball (R20 vs R140).
- K=3 multi-direction averaging: equal-compute loses K=1 (Pareto trade-off, not improvement).
- ε(t) warmup schedules: robustly lose const ε=10⁻³ across 16+ cells.
- Asymptotic momentum acceleration under PL+ZO: forbidden by Bottou-Curtis-Nocedal 2018 T5.1.

Each is a clean falsification with mechanistic explanation. See `docs/robustness_matrix.md` for full classification.

## Citing

If you use this work, please cite:

```bibtex
@misc{sukhatsky2026dmezon,
  author = {Sukhatsky, Maxim},
  title = {D-MeZO-N: Decentralized Federated MeZO with Nesterov Stabilization},
  year = {2026},
  url = {https://github.com/Siesher/dmezo}
}
```

## License

MIT (project code). Underlying models follow their own licenses (Qwen3 / Qwen3.5: Apache 2.0).
