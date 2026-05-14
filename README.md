# D-MeZO: Decentralized Federated MeZO with Nesterov Acceleration

Research project: peer-to-peer federated fine-tuning of LLMs without backpropagation, with Nesterov-style acceleration.

**Status:** scaffold, Day 1 sanity-check ready.

## What this is

We combine three ideas:

1. **MeZO** (Malladi et al., NeurIPS 2023) — fine-tuning LLMs using only forward passes via SPSA-style gradient estimates.
2. **Decentralized stochastic optimization** with consensus mixing (Koloskova et al. 2020) — peer-to-peer training without a central server.
3. **Nesterov-style acceleration** adapted to zeroth-order updates.

The result is **D-MeZO-N** — a federated fine-tuning algorithm where each communication round exchanges *one scalar and one seed* per neighbor pair, instead of gigabytes of weight deltas.

The closest published competitor is **FedKSeed** (Qin et al., ICML 2024), which uses federated MeZO with star topology and no momentum. Our delta: peer-to-peer + Nesterov.

## Quick start

```bash
# Install (Python 3.11 recommended)
pip install -e .

# Day 1: sanity-check that MeZO converges on Qwen3-4B / SST-2
python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml

# Run tests
pytest tests/ -v
```

For Colab Pro+: open `notebooks/bootstrap_colab.ipynb`.

## Project layout

```
dmezo/
├── CLAUDE.md                   # Claude Code project memory
├── README.md                   # this file
├── pyproject.toml              # dependencies and packaging
├── src/dmezo/
│   ├── mezo/                   # MeZO step, perturbation, Nesterov
│   ├── federated/              # client, topology, consensus, simulator
│   ├── models/                 # HF model loading utilities
│   ├── data/                   # SuperGLUE tasks + IID/non-IID partitioning
│   └── utils/                  # logging, checkpoint
├── scripts/                    # entry-point scripts for each experiment
├── configs/                    # Hydra YAML configs
├── docs/                       # algorithm spec, theory template, plans
├── notebooks/                  # Colab-ready notebook
├── tests/                      # unit tests
└── experiments/                # outputs (gitignored)
```

## Default model

**Qwen3-4B** (`Qwen/Qwen3-4B`, Apache 2.0). Standard transformer architecture, ~8 GB FP16. Compatible with the Princeton MeZO codebase without modification.

Upgrade path: Qwen3-8B for scale, Qwen3.5-4B for novel-architecture experiment (Gated DeltaNet + attention hybrid — MeZO behavior on this is an open question and a potential research angle).

## Roadmap (week 1)

See `docs/05-week1-plan.md`. Summary:

- Day 1 — sanity check MeZO on Qwen3-4B
- Day 2 — literature deep-dive + centralized baselines
- Day 3 — theorem template
- Day 4 — D-MeZO 2 clients
- Day 5 — 4 clients + topologies + non-IID
- Day 6 — Qwen3-8B / Qwen3.5-4B stretch
- Day 7 — one-pager + ablations

## License

MIT (project code). Underlying models follow their own licenses (Qwen3: Apache 2.0).
