# Figures index

Registry of all 41 PNG figures in `docs/figures/`. Split into **paper figures** (fig0-fig19, embedded in `paper_ru.md`) and **equation snippets** (eq_*, для docx build via OMML). Sorted by paper section.

## Paper figures by section

### §0 Master summary
| Fig | File | Description |
|---|---|---|
| **fig00** | `fig00_paper_summary.png` | Master overview C1–C6 contributions с headline numbers (one-page summary for abstract/intro) |

### §3 Algorithm
| Fig | File | Description |
|---|---|---|
| fig5 | `fig5_algorithm_schematic.png` | D-MeZO-N algorithm schematic (EN) |
| fig5_ru | `fig5_algorithm_schematic_ru.png` | Same in Russian |

### §5 Empirical results

#### §5.2 Day 5 federated grid
| Fig | File | Description |
|---|---|---|
| fig1 | `fig1_day5_grid.png` | 4-cell grid (2 topologies × 2 partitions) on Qwen3.5-4B-Base / SST-2 |
| fig3 | `fig3_federated_vs_centralized.png` | Federated D-MeZO-N vs centralized baseline summary |

#### §5.4 Nesterov phase diagram
| Fig | File | Description |
|---|---|---|
| fig2 | `fig2_nesterov_phase_diagram.png` | β=0.9 / clip200 / clip50 / β-decay phases on worst Day 5 cell |
| fig4 | `fig4_r1d_detailed.png` | D-MeZO-N v1 (R1d) detailed trajectory |

#### §5.5–§5.6 Cross-task results
| Fig | File | Description |
|---|---|---|
| fig6 | `fig6_cross_domain_trajectories.png` | HellaSwag + MathLogicQA trajectories side-by-side |
| fig7 | `fig7_cross_task_summary.png` | Cross-task improvement bar chart |

### §6.4 Batch variance
| Fig | File | Description |
|---|---|---|
| fig8 | `fig8_batch_variance.png` | ρ̂ distribution at B ∈ {1,2,4,8,16,32}, σ saturates at B≥8. **Caveat**: cross-panel mean-shift — sampling artifact (see §6.4 text) |

### §6.7 ε autotuner / scheduling
| Fig | File | Description |
|---|---|---|
| fig9 | `fig9_eps_warmup_Qwen_Qwen3-0p6B.png` | Per-arch ε autotuner sweep (full-attn) |
| fig9 | `fig9_eps_warmup_Qwen_Qwen3p5-0p8B.png` | Same (hybrid linear-attn) |
| fig10 | `fig10_eps_warmup_cross_arch.png` | Cross-arch autotuner composite |
| fig11 | `fig11_eps_validate_Qwen_Qwen3-0p6B.png` | Downstream validation: full-attn |
| fig11 | `fig11_eps_validate_Qwen_Qwen3p5-0p8B.png` | Downstream validation: hybrid |
| fig12 | `fig12_eps_autotuner_paradox.png` | **Cross-arch paradox**: autotuner picks ε* that loses downstream (4-panel composite) |
| fig13 | `fig13_eps_schedule_Qwen_Qwen3-0p6B.png` | ε(t) schedule per-arch (5 schedules × 100 steps) |
| fig13 | `fig13_eps_schedule_Qwen_Qwen3p5-0p8B.png` | Same on hybrid |
| fig14 | `fig14_eps_schedule_cross_arch.png` | Cross-arch ε(t) ranking composite |
| fig15 | `fig15_eps_schedule_dmezo_composite.png` | 2×2 (vanilla × D-MeZO-N) × (0.6B × 1.7B) composite |
| fig15 | `fig15_eps_schedule_dmezo_n_Qwen_Qwen3-0p6B.png` | D-MeZO-N variant on 0.6B |
| fig15 | `fig15_eps_schedule_dmezo_n_Qwen_Qwen3-1p7B.png` | D-MeZO-N variant on 1.7B |
| fig15 | `fig15_eps_schedule_vanilla_Qwen_Qwen3-1p7B.png` | Vanilla on 1.7B (control) |
| fig17 | `fig17_richardson_vs_2point_Qwen_Qwen3-0p6B.png` | 2-pt vs 4-pt vs 6-pt finite-diff on full-attn |
| fig17 | `fig17_richardson_vs_2point_Qwen_Qwen3p5-0p8B.png` | Same on hybrid |

### §6.8 Joint sweep on HellaSwag/Qwen3.5-4B-Base
| Fig | File | Description |
|---|---|---|
| fig16 | `fig16_sweep_lr_eps_hellaswag_Qwen_Qwen3p5-4B-Base.png` | Original 100-step local pilot (null result, kept as historical artifact) |
| fig16 | `fig16_sweep_lr_eps_hellaswag_vanilla_Qwen_Qwen3p5-4B-Base.png` | Vanilla 500-step Colab sweep raw figure |
| fig16 | `fig16_sweep_lr_eps_hellaswag_dmezo_n_Qwen_Qwen3p5-4B-Base.png` | D-MeZO-N 500-step Colab sweep raw figure |
| **fig18** | `fig18_joint_sweep_colab.png` | **Headline joint sweep**: 2-row (variant) × 4-col (lr) composite, all 3 schedules, with divergence markers. §6.8 main figure. |

### §5.5 multi-seed validation (pending)
| Fig | File | Description |
|---|---|---|
| fig19 | `fig19_multiseed_validation_Qwen_Qwen3-4B.png` | **PENDING**: 3-seed re-validation of §5.5 D-MeZO-N rescue with bootstrap CI |

## Equation snippets (LaTeX → PNG for docx build)

Используются скриптом `99_build_paper_docx_ru.py` для встраивания формул в .docx (Word OMML pipeline).

| File | Equation |
|---|---|
| `eq_mezo_grad.png` | ρ̂ = (L(θ+εz) − L(θ−εz))/(2ε) |
| `eq_mezo_update.png` | θ ← θ − η·ρ·z |
| `eq_round_step.png` | One federated round update |
| `eq_consensus_error.png` | Bound on per-round consensus error |
| `eq_spectral_gap.png` | ρ_W = ‖I − W‖₂ |
| `eq_pl_condition.png` | μ-PL inequality |
| `eq_pl_descent.png` | PL-descent rate |
| `eq_zo_variance.png` | Var[ρ̂] bound |
| `eq_clip_variance.png` | Variance with ρ-clipping |
| `eq_communication.png` | Communication cost |
| `eq_theorem1_bound.png` | Theorem 1 final bound |
| `eq_theorem2_bound.png` | Theorem 2 final bound |

## Statistics

- **41 PNG total**
- **24 paper figures** (fig0-fig19, multi-version counts)
- **12 equation snippets** (eq_*)
- **~5 figure-files per major section** in §5/§6
- All figures regenerable from `scripts/compose_*.py` + `experiments/diagnostics/*.json`

## Conventions

- `figN_topic_model.png` for per-model raw figures (e.g. `fig17_richardson_vs_2point_Qwen_Qwen3-0p6B.png`)
- `figN_topic_cross_*.png` for cross-arch / cross-variant composites
- `figN_*_composite.png` for multi-panel layouts
- All saved at `dpi=300`, `bbox_inches="tight"` (publication-ready)
- Russian text in figures uses `font.family="DejaVu Sans"` для compatibility
