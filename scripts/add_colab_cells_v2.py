"""Append new Colab cells for D-MeZO-N v2 (B1+B5 combo) + HellaSwag rescue + DP-MeZO sweep.

Adds sections 22-24 to ``notebooks/bootstrap_colab.ipynb``:
    22. D-MeZO-N v2 (B1+B5 combo) replication: Qwen3.5-4B-Base + MathLogicQA, 3 seeds
    23. D-MeZO-N v2 on HellaSwag: rescue regime test (Qwen3-4B, 3 seeds)
    24. DP-MeZO sigma-sweep: privacy/utility frontier (Qwen3.5-0.8B for speed)

Each section: markdown header + run cell + persist markdown + drive-copy cell.

Run:
    .venv/Scripts/python scripts/add_colab_cells_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    """Build a markdown cell with text split into lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=False),
    }


def code_cell(text: str) -> dict:
    """Build a code cell with text split into lines."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=False),
    }


def main() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "bootstrap_colab.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    n_before = len(nb["cells"])
    print(f"Notebook has {n_before} cells before append.")

    new_cells = []

    # ------------------------------------------------------------------
    # Section 22: D-MeZO-N v2 (B1+B5 combo) on Qwen3.5-4B-Base + MathLogicQA
    # ------------------------------------------------------------------
    new_cells.append(md_cell(
        "## 22. ⭐ D-MeZO-N v2 (B1+B5 combo) replication: Qwen3.5-4B-Base + MathLogicQA\n"
        "\n"
        "**Goal:** Replicate the local **B1+B5 combo finding** at paper scale "
        "(Qwen3.5-4B-Base instead of Qwen3.5-0.8B). Tests whether the local "
        "result (combo reaches vanilla parity, beats D-MeZO-N v1 by -29% loss / +6.25pp acc CI-significant) "
        "holds at 5× larger model.\n"
        "\n"
        "**Setup:** 5 variants × 3 seeds × 1000 rounds = 15 cells on Colab Blackwell.\n"
        "\n"
        "| Variant | Description |\n"
        "|---|---|\n"
        "| `vanilla` | plain MeZO (no momentum, no clip) |\n"
        "| `dmezo_n` | D-MeZO-N v1 (β-decay 0.9→0 + fixed C=50) |\n"
        "| `dmezo_n_drift` | B5 alone (drift-reset on top of v1) |\n"
        "| `dmezo_n_adaptive_clip` | B1 alone (adaptive clip replaces fixed C) |\n"
        "| `dmezo_n_combo` | **B1+B5 combo (v2 proposed)** |\n"
        "\n"
        "Local Qwen3.5-0.8B finding (`docs/multiseed_analysis.md` + `project_combo_b1_b5_works.md`):\n"
        "\n"
        "- combo vs vanilla: Δloss = −0.0004 (TIE), Δacc CI [-0.10, +0.00] (TIE)\n"
        "- combo vs D-MeZO-N v1: Δloss = −0.61, Δacc = +0.0625, CI [+0.025, +0.100] (significant)\n"
        "\n"
        "**Compute estimate (Colab A100/L4):** ≈15 cells × ≈45 min/cell = **≈11 hours**.\n"
        "Recommend splitting into two runs (variants 1-3, then 4-5) or reducing to 2 seeds for first pass.\n"
        "**Faster preview:** drop to 2 seeds (10 cells ≈ 7.5h) or 500 rounds (≈5h)."
    ))
    new_cells.append(code_cell(
        "# 22a. Replicate B1+B5 combo at paper scale (Qwen3.5-4B-Base / MathLogicQA / 3 seeds).\n"
        "# Make sure fla is installed (section 18a) for Qwen3.5 hybrid fast path.\n"
        "!cd /content/dmezo && git pull origin main\n"
        "\n"
        "# Full run (3 seeds, 5 variants, 1000 rounds). ETA ~11h on A100.\n"
        "# To shorten, reduce --seeds or --num-rounds (see comments below).\n"
        "!cd /content/dmezo && python scripts/local_test_improvements.py \\\n"
        "    --model Qwen/Qwen3.5-4B-Base \\\n"
        "    --task mathlogicqa \\\n"
        "    --seeds 42 43 44 \\\n"
        "    --variants vanilla dmezo_n dmezo_n_drift dmezo_n_adaptive_clip dmezo_n_combo \\\n"
        "    --n-clients 4 \\\n"
        "    --num-rounds 1000 \\\n"
        "    --lr 3e-7 --eps 1e-3 \\\n"
        "    --num-train-examples 500 --num-eval-examples 100 \\\n"
        "    --batch-size 8 --max-length 256 \\\n"
        "    --eval-every 100 --eval-batches 13 \\\n"
        "    --dtype bfloat16 \\\n"
        "    --rho-clip 50 --beta-start 0.9 --beta-end 0.0 \\\n"
        "    --drift-window 50 --drift-threshold 0.1 \\\n"
        "    --ac-window 30 --ac-quantile 0.95 --ac-alpha 1.3 \\\n"
        "    --dp-sigma 0.5\n"
        "\n"
        "# QUICK PREVIEW (2 seeds, 500 rounds, ~3.5h):\n"
        "#     --seeds 42 43 --num-rounds 500"
    ))
    new_cells.append(md_cell("### 22b. Persist D-MeZO-N v2 replication results to Drive"))
    new_cells.append(code_cell(
        "# 22b. Copy combo replication results to Drive.\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "drive_dest = Path('/content/drive/MyDrive/dmezo_runs/v2_combo_replication')\n"
        "drive_dest.mkdir(parents=True, exist_ok=True)\n"
        "for src in [\n"
        "    Path('/content/dmezo/experiments/diagnostics/local_test_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.json'),\n"
        "    Path('/content/dmezo/docs/figures/fig_local_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.png'),\n"
        "]:\n"
        "    if src.exists():\n"
        "        shutil.copy(src, drive_dest / src.name)\n"
        "        print(f'  copied {src.name} ({src.stat().st_size / 1024:.1f} KB)')\n"
        "    else:\n"
        "        print(f'  MISSING: {src.name}')\n"
        "print(f'Drive dest: {drive_dest}')\n"
        "\n"
        "# Run analysis script.\n"
        "!cd /content/dmezo && python scripts/analyze_local_test.py \\\n"
        "    experiments/diagnostics/local_test_improvements_Qwen_Qwen3p5-4B-Base_mathlogicqa.json"
    ))

    # ------------------------------------------------------------------
    # Section 23: D-MeZO-N v2 on HellaSwag (rescue regime, Qwen3-4B)
    # ------------------------------------------------------------------
    new_cells.append(md_cell(
        "## 23. 🔥 D-MeZO-N v2 on HellaSwag (rescue regime test)\n"
        "\n"
        "**Goal:** Verify the B1+B5 combo does NOT break the **rescue ability** documented in §5.5 "
        "(Qwen3-4B + HellaSwag: vanilla MeZO **diverges**, D-MeZO-N v1 **rescues**).\n"
        "\n"
        "The risk: B1 adaptive-clip allows much larger |ρ| values (8–10× the fixed C=50) through to "
        "the velocity buffer. On rescue regime, vanilla diverged because of unbounded |ρ| spikes — "
        "so combo's looser clip might re-introduce divergence even with B5 safety net. This test "
        "checks if combo's drift-reset is fast enough to catch rescue-regime divergence.\n"
        "\n"
        "**Setup:** Qwen3-4B (full-attn, matching §5.5 paper setup), HellaSwag, 4 clients IID, "
        "3 seeds, 1000 rounds.\n"
        "\n"
        "**Variants to compare:**\n"
        "- `vanilla` (expected to diverge per §5.5)\n"
        "- `dmezo_n` (D-MeZO-N v1 rescue baseline, expected to converge per §5.5)\n"
        "- `dmezo_n_combo` (B1+B5 v2, **the test** — does it still rescue?)\n"
        "\n"
        "**Three outcomes:**\n"
        "1. **Combo rescues = vanilla parity confirmed AND rescue preserved.** Best case. v2 is universal.\n"
        "2. **Combo rescues better than v1.** Even better — adaptive clip absorbs the divergence faster.\n"
        "3. **Combo diverges.** Need to add a min-clip floor (C >= 50) or use lower α in B1.\n"
        "\n"
        "**Compute estimate:** 3 variants × 3 seeds = 9 cells × ≈30 min/cell = **≈4.5 hours** on A100."
    ))
    new_cells.append(code_cell(
        "# 23a. D-MeZO-N v2 (combo) test on rescue regime.\n"
        "!cd /content/dmezo && git pull origin main\n"
        "\n"
        "# Full run (3 seeds × 3 variants × 1000 rounds, ~4.5h on A100).\n"
        "!cd /content/dmezo && python scripts/local_test_improvements.py \\\n"
        "    --model Qwen/Qwen3-4B \\\n"
        "    --task hellaswag \\\n"
        "    --seeds 42 43 44 \\\n"
        "    --variants vanilla dmezo_n dmezo_n_combo \\\n"
        "    --n-clients 4 \\\n"
        "    --num-rounds 1000 \\\n"
        "    --lr 3e-7 --eps 1e-3 \\\n"
        "    --num-train-examples 2000 --num-eval-examples 500 \\\n"
        "    --batch-size 8 --max-length 256 \\\n"
        "    --eval-every 100 --eval-batches 63 \\\n"
        "    --dtype bfloat16 \\\n"
        "    --rho-clip 50 --beta-start 0.9 --beta-end 0.0 \\\n"
        "    --drift-window 50 --drift-threshold 0.1 \\\n"
        "    --ac-window 30 --ac-quantile 0.95 --ac-alpha 1.3 \\\n"
        "    --dp-sigma 0.5\n"
        "\n"
        "# QUICK PREVIEW (1 seed, 500 rounds, ~45 min):\n"
        "#     --seeds 42 --num-rounds 500"
    ))
    new_cells.append(md_cell("### 23b. Persist HellaSwag rescue test results to Drive"))
    new_cells.append(code_cell(
        "# 23b. Copy results to Drive.\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "drive_dest = Path('/content/drive/MyDrive/dmezo_runs/v2_combo_hellaswag_rescue')\n"
        "drive_dest.mkdir(parents=True, exist_ok=True)\n"
        "for src in [\n"
        "    Path('/content/dmezo/experiments/diagnostics/local_test_improvements_Qwen_Qwen3-4B_hellaswag.json'),\n"
        "    Path('/content/dmezo/docs/figures/fig_local_improvements_Qwen_Qwen3-4B_hellaswag.png'),\n"
        "]:\n"
        "    if src.exists():\n"
        "        shutil.copy(src, drive_dest / src.name)\n"
        "        print(f'  copied {src.name} ({src.stat().st_size / 1024:.1f} KB)')\n"
        "    else:\n"
        "        print(f'  MISSING: {src.name}')\n"
        "print(f'Drive dest: {drive_dest}')\n"
        "\n"
        "!cd /content/dmezo && python scripts/analyze_local_test.py \\\n"
        "    experiments/diagnostics/local_test_improvements_Qwen_Qwen3-4B_hellaswag.json"
    ))

    # ------------------------------------------------------------------
    # Section 24: DP-MeZO sigma sweep (privacy/utility frontier)
    # ------------------------------------------------------------------
    new_cells.append(md_cell(
        "## 24. 🔐 DP-MeZO σ-sweep: privacy/utility frontier\n"
        "\n"
        "**Goal:** Map the (ε, δ)-DP **privacy/utility frontier** for D-MeZO-N + Gaussian noise (D2 improvement). "
        "Determine whether D-MeZO-N can achieve **publishable DP** (ε ≤ 10) with acceptable utility loss.\n"
        "\n"
        "**Setup:** Qwen/Qwen3.5-0.8B (smaller for faster sweep — σ-sensitivity is model-size-invariant), "
        "MathLogicQA, 4 clients IID, 2 seeds, 200 rounds.\n"
        "\n"
        "**σ → ε mapping (with C=50, δ=10⁻³):**\n"
        "\n"
        "| σ | ε | Privacy class |\n"
        "|---|---|---|\n"
        "| 0.5 | 378 | No privacy (reference) |\n"
        "| 2.0 | 94 | Trivial |\n"
        "| 5.0 | 38 | Trivial |\n"
        "| 10.0 | 19 | Weak |\n"
        "| **19.0** | **10** | **Medium ★ paper threshold** |\n"
        "| 50.0 | 4 | Medium-strong |\n"
        "\n"
        "**Compute estimate:** 8 cells (6 σ + 2 baselines) × 2 seeds = 16 cells × ≈3 min/cell on Qwen3.5-0.8B "
        "= **≈50 min total** on A100.\n"
        "\n"
        "**Paper claim if successful:** \"First privacy-preserving decentralized federated ZO optimizer for LLMs, "
        "achieving ε≤10 with X% utility retention\". See `docs/dp_sigma_sweep_plan.md`."
    ))
    new_cells.append(code_cell(
        "# 24a. DP-MeZO sigma-sweep on Qwen3.5-0.8B / MathLogicQA.\n"
        "!cd /content/dmezo && git pull origin main\n"
        "\n"
        "!cd /content/dmezo && python scripts/sweep_dp_sigma.py \\\n"
        "    --model Qwen/Qwen3.5-0.8B \\\n"
        "    --task mathlogicqa \\\n"
        "    --seeds 42 43 \\\n"
        "    --sigmas 0.5 2 5 10 19 50 \\\n"
        "    --n-clients 4 \\\n"
        "    --num-rounds 200 \\\n"
        "    --lr 3e-7 --eps 1e-3 \\\n"
        "    --num-train-examples 500 --num-eval-examples 100 \\\n"
        "    --batch-size 8 --max-length 256 \\\n"
        "    --eval-every 25 --eval-batches 13 \\\n"
        "    --dtype bfloat16 \\\n"
        "    --rho-clip 50 --beta-start 0.9 --beta-end 0.0 \\\n"
        "    --dp-delta 1e-3 \\\n"
        "    --include-baselines\n"
        "\n"
        "# QUICK PREVIEW (1 seed, 4 sigmas, 100 rounds, ~10 min):\n"
        "#     --seeds 42 --sigmas 0.5 10 19 50 --num-rounds 100 \\\n"
        "#     --num-train-examples 200 --num-eval-examples 50"
    ))
    new_cells.append(md_cell("### 24b. Persist DP sweep results to Drive"))
    new_cells.append(code_cell(
        "# 24b. Copy DP sweep results to Drive.\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "drive_dest = Path('/content/drive/MyDrive/dmezo_runs/dp_sigma_sweep')\n"
        "drive_dest.mkdir(parents=True, exist_ok=True)\n"
        "for src in [\n"
        "    Path('/content/dmezo/experiments/diagnostics/sweep_dp_sigma_Qwen_Qwen3p5-0p8B_mathlogicqa.json'),\n"
        "    Path('/content/dmezo/docs/figures/fig_sweep_dp_sigma_trajectories_Qwen_Qwen3p5-0p8B_mathlogicqa.png'),\n"
        "    Path('/content/dmezo/docs/figures/fig_sweep_dp_sigma_frontier_Qwen_Qwen3p5-0p8B_mathlogicqa.png'),\n"
        "]:\n"
        "    if src.exists():\n"
        "        shutil.copy(src, drive_dest / src.name)\n"
        "        print(f'  copied {src.name} ({src.stat().st_size / 1024:.1f} KB)')\n"
        "    else:\n"
        "        print(f'  MISSING: {src.name}')\n"
        "print(f'Drive dest: {drive_dest}')"
    ))

    # ------------------------------------------------------------------
    # Section 25: Final summary cell
    # ------------------------------------------------------------------
    new_cells.append(md_cell(
        "## 25. 📊 Summary of v2 / DP-MeZO experiments\n"
        "\n"
        "After sections 22-24 complete, the following should be on Drive at `/content/drive/MyDrive/dmezo_runs/`:\n"
        "\n"
        "1. `v2_combo_replication/` — Qwen3.5-4B-Base / MathLogicQA combo at paper scale. **Critical:** confirms combo ≥ vanilla parity at 5× model size.\n"
        "2. `v2_combo_hellaswag_rescue/` — Qwen3-4B / HellaSwag combo. **Critical:** confirms combo still rescues divergent regime.\n"
        "3. `dp_sigma_sweep/` — DP σ vs utility frontier. **Critical:** establishes feasible ε threshold for paper-grade privacy.\n"
        "\n"
        "**Three possible outcome combinations:**\n"
        "\n"
        "| Combo replicates? | Combo rescues? | DP ≤ 10 viable? | Paper position |\n"
        "|---|---|---|---|\n"
        "| YES | YES | YES | **Strong**: v2 universal + private. Major paper-changing direction. |\n"
        "| YES | YES | NO | **Good**: v2 is the new default; DP work as future direction. |\n"
        "| YES | NO  | * | **Hybrid**: combo for convergent tasks, v1 for rescue (recipe selector). |\n"
        "| NO  | *   | * | **Conservative**: report locally-observed result as Qwen3.5-0.8B-specific; investigate scale-dependence. |"
    ))

    nb["cells"].extend(new_cells)
    n_after = len(nb["cells"])
    print(f"Added {len(new_cells)} new cells. Notebook now has {n_after} cells.")

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {nb_path}")


if __name__ == "__main__":
    main()
