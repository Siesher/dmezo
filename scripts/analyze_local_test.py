"""Analyze local_test_improvements_*.json — paired CI, signal-to-noise per improvement.

Reports:
    - Per-variant mean ± std on final loss/acc across seeds.
    - Paired d (B5 - vanilla, B1 - vanilla, D2 - vanilla, each - dmezo_n).
    - Bootstrap 95% CI on each dacc.
    - Adaptive-clip threshold trajectory (if available).
    - Drift-reset counts.
    - DP epsilon report.

Usage:
    .venv/Scripts/python scripts/analyze_local_test.py \\
        experiments/diagnostics/local_test_improvements_Qwen_Qwen3p5-0p8B_sst2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bootstrap_ci(values, n_boot=10000, ci=0.95, rng_seed=0):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=str)
    args = p.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    cells = data["cells"]
    variants = data["variants"]
    seeds = data["seeds"]

    print("=" * 100)
    print(f"Local improvements test: {data['model']} / {data['task']} / "
          f"{len(seeds)} seeds x {data['num_rounds']} rounds")
    print(f"Hyperparams: lr={data['lr']}, eps={data['eps']}, rho_clip={data['rho_clip']}, "
          f"beta={data['beta_start']}->{data['beta_end']}")
    print("=" * 100)

    # ---- Per-variant aggregate.
    print(f"\n{'Variant':<28}{'mean_loss':>12}{'std_loss':>11}{'mean_acc':>11}"
          f"{'std_acc':>10}{'best_acc':>11}{'resets':>9}{'DPeps':>10}")
    print("-" * 100)
    variant_stats = {}
    for v in variants:
        losses = [cells[f"{v}|seed={s}"]["eval_losses"][-1]
                  for s in seeds if f"{v}|seed={s}" in cells]
        accs_final = [cells[f"{v}|seed={s}"]["eval_accs"][-1]
                      for s in seeds if f"{v}|seed={s}" in cells]
        accs_best = [max(cells[f"{v}|seed={s}"]["eval_accs"])
                     for s in seeds if f"{v}|seed={s}" in cells]
        resets = sum(cells[f"{v}|seed={s}"]["n_drift_resets"]
                     for s in seeds if f"{v}|seed={s}" in cells)
        dp_eps_vals = [cells[f"{v}|seed={s}"]["dp_epsilon"]
                       for s in seeds if f"{v}|seed={s}" in cells
                       and cells[f"{v}|seed={s}"]["dp_epsilon"] is not None]
        dp_str = f"{dp_eps_vals[0]:.1f}" if dp_eps_vals else "N/A"
        variant_stats[v] = {
            "losses": losses, "accs_final": accs_final, "accs_best": accs_best,
        }
        if losses:
            print(f"{v:<28}{np.mean(losses):>12.4f}{np.std(losses):>11.4f}"
                  f"{np.mean(accs_final):>11.4f}{np.std(accs_final):>10.4f}"
                  f"{np.mean(accs_best):>11.4f}{resets:>9d}{dp_str:>10}")

    # ---- Paired d analyses.
    print("\nPaired d analyses (improvement minus baseline, bootstrap 95% CI):")
    print("-" * 100)
    print(f"{'comparison':<46}{'dloss mean':>14}{'dacc mean':>14}{'dacc 95% CI':>26}")
    print("-" * 100)

    baselines = ["vanilla", "dmezo_n"]
    improvements = [
        "dmezo_n_drift", "dmezo_n_adaptive_clip", "dmezo_n_dp", "dmezo_n_combo",
    ]
    for baseline in baselines:
        if baseline not in variants:
            continue
        for imp in improvements:
            if imp not in variants:
                continue
            # Paired: same seed for both.
            diffs_loss = []
            diffs_acc = []
            for s in seeds:
                ka = f"{imp}|seed={s}"
                kb = f"{baseline}|seed={s}"
                if ka not in cells or kb not in cells:
                    continue
                # dloss: positive = baseline wins on loss (so neg = improvement wins).
                # Convention: dloss = imp - baseline (negative = imp lower -> wins).
                diffs_loss.append(
                    cells[ka]["eval_losses"][-1] - cells[kb]["eval_losses"][-1]
                )
                diffs_acc.append(
                    cells[ka]["eval_accs"][-1] - cells[kb]["eval_accs"][-1]
                )
            if not diffs_loss:
                continue
            m_loss = float(np.mean(diffs_loss))
            m_acc = float(np.mean(diffs_acc))
            ci_lo, ci_hi = _bootstrap_ci(diffs_acc)
            verdict_loss = "+" if m_loss < 0 else " "
            ci_excludes_zero = "*" if (ci_lo > 0 or ci_hi < 0) else " "
            print(f"{imp:<22} vs {baseline:<20}{verdict_loss}{m_loss:>+12.4f}"
                  f"{m_acc:>+14.4f}    [{ci_lo:+.4f}, {ci_hi:+.4f}]{ci_excludes_zero}")

    # ---- Best variant per metric.
    print("\nBest variants (mean over seeds):")
    print("-" * 100)
    best_loss = min(variant_stats.items(), key=lambda kv: np.mean(kv[1]["losses"]) if kv[1]["losses"] else 1e9)
    best_acc_final = max(variant_stats.items(), key=lambda kv: np.mean(kv[1]["accs_final"]) if kv[1]["accs_final"] else -1)
    best_acc_best = max(variant_stats.items(), key=lambda kv: np.mean(kv[1]["accs_best"]) if kv[1]["accs_best"] else -1)
    print(f"  Lowest final loss:  {best_loss[0]:<28} (mean = {np.mean(best_loss[1]['losses']):.4f})")
    print(f"  Highest final acc:  {best_acc_final[0]:<28} (mean = {np.mean(best_acc_final[1]['accs_final']):.4f})")
    print(f"  Highest peak  acc:  {best_acc_best[0]:<28} (mean = {np.mean(best_acc_best[1]['accs_best']):.4f})")

    # ---- Adaptive-clip threshold trajectory (only for adaptive_clip variant).
    if "dmezo_n_adaptive_clip" in variants:
        print("\nAdaptive clip threshold trajectory (per seed, every 25 rounds):")
        print("-" * 100)
        for s in seeds:
            cell = cells.get(f"dmezo_n_adaptive_clip|seed={s}")
            if cell is None:
                continue
            # Reconstruct trajectory from eval_steps if available.
            # We logged ac_thr in stdout but didn't save to JSON. Just note presence.
            print(f"  seed={s}: (see training log for ac_thr trajectory)")

    print("\nLegend:")
    print("  + before dloss = improvement wins (lower loss).")
    print("  * after CI = 95% CI excludes 0 (statistically significant).")
    print("=" * 100)


if __name__ == "__main__":
    main()
