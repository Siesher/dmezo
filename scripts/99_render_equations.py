"""Render LaTeX equations to PNG for embedding in the .docx paper.

Output: docs/figures/eq_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

EQUATIONS = {
    "mezo_grad": r"\hat g_i^t = \frac{L(\theta_i^t + \epsilon z) - L(\theta_i^t - \epsilon z)}{2\epsilon}",
    "mezo_update": r"\theta \leftarrow \theta - \eta \cdot \hat\rho \cdot z",
    "spectral_gap": r"\rho(W) := \| W - n^{-1} 11^\top \|_{op} \in [0,1)",
    "round_step": r"v_i^{t+1} = \beta_t v_i^t + clip(\hat\rho_i^t, \pm C)\,z_{s_i^t}, \quad \theta_i^{t+1/2} = \theta_i^t - \eta v_i^{t+1}, \quad \theta_i^{t+1} = \sum_{j} W_{ij}\,\theta_j^{t+1/2}",
    "theorem1_bound": r"E[L(\bar\theta_T) - L^*] \leq \tilde{O}\left(\sqrt{\frac{L\,r(H)\,\Delta_0}{n\,T}}\right) + \tilde{O}\left(\frac{\rho^{2}\,C^{2}\,r(H)}{(1-\bar\beta)^{2}\,T}\right) + O(\epsilon^{2} L^{2} r(H))",
    "theorem2_bound": r"E[L(\bar\theta_T) - L^*] \leq (1-\eta\mu)^{T} \Delta_{0} + \tilde{O}\left(\frac{\eta L\,r(H)\,G^{2}}{\mu n}\right) + \tilde{O}\left(\frac{\eta^{2}\rho^{2}L^{2}r(H)G^{2}}{\mu(1-\rho)^{2}}\right) + O\left(\frac{\epsilon^{2} L^{2} r(H)}{\mu}\right)",
    "pl_condition": r"\| \nabla L(\theta) \|^{2} \geq 2\mu (L(\theta) - L^*) \quad \forall \theta \in R^{d}",
    "zo_variance": r"E_{z}\,\| \hat\rho \cdot z \|^{2} \leq 2(r(H)+1) \|\nabla L\|^{2} + \epsilon^{2} L^{2}\,r(H)",
    "clip_variance": r"E\,\| \tilde\rho\,z \|^{2} \leq \min(E\,\| \hat\rho\,z \|^{2},\; C^{2}\,d)",
    "consensus_error": r"\frac{1}{n}\sum_{i} \| \theta_i^{t+1} - \bar\theta_{t+1} \|^{2} \leq \frac{\rho^{2}}{(1-\rho)^{2}} \eta^{2} (G^{2} r(H) + \zeta^{2})",
    "pl_descent": r"E[f(\theta_{t+1}) - f^*] \leq (1-\eta\mu)\,E[f(\theta_t) - f^*] + \frac{\eta^{2}L\sigma^{2}}{2} + \frac{\eta\delta^{2}}{\mu}",
    "communication": r"Comm = O(1)\ scalar + 1\ int\ seed\ per\ neighbour\ per\ round",
}


def render(eq_name: str, latex: str, font_size: int = 14, scale: float = 1.0):
    fig, ax = plt.subplots(figsize=(7 * scale, 0.9 * scale))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"${latex}$",
        ha="center",
        va="center",
        fontsize=font_size,
        color="black",
    )
    out_path = OUT / f"eq_{eq_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print("Rendering equations...")
    # Larger size for the main theorems (longer formulas).
    big_eqs = {"theorem1_bound", "theorem2_bound", "round_step"}
    for name, latex in EQUATIONS.items():
        scale = 1.4 if name in big_eqs else 1.0
        path = render(name, latex, scale=scale)
        print(f"  {path}")
    print(f"\nDone. {len(EQUATIONS)} equations saved to {OUT}")
