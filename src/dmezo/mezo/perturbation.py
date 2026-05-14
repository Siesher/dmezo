"""In-place seed-based perturbation for MeZO.

This module implements the canonical MeZO trick: given a seed, deterministically
regenerate a Gaussian perturbation `z` layer-by-layer and apply it in-place to
the model parameters. Because `z` is regenerated rather than stored, memory cost
stays at inference level.

Reference: princeton-nlp/MeZO `large_models/trainer.py::zo_perturb_parameters`.

Critical invariants:
    - Always use `param.data = param.data + ...` (NOT `param = ...` or `+=`):
      we must keep tensor identities so optimizer/scheduler references stay valid.
    - The same seed MUST produce the same `z` across calls. Hence `torch.manual_seed`
      is called once at the start of perturbation, and we iterate parameters in a
      deterministic order.
    - Bias / LayerNorm parameters are typically perturbed too (Malladi 2023 does
      not exclude them from perturbation, only from weight decay in the update).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


def perturb_parameters(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    *,
    seed: int,
    scaling_factor: float,
    eps: float,
) -> None:
    """Apply scaled in-place Gaussian perturbation to parameters.

    Updates each parameter as::

        param.data <- param.data + scaling_factor * z * eps

    where ``z ~ N(0, I)`` is regenerated deterministically from ``seed``.

    Args:
        named_params: Iterable of ``(name, parameter)`` pairs. MUST be iterated
            in a deterministic order across calls with the same seed, otherwise
            the regenerated ``z`` will not match.
        seed: Integer seed for ``torch.manual_seed`` controlling ``z``.
        scaling_factor: Multiplier applied to ``z * eps`` (typically +1, -2, +1
            during one MeZO step to do +eps perturb, -eps perturb, restore).
        eps: MeZO perturbation magnitude (typically 1e-3).

    Note:
        Per-parameter ``torch.normal`` calls are batched here. This matches the
        Princeton reference. Using a single global RNG state ensures ``z`` is
        identical given the same seed.
    """
    torch.manual_seed(seed)
    for _name, param in named_params:
        z = torch.normal(
            mean=0.0,
            std=1.0,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
        )
        param.data.add_(z, alpha=scaling_factor * eps)


def regenerate_z_inplace(
    param: nn.Parameter,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Convenience: regenerate one parameter's `z` matching `perturb_parameters`.

    Used during the update step where we need `z` per parameter as we iterate.
    Assumes the caller has already called ``torch.manual_seed(seed)`` at the
    same point in the iteration order.

    Args:
        param: Parameter to generate `z` for (uses its shape/device/dtype).
        generator: Optional explicit generator. If None, uses global PRNG state.

    Returns:
        Tensor of the same shape/device/dtype as ``param.data``.
    """
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=param.data.size(),
        device=param.data.device,
        dtype=param.data.dtype,
        generator=generator,
    )
