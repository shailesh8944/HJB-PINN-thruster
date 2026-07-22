"""
SECTION 6: AUTOGRAD — COMPUTING DERIVATIVES
===========================================
The PINN needs ∂V/∂t, ∇_ζ V (7 first derivatives),
and ΔV (7 second derivatives) via automatic differentiation.
"""

from typing import Tuple

import torch

from .network import ValueNetwork


def compute_gradients(
    model: ValueNetwork,
    zeta: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all required derivatives of V_θ(ζ, t) via autograd.

    Returns:
        V:         (batch, 1) — value function
        dV_dt:     (batch, 1) — ∂V/∂t
        grad_zeta: (batch, 7) — ∇_ζ V = (∂V/∂X, ∂V/∂Y, ..., ∂V/∂r_i)
        laplacian: (batch, 1) — ΔV = Σ_j ∂²V/∂ζ_j²
    """
    # Forward pass
    V = model(zeta, t)  # (batch, 1)

    # --- First derivatives ---
    # ∂V/∂ζ (all 7 components at once)
    grad_zeta = torch.autograd.grad(
        outputs=V,
        inputs=zeta,
        grad_outputs=torch.ones_like(V),
        create_graph=True,  # Need graph for second derivatives
        retain_graph=True,
    )[0]  # (batch, 7)

    # ∂V/∂t
    dV_dt = torch.autograd.grad(
        outputs=V,
        inputs=t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]  # (batch, 1)

    # --- Second derivatives (Laplacian) ---
    # ΔV = Σ_j ∂²V/∂ζ_j²
    # We compute each ∂²V/∂ζ_j² separately and sum
    laplacian = torch.zeros_like(V)  # (batch, 1)

    for j in range(7):
        # ∂V/∂ζ_j
        dV_dj = grad_zeta[:, j:j+1]  # (batch, 1)

        # ∂²V/∂ζ_j²
        d2V_dj2 = torch.autograd.grad(
            outputs=dV_dj,
            inputs=zeta,
            grad_outputs=torch.ones_like(dV_dj),
            create_graph=True,
            retain_graph=True,
        )[0][:, j:j+1]  # (batch, 1) — only the j-th component

        laplacian = laplacian + d2V_dj2

    return V, dV_dt, grad_zeta, laplacian
