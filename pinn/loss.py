"""
SECTION 7: PINN LOSS FUNCTION
=============================
"""

from typing import Tuple

import torch

from .config import GameConfig
from .gradients import compute_gradients
from .hamiltonian import compute_full_hamiltonian
from .network import ValueNetwork
from .sampling import DomainSampler


def pinn_loss(
    model: ValueNetwork,
    sampler: DomainSampler,
    config: GameConfig,
    epsilon: float,
    pursuer_idx: int = 0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the total PINN loss:

        L = L_pde + λ_tc · L_tc + λ_ic · L_ic

    where:
        L_pde = (1/N) Σ |r(ζ, t; θ)|²
              = (1/N) Σ |∂V/∂t + min[0, H*] - ε ΔV|²

        L_tc  = (1/N) Σ |V(ζ, 0) - (√(X²+Y²) - ρ)|²

        L_ic  = (1/N) Σ |soft constraint near initial state|²

    Args:
        model: the neural network V_θ
        sampler: generates collocation points
        config: problem parameters
        epsilon: current diffusion coefficient
        pursuer_idx: which pursuer (0, 1, or 2) we're solving for
    Returns:
        loss: scalar total loss
        info: dict with component losses for logging
    """

    # ----- PDE RESIDUAL LOSS -----
    # Sample collocation points in the interior
    zeta_pde, t_pde = sampler.sample_pde(config.n_pde)

    # Compute all derivatives
    V_pde, dV_dt, grad_zeta, laplacian = compute_gradients(model, zeta_pde, t_pde)

    # Compute the clamped Hamiltonian: min[0, H*]
    H_clamped = compute_full_hamiltonian(zeta_pde, grad_zeta, config)

    # PDE residual: ∂V/∂t + min[0, H*] - ε ΔV = 0
    residual = dV_dt + H_clamped - epsilon * laplacian

    L_pde = torch.mean(residual ** 2)

    # ----- TERMINAL CONDITION LOSS -----
    # At t = 0: V(ζ, 0) = √(X² + Y²) - ρ
    zeta_tc = sampler.sample_terminal(config.n_tc)
    t_tc = torch.zeros(config.n_tc, 1, device=zeta_tc.device)

    V_tc = model(zeta_tc, t_tc)

    # Target: signed distance to capture
    X_tc = zeta_tc[:, 0:1]
    Y_tc = zeta_tc[:, 1:2]
    target_tc = torch.sqrt(X_tc**2 + Y_tc**2) - config.rho

    L_tc = torch.mean((V_tc - target_tc) ** 2)

    # ----- INITIAL CONDITION LOSS (soft regularization) -----
    # Near the actual game start: pursuers at triangle vertices,
    # all velocities zero, t = -T_f
    zeta_ic = sampler.sample_initial_condition(config.n_ic, pursuer_idx)
    t_ic = torch.full(
        (config.n_ic, 1), -config.T_f, device=zeta_ic.device
    )

    V_ic = model(zeta_ic, t_ic)

    # At the initial time, V should be positive (pursuer hasn't caught evader yet)
    # because the pursuer starts at distance R >> ρ.
    # We use a soft constraint: V should be approximately R - ρ
    X_ic = zeta_ic[:, 0:1]
    Y_ic = zeta_ic[:, 1:2]
    target_ic = torch.sqrt(X_ic**2 + Y_ic**2) - config.rho
    # Only penalize if V is very far from the geometric distance
    L_ic = torch.mean((V_ic - target_ic) ** 2)

    # ----- TOTAL LOSS -----
    loss = L_pde + config.lambda_tc * L_tc + config.lambda_ic * L_ic

    info = {
        'L_pde': L_pde.item(),
        'L_tc': L_tc.item(),
        'L_ic': L_ic.item(),
        'L_total': loss.item(),
        'epsilon': epsilon,
        'V_mean': V_pde.mean().item(),
        'residual_max': residual.abs().max().item(),
    }

    return loss, info
