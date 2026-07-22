"""
SECTION 5: HAMILTONIAN COMPUTATION
==================================
This is the core physics. Every term maps to the equations of motion.
"""

import torch

from .config import GameConfig
from .vessel_params import VesselParams


def compute_drift_S(
    zeta: torch.Tensor,
    p: torch.Tensor,
    evader: VesselParams,
    pursuer: VesselParams,
) -> torch.Tensor:
    """
    Compute the control-free drift Hamiltonian S(ζ, p).

    S = p_X (V_i cos ψ_rel - V_e + r_e Y)
      + p_Y (V_i sin ψ_rel - r_e X)
      + p_ψ (r_i - r_e)
      - p_Ve V_e / T_V^e
      - p_Vi V_i / T_V^i
      - p_re r_e / T_r^e
      - p_ri r_i / T_r^i

    Args:
        zeta: (batch, 7) — [X, Y, ψ_rel, V_e, V_i, r_e, r_i]
        p:    (batch, 7) — costate [p_X, p_Y, p_ψ, p_Ve, p_Vi, p_re, p_ri]
    Returns:
        S:    (batch, 1)
    """
    # Unpack states
    X       = zeta[:, 0:1]
    Y       = zeta[:, 1:2]
    psi_rel = zeta[:, 2:3]
    V_e     = zeta[:, 3:4]
    V_i     = zeta[:, 4:5]
    r_e     = zeta[:, 5:6]
    r_i     = zeta[:, 6:7]

    # Unpack costates
    p_X   = p[:, 0:1]
    p_Y   = p[:, 1:2]
    p_psi = p[:, 2:3]
    p_Ve  = p[:, 3:4]
    p_Vi  = p[:, 4:5]
    p_re  = p[:, 5:6]
    p_ri  = p[:, 6:7]

    # Compute each term of S
    # Term 1: p_X · (V_i cos ψ_rel - V_e + r_e · Y)
    S = p_X * (V_i * torch.cos(psi_rel) - V_e + r_e * Y)

    # Term 2: p_Y · (V_i sin ψ_rel - r_e · X)
    S = S + p_Y * (V_i * torch.sin(psi_rel) - r_e * X)

    # Term 3: p_ψ · (r_i - r_e)
    S = S + p_psi * (r_i - r_e)

    # Term 4: -p_Ve · V_e / T_V^e  (evader speed decay)
    S = S - p_Ve * V_e / evader.T_V

    # Term 5: -p_Vi · V_i / T_V^i  (pursuer speed decay)
    S = S - p_Vi * V_i / pursuer.T_V

    # Term 6: -p_re · r_e / T_r^e  (evader yaw decay)
    S = S - p_re * r_e / evader.T_r

    # Term 7: -p_ri · r_i / T_r^i  (pursuer yaw decay)
    S = S - p_ri * r_i / pursuer.T_r

    return S


def compute_Phi_evader(
    p: torch.Tensor,
    evader: VesselParams,
) -> torch.Tensor:
    """
    Compute Φ_E = max(M_A^e, M_B^e, M_C^e, 0) · n_max^e

    The evader MAXIMIZES the Hamiltonian (wants to escape).

    α_V^e = p_Ve · k_V^e / T_V^e
    α_r^e = p_re · k_r^e / T_r^e

    Four diamond vertices give:
        M_A = 2 α_V^e · n_max    (full speed, no turn)
        M_B = (α_V^e + α_r^e) · n_max  (port turn)
        M_C = (α_V^e - α_r^e) · n_max  (starboard turn)
        M_D = 0                  (dead stop)

    Args:
        p: (batch, 7) — costate vector
    Returns:
        Phi_E: (batch, 1) — evader's optimal contribution
    """
    p_Ve = p[:, 3:4]
    p_re = p[:, 5:6]

    alpha_V_e = p_Ve * evader.k_V / evader.T_V
    alpha_r_e = p_re * evader.k_r / evader.T_r

    n = evader.n_max

    # Evaluate at all 4 vertices
    M_A = 2 * alpha_V_e * n
    M_B = (alpha_V_e + alpha_r_e) * n
    M_C = (alpha_V_e - alpha_r_e) * n
    M_D = torch.zeros_like(M_A)

    # Evader maximizes → take the max across vertices
    # stack: (batch, 1, 4) → max over dim=-1 → (batch, 1)
    Phi_E = torch.max(torch.stack([M_A, M_B, M_C, M_D], dim=-1), dim=-1).values
    return Phi_E


def compute_Phi_pursuer(
    p: torch.Tensor,
    pursuer: VesselParams,
) -> torch.Tensor:
    """
    Compute Φ_P = min(M_A^i, M_B^i, M_C^i, 0) · n_max^i

    The pursuer MINIMIZES the Hamiltonian (wants to capture).

    Same α formulas, same vertices, but min instead of max.

    Args:
        p: (batch, 7) — costate vector
    Returns:
        Phi_P: (batch, 1) — pursuer's optimal contribution
    """
    p_Vi = p[:, 4:5]
    p_ri = p[:, 6:7]

    alpha_V_i = p_Vi * pursuer.k_V / pursuer.T_V
    alpha_r_i = p_ri * pursuer.k_r / pursuer.T_r

    n = pursuer.n_max

    # Evaluate at all 4 vertices
    M_A = 2 * alpha_V_i * n
    M_B = (alpha_V_i + alpha_r_i) * n
    M_C = (alpha_V_i - alpha_r_i) * n
    M_D = torch.zeros_like(M_A)

    # Pursuer minimizes → take the min across vertices
    # stack: (batch, 1, 4) → min over dim=-1 → (batch, 1)
    Phi_P = torch.min(torch.stack([M_A, M_B, M_C, M_D], dim=-1), dim=-1).values
    return Phi_P


def compute_full_hamiltonian(
    zeta: torch.Tensor,
    p: torch.Tensor,
    config: GameConfig,
) -> torch.Tensor:
    """
    Compute the complete saddle-point Hamiltonian:
        H* = S(ζ, p) + Φ_E(p) + Φ_P(p)

    Then apply the variational inequality:
        H_clamped = min(0, H*)

    The min[0, ...] enforces irreversible capture —
    once V ≤ 0, it cannot increase.

    Args:
        zeta: (batch, 7) — states
        p:    (batch, 7) — costates (= ∇_ζ V)
        config: GameConfig
    Returns:
        H_clamped: (batch, 1)
    """
    S = compute_drift_S(zeta, p, config.evader, config.pursuer)
    Phi_E = compute_Phi_evader(p, config.evader)
    Phi_P = compute_Phi_pursuer(p, config.pursuer)

    H_star = S + Phi_E + Phi_P

    # Variational inequality: min(0, H*)
    H_clamped = torch.min(torch.zeros_like(H_star), H_star)

    return H_clamped
