"""
SECTION 10: OPTIMAL CONTROL EXTRACTION
======================================
"""

from typing import List, Tuple

import torch

from .config import GameConfig
from .hamiltonian import compute_drift_S
from .network import NormalizedValueNetwork


def extract_optimal_controls(
    model: NormalizedValueNetwork,
    zeta: torch.Tensor,
    t: torch.Tensor,
    config: GameConfig,
) -> dict:
    """
    Given a trained model, extract the optimal (s*, w*) for
    both evader and pursuer at any state.

    Process:
        1. Forward pass → V_θ(ζ, t)
        2. Autograd → ∇_ζ V_θ
        3. Compute α_V, α_r from the gradient
        4. Evaluate 4 diamond vertices
        5. Evader picks max, pursuer picks min

    Args:
        model: trained PINN
        zeta: (batch, 7) — query states
        t:    (batch, 1) — query times
        config: problem parameters
    Returns:
        dict with optimal controls and diagnostics
    """
    zeta = zeta.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    V = model(zeta, t)

    # ∇_ζ V
    grad_zeta = torch.autograd.grad(
        V, zeta, grad_outputs=torch.ones_like(V),
        create_graph=False, retain_graph=False,
    )[0]  # (batch, 7)

    p_Ve = grad_zeta[:, 3:4]
    p_Vi = grad_zeta[:, 4:5]
    p_re = grad_zeta[:, 5:6]
    p_ri = grad_zeta[:, 6:7]

    e = config.evader
    p_cfg = config.pursuer

    # Evader α values
    alpha_V_e = p_Ve * e.k_V / e.T_V
    alpha_r_e = p_re * e.k_r / e.T_r

    # Pursuer α values
    alpha_V_i = p_Vi * p_cfg.k_V / p_cfg.T_V
    alpha_r_i = p_ri * p_cfg.k_r / p_cfg.T_r

    # --- Evader: pick the vertex that maximizes α_V s + α_r w ---
    M_A_e = 2 * alpha_V_e * e.n_max
    M_B_e = (alpha_V_e + alpha_r_e) * e.n_max
    M_C_e = (alpha_V_e - alpha_r_e) * e.n_max
    M_D_e = torch.zeros_like(M_A_e)

    vertices_e = torch.stack([M_A_e, M_B_e, M_C_e, M_D_e], dim=-1)
    best_e = torch.argmax(vertices_e.squeeze(-2), dim=-1)  # Index of winning vertex

    # Map index to physical controls (must match zeta / best_* device)
    device = zeta.device
    dtype = zeta.dtype
    vertex_map_e = torch.tensor([
        [2 * e.n_max, 0.0],            # A: full speed
        [e.n_max, e.n_max],             # B: port turn
        [e.n_max, -e.n_max],            # C: starboard turn
        [0.0, 0.0],                     # D: dead stop
    ], device=device, dtype=dtype)

    s_e_star = vertex_map_e[best_e, 0]
    w_e_star = vertex_map_e[best_e, 1]

    # --- Pursuer: pick the vertex that minimizes ---
    M_A_i = 2 * alpha_V_i * p_cfg.n_max
    M_B_i = (alpha_V_i + alpha_r_i) * p_cfg.n_max
    M_C_i = (alpha_V_i - alpha_r_i) * p_cfg.n_max
    M_D_i = torch.zeros_like(M_A_i)

    vertices_i = torch.stack([M_A_i, M_B_i, M_C_i, M_D_i], dim=-1)
    best_i = torch.argmin(vertices_i.squeeze(-2), dim=-1)

    vertex_map_i = torch.tensor([
        [2 * p_cfg.n_max, 0.0],
        [p_cfg.n_max, p_cfg.n_max],
        [p_cfg.n_max, -p_cfg.n_max],
        [0.0, 0.0],
    ], device=device, dtype=dtype)

    s_i_star = vertex_map_i[best_i, 0]
    w_i_star = vertex_map_i[best_i, 1]

    return {
        'V': V.detach(),
        'grad': grad_zeta.detach(),
        'evader': {
            's_star': s_e_star, 'w_star': w_e_star,
            'vertex': best_e,
            'alpha_V': alpha_V_e.detach(), 'alpha_r': alpha_r_e.detach(),
        },
        'pursuer': {
            's_star': s_i_star, 'w_star': w_i_star,
            'vertex': best_i,
            'alpha_V': alpha_V_i.detach(), 'alpha_r': alpha_r_i.detach(),
        },
    }


def extract_multi_pursuer_controls(
    model: NormalizedValueNetwork,
    zetas: torch.Tensor,
    t: torch.Tensor,
    config: GameConfig,
) -> dict:
    """
    Multi-pursuer optimal control: the evader maximizes min_i H_i(v_E)
    across ALL pairs simultaneously, not just the most threatening one.

    For each evader diamond vertex v_E ∈ {A, B, C, D}:
      For each pair i ∈ {1, 2, 3}:
        H_i(v_E) = S_i + Φ_E^i(v_E) + Φ_P^i*
      score(v_E) = min_i H_i(v_E)
    Evader picks: v_E* = argmax over v_E of score(v_E)

    This is 4 × 3 = 12 scalar evaluations. The key difference from
    single-pair control: the evader accounts for ALL pursuers when
    choosing its escape direction, not just the closest one.

    Pursuer controls remain per-pair (each minimizes its own H_i).

    Args:
        model:  trained PINN
        zetas:  (n_pursuers, 7) — relative states for each pair
        t:      (n_pursuers, 1) — PINN time for each pair
        config: game parameters
    Returns:
        dict with evader (s*, w*, vertex) and per-pursuer controls
    """
    n_p = zetas.shape[0]  # number of pursuers
    device = zetas.device
    dtype = zetas.dtype
    e = config.evader
    p_cfg = config.pursuer

    zetas_g = zetas.clone().requires_grad_(True)
    t_g = t.clone().requires_grad_(True)

    V = model(zetas_g, t_g)  # (n_p, 1)

    grad_zeta = torch.autograd.grad(
        V, zetas_g, grad_outputs=torch.ones_like(V),
        create_graph=False, retain_graph=False,
    )[0]  # (n_p, 7)

    # ------------------------------------------------------------------
    # Per-pair α values
    # ------------------------------------------------------------------
    # Evader α (from ∂V_i/∂V_e and ∂V_i/∂r_e — DIFFERENT for each pair)
    alpha_V_e = grad_zeta[:, 3:4] * e.k_V / e.T_V   # (n_p, 1)
    alpha_r_e = grad_zeta[:, 5:6] * e.k_r / e.T_r   # (n_p, 1)

    # Pursuer α (each pursuer has its own gradient)
    alpha_V_i = grad_zeta[:, 4:5] * p_cfg.k_V / p_cfg.T_V  # (n_p, 1)
    alpha_r_i = grad_zeta[:, 6:7] * p_cfg.k_r / p_cfg.T_r  # (n_p, 1)

    # ------------------------------------------------------------------
    # Compute S_i + Φ_P^i* for each pair (evader-control-independent)
    # ------------------------------------------------------------------
    S = compute_drift_S(zetas_g.detach(), grad_zeta.detach(), e, p_cfg)  # (n_p, 1)

    # Pursuer optimal: min over 4 vertices
    M_A_p = 2 * alpha_V_i * p_cfg.n_max
    M_B_p = (alpha_V_i + alpha_r_i) * p_cfg.n_max
    M_C_p = (alpha_V_i - alpha_r_i) * p_cfg.n_max
    M_D_p = torch.zeros_like(M_A_p)
    Phi_P = torch.min(
        torch.stack([M_A_p, M_B_p, M_C_p, M_D_p], dim=-1), dim=-1
    ).values  # (n_p, 1)

    # Pursuer vertex indices for output
    best_p = torch.argmin(
        torch.stack([M_A_p, M_B_p, M_C_p, M_D_p], dim=-1).squeeze(-2), dim=-1
    )  # (n_p,)

    # Baseline per pair (everything except evader control)
    baseline = S + Phi_P  # (n_p, 1)

    # ------------------------------------------------------------------
    # Evader: evaluate all 4 vertices against ALL pairs
    # ------------------------------------------------------------------
    # For vertex v_E = (s_e, w_e), contribution to pair i:
    #   Φ_E^i(v_E) = α_V^{e,i} · s_e + α_r^{e,i} · w_e
    #
    # H_i(v_E) = baseline_i + Φ_E^i(v_E)
    #
    # Evader picks: argmax_{v_E} min_i H_i(v_E)

    evader_vertices_sw = torch.tensor([
        [2 * e.n_max, 0.0],          # A: full speed
        [e.n_max,     e.n_max],      # B: port turn
        [e.n_max,    -e.n_max],      # C: starboard turn
        [0.0,         0.0],          # D: dead stop
    ], device=device, dtype=dtype)  # (4, 2)

    # Φ_E^i(v_E) for all pairs × all vertices
    # alpha_V_e: (n_p, 1), evader_vertices_sw[:, 0]: (4,)
    # → Phi_E_all: (n_p, 4)
    Phi_E_all = (
        alpha_V_e * evader_vertices_sw[:, 0].unsqueeze(0)  # (n_p, 4)
        + alpha_r_e * evader_vertices_sw[:, 1].unsqueeze(0)  # (n_p, 4)
    )

    # H_i(v_E) for all pairs × all vertices: (n_p, 4)
    H_all = baseline + Phi_E_all  # broadcasting (n_p, 1) + (n_p, 4) → (n_p, 4)

    # For each vertex, take min over pairs (worst case for evader)
    # H_all: (n_p, 4) → min over dim=0 → (4,)
    worst_case_per_vertex = H_all.min(dim=0).values  # (4,)

    # Evader picks the vertex that maximizes the worst case
    best_e_vertex = torch.argmax(worst_case_per_vertex)  # scalar

    s_e_star = evader_vertices_sw[best_e_vertex, 0]
    w_e_star = evader_vertices_sw[best_e_vertex, 1]

    # ------------------------------------------------------------------
    # Pursuer controls: per-pair (each minimizes its own H_i)
    # ------------------------------------------------------------------
    pursuer_vertices_sw = torch.tensor([
        [2 * p_cfg.n_max, 0.0],
        [p_cfg.n_max,     p_cfg.n_max],
        [p_cfg.n_max,    -p_cfg.n_max],
        [0.0,             0.0],
    ], device=device, dtype=dtype)

    s_i_star = pursuer_vertices_sw[best_p, 0]  # (n_p,)
    w_i_star = pursuer_vertices_sw[best_p, 1]  # (n_p,)

    return {
        'V': V.detach(),
        'grad': grad_zeta.detach(),
        'evader': {
            's_star': s_e_star,
            'w_star': w_e_star,
            'vertex': best_e_vertex,
            'H_per_vertex': worst_case_per_vertex.detach(),  # for diagnostics
        },
        'pursuer': {
            's_star': s_i_star,
            'w_star': w_i_star,
            'vertex': best_p,
            'alpha_V': alpha_V_i.detach(),
            'alpha_r': alpha_r_i.detach(),
        },
    }
