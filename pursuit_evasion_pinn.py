"""
PINN Solver for Multi-Pursuer Single-Evader Differential Game
==============================================================

Solves the Hamilton-Jacobi-Isaacs Variational Inequality (HJI-VI):

    ∂V/∂t + min[0, H*(ζ, ∇V)] = ε ΔV      (vanishing viscosity form)

where:
    H* = S(ζ, p) + Φ_E(α_V^e, α_r^e) + Φ_P(α_V^i, α_r^i)

Terminal condition:
    V(ζ, 0) = √(X² + Y²) - ρ

State vector (7D, in evader body frame):
    ζ = (X_i, Y_i, ψ_rel, V_e, V_i, r_e, r_i)

Both evader and each pursuer use twin thrusters with diamond constraints.

Setup:
    - 3 pursuers at vertices of equilateral triangle (radius R)
    - Evader at center
    - All initial velocities and yaw rates = 0
    - Solved pairwise: one V_i per pursuer-evader pair

Author: Shailesh Yadav (MAVLab, IIT Madras)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple
import time
import os


# ============================================================
# SECTION 1: VESSEL PARAMETERS
# ============================================================
# These come from your Nomoto identification and thruster specs.
# Both players have the SAME control structure (twin thrusters,
# diamond constraint) but may have different parameters.

@dataclass
class VesselParams:
    """Second-order Nomoto vessel parameters for one agent."""
    n_max: float    # Max single thruster RPM
    k_V: float      # Speed gain: steady-state speed = k_V * s
    T_V: float      # Speed time constant [s]
    k_r: float      # Yaw gain: steady-state yaw rate = k_r * w
    T_r: float      # Yaw time constant [s]

    @property
    def V_max(self) -> float:
        """Maximum steady-state speed (both thrusters full)."""
        return self.k_V * 2 * self.n_max

    @property
    def r_max(self) -> float:
        """Maximum steady-state yaw rate (one thruster full, other off)."""
        return self.k_r * self.n_max


# ============================================================
# SECTION 2: PROBLEM CONFIGURATION
# ============================================================

@dataclass
class GameConfig:
    """Full problem specification."""

    # Capture radius [m]
    rho: float = 50.0

    # Time horizon [s] — game runs backward from t=0 to t=-T_f
    T_f: float = 300.0

    # Equilateral triangle radius [m] — distance from center to each pursuer
    R_triangle: float = 500.0

    # --- Vessel parameters ---
    # Evader (same type of vessel as pursuers in this setup)
    evader: VesselParams = field(default_factory=lambda: VesselParams(
        n_max=50.0,     # RPM
        k_V=0.10,       # m/s per RPM-sum
        T_V=20.0,       # s (ships are slow to accelerate)
        k_r=0.02,       # rad/s per RPM-diff
        T_r=5.0,        # s
    ))

    # Pursuer (identical vessels — change if pursuers differ)
    pursuer: VesselParams = field(default_factory=lambda: VesselParams(
        n_max=50.0,
        k_V=0.10,
        T_V=20.0,
        k_r=0.02,
        T_r=5.0,
    ))

    # --- PINN training ---
    n_pde: int = 10000          # PDE collocation points per batch
    n_tc: int = 2000            # Terminal condition points per batch
    n_ic: int = 1000            # Initial condition points (for regularization)
    lambda_tc: float = 10.0     # Weight on terminal condition loss
    lambda_ic: float = 1.0      # Weight on initial condition loss
    lr: float = 1e-3            # Learning rate
    epochs_per_eps: int = 5000  # Training epochs per ε stage
    eps_schedule: list = field(default_factory=lambda: [
        1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001
    ])

    def initial_pursuer_positions(self) -> list:
        """
        Returns [(X1, Y1), (X2, Y2), (X3, Y3)] — pursuer positions
        in evader's body frame at t = -T_f.

        Equilateral triangle centered at origin:
          Pursuer 1: (R, 0)         — directly ahead
          Pursuer 2: (-R/2, R√3/2)  — port quarter
          Pursuer 3: (-R/2, -R√3/2) — starboard quarter
        """
        R = self.R_triangle
        return [
            (R, 0.0),
            (-R / 2, R * np.sqrt(3) / 2),
            (-R / 2, -R * np.sqrt(3) / 2),
        ]


# ============================================================
# SECTION 3: NEURAL NETWORK
# ============================================================
# V_θ(ζ, t) : R^8 → R
# Input:  (X, Y, ψ_rel, V_e, V_i, r_e, r_i, t)
# Output: scalar value function

class ValueNetwork(nn.Module):
    """
    Feedforward neural network approximating V(ζ, t).

    Architecture: 8 inputs → [hidden layers with tanh] → 1 output

    Why tanh?
    - Smooth and infinitely differentiable (C^∞)
    - Required because we need ∂²V/∂ζ² via autograd
    - ReLU would give zero second derivatives everywhere
    """

    def __init__(self, n_hidden: int = 128, n_layers: int = 5):
        super().__init__()

        layers = []
        # Input layer: 8 inputs (7 states + time)
        layers.append(nn.Linear(8, n_hidden))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())

        # Output layer: 1 output (value function)
        layers.append(nn.Linear(n_hidden, 1))

        self.net = nn.Sequential(*layers)

        # Xavier initialization for better training with tanh
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, zeta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zeta: (batch, 7) — state vector [X, Y, ψ_rel, V_e, V_i, r_e, r_i]
            t:    (batch, 1) — time (runs from -T_f to 0)
        Returns:
            V:    (batch, 1) — value function
        """
        # Concatenate state and time → (batch, 8)
        inputs = torch.cat([zeta, t], dim=-1)
        return self.net(inputs)


# ============================================================
# SECTION 4: DOMAIN SAMPLING
# ============================================================
# We need random points in the 7D+time domain for training.

class DomainSampler:
    """
    Samples collocation points in the state-time domain.

    Bounds for each state dimension:
        X_i:    [-2R, 2R]          (pursuer can be up to 2× triangle radius away)
        Y_i:    [-2R, 2R]
        ψ_rel:  [-π, π]            (full heading range)
        V_e:    [0, V_max^e]       (speed is non-negative)
        V_i:    [0, V_max^i]
        r_e:    [-r_max^e, r_max^e] (yaw rate can be positive or negative)
        r_i:    [-r_max^i, r_max^i]
        t:      [-T_f, 0]          (backward time)
    """

    def __init__(self, config: GameConfig, device: torch.device):
        self.config = config
        self.device = device

        R = config.R_triangle
        e = config.evader
        p = config.pursuer

        # State bounds: (7, 2) — [lower, upper] for each dimension
        self.zeta_low = torch.tensor([
            -2 * R,         # X
            -2 * R,         # Y
            -np.pi,         # ψ_rel
            0.0,            # V_e (speed >= 0)
            0.0,            # V_i
            -e.r_max,       # r_e
            -p.r_max,       # r_i
        ], device=device)

        self.zeta_high = torch.tensor([
            2 * R,          # X
            2 * R,          # Y
            np.pi,          # ψ_rel
            e.V_max,        # V_e
            p.V_max,        # V_i
            e.r_max,        # r_e
            p.r_max,        # r_i
        ], device=device)

    def sample_pde(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n collocation points in the interior of the domain.
        Returns: (zeta, t) — both require grad for autograd.
        """
        # Uniform random in [0, 1], then scale to bounds
        zeta = (
            torch.rand(n, 7, device=self.device)
            * (self.zeta_high - self.zeta_low)
            + self.zeta_low
        )
        # Time: uniform in [-T_f, 0]
        t = torch.rand(n, 1, device=self.device) * (-self.config.T_f)

        zeta.requires_grad_(True)
        t.requires_grad_(True)
        return zeta, t

    def sample_terminal(self, n: int) -> torch.Tensor:
        """
        Sample n points at terminal time t = 0.
        Returns: zeta (no grad needed — this is the boundary condition).
        """
        zeta = (
            torch.rand(n, 7, device=self.device)
            * (self.zeta_high - self.zeta_low)
            + self.zeta_low
        )
        return zeta

    def sample_initial_condition(self, n: int, pursuer_idx: int) -> torch.Tensor:
        """
        Sample n points near the actual initial state at t = -T_f.
        Pursuer at equilateral triangle vertex, evader at origin.
        All velocities and yaw rates = 0 (with small perturbation for coverage).

        This is a soft constraint to help the PINN learn the physically
        relevant region of state space (not just random points).
        """
        positions = self.config.initial_pursuer_positions()
        X0, Y0 = positions[pursuer_idx]

        # Small Gaussian perturbation around the true initial state
        sigma = torch.tensor([
            100.0,      # X: ±100m around true position
            100.0,      # Y
            0.5,        # ψ_rel: ±0.5 rad
            0.5,        # V_e: near 0
            0.5,        # V_i: near 0
            0.1,        # r_e: near 0
            0.1,        # r_i: near 0
        ], device=self.device)

        mean = torch.tensor([
            X0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], device=self.device)

        zeta = mean + sigma * torch.randn(n, 7, device=self.device)

        # Clamp speeds to be non-negative
        zeta[:, 3] = torch.clamp(zeta[:, 3], min=0.0)
        zeta[:, 4] = torch.clamp(zeta[:, 4], min=0.0)

        return zeta


# ============================================================
# SECTION 5: HAMILTONIAN COMPUTATION
# ============================================================
# This is the core physics. Every term maps to the equations of motion.

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


# ============================================================
# SECTION 6: AUTOGRAD — COMPUTING DERIVATIVES
# ============================================================
# The PINN needs ∂V/∂t, ∇_ζ V (7 first derivatives),
# and ΔV (7 second derivatives) via automatic differentiation.

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


# ============================================================
# SECTION 7: PINN LOSS FUNCTION
# ============================================================

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


# ============================================================
# SECTION 8: INPUT NORMALIZATION
# ============================================================
# Raw state values span wildly different ranges:
#   X ∈ [-1000, 1000], ψ_rel ∈ [-π, π], r ∈ [-1, 1]
# Normalizing to [-1, 1] helps the network train.

class NormalizedValueNetwork(nn.Module):
    """
    Wraps ValueNetwork with input normalization.
    Maps each state dimension to approximately [-1, 1] before
    feeding into the network.
    """

    def __init__(self, config: GameConfig, n_hidden: int = 128, n_layers: int = 5):
        super().__init__()
        self.net = ValueNetwork(n_hidden, n_layers)

        R = config.R_triangle
        e = config.evader
        p = config.pursuer

        # Normalization: center and scale for each dimension
        self.register_buffer('zeta_mean', torch.tensor([
            0.0, 0.0, 0.0,
            e.V_max / 2, p.V_max / 2,
            0.0, 0.0
        ]))
        self.register_buffer('zeta_scale', torch.tensor([
            R, R, np.pi,
            e.V_max / 2, p.V_max / 2,
            e.r_max, p.r_max
        ]))
        self.register_buffer('t_mean', torch.tensor([-config.T_f / 2]))
        self.register_buffer('t_scale', torch.tensor([config.T_f / 2]))

    def forward(self, zeta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        zeta_norm = (zeta - self.zeta_mean) / self.zeta_scale
        t_norm = (t - self.t_mean) / self.t_scale
        return self.net(zeta_norm, t_norm)


# ============================================================
# SECTION 9: TRAINING LOOP WITH ε-SCHEDULING
# ============================================================

def train_pinn(
    config: GameConfig,
    pursuer_idx: int = 0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> NormalizedValueNetwork:
    """
    Train the PINN with ε-scheduling (vanishing viscosity).

    Schedule: ε = 1.0 → 0.5 → 0.25 → ... → 0.001
    At each stage, warm-start from the previous converged weights.

    Args:
        config: full problem configuration
        pursuer_idx: which pursuer-evader pair (0, 1, or 2)
        device: CPU or CUDA
        verbose: print training progress
    Returns:
        model: trained NormalizedValueNetwork
    """
    model = NormalizedValueNetwork(config).to(device)
    sampler = DomainSampler(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2
    )

    history = []

    for stage, epsilon in enumerate(config.eps_schedule):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ε-STAGE {stage + 1}/{len(config.eps_schedule)}: ε = {epsilon}")
            print(f"{'='*60}")

        for epoch in range(config.epochs_per_eps):
            optimizer.zero_grad()

            loss, info = pinn_loss(
                model, sampler, config, epsilon, pursuer_idx
            )

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            history.append(info)

            if verbose and (epoch + 1) % 500 == 0:
                print(
                    f"  Epoch {epoch+1:5d}/{config.epochs_per_eps} | "
                    f"L_total={info['L_total']:.4e} | "
                    f"L_pde={info['L_pde']:.4e} | "
                    f"L_tc={info['L_tc']:.4e} | "
                    f"residual_max={info['residual_max']:.4e}"
                )

    if verbose:
        print(f"\nTraining complete. Final loss: {history[-1]['L_total']:.4e}")

    return model, history


# ============================================================
# SECTION 10: OPTIMAL CONTROL EXTRACTION
# ============================================================

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

    # Map index to physical controls
    vertex_map_e = torch.tensor([
        [2 * e.n_max, 0.0],            # A: full speed
        [e.n_max, e.n_max],             # B: port turn
        [e.n_max, -e.n_max],            # C: starboard turn
        [0.0, 0.0],                     # D: dead stop
    ])

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
    ])

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


# ============================================================
# SECTION 11: VISUALIZATION
# ============================================================

def plot_value_function_slice(
    model: NormalizedValueNetwork,
    config: GameConfig,
    t_query: float = -150.0,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'value_function_slice.png',
):
    """
    Plot V(X, Y) for fixed (ψ_rel=0, V_e=0, V_i=0, r_e=0, r_i=0)
    at a specific time t_query.

    This is a 2D slice through the 7D value function —
    the zero-level set shows the capture boundary in position space.
    """
    n_grid = 200
    R = config.R_triangle
    x_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    y_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    XX, YY = torch.meshgrid(x_range, y_range, indexing='ij')

    # Flatten to (n_grid², 7)
    X_flat = XX.reshape(-1, 1)
    Y_flat = YY.reshape(-1, 1)
    n_pts = X_flat.shape[0]

    # Fixed states: ψ_rel=0, V_e=0, V_i=0, r_e=0, r_i=0
    zeta = torch.cat([
        X_flat,
        Y_flat,
        torch.zeros(n_pts, 1, device=device),  # ψ_rel
        torch.zeros(n_pts, 1, device=device),  # V_e
        torch.zeros(n_pts, 1, device=device),  # V_i
        torch.zeros(n_pts, 1, device=device),  # r_e
        torch.zeros(n_pts, 1, device=device),  # r_i
    ], dim=1)

    t = torch.full((n_pts, 1), t_query, device=device)

    with torch.no_grad():
        V = model(zeta, t).reshape(n_grid, n_grid)

    V_np = V.cpu().numpy()
    XX_np = XX.cpu().numpy()
    YY_np = YY.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Color map: blue = evader escapes (V > 0), red = capture (V < 0)
    cf = ax.contourf(XX_np, YY_np, V_np, levels=50, cmap='RdBu')
    plt.colorbar(cf, ax=ax, label='V(X, Y)')

    # Zero contour = capture boundary
    ax.contour(XX_np, YY_np, V_np, levels=[0], colors='black', linewidths=2)

    # Mark evader position (origin)
    ax.plot(0, 0, 'g*', markersize=15, label='Evader')

    # Mark pursuer initial positions
    for idx, (xp, yp) in enumerate(config.initial_pursuer_positions()):
        ax.plot(xp, yp, 'r^', markersize=12, label=f'Pursuer {idx+1}' if idx == 0 else None)

    # Capture circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(config.rho * np.cos(theta), config.rho * np.sin(theta),
            'k--', linewidth=1, alpha=0.5, label=f'Capture radius ρ={config.rho}m')

    ax.set_xlabel('X [m] (ahead of evader)')
    ax.set_ylabel('Y [m] (port of evader)')
    ax.set_title(
        f'Value function slice at t = {t_query:.0f}s\n'
        f'(ψ_rel=0, V_e=V_i=0, r_e=r_i=0)\n'
        f'Black contour = capture boundary (V=0)'
    )
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(
    history: list,
    save_path: str = 'training_history.png',
):
    """Plot loss curves across all ε stages."""
    epochs = range(len(history))
    L_pde = [h['L_pde'] for h in history]
    L_tc = [h['L_tc'] for h in history]
    L_total = [h['L_total'] for h in history]
    epsilons = [h['epsilon'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogy(epochs, L_total, 'b-', alpha=0.7, linewidth=0.5, label='Total')
    ax1.semilogy(epochs, L_pde, 'r-', alpha=0.5, linewidth=0.5, label='PDE')
    ax1.semilogy(epochs, L_tc, 'g-', alpha=0.5, linewidth=0.5, label='Terminal')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend()
    ax1.set_title('PINN Training History')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, epsilons, 'k-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ε (diffusion)')
    ax2.set_title('ε-Schedule (vanishing viscosity)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_optimal_controls(
    model: NormalizedValueNetwork,
    config: GameConfig,
    t_query: float = -150.0,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'optimal_controls.png',
):
    """
    Plot which diamond vertex is optimal across the (X, Y) plane.
    Shows the switching surfaces between control strategies.
    """
    n_grid = 150
    R = config.R_triangle
    x_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    y_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    XX, YY = torch.meshgrid(x_range, y_range, indexing='ij')

    X_flat = XX.reshape(-1, 1)
    Y_flat = YY.reshape(-1, 1)
    n_pts = X_flat.shape[0]

    zeta = torch.cat([
        X_flat, Y_flat,
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
    ], dim=1)

    t = torch.full((n_pts, 1), t_query, device=device)

    controls = extract_optimal_controls(model, zeta, t, config)

    vertex_names = ['A: Full speed', 'B: Port turn', 'C: Starboard turn', 'D: Stop']
    cmap = plt.cm.get_cmap('Set1', 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Evader control map
    ev = controls['evader']['vertex'].reshape(n_grid, n_grid).cpu().numpy()
    im1 = ax1.pcolormesh(XX.cpu(), YY.cpu(), ev, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax1.set_title(f'Evader optimal vertex at t={t_query:.0f}s')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
    cbar1.set_ticklabels(vertex_names)

    # Pursuer control map
    pv = controls['pursuer']['vertex'].reshape(n_grid, n_grid).cpu().numpy()
    im2 = ax2.pcolormesh(XX.cpu(), YY.cpu(), pv, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax2.set_title(f'Pursuer optimal vertex at t={t_query:.0f}s')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
    cbar2.set_ticklabels(vertex_names)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# SECTION 12: MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  PINN Solver: Pursuit-Evasion Viscosity Solution")
    print("  3 Pursuers (equilateral triangle) vs 1 Evader")
    print("  Diamond constraint on all vessels")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # --- Configuration ---
    # For a quick test run, use reduced settings.
    # For a real solve, increase n_pde, epochs_per_eps, and use full eps_schedule.
    config = GameConfig(
        rho=50.0,
        T_f=300.0,
        R_triangle=500.0,
        n_pde=4000,             # Reduce for quick test (use 10000+ for real)
        n_tc=1000,
        n_ic=500,
        epochs_per_eps=2000,    # Reduce for quick test (use 5000+ for real)
        eps_schedule=[1.0, 0.5, 0.1, 0.01],  # Shortened for test
    )

    print(f"\nProblem setup:")
    print(f"  Capture radius: {config.rho} m")
    print(f"  Time horizon: {config.T_f} s")
    print(f"  Triangle radius: {config.R_triangle} m")
    print(f"  Evader V_max: {config.evader.V_max:.1f} m/s")
    print(f"  Pursuer V_max: {config.pursuer.V_max:.1f} m/s")
    print(f"  ε-schedule: {config.eps_schedule}")

    # --- Solve for each pursuer-evader pair ---
    # Due to the equilateral symmetry and identical vessels,
    # we only need to solve pair 0 — the others are related by
    # 120° rotation. But the code supports solving all 3.

    pursuer_idx = 0
    pos = config.initial_pursuer_positions()
    print(f"\nSolving pair {pursuer_idx}: pursuer at ({pos[pursuer_idx][0]:.0f}, {pos[pursuer_idx][1]:.0f})")

    t_start = time.time()
    model, history = train_pinn(config, pursuer_idx, device, verbose=True)
    t_elapsed = time.time() - t_start
    print(f"\nTraining time: {t_elapsed:.1f}s")

    # --- Save model ---
    save_dir = '/mnt/user-data/outputs'
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'pinn_model_pair0.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'history': history,
    }, model_path)
    print(f"Model saved: {model_path}")

    # --- Visualize ---
    print("\nGenerating plots...")

    plot_training_history(
        history,
        save_path=os.path.join(save_dir, 'training_history.png')
    )

    plot_value_function_slice(
        model, config,
        t_query=-config.T_f / 2,  # Mid-game
        device=device,
        save_path=os.path.join(save_dir, 'value_function_slice.png')
    )

    plot_optimal_controls(
        model, config,
        t_query=-config.T_f / 2,
        device=device,
        save_path=os.path.join(save_dir, 'optimal_controls.png')
    )

    # --- Extract control at a specific state ---
    print("\n--- Example control extraction ---")
    test_zeta = torch.tensor([[
        500.0, 0.0,     # Pursuer directly ahead at 500m
        0.0,            # Same heading as evader
        0.0, 0.0,       # Both at rest
        0.0, 0.0,       # No yaw
    ]], device=device)
    test_t = torch.tensor([[-150.0]], device=device)

    ctrl = extract_optimal_controls(model, test_zeta, test_t, config)

    vertex_names = ['A (full speed)', 'B (port turn)', 'C (starboard turn)', 'D (stop)']
    print(f"  State: pursuer at (500, 0)m, all velocities zero, t=-150s")
    print(f"  V = {ctrl['V'].item():.2f}")
    print(f"  Evader: vertex {vertex_names[ctrl['evader']['vertex'].item()]}")
    print(f"    s* = {ctrl['evader']['s_star'].item():.1f}, w* = {ctrl['evader']['w_star'].item():.1f}")
    print(f"  Pursuer: vertex {vertex_names[ctrl['pursuer']['vertex'].item()]}")
    print(f"    s* = {ctrl['pursuer']['s_star'].item():.1f}, w* = {ctrl['pursuer']['w_star'].item():.1f}")

    print("\nDone.")
