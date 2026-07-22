"""
SECTION 4: DOMAIN SAMPLING
==========================
We need random points in the 7D+time domain for training.
"""

from typing import Tuple

import numpy as np
import torch

from .config import GameConfig


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
        # NOTE: sigma must match domain scale (R_triangle=30m, domain=[-60,60])
        sigma = torch.tensor([
            10.0,       # X: ±10m around true position (was 100 — way too large)
            10.0,       # Y
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
