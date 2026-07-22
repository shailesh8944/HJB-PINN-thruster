"""
SECTION 2: PROBLEM CONFIGURATION
================================
"""

from dataclasses import dataclass, field

import numpy as np

from .vessel_params import VesselParams


@dataclass
class GameConfig:
    """Full problem specification."""

    # Capture radius [m] — USV scale (sookshma ~1 m long)
    rho: float = 2.0

    # Time horizon [s] — game runs backward from t=0 to t=-T_f
    T_f: float = 90.0

    # Equilateral triangle radius [m] — distance from center to each pursuer
    R_triangle: float = 30.0

    # --- Vessel parameters (sookshma, identified from Makara open-loop steps) ---
    # Units: thruster commands are Makara actuators in [-1, 1], so n_max=1.
    # Re-identify with:  cd files && python3 -m nomoto_id.cli
    # Result file:       files/nomoto_id/results/nomoto_params.yml
    evader: VesselParams = field(default_factory=lambda: VesselParams(
        n_max=1.0,
        k_V=0.870970,   # V_ss = k_V * (n_port + n_stbd)
        T_V=3.2596,     # s
        k_r=1.020237,   # r_ss = k_r * (n_port - n_stbd)
        T_r=0.9236,     # s
    ))

    # Pursuer (identical vessels — change if pursuers differ)
    pursuer: VesselParams = field(default_factory=lambda: VesselParams(
        n_max=1.0,
        k_V=0.870970,
        T_V=3.2596,
        k_r=1.020237,
        T_r=0.9236,
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
