"""
SECTION 3: NEURAL NETWORK
=========================
V_θ(ζ, t) : R^8 → R
Input:  (X, Y, ψ_rel, V_e, V_i, r_e, r_i, t)
Output: scalar value function

SECTION 8: INPUT NORMALIZATION
==============================
Raw state values span wildly different ranges:
  X ∈ [-1000, 1000], ψ_rel ∈ [-π, π], r ∈ [-1, 1]
Normalizing to [-1, 1] helps the network train.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import GameConfig


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
