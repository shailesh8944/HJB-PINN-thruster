"""
PINN Solver for Multi-Pursuer Single-Evader Differential Game.

Package modules mirror the original monolithic sections.
Equations and training flow are unchanged — only file layout.
"""

from .vessel_params import VesselParams
from .config import GameConfig
from .network import ValueNetwork, NormalizedValueNetwork
from .sampling import DomainSampler
from .hamiltonian import (
    compute_drift_S,
    compute_Phi_evader,
    compute_Phi_pursuer,
    compute_full_hamiltonian,
)
from .gradients import compute_gradients
from .loss import pinn_loss
from .training import train_pinn
from .controls import extract_optimal_controls
from .visualization import (
    plot_value_function_slice,
    plot_training_history,
    plot_optimal_controls,
)

__all__ = [
    'VesselParams',
    'GameConfig',
    'ValueNetwork',
    'NormalizedValueNetwork',
    'DomainSampler',
    'compute_drift_S',
    'compute_Phi_evader',
    'compute_Phi_pursuer',
    'compute_full_hamiltonian',
    'compute_gradients',
    'pinn_loss',
    'train_pinn',
    'extract_optimal_controls',
    'plot_value_function_slice',
    'plot_training_history',
    'plot_optimal_controls',
]
