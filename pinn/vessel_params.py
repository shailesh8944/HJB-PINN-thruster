"""
SECTION 1: VESSEL PARAMETERS
============================
These come from your Nomoto identification and thruster specs.
Both players have the SAME control structure (twin thrusters,
diamond constraint) but may have different parameters.
"""

from dataclasses import dataclass


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
