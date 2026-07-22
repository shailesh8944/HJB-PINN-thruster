"""Reduced-order twin-thruster Nomoto vessel model."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class VesselParams:
    """Second-order Nomoto parameters (same layout as files/pinn/vessel_params.py)."""

    n_max: float  # Max |actuator| or RPM for one thruster
    k_V: float  # V_ss = k_V * (n_port + n_stbd)
    T_V: float  # Surge time constant [s]
    k_r: float  # r_ss = k_r * (n_port - n_stbd)
    T_r: float  # Yaw time constant [s]

    @property
    def V_max(self) -> float:
        return self.k_V * 2.0 * self.n_max

    @property
    def r_max(self) -> float:
        return self.k_r * self.n_max

    def to_dict(self):
        d = asdict(self)
        d["V_max"] = self.V_max
        d["r_max"] = self.r_max
        return d


class NomotoVessel:
    """
    Planar twin-thruster Nomoto dynamics:

        V_dot = (-V + k_V * (n_p + n_s)) / T_V
        r_dot = (-r + k_r * (n_p - n_s)) / T_r
        psi_dot = r
        x_dot = V * cos(psi)
        y_dot = V * sin(psi)

    Thruster commands n_p, n_s are clipped to [-n_max, n_max].
    """

    def __init__(self, params: VesselParams, dt: float = 0.1):
        self.params = params
        self.dt = float(dt)
        self.reset()

    def reset(self, x=0.0, y=0.0, psi=0.0, V=0.0, r=0.0):
        self.x = float(x)
        self.y = float(y)
        self.psi = float(psi)
        self.V = float(V)
        self.r = float(r)
        self.t = 0.0

    def step(self, n_port: float, n_stbd: float):
        p = self.params
        n_p = float(np.clip(n_port, -p.n_max, p.n_max))
        n_s = float(np.clip(n_stbd, -p.n_max, p.n_max))

        s = n_p + n_s
        w = n_p - n_s

        self.V += self.dt * (-self.V + p.k_V * s) / max(p.T_V, 1e-6)
        self.r += self.dt * (-self.r + p.k_r * w) / max(p.T_r, 1e-6)
        self.psi += self.dt * self.r
        self.x += self.dt * self.V * np.cos(self.psi)
        self.y += self.dt * self.V * np.sin(self.psi)
        self.t += self.dt

        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "psi": self.psi,
            "V": self.V,
            "r": self.r,
            "n_port": n_p,
            "n_stbd": n_s,
        }

    def simulate(self, n_port, n_stbd, duration, reset_state=True):
        """Constant thruster commands for `duration` seconds."""
        if reset_state:
            self.reset()
        n_steps = int(np.ceil(duration / self.dt))
        log = {
            "t": [],
            "V": [],
            "r": [],
            "psi": [],
            "x": [],
            "y": [],
            "n_port": [],
            "n_stbd": [],
        }
        for _ in range(n_steps):
            s = self.step(n_port, n_stbd)
            for k in log:
                log[k].append(s[k])
        for k in log:
            log[k] = np.asarray(log[k], dtype=float)
        return log
