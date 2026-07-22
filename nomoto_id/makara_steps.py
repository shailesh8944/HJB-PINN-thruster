"""Open-loop Makara thruster step experiments for Nomoto ID."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES_DIR = HERE.parent
PROJECT_ROOT = FILES_DIR.parent
SINGLE_VESSEL = PROJECT_ROOT / "appolonius-fleet_thruster" / "single_vessel"
MAV_SIM_SRC = PROJECT_ROOT / "appolonius-fleet_thruster" / "makara" / "ros2_ws" / "src" / "mav_simulator"

for p in (str(MAV_SIM_SRC), str(SINGLE_VESSEL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import read_input_local  # noqa: E402
from mav_simulator.class_vessel import Vessel  # noqa: E402
from vessel_state import apply_rudder, apply_thrusters  # noqa: E402


DEFAULT_SIM_CONFIG = SINGLE_VESSEL / "config" / "simulation_input_single.yml"


def build_makara_vessel(sim_config: str | Path | None = None, verbose: bool = False):
    """Create sookshma Vessel with maintain_speed disabled for thruster ID."""
    sim_config = Path(sim_config or DEFAULT_SIM_CONFIG).resolve()
    _, agents = read_input_local(str(sim_config))
    agent = agents[0]
    agent["vessel_config"]["maintain_speed"] = False
    if not verbose:
        # Reduce thruster debug spam by raising log limit after init
        pass
    vessel = Vessel(agent["vessel_config"], agent["hydrodynamics"], vessel_id=0, ros_flag=False)
    vessel.maintain_speed = False
    # Silence most thrust logs after construction
    vessel._thrust_log_limit = 0
    return vessel


def run_open_loop(vessel, n_port, n_stbd, duration, settle_zero=True):
    """
    Run constant thruster commands from rest.

    Returns dict with t, u, r, x, y, psi, n_port, n_stbd.
    """
    vessel.reset()
    apply_rudder(vessel, 0.0)

    if settle_zero:
        # Brief zero-thrust settle
        apply_thrusters(vessel, 0.0, 0.0)
        for _ in range(5):
            vessel.step()
        vessel.reset()
        apply_rudder(vessel, 0.0)

    dt = float(vessel.dt)
    n_steps = int(np.ceil(duration / dt))
    log = {k: [] for k in ("t", "u", "r", "x", "y", "psi", "n_port", "n_stbd")}

    for _ in range(n_steps):
        apply_thrusters(vessel, n_port, n_stbd)
        apply_rudder(vessel, 0.0)
        vessel.step()
        s = vessel.current_state
        log["t"].append(float(vessel.t))
        log["u"].append(float(s[0]))
        log["r"].append(float(s[5]))
        log["x"].append(float(s[6]))
        log["y"].append(float(s[7]))
        log["psi"].append(float(s[11]))
        log["n_port"].append(float(n_port))
        log["n_stbd"].append(float(n_stbd))

    return {k: np.asarray(v, dtype=float) for k, v in log.items()}


def run_surge_steps(vessel, levels, duration=40.0):
    """Equal port/stbd thruster steps — identify k_V, T_V."""
    results = []
    for act in levels:
        act = float(act)
        log = run_open_loop(vessel, act, act, duration)
        results.append(
            {
                "kind": "surge",
                "n_port": act,
                "n_stbd": act,
                "s": act + act,  # n_p + n_s
                "log": log,
            }
        )
        print(f"  surge act={act:.2f}: u_final={log['u'][-1]:.4f} m/s")
    return results


def run_yaw_steps(vessel, levels, duration=30.0):
    """Pure differential steps — identify k_r, T_r."""
    results = []
    for act in levels:
        act = float(act)
        # Positive w = n_p - n_s = 2*act  => CCW for sookshma allocation
        log = run_open_loop(vessel, act, -act, duration)
        results.append(
            {
                "kind": "yaw",
                "n_port": act,
                "n_stbd": -act,
                "w": act - (-act),  # 2*act
                "log": log,
            }
        )
        print(f"  yaw  act=±{act:.2f}: r_final={log['r'][-1]:.4f} rad/s")
    return results
