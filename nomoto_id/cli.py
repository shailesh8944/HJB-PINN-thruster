#!/usr/bin/env python3
"""
Identify Nomoto twin-thruster parameters from Makara sookshma open-loop steps,
then optionally replay the reduced Nomoto model for validation.

Usage (from repo root or files/):
    python -m nomoto_id.cli
    python -m nomoto_id.cli --surge-duration 50 --no-plots
    python -m nomoto_id.cli --simulate   # also run a short Nomoto square path demo

Outputs:
    files/nomoto_id/results/nomoto_params.yml
    files/nomoto_id/results/nomoto_params.json
    files/nomoto_id/results/nomoto_validation.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python files/nomoto_id/cli.py` and `python -m nomoto_id.cli`
_HERE = Path(__file__).resolve().parent
_FILES = _HERE.parent
if str(_FILES) not in sys.path:
    sys.path.insert(0, str(_FILES))

from nomoto_id.identify import identify_nomoto
from nomoto_id.model import NomotoVessel


def _demo_square(params, duration_leg=20.0, dt=0.1, output_dir=None):
    """Drive Nomoto around a rough square with constant legs + spot turns."""
    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skip square demo")
        return

    vessel = NomotoVessel(params, dt=dt)
    vessel.reset()
    xs, ys = [0.0], [0.0]

    def cruise(n, T):
        steps = int(T / dt)
        for _ in range(steps):
            vessel.step(n, n)
            xs.append(vessel.x)
            ys.append(vessel.y)

    def turn90(n, T):
        # differential turn with slight forward bias
        steps = int(T / dt)
        for _ in range(steps):
            vessel.step(n, -0.2 * n)
            xs.append(vessel.x)
            ys.append(vessel.y)

    n = min(0.65, params.n_max)
    for _ in range(4):
        cruise(n, duration_leg)
        turn90(n, 8.0)

    output_dir = Path(output_dir or _HERE / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, lw=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Nomoto reduced model — open-loop square demo")
    path = output_dir / "nomoto_square_demo.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Identify Nomoto params from Makara")
    parser.add_argument("--sim-config", default=None, help="Makara simulation_input.yml")
    parser.add_argument("--surge-levels", nargs="+", type=float, default=[0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--yaw-levels", nargs="+", type=float, default=[0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--surge-duration", type=float, default=45.0)
    parser.add_argument("--yaw-duration", type=float, default=35.0)
    parser.add_argument("--n-max", type=float, default=1.0, help="Actuator limit (default 1.0)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Also plot Nomoto square demo")
    args = parser.parse_args()

    params, _ = identify_nomoto(
        sim_config=args.sim_config,
        surge_levels=args.surge_levels,
        yaw_levels=args.yaw_levels,
        surge_duration=args.surge_duration,
        yaw_duration=args.yaw_duration,
        n_max=args.n_max,
        output_dir=args.output,
        make_plots=not args.no_plots,
    )

    if args.simulate:
        _demo_square(params, output_dir=args.output)


if __name__ == "__main__":
    main()
