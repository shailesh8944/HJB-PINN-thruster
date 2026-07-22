"""Identify Nomoto VesselParams from Makara open-loop data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .fit import aggregate_gains, fit_first_order
from .makara_steps import build_makara_vessel, run_surge_steps, run_yaw_steps
from .model import NomotoVessel, VesselParams


def _trim_transient(log, skip_s=0.5):
    t = log["t"]
    mask = t >= (t[0] + skip_s)
    out = {k: v[mask] for k, v in log.items()}
    out["t"] = out["t"] - out["t"][0]
    return out


def identify_nomoto(
    sim_config=None,
    surge_levels=None,
    yaw_levels=None,
    surge_duration=45.0,
    yaw_duration=35.0,
    n_max=1.0,
    output_dir=None,
    make_plots=True,
):
    """
    Run Makara thruster steps, fit Nomoto gains, validate with Nomoto sim.

    Parameters use *actuator* units in [-1, 1] (Makara PWM command),
    so n_max defaults to 1.0.
    """
    surge_levels = surge_levels or [0.45, 0.55, 0.65, 0.75]
    yaw_levels = yaw_levels or [0.45, 0.55, 0.65, 0.75]
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building Makara vessel (maintain_speed=False)...")
    vessel = build_makara_vessel(sim_config)

    print("\n=== Surge step tests (equal thrusters) ===")
    surge = run_surge_steps(vessel, surge_levels, duration=surge_duration)

    print("\n=== Yaw step tests (differential thrusters) ===")
    yaw = run_yaw_steps(vessel, yaw_levels, duration=yaw_duration)

    # --- Fit surge ---
    s_inputs, V_ss_list, T_V_list = [], [], []
    surge_fits = []
    for exp in surge:
        log = _trim_transient(exp["log"])
        fit = fit_first_order(log["t"], log["u"])
        fit["s"] = exp["s"]
        fit["n_port"] = exp["n_port"]
        fit["n_stbd"] = exp["n_stbd"]
        surge_fits.append(fit)
        if fit["ok"]:
            s_inputs.append(exp["s"])
            V_ss_list.append(fit["y_ss"])
            T_V_list.append(fit["T"])
            print(
                f"  fit surge s={exp['s']:.2f}: V_ss={fit['y_ss']:.4f}, "
                f"T_V={fit['T']:.3f}s, R²={fit['r2']:.3f}"
            )

    k_V, T_V, _ = aggregate_gains(s_inputs, V_ss_list, T_V_list)

    # --- Fit yaw ---
    w_inputs, r_ss_list, T_r_list = [], [], []
    yaw_fits = []
    for exp in yaw:
        log = _trim_transient(exp["log"])
        fit = fit_first_order(log["t"], log["r"])
        fit["w"] = exp["w"]
        fit["n_port"] = exp["n_port"]
        fit["n_stbd"] = exp["n_stbd"]
        yaw_fits.append(fit)
        if fit["ok"]:
            w_inputs.append(exp["w"])
            r_ss_list.append(fit["y_ss"])
            T_r_list.append(fit["T"])
            print(
                f"  fit yaw  w={exp['w']:.2f}: r_ss={fit['y_ss']:.4f}, "
                f"T_r={fit['T']:.3f}s, R²={fit['r2']:.3f}"
            )

    k_r, T_r, _ = aggregate_gains(w_inputs, r_ss_list, T_r_list)

    if not np.isfinite(k_V) or not np.isfinite(T_V):
        raise RuntimeError(
            "Surge identification failed. Use actuator levels above PWM deadband (~0.40)."
        )
    if not np.isfinite(k_r) or not np.isfinite(T_r):
        raise RuntimeError("Yaw identification failed.")

    params = VesselParams(
        n_max=float(n_max),
        k_V=float(k_V),
        T_V=float(T_V),
        k_r=float(k_r),
        T_r=float(T_r),
    )

    print("\n=== Identified Nomoto parameters (actuator units) ===")
    print(f"  n_max = {params.n_max:.4f}")
    print(f"  k_V   = {params.k_V:.6f}   (V_ss = k_V * (n_p + n_s))")
    print(f"  T_V   = {params.T_V:.4f} s")
    print(f"  k_r   = {params.k_r:.6f}   (r_ss = k_r * (n_p - n_s))")
    print(f"  T_r   = {params.T_r:.4f} s")
    print(f"  V_max = {params.V_max:.4f} m/s")
    print(f"  r_max = {params.r_max:.4f} rad/s ({np.rad2deg(params.r_max):.2f} deg/s)")

    # Validation: replay one surge + one yaw on Nomoto and Makara
    dt = float(vessel.dt)
    nomoto = NomotoVessel(params, dt=dt)
    val_act = float(surge_levels[min(1, len(surge_levels) - 1)])
    makara_surge = surge[min(1, len(surge) - 1)]["log"]
    nomoto_surge = nomoto.simulate(val_act, val_act, duration=surge_duration)

    val_yaw = float(yaw_levels[min(1, len(yaw_levels) - 1)])
    makara_yaw = yaw[min(1, len(yaw) - 1)]["log"]
    nomoto_yaw = nomoto.simulate(val_yaw, -val_yaw, duration=yaw_duration)

    result = {
        "params": params.to_dict(),
        "notes": {
            "units": "n_port/n_stbd are Makara thruster actuators in [-1, 1]",
            "n_max": "Set to 1.0 for actuator-normalized control (recommended for PINN)",
            "source": "Makara sookshma 6-DOF open-loop thruster steps",
            "model": "V_dot=(-V+k_V*(np+ns))/T_V; r_dot=(-r+k_r*(np-ns))/T_r",
        },
        "surge_levels": list(map(float, surge_levels)),
        "yaw_levels": list(map(float, yaw_levels)),
        "pinn_snippet": (
            f"VesselParams(n_max={params.n_max:.4f}, k_V={params.k_V:.6f}, "
            f"T_V={params.T_V:.4f}, k_r={params.k_r:.6f}, T_r={params.T_r:.4f})"
        ),
    }

    # Save YAML + JSON
    yaml_path = output_dir / "nomoto_params.yml"
    json_path = output_dir / "nomoto_params.json"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Save raw arrays (without fit y_hat which is large in nested)
    np.savez_compressed(
        output_dir / "identification_data.npz",
        surge_t=makara_surge["t"],
        surge_u_makara=makara_surge["u"],
        surge_u_nomoto=nomoto_surge["V"],
        yaw_t=makara_yaw["t"],
        yaw_r_makara=makara_yaw["r"],
        yaw_r_nomoto=nomoto_yaw["r"],
        surge_act=val_act,
        yaw_act=val_yaw,
    )

    if make_plots:
        _save_plots(
            output_dir,
            makara_surge,
            nomoto_surge,
            makara_yaw,
            nomoto_yaw,
            params,
            val_act,
            val_yaw,
        )

    print(f"\nSaved: {yaml_path}")
    print(f"Saved: {json_path}")
    print(f"PINN paste-in:\n  {result['pinn_snippet']}")
    return params, result


def _save_plots(output_dir, m_surge, n_surge, m_yaw, n_yaw, params, act_s, act_y):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(m_surge["t"], m_surge["u"], label="Makara u", lw=2)
    axes[0].plot(n_surge["t"], n_surge["V"], "--", label="Nomoto V", lw=2)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Surge speed [m/s]")
    axes[0].set_title(f"Surge step (n_p=n_s={act_s:.2f})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(m_yaw["t"], m_yaw["r"], label="Makara r", lw=2)
    axes[1].plot(n_yaw["t"], n_yaw["r"], "--", label="Nomoto r", lw=2)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Yaw rate [rad/s]")
    axes[1].set_title(f"Yaw step (n_p={act_y:.2f}, n_s={-act_y:.2f})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Nomoto ID: k_V={params.k_V:.4f}, T_V={params.T_V:.2f}s, "
        f"k_r={params.k_r:.4f}, T_r={params.T_r:.2f}s"
    )
    fig.tight_layout()
    path = output_dir / "nomoto_validation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved: {path}")
