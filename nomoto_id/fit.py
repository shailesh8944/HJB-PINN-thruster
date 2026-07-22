"""First-order step-response fitting utilities."""

import numpy as np


def _first_order_response(t, amp, tau, y0=0.0):
    """y(t) = y0 + (amp - y0) * (1 - exp(-t/tau))."""
    tau = max(float(tau), 1e-6)
    return y0 + (amp - y0) * (1.0 - np.exp(-t / tau))


def fit_first_order(t, y, t_settle_frac=0.25):
    """
    Fit y(t) ≈ y0 + (y_ss - y0) * (1 - exp(-t/T)).

    Returns dict with y0, y_ss, T, rmse, r2.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(t) < 5:
        raise ValueError("Need at least 5 samples to fit.")

    t = t - t[0]
    y0 = float(y[0])

    n_tail = max(5, int(len(y) * t_settle_frac))
    y_ss = float(np.mean(y[-n_tail:]))

    amp = y_ss - y0
    if abs(amp) < 1e-8:
        return {
            "y0": y0,
            "y_ss": y_ss,
            "T": float("nan"),
            "rmse": 0.0,
            "r2": 1.0,
            "ok": False,
            "reason": "no steady-state change",
        }

    # Linearized: ln|1 - (y-y0)/amp| = -t/T
    frac = np.clip((y - y0) / amp, 1e-6, 1.0 - 1e-6)
    # Use only early-to-mid rise (10%–90%)
    mask = (frac > 0.05) & (frac < 0.95)
    if np.count_nonzero(mask) < 5:
        mask = np.ones_like(frac, dtype=bool)

    z = -np.log(1.0 - frac[mask])
    tt = t[mask]
    # z = tt / T  =>  T = mean(tt / z) with positivity
    with np.errstate(divide="ignore", invalid="ignore"):
        T_est = tt / z
    T_est = T_est[np.isfinite(T_est) & (T_est > 0)]
    if len(T_est) == 0:
        return {
            "y0": y0,
            "y_ss": y_ss,
            "T": float("nan"),
            "rmse": float("nan"),
            "r2": 0.0,
            "ok": False,
            "reason": "failed time-constant estimate",
        }

    T = float(np.median(T_est))

    # Optional refinement: 1-D grid search around T
    candidates = T * np.linspace(0.4, 2.5, 45)
    best_T, best_err = T, np.inf
    for Tc in candidates:
        y_hat = _first_order_response(t, y_ss, Tc, y0)
        err = float(np.mean((y - y_hat) ** 2))
        if err < best_err:
            best_err = err
            best_T = float(Tc)

    y_hat = _first_order_response(t, y_ss, best_T, y0)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)

    return {
        "y0": y0,
        "y_ss": y_ss,
        "T": best_T,
        "rmse": rmse,
        "r2": r2,
        "ok": True,
        "y_hat": y_hat,
    }


def aggregate_gains(inputs, steady_values, time_constants):
    """
    Aggregate k = y_ss / u over multiple steps.

    Prefer median of valid positive points.
    """
    ks = []
    Ts = []
    for u, yss, T in zip(inputs, steady_values, time_constants):
        if abs(u) < 1e-9 or not np.isfinite(yss) or not np.isfinite(T):
            continue
        if T <= 0:
            continue
        ks.append(yss / u)
        Ts.append(T)
    if not ks:
        return float("nan"), float("nan"), []
    return float(np.median(ks)), float(np.median(Ts)), list(zip(ks, Ts))
