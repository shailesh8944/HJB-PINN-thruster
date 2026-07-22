"""
Closed-loop 3-pursuer / 1-evader simulation using Nomoto dynamics
and (optional) PINN optimal diamond controls.

Relative state ζ matches the PINN:
  [X, Y, ψ_rel, V_e, V_i, r_e, r_i]  — pursuer pose in evader body frame.
Game clock for the network: t_pinn = -T_f + t_sim  ∈ [-T_f, 0].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nomoto_id.model import NomotoVessel, VesselParams
from pinn.config import GameConfig
from pinn.controls import extract_multi_pursuer_controls, extract_optimal_controls
from pinn.network import NormalizedValueNetwork


def ssa(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def sw_to_thrusters(s: float, w: float, n_max: float) -> Tuple[float, float]:
    """Diamond (s, w) → (n_port, n_stbd)."""
    n_p = 0.5 * (s + w)
    n_s = 0.5 * (s - w)
    return float(np.clip(n_p, -n_max, n_max)), float(np.clip(n_s, -n_max, n_max))


def heading_to_thrusters(
    vessel: NomotoVessel,
    desired_heading: float,
    n_max: float,
    aggressiveness: float = 1.0,
    kp_psi: float = 2.2,
    kd_r: float = 0.8,
) -> Tuple[float, float]:
    """
    Aim the bow at desired_heading and surge forward.

    Raw diamond vertices (port/stbd only) make the boat spin in place.
    This maps a desired world heading into differential thrusters with
    enough common-mode thrust to actually translate.
    """
    aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
    if aggressiveness < 1e-3:
        return 0.0, 0.0

    hdg_err = ssa(desired_heading - vessel.psi)
    yaw = float(np.clip(kp_psi * hdg_err - kd_r * vessel.r, -n_max, n_max))

    # Slow while badly misaligned; keep a floor so we don't stall
    align = max(0.25, float(np.cos(hdg_err)))
    base = aggressiveness * n_max * align
    base = max(base, 0.35 * aggressiveness * n_max)

    port = float(np.clip(base + yaw, -n_max, n_max))
    stbd = float(np.clip(base - yaw, -n_max, n_max))
    return port, stbd


def world_to_relative(evader: NomotoVessel, pursuer: NomotoVessel) -> np.ndarray:
    """Pursuer state in evader body frame → ζ (7,)."""
    dx = pursuer.x - evader.x
    dy = pursuer.y - evader.y
    c, s = np.cos(evader.psi), np.sin(evader.psi)
    X = c * dx + s * dy
    Y = -s * dx + c * dy
    psi_rel = ssa(pursuer.psi - evader.psi)
    return np.array(
        [X, Y, psi_rel, evader.V, pursuer.V, evader.r, pursuer.r],
        dtype=float,
    )


@dataclass
class GameResult:
    captured: bool
    capture_time: Optional[float]
    reason: str
    t: np.ndarray
    evader_xy: np.ndarray
    pursuer_xy: List[np.ndarray]
    distances: np.ndarray
    V_values: np.ndarray


class PursuitEvasionSim:
    def __init__(
        self,
        config: GameConfig,
        model: Optional[NormalizedValueNetwork] = None,
        device: Optional[torch.device] = None,
        dt: float = 0.05,
        mode: str = "hjb",
    ):
        """
        mode:
          'hjb'       — pure HJB diamond (s*,w*) from ∇V  [theory-correct]
          'guided'    — PINN threat/aggression + geometric LOS (demo movie)
          'heuristic' — chase / flee, no network
        """
        self.config = config
        self.model = model
        self.device = device or torch.device("cpu")
        self.dt = float(dt)
        self.mode = mode
        if mode in ("hjb", "guided", "pinn") and model is None:
            raise ValueError(f"{mode} mode requires a loaded model")
        # alias: old name 'pinn' → guided
        if mode == "pinn":
            self.mode = "guided"

        pe = config.evader
        pp = config.pursuer
        self.evader_params = VesselParams(pe.n_max, pe.k_V, pe.T_V, pe.k_r, pe.T_r)
        self.pursuer_params = VesselParams(pp.n_max, pp.k_V, pp.T_V, pp.k_r, pp.T_r)
        self._escape_heading = None  # used only in guided mode

    def _reset_agents(self) -> Tuple[NomotoVessel, List[NomotoVessel]]:
        self._escape_heading = None
        e = NomotoVessel(self.evader_params, dt=self.dt)
        e.reset(0.0, 0.0, 0.0, 0.0, 0.0)
        pursuers = []
        for (X, Y) in self.config.initial_pursuer_positions():
            # Triangle positions are in evader body frame at t=-T_f with psi_e=0
            p = NomotoVessel(self.pursuer_params, dt=self.dt)
            # Point roughly toward origin
            heading = float(np.arctan2(-Y, -X))
            p.reset(X, Y, heading, 0.0, 0.0)
            pursuers.append(p)
        return e, pursuers

    @staticmethod
    def _vertex_aggressiveness(vertex_idx: int) -> float:
        """Map diamond vertex → how hard to surge (0=stop … 1=full)."""
        # 0=A full, 1=B port, 2=C stbd, 3=D stop
        return {0: 1.0, 1: 0.75, 2: 0.75, 3: 0.15}.get(int(vertex_idx), 0.7)

    def _escape_heading_cmd(self, evader: NomotoVessel, pursuers: List[NomotoVessel]) -> float:
        """
        Desired world heading for the evader: into the largest angular gap.

        Bearings are from evader → each pursuer. The open sector midpoint is
        the flee direction (do NOT add π — that pointed back into the pack
        and made the boat circle).
        """
        angles = np.sort(
            np.array(
                [float(np.arctan2(p.y - evader.y, p.x - evader.x)) for p in pursuers],
                dtype=float,
            )
        )
        gaps = np.diff(angles, append=angles[0] + 2.0 * np.pi)
        k = int(np.argmax(gaps))
        raw = ssa(float(angles[k] + 0.5 * gaps[k]))

        # Also push away from pursuer centroid when not co-located
        cx = float(np.mean([p.x for p in pursuers]))
        cy = float(np.mean([p.y for p in pursuers]))
        dx_c, dy_c = evader.x - cx, evader.y - cy
        if np.hypot(dx_c, dy_c) > 0.5:
            away_c = float(np.arctan2(dy_c, dx_c))
            # Blend on unit circle (angle lerp is wrong across wrap)
            vx = 0.65 * np.cos(raw) + 0.35 * np.cos(away_c)
            vy = 0.65 * np.sin(raw) + 0.35 * np.sin(away_c)
            raw = float(np.arctan2(vy, vx))

        # Low-pass filter so the aim doesn't chatter → circling
        if self._escape_heading is None:
            self._escape_heading = raw
        else:
            self._escape_heading = ssa(
                self._escape_heading + 0.2 * ssa(raw - self._escape_heading)
            )
        return float(self._escape_heading)

    def _pinn_controls(
        self, evader: NomotoVessel, pursuers: List[NomotoVessel], t_sim: float
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float]], float]:
        """
        PINN-guided LOS control.

        Raw (s*, w*) diamond commands often pick pure differential thrust, so
        the boat spins and never closes range. Here the network still chooses:
          - which pursuer is most threatening (min V)
          - aggression / stop vs go (vertex)
        while geometry sets the aim direction so agents actually translate.
        """
        t_pinn = float(np.clip(-self.config.T_f + t_sim, -self.config.T_f, 0.0))

        zetas = [world_to_relative(evader, p) for p in pursuers]
        zeta_t = torch.tensor(np.stack(zetas), dtype=torch.float32, device=self.device)
        t_t = torch.full((len(pursuers), 1), t_pinn, dtype=torch.float32, device=self.device)

        ctrl = extract_optimal_controls(self.model, zeta_t, t_t, self.config)
        V = ctrl["V"].detach().cpu().numpy().ravel()
        threat = int(np.argmin(V))

        escape_dir = self._escape_heading_cmd(evader, pursuers)
        v_e = int(ctrl["evader"]["vertex"][threat].detach().cpu())
        # Light PINN turn bias only (don't override flee geometry)
        if v_e == 1:
            escape_dir = ssa(escape_dir + 0.15)
        elif v_e == 2:
            escape_dir = ssa(escape_dir - 0.15)

        # Always full surge while fleeing — circling came from half-thrust + wrong aim
        n_e = heading_to_thrusters(
            evader, escape_dir, self.evader_params.n_max,
            aggressiveness=1.0, kp_psi=1.8, kd_r=1.0,
        )

        n_p_list = []
        for i, p in enumerate(pursuers):
            toward = float(np.arctan2(evader.y - p.y, evader.x - p.x))
            v_p = int(ctrl["pursuer"]["vertex"][i].detach().cpu())
            if v_p == 1:
                toward = ssa(toward + 0.2)
            elif v_p == 2:
                toward = ssa(toward - 0.2)
            agg_p = self._vertex_aggressiveness(v_p)
            dist = float(np.hypot(evader.x - p.x, evader.y - p.y))
            if dist > 2.5 * self.config.rho:
                agg_p = max(agg_p, 0.85)
            n_p_list.append(
                heading_to_thrusters(p, toward, self.pursuer_params.n_max, agg_p)
            )

        return n_e, n_p_list, float(V[threat])

    def _hjb_controls(
        self, evader: NomotoVessel, pursuers: List[NomotoVessel], t_sim: float
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float]], float]:
        """
        Theory-correct closed loop from the HJB saddle, with
        MULTI-PURSUER evader control.

        Evader:  argmax_{v_E} min_i H_i(v_E)  — maximizes worst-case
                 Hamiltonian across ALL pairs simultaneously.
        Pursuer: argmin_vertex per-pair  — each minimizes its own H_i.

        This prevents the evader from fleeing one pursuer straight
        into another.
        """
        t_pinn = float(np.clip(-self.config.T_f + t_sim, -self.config.T_f, 0.0))
        zetas = [world_to_relative(evader, p) for p in pursuers]
        zeta_t = torch.tensor(np.stack(zetas), dtype=torch.float32, device=self.device)
        t_t = torch.full((len(pursuers), 1), t_pinn, dtype=torch.float32, device=self.device)

        ctrl = extract_multi_pursuer_controls(self.model, zeta_t, t_t, self.config)
        V = ctrl["V"].detach().cpu().numpy().ravel()

        # Evader: single control chosen against ALL pursuers jointly
        s_e = float(ctrl["evader"]["s_star"].detach().cpu())
        w_e = float(ctrl["evader"]["w_star"].detach().cpu())
        n_e = sw_to_thrusters(s_e, w_e, self.evader_params.n_max)

        # Pursuers: per-pair optimal
        n_p_list = []
        for i in range(len(pursuers)):
            s_i = float(ctrl["pursuer"]["s_star"][i].detach().cpu())
            w_i = float(ctrl["pursuer"]["w_star"][i].detach().cpu())
            n_p_list.append(sw_to_thrusters(s_i, w_i, self.pursuer_params.n_max))

        return n_e, n_p_list, float(V.min())

    def _heuristic_controls(
        self, evader: NomotoVessel, pursuers: List[NomotoVessel]
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float]], float]:
        """Simple demo: each pursuer steers toward evader; evader flees closest pursuer."""
        n_max = self.pursuer_params.n_max
        dists = [np.hypot(p.x - evader.x, p.y - evader.y) for p in pursuers]
        nearest = int(np.argmin(dists))

        p_near = pursuers[nearest]
        away = float(np.arctan2(evader.y - p_near.y, evader.x - p_near.x))
        n_e = heading_to_thrusters(evader, away, n_max, aggressiveness=1.0)

        n_p_list = []
        for p in pursuers:
            toward = float(np.arctan2(evader.y - p.y, evader.x - p.x))
            n_p_list.append(heading_to_thrusters(p, toward, n_max, aggressiveness=1.0))

        return n_e, n_p_list, float(min(dists) - self.config.rho)

    def run(self, max_time: Optional[float] = None) -> GameResult:
        max_time = float(max_time if max_time is not None else self.config.T_f)
        evader, pursuers = self._reset_agents()
        rho = self.config.rho

        t_hist = [0.0]
        e_hist = [[evader.x, evader.y]]
        p_hist = [[[p.x, p.y] for p in pursuers]]
        d_hist = [[np.hypot(p.x - evader.x, p.y - evader.y) for p in pursuers]]
        v_hist = [0.0]

        captured = False
        capture_time = None
        reason = "time_expired"

        n_steps = int(np.ceil(max_time / self.dt))
        for k in range(n_steps):
            t_sim = (k + 1) * self.dt
            if self.mode == "hjb":
                n_e, n_ps, V_val = self._hjb_controls(evader, pursuers, t_sim - self.dt)
            elif self.mode == "guided":
                n_e, n_ps, V_val = self._pinn_controls(evader, pursuers, t_sim - self.dt)
            else:
                n_e, n_ps, V_val = self._heuristic_controls(evader, pursuers)

            evader.step(*n_e)
            for p, n_cmd in zip(pursuers, n_ps):
                p.step(*n_cmd)

            dists = [np.hypot(p.x - evader.x, p.y - evader.y) for p in pursuers]
            t_hist.append(t_sim)
            e_hist.append([evader.x, evader.y])
            p_hist.append([[p.x, p.y] for p in pursuers])
            d_hist.append(dists)
            v_hist.append(V_val)

            if min(dists) <= rho:
                captured = True
                capture_time = t_sim
                reason = "captured"
                break

        # Escape if horizon ended without capture
        if not captured:
            reason = "escaped"

        return GameResult(
            captured=captured,
            capture_time=capture_time,
            reason=reason,
            t=np.asarray(t_hist),
            evader_xy=np.asarray(e_hist),
            pursuer_xy=[np.asarray([row[i] for row in p_hist]) for i in range(len(pursuers))],
            distances=np.asarray(d_hist),
            V_values=np.asarray(v_hist),
        )


def load_pinn_model(
    checkpoint: Path,
    config: GameConfig,
    device: torch.device,
) -> NormalizedValueNetwork:
    model = NormalizedValueNetwork(config).to(device)
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model


def plot_game(result: GameResult, config: GameConfig, save_path: Path, title: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]

    ax.plot(result.evader_xy[:, 0], result.evader_xy[:, 1], "g-", lw=2, label="Evader")
    ax.plot(result.evader_xy[0, 0], result.evader_xy[0, 1], "g*", ms=14)
    ax.plot(result.evader_xy[-1, 0], result.evader_xy[-1, 1], "go", ms=8)

    colors = ["C3", "C1", "C4"]
    for i, traj in enumerate(result.pursuer_xy):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i % 3], lw=1.8, label=f"Pursuer {i+1}")
        ax.plot(traj[0, 0], traj[0, 1], "^", color=colors[i % 3], ms=10)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=colors[i % 3], ms=7)
        ax.add_patch(
            Circle(
                (traj[-1, 0], traj[-1, 1]),
                config.rho,
                fill=False,
                ls="--",
                color=colors[i % 3],
                alpha=0.35,
            )
        )

    ax.add_patch(
        Circle(
            (result.evader_xy[-1, 0], result.evader_xy[-1, 1]),
            config.rho,
            fill=False,
            ls=":",
            color="g",
            alpha=0.5,
        )
    )
    status = "CAPTURED" if result.captured else "ESCAPED"
    ax.set_title(title or f"Pursuit–evasion — {status}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    for i in range(result.distances.shape[1]):
        ax2.plot(result.t, result.distances[:, i], color=colors[i % 3], label=f"d → P{i+1}")
    ax2.axhline(config.rho, color="k", ls="--", label=f"capture ρ={config.rho} m")
    if result.captured and result.capture_time is not None:
        ax2.axvline(result.capture_time, color="r", ls=":", label="capture time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Distance [m]")
    ax2.set_title("Range to each pursuer")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {save_path}")


def animate_game(
    result: GameResult,
    config: GameConfig,
    save_path: Path,
    fps: int = 20,
    stride: int = 2,
):
    """Save an MP4/GIF animation of the chase."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    n = len(result.t)
    idx = np.arange(0, n, max(1, stride))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    xs = [result.evader_xy[:, 0]] + [p[:, 0] for p in result.pursuer_xy]
    ys = [result.evader_xy[:, 1]] + [p[:, 1] for p in result.pursuer_xy]
    pad = 5.0
    ax.set_xlim(min(x.min() for x in xs) - pad, max(x.max() for x in xs) + pad)
    ax.set_ylim(min(y.min() for y in ys) - pad, max(y.max() for y in ys) + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    colors = ["C3", "C1", "C4"]
    e_line, = ax.plot([], [], "g-", lw=2, label="Evader")
    e_dot, = ax.plot([], [], "g*", ms=14)
    p_lines = []
    p_dots = []
    for i in range(len(result.pursuer_xy)):
        (ln,) = ax.plot([], [], color=colors[i], lw=1.5, label=f"Pursuer {i+1}")
        (dt,) = ax.plot([], [], "^", color=colors[i], ms=10)
        p_lines.append(ln)
        p_dots.append(dt)
    capture_circle = Circle((0, 0), config.rho, fill=False, ls="--", color="r", lw=1.5)
    ax.add_patch(capture_circle)
    clock = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=8)

    def update(frame_i):
        k = int(idx[frame_i])
        e_line.set_data(result.evader_xy[: k + 1, 0], result.evader_xy[: k + 1, 1])
        e_dot.set_data([result.evader_xy[k, 0]], [result.evader_xy[k, 1]])
        for i, traj in enumerate(result.pursuer_xy):
            p_lines[i].set_data(traj[: k + 1, 0], traj[: k + 1, 1])
            p_dots[i].set_data([traj[k, 0]], [traj[k, 1]])
        # Draw capture circle on closest pursuer
        d = result.distances[k]
        j = int(np.argmin(d))
        capture_circle.center = (result.pursuer_xy[j][k, 0], result.pursuer_xy[j][k, 1])
        status = "CAPTURED" if (result.captured and result.capture_time and result.t[k] >= result.capture_time) else ""
        clock.set_text(f"t = {result.t[k]:.1f} s   min d = {d.min():.2f} m   {status}")
        return [e_line, e_dot, *p_lines, *p_dots, capture_circle, clock]

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".gif":
        anim.save(save_path, writer=PillowWriter(fps=fps))
    else:
        try:
            anim.save(save_path, fps=fps, dpi=120)
        except Exception:
            gif = save_path.with_suffix(".gif")
            anim.save(gif, writer=PillowWriter(fps=fps))
            save_path = gif
    plt.close(fig)
    print(f"Saved animation: {save_path}")
    return save_path
