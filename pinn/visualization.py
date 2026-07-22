"""
SECTION 11: VISUALIZATION
=========================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import GameConfig
from .controls import extract_optimal_controls
from .network import NormalizedValueNetwork


def plot_value_function_slice(
    model: NormalizedValueNetwork,
    config: GameConfig,
    t_query: float = -150.0,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'value_function_slice.png',
):
    """
    Plot V(X, Y) for fixed (ψ_rel=0, V_e=0, V_i=0, r_e=0, r_i=0)
    at a specific time t_query.

    This is a 2D slice through the 7D value function —
    the zero-level set shows the capture boundary in position space.
    """
    n_grid = 200
    R = config.R_triangle
    x_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    y_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    XX, YY = torch.meshgrid(x_range, y_range, indexing='ij')

    # Flatten to (n_grid², 7)
    X_flat = XX.reshape(-1, 1)
    Y_flat = YY.reshape(-1, 1)
    n_pts = X_flat.shape[0]

    # Fixed states: ψ_rel=0, V_e=0, V_i=0, r_e=0, r_i=0
    zeta = torch.cat([
        X_flat,
        Y_flat,
        torch.zeros(n_pts, 1, device=device),  # ψ_rel
        torch.zeros(n_pts, 1, device=device),  # V_e
        torch.zeros(n_pts, 1, device=device),  # V_i
        torch.zeros(n_pts, 1, device=device),  # r_e
        torch.zeros(n_pts, 1, device=device),  # r_i
    ], dim=1)

    t = torch.full((n_pts, 1), t_query, device=device)

    with torch.no_grad():
        V = model(zeta, t).reshape(n_grid, n_grid)

    V_np = V.cpu().numpy()
    XX_np = XX.cpu().numpy()
    YY_np = YY.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Color map: blue = evader escapes (V > 0), red = capture (V < 0)
    cf = ax.contourf(XX_np, YY_np, V_np, levels=50, cmap='RdBu')
    plt.colorbar(cf, ax=ax, label='V(X, Y)')

    # Zero contour = capture boundary
    ax.contour(XX_np, YY_np, V_np, levels=[0], colors='black', linewidths=2)

    # Mark evader position (origin)
    ax.plot(0, 0, 'g*', markersize=15, label='Evader')

    # Mark pursuer initial positions
    for idx, (xp, yp) in enumerate(config.initial_pursuer_positions()):
        ax.plot(xp, yp, 'r^', markersize=12, label=f'Pursuer {idx+1}' if idx == 0 else None)

    # Capture circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(config.rho * np.cos(theta), config.rho * np.sin(theta),
            'k--', linewidth=1, alpha=0.5, label=f'Capture radius ρ={config.rho}m')

    ax.set_xlabel('X [m] (ahead of evader)')
    ax.set_ylabel('Y [m] (port of evader)')
    ax.set_title(
        f'Value function slice at t = {t_query:.0f}s\n'
        f'(ψ_rel=0, V_e=V_i=0, r_e=r_i=0)\n'
        f'Black contour = capture boundary (V=0)'
    )
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(
    history: list,
    save_path: str = 'training_history.png',
):
    """Plot loss curves across all ε stages."""
    epochs = range(len(history))
    L_pde = [h['L_pde'] for h in history]
    L_tc = [h['L_tc'] for h in history]
    L_total = [h['L_total'] for h in history]
    epsilons = [h['epsilon'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogy(epochs, L_total, 'b-', alpha=0.7, linewidth=0.5, label='Total')
    ax1.semilogy(epochs, L_pde, 'r-', alpha=0.5, linewidth=0.5, label='PDE')
    ax1.semilogy(epochs, L_tc, 'g-', alpha=0.5, linewidth=0.5, label='Terminal')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend()
    ax1.set_title('PINN Training History')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, epsilons, 'k-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ε (diffusion)')
    ax2.set_title('ε-Schedule (vanishing viscosity)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_optimal_controls(
    model: NormalizedValueNetwork,
    config: GameConfig,
    t_query: float = -150.0,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'optimal_controls.png',
):
    """
    Plot which diamond vertex is optimal across the (X, Y) plane.
    Shows the switching surfaces between control strategies.
    """
    n_grid = 150
    R = config.R_triangle
    x_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    y_range = torch.linspace(-1.5 * R, 1.5 * R, n_grid, device=device)
    XX, YY = torch.meshgrid(x_range, y_range, indexing='ij')

    X_flat = XX.reshape(-1, 1)
    Y_flat = YY.reshape(-1, 1)
    n_pts = X_flat.shape[0]

    zeta = torch.cat([
        X_flat, Y_flat,
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
        torch.zeros(n_pts, 1, device=device),
    ], dim=1)

    t = torch.full((n_pts, 1), t_query, device=device)

    controls = extract_optimal_controls(model, zeta, t, config)

    vertex_names = ['A: Full speed', 'B: Port turn', 'C: Starboard turn', 'D: Stop']
    cmap = plt.colormaps['Set1'].resampled(4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Evader control map
    ev = controls['evader']['vertex'].reshape(n_grid, n_grid).cpu().numpy()
    im1 = ax1.pcolormesh(XX.cpu(), YY.cpu(), ev, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax1.set_title(f'Evader optimal vertex at t={t_query:.0f}s')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
    cbar1.set_ticklabels(vertex_names)

    # Pursuer control map
    pv = controls['pursuer']['vertex'].reshape(n_grid, n_grid).cpu().numpy()
    im2 = ax2.pcolormesh(XX.cpu(), YY.cpu(), pv, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax2.set_title(f'Pursuer optimal vertex at t={t_query:.0f}s')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
    cbar2.set_ticklabels(vertex_names)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
