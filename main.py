"""
Main entry point for the PINN pursuit-evasion solver.

Run:
    python main.py

Equations, loss, and training flow are unchanged from the original
monolithic script — only the file layout was split into pinn/*.py.
"""

import os
import time

import torch

from pinn import (
    GameConfig,
    extract_optimal_controls,
    plot_optimal_controls,
    plot_training_history,
    plot_value_function_slice,
    train_pinn,
)


def main():
    print("=" * 60)
    print("  PINN Solver: Pursuit-Evasion Viscosity Solution")
    print("  3 Pursuers (equilateral triangle) vs 1 Evader")
    print("  Diamond constraint on all vessels")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # --- Configuration ---
    # For a quick test run, use reduced settings.
    # For a real solve, increase n_pde, epochs_per_eps, and use full eps_schedule.
    config = GameConfig(
        rho=50.0,
        T_f=300.0,
        R_triangle=500.0,
        n_pde=4000,             # Reduce for quick test (use 10000+ for real)
        n_tc=1000,
        n_ic=500,
        epochs_per_eps=2000,    # Reduce for quick test (use 5000+ for real)
        eps_schedule=[1.0, 0.5, 0.1, 0.01],  # Shortened for test
    )

    print(f"\nProblem setup:")
    print(f"  Capture radius: {config.rho} m")
    print(f"  Time horizon: {config.T_f} s")
    print(f"  Triangle radius: {config.R_triangle} m")
    print(f"  Evader V_max: {config.evader.V_max:.1f} m/s")
    print(f"  Pursuer V_max: {config.pursuer.V_max:.1f} m/s")
    print(f"  ε-schedule: {config.eps_schedule}")

    # --- Solve for each pursuer-evader pair ---
    # Due to the equilateral symmetry and identical vessels,
    # we only need to solve pair 0 — the others are related by
    # 120° rotation. But the code supports solving all 3.

    pursuer_idx = 0
    pos = config.initial_pursuer_positions()
    print(f"\nSolving pair {pursuer_idx}: pursuer at ({pos[pursuer_idx][0]:.0f}, {pos[pursuer_idx][1]:.0f})")

    t_start = time.time()
    model, history = train_pinn(config, pursuer_idx, device, verbose=True)
    t_elapsed = time.time() - t_start
    print(f"\nTraining time: {t_elapsed:.1f}s")

    # --- Save model ---
    save_dir = '/mnt/user-data/outputs'
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'pinn_model_pair0.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'history': history,
    }, model_path)
    print(f"Model saved: {model_path}")

    # --- Visualize ---
    print("\nGenerating plots...")

    plot_training_history(
        history,
        save_path=os.path.join(save_dir, 'training_history.png')
    )

    plot_value_function_slice(
        model, config,
        t_query=-config.T_f / 2,  # Mid-game
        device=device,
        save_path=os.path.join(save_dir, 'value_function_slice.png')
    )

    plot_optimal_controls(
        model, config,
        t_query=-config.T_f / 2,
        device=device,
        save_path=os.path.join(save_dir, 'optimal_controls.png')
    )

    # --- Extract control at a specific state ---
    print("\n--- Example control extraction ---")
    test_zeta = torch.tensor([[
        500.0, 0.0,     # Pursuer directly ahead at 500m
        0.0,            # Same heading as evader
        0.0, 0.0,       # Both at rest
        0.0, 0.0,       # No yaw
    ]], device=device)
    test_t = torch.tensor([[-150.0]], device=device)

    ctrl = extract_optimal_controls(model, test_zeta, test_t, config)

    vertex_names = ['A (full speed)', 'B (port turn)', 'C (starboard turn)', 'D (stop)']
    print(f"  State: pursuer at (500, 0)m, all velocities zero, t=-150s")
    print(f"  V = {ctrl['V'].item():.2f}")
    print(f"  Evader: vertex {vertex_names[ctrl['evader']['vertex'].item()]}")
    print(f"    s* = {ctrl['evader']['s_star'].item():.1f}, w* = {ctrl['evader']['w_star'].item():.1f}")
    print(f"  Pursuer: vertex {vertex_names[ctrl['pursuer']['vertex'].item()]}")
    print(f"    s* = {ctrl['pursuer']['s_star'].item():.1f}, w* = {ctrl['pursuer']['w_star'].item():.1f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
