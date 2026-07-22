#!/usr/bin/env python3
"""
Train the pursuit-evasion PINN with sookshma Nomoto parameters.

Usage:
    cd /mnt/newvolume/HJB_thruster/files

    # Quick smoke test (~few minutes on CPU)
    python3 train_pinn.py --quick

    # Full solve (long — hours on CPU, much faster on GPU)
    python3 train_pinn.py

    # GPU if available
    python3 train_pinn.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from pinn import GameConfig, train_pinn
from pinn.visualization import (
    plot_optimal_controls,
    plot_training_history,
    plot_value_function_slice,
)


def build_config(quick: bool) -> GameConfig:
    cfg = GameConfig()  # sookshma Nomoto + USV-scale geometry
    if quick:
        cfg.n_pde = 2000
        cfg.n_tc = 500
        cfg.n_ic = 250
        cfg.epochs_per_eps = 300
        cfg.eps_schedule = [1.0, 0.5, 0.1]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train pursuit-evasion PINN")
    parser.add_argument("--quick", action="store_true", help="Short smoke-test schedule")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--pursuer-idx", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "pinn_runs" / "latest"),
        help="Directory for model + plots",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = build_config(args.quick)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PINN: 3 pursuers vs 1 evader (sookshma Nomoto)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mode:   {'QUICK' if args.quick else 'FULL'}")
    print(f"rho={config.rho} m  R={config.R_triangle} m  T_f={config.T_f} s")
    print(f"Evader  V_max={config.evader.V_max:.3f} m/s  r_max={config.evader.r_max:.3f} rad/s")
    print(f"Pursuer V_max={config.pursuer.V_max:.3f} m/s  r_max={config.pursuer.r_max:.3f} rad/s")
    print(f"k_V={config.evader.k_V:.4f}  T_V={config.evader.T_V:.3f}  "
          f"k_r={config.evader.k_r:.4f}  T_r={config.evader.T_r:.3f}")
    print(f"epochs/eps={config.epochs_per_eps}  eps={config.eps_schedule}")
    print(f"Output → {out}")

    t0 = time.time()
    model, history = train_pinn(config, args.pursuer_idx, device, verbose=True)
    print(f"\nTraining time: {time.time() - t0:.1f} s")

    model_path = out / f"pinn_model_pair{args.pursuer_idx}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "pursuer_idx": args.pursuer_idx,
            "config": {
                "rho": config.rho,
                "T_f": config.T_f,
                "R_triangle": config.R_triangle,
                "evader": config.evader.to_dict() if hasattr(config.evader, "to_dict") else {
                    "n_max": config.evader.n_max,
                    "k_V": config.evader.k_V,
                    "T_V": config.evader.T_V,
                    "k_r": config.evader.k_r,
                    "T_r": config.evader.T_r,
                    "V_max": config.evader.V_max,
                    "r_max": config.evader.r_max,
                },
                "pursuer": {
                    "n_max": config.pursuer.n_max,
                    "k_V": config.pursuer.k_V,
                    "T_V": config.pursuer.T_V,
                    "k_r": config.pursuer.k_r,
                    "T_r": config.pursuer.T_r,
                    "V_max": config.pursuer.V_max,
                    "r_max": config.pursuer.r_max,
                },
                "eps_schedule": list(config.eps_schedule),
                "epochs_per_eps": config.epochs_per_eps,
            },
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_loss": history[-1] if history else None,
                "n_epochs_logged": len(history),
                "device": str(device),
                "quick": args.quick,
            },
            f,
            indent=2,
            default=float,
        )

    print("Generating plots...")
    plot_training_history(history, save_path=str(out / "training_history.png"))
    plot_value_function_slice(
        model,
        config,
        t_query=-config.T_f / 2,
        device=device,
        save_path=str(out / "value_function_slice.png"),
    )
    plot_optimal_controls(
        model,
        config,
        t_query=-config.T_f / 2,
        device=device,
        save_path=str(out / "optimal_controls.png"),
    )
    print(f"Done. Artifacts in {out}")


if __name__ == "__main__":
    # Avoid OpenMP oversubscription on shared machines
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
