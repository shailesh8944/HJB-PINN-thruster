#!/usr/bin/env python3
"""
Run a closed-loop pursuit–evasion simulation and see CAPTURED vs ESCAPED.

Examples:
    cd /mnt/newvolume/HJB_thruster/files

    # Theory-correct: diamond (s*,w*) from ∇V (HJB)
    .venv/bin/python run_game_sim.py --mode hjb --gif

    # Demo movie: geometric LOS + PINN threat ranking
    .venv/bin/python run_game_sim.py --mode guided --gif

    # Baseline chase/flee (no network)
    .venv/bin/python run_game_sim.py --mode heuristic --gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pinn.config import GameConfig
from pinn.simulate_game import (
    PursuitEvasionSim,
    animate_game,
    load_pinn_model,
    plot_game,
)


def main():
    parser = argparse.ArgumentParser(description="Simulate 3 pursuers vs 1 evader")
    parser.add_argument(
        "--mode",
        choices=["hjb", "guided", "pinn", "heuristic"],
        default="hjb",
        help="hjb=pure HJB diamond (theory); guided=LOS demo; heuristic=chase/flee",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "pinn_runs" / "latest" / "pinn_model_pair0.pt"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-time", type=float, default=None, help="Sim time [s] (default T_f)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "pinn_runs" / "latest" / "game_sim"),
    )
    parser.add_argument("--gif", action="store_true", help="Also save GIF animation")
    parser.add_argument("--no-anim", action="store_true")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GameConfig()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    mode = "guided" if args.mode == "pinn" else args.mode
    model = None
    if mode in ("hjb", "guided"):
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Train first:  .venv/bin/python train_pinn.py --device cuda\n"
                "Or use demo:  .venv/bin/python run_game_sim.py --mode heuristic"
            )
        print(f"Loading PINN: {ckpt}  (device={device})")
        model = load_pinn_model(ckpt, config, device)

    sim = PursuitEvasionSim(config, model=model, device=device, dt=args.dt, mode=mode)
    print(
        f"Running {mode} sim  rho={config.rho} m  R={config.R_triangle} m  "
        f"T_f={config.T_f} s  V_max={config.evader.V_max:.2f} m/s"
    )
    result = sim.run(max_time=args.max_time)

    if result.captured:
        print(f"\n*** CAPTURED at t = {result.capture_time:.2f} s ***")
        print(f"    Final min distance = {result.distances[-1].min():.3f} m  (ρ = {config.rho})")
    else:
        print(f"\n*** ESCAPED (horizon ended at t = {result.t[-1]:.2f} s) ***")
        print(f"    Final min distance = {result.distances[-1].min():.3f} m  (ρ = {config.rho})")

    plot_path = out / f"trajectories_{mode}.png"
    plot_game(result, config, plot_path, title=f"{mode.upper()} — {result.reason.upper()}")

    summary = {
        "mode": mode,
        "captured": result.captured,
        "capture_time": result.capture_time,
        "reason": result.reason,
        "final_min_distance": float(result.distances[-1].min()),
        "rho": config.rho,
        "sim_time": float(result.t[-1]),
        "V_max": config.evader.V_max,
    }
    with open(out / f"result_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out / f'result_{mode}.json'}")

    if not args.no_anim:
        ext = ".gif" if args.gif else ".mp4"
        anim_path = out / f"animation_{mode}{ext}"
        animate_game(result, config, anim_path, fps=20, stride=max(1, int(0.1 / args.dt)))


if __name__ == "__main__":
    main()
