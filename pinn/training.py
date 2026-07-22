"""
SECTION 9: TRAINING LOOP WITH ε-SCHEDULING
==========================================
"""

import torch

from .config import GameConfig
from .loss import pinn_loss
from .network import NormalizedValueNetwork
from .sampling import DomainSampler


def train_pinn(
    config: GameConfig,
    pursuer_idx: int = 0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> NormalizedValueNetwork:
    """
    Train the PINN with ε-scheduling (vanishing viscosity).

    Schedule: ε = 1.0 → 0.5 → 0.25 → ... → 0.001
    At each stage, warm-start from the previous converged weights.

    Args:
        config: full problem configuration
        pursuer_idx: which pursuer-evader pair (0, 1, or 2)
        device: CPU or CUDA
        verbose: print training progress
    Returns:
        model: trained NormalizedValueNetwork
    """
    model = NormalizedValueNetwork(config).to(device)
    sampler = DomainSampler(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2
    )

    history = []

    for stage, epsilon in enumerate(config.eps_schedule):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ε-STAGE {stage + 1}/{len(config.eps_schedule)}: ε = {epsilon}")
            print(f"{'='*60}")

        for epoch in range(config.epochs_per_eps):
            optimizer.zero_grad()

            loss, info = pinn_loss(
                model, sampler, config, epsilon, pursuer_idx
            )

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            history.append(info)

            if verbose and (epoch + 1) % 500 == 0:
                print(
                    f"  Epoch {epoch+1:5d}/{config.epochs_per_eps} | "
                    f"L_total={info['L_total']:.4e} | "
                    f"L_pde={info['L_pde']:.4e} | "
                    f"L_tc={info['L_tc']:.4e} | "
                    f"residual_max={info['residual_max']:.4e}"
                )

    if verbose:
        print(f"\nTraining complete. Final loss: {history[-1]['L_total']:.4e}")

    return model, history
