# NOESIS-Σ Core Tests — Low-rank K
# --------------------------------
# Basic correctness and safety checks for LowRankK:
#   - shape preservation
#   - projector behaviour
#   - spectral norm estimate <= lambda_cap (within small tolerance)

import math

import torch

from Core.OSC.k_lowrank import LowRankK, LowRankKConfig  # adjust import if needed


def test_lowrankk_shapes_cpu() -> None:
    d, r = 128, 16
    cfg = LowRankKConfig(
        d=d,
        rank=r,
        lambda_cap=0.5,
        projector_radius=1.0,
        enable_projector=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    op = LowRankK(cfg)

    x = torch.randn(4, d, dtype=torch.float32)
    y = op.apply_and_project(x, dt=0.01)

    assert y.shape == x.shape
    # Projector should not explode values
    assert torch.all(torch.abs(y) <= 1.0 + 1e-5)


def test_lowrankk_spectral_cap_estimate() -> None:
    d, r = 64, 8
    cfg = LowRankKConfig(
        d=d,
        rank=r,
        lambda_cap=0.7,
        projector_radius=None,
        enable_projector=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    op = LowRankK(cfg)

    est = op.spectral_norm_estimate(n_power_iters=10, batch_size=4, seed=123)
    # allow a small overshoot due to approximation
    assert est <= 0.7 * 1.05, f"spectral estimate {est} exceeds cap"
