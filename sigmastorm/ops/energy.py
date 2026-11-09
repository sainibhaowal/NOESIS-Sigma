"""Learned energy potential and its analytic gradient (the ``-grad E`` term).

Honest scope: with a positive-definite potential ``E(x) = 1/2 x^T D x + 1/2 ||L^T x||^2`` the
gradient is ``D x + L L^T x``, i.e. a learned anisotropic linear contraction. It is a *stabiliser*
that pulls the latent toward a learned basin, not an optimiser that solves for an equilibrium, and
it is not a multi-basin associative memory (strong convexity forbids that). Its value is that it
keeps the bounded state well-conditioned and gives a real, measurable energy signal for telemetry.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EnergyPotential(nn.Module):
    def __init__(self, width: int, rank: int = 32, floor: float = 1e-2) -> None:
        super().__init__()
        self.diagonal = nn.Parameter(torch.zeros(width))
        self.low_rank = nn.Parameter(torch.randn(width, rank) * width**-0.5)
        self.floor = floor

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        diagonal = F.softplus(self.diagonal) + self.floor
        return x * diagonal + (x @ self.low_rank) @ self.low_rank.transpose(0, 1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        diagonal = F.softplus(self.diagonal) + self.floor
        return 0.5 * (x.square() * diagonal).sum(dim=-1) + 0.5 * (x @ self.low_rank).square().sum(dim=-1)
