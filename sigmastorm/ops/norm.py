"""Normalisation and the bounded-state projection from the Sigma transition law."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def project_norm_ball(x: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale rows of ``x`` onto the ball of ``radius``.

    Returns ``(projected, hit)`` where ``hit`` marks the rows that were rescaled. The projection is
    the bounded-state constraint of the transition: it is applied to the *latent*, never to the
    unconstrained residual stream.
    """

    norm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (radius / norm).clamp(max=1.0)
    hit = scale.squeeze(-1) < 1.0
    return x * scale.to(x.dtype), hit
