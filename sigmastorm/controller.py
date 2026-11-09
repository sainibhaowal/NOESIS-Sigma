"""The routing and budgeting controller ``q_t``.

Design note that matters for training: the gates are **independent sigmoids**, not a softmax over
operators. A softmax makes the operators compete for a fixed total, so the cheapest operator (the
local convolution) wins within a few hundred steps and the associative memory is starved to ~5%
weight — which is exactly what happened in the first version of this package and why recall stayed
at chance. Independent gates let every operator contribute at full strength; the bounded-state
projection keeps the sum well conditioned.

Honesty notes:

* The gates are mixture weights, not compute skipping. Setting a gate to zero does not skip a
  kernel in dense PyTorch; gather-based token sparsity is a later optimisation.
* ``budget`` is the expected cost of the mixture, logged as a diagnostic.
"""

from __future__ import annotations

import torch
from torch import nn


class SigmaController(nn.Module):
    def __init__(self, width: int, n_ops: int, hidden: int = 0, costs: list[float] | None = None) -> None:
        super().__init__()
        hidden = hidden or width
        self.n_ops = n_ops
        self.route = nn.Sequential(nn.Linear(width, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, n_ops))
        self.step = nn.Sequential(nn.Linear(width, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, 1))
        costs = costs or [1.0] * n_ops
        if len(costs) != n_ops:
            raise ValueError("one cost per operator is required")
        self.register_buffer("costs", torch.tensor(costs, dtype=torch.float32))
        nn.init.zeros_(self.route[-1].bias)
        nn.init.zeros_(self.route[-1].weight)

    def forward(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        gates = torch.sigmoid(self.route(u))
        budget = (gates * self.costs.to(gates.dtype)).sum(dim=-1).mean()
        share = gates.mean(dim=(0, 1)) + 1e-6
        share = share / share.sum()
        entropy = -(share * share.log()).sum()
        return {
            "gates": gates,
            "step": torch.sigmoid(self.step(u)),
            "budget": budget,
            "entropy": entropy,
        }
