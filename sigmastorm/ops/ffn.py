"""Gated feed-forward layer and an optional sparse mixture-of-experts."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedFFN(nn.Module):
    """SwiGLU feed-forward: 3 projections per token, ``hidden = ffn_mult * width``."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """Top-k routed experts plus optional always-on shared experts.

    Routing probabilities are returned so the trainer can apply a load-balancing loss; without one
    the router collapses onto a handful of experts within a few hundred steps.
    """

    def __init__(
        self,
        width: int,
        hidden: int,
        n_experts: int,
        top_k: int = 1,
        n_shared: int = 1,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(width, n_experts, bias=False)
        self.experts = nn.ModuleList(GatedFFN(width, hidden) for _ in range(n_experts))
        self.shared = nn.ModuleList(GatedFFN(width, hidden) for _ in range(n_shared))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            expert_ids = indices[:, slot]
            for expert_index, expert in enumerate(self.experts):
                selected = expert_ids == expert_index
                if selected.any():
                    out[selected] += weights[selected, slot].unsqueeze(-1) * expert(flat[selected])
        for expert in self.shared:
            out = out + expert(flat)

        # Load balance: fraction of tokens per expert times its mean routing probability.
        one_hot = F.one_hot(indices[:, 0], self.n_experts).to(probs.dtype)
        fraction = one_hot.mean(dim=0)
        aux = (fraction * probs.mean(dim=0)).sum() * self.n_experts
        return out.view_as(x), aux
