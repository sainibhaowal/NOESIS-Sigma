"""Depthwise causal short convolution — the ``d_fast`` proposal."""

from __future__ import annotations

import torch
from torch import nn


class ShortConv(nn.Module):
    """Gated depthwise convolution over the last ``kernel_size`` tokens.

    State is the previous ``kernel_size - 1`` inputs, so a chunk of any length can be processed in
    one call and the state can be carried to the next chunk or decode step.
    """

    def __init__(self, width: int, kernel_size: int = 4) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.kernel_size = kernel_size
        self.value = nn.Conv1d(width, width, kernel_size, groups=width, bias=True)
        self.gate = nn.Conv1d(width, width, kernel_size, groups=width, bias=True)
        self.out = nn.Linear(width, width, bias=False)

    def initial_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch, self.kernel_size - 1, self.value.weight.shape[0], device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tail is None:
            tail = x.new_zeros(x.shape[0], self.kernel_size - 1, x.shape[-1])
        seq = torch.cat([tail.to(x.dtype), x], dim=1).transpose(1, 2)
        value = self.value(seq)
        gate = torch.sigmoid(self.gate(seq))
        y = (value * gate).transpose(1, 2)
        return self.out(y), seq.transpose(1, 2)[:, -(self.kernel_size - 1) :]
