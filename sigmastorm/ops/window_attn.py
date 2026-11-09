"""Sliding-window causal attention — the optional ``d_select`` interaction.

Included on a minority of blocks (default 1 in 4) because precise local copying (induction heads) is
the one thing recurrence is reliably bad at. Its cost is ``O(T * window)`` and its state is a fixed
ring of keys and values, so it does not make the model a transformer: there is no full-sequence
attention, no KV growth with context, and it occupies a quarter of the layers.

``WindowAttention`` is a single component of the stack. Removing it entirely is a config switch
(``local_attn_every = 0``); the rest of the model is unaffected.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    def __init__(self, width: int, heads: int, window: int = 256) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.window = window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def initial_state(self, batch: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.heads, self.window, self.head_dim)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        history_len: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int]:
        batch, length, _ = x.shape
        if state is None:
            hist_k, hist_v = self.initial_state(batch, x.device, x.dtype)
        else:
            hist_k, hist_v = state
        keep = self.window - 1
        hist_k, hist_v = hist_k[:, :, -keep:], hist_v[:, :, -keep:]

        q, k, v = self.qkv(x).split(self.width, dim=-1)

        def shape(t):
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)
        keys = torch.cat([hist_k, k], dim=2)
        values = torch.cat([hist_v, v], dim=2)

        total = keys.shape[2]
        # Absolute position of each key column; the first entries of the ring are dummy zeros until
        # the buffer has been filled.
        key_pos = torch.arange(total, device=x.device) - keep + history_len
        query_pos = torch.arange(length, device=x.device) + history_len
        mask = (
            (key_pos[None, :] >= 0)
            & (key_pos[None, :] <= query_pos[:, None])
            & (key_pos[None, :] > query_pos[:, None] - self.window)
        )
        attn = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask[None, None].expand(batch, self.heads, length, total)
        )
        out = self.out(attn.transpose(1, 2).reshape(batch, length, self.width))
        return out, (keys[:, :, -keep:], values[:, :, -keep:]), history_len + length
