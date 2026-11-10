"""Chunk-summary attention — a bounded summary path (Gap C).

This operator keeps one key/value vector per pooled chunk: a fixed ring of
summary_slots summaries, each produced by averaging a completed block of summary_chunk tokens
and projecting it. Every token queries the retained ring:

    q_t  = normalize(W_q x_t)
    s_tj = q_t . k_j / sqrt(d) - softplus(age) * (index - pos_j)
    o_t  = W_o sum_j softmax_j(s_tj) v_j

Cost is O(T * summary_slots * width) for fixed configuration — linear over the processed tokens
with a bounded ring — and state is constant in context length. The age bias is learned per head,
so a retained summary can persist or fade. The ring bounds memory; it is not an unbounded token
index.

The implementation writes a summary only after a full block of summary_chunk tokens and carries
the partial block and ring cursor in SummaryState. This is the state-carrying condition tested by
the paper's split-equivalence gate; that gate is not a claim over every possible boundary, length,
or backend. A token reads summaries from earlier completed blocks, never its own.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..state import SummaryState


class SummaryAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        chunk: int = 256,
        slots: int = 64,
        age_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if chunk < 1 or slots < 1:
            raise ValueError("chunk and slots must be positive")
        if age_decay < 0.0:
            raise ValueError("age_decay must be non-negative")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.chunk = chunk
        self.slots = slots
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        # Per-head persistence of a summary: softplus(age_bias) is the logit cost per summary of
        # age. It starts small (a summary a few dozen chunks old is still visible) and is learned.
        init = math.log(math.expm1(age_decay)) if age_decay > 0 else -10.0
        self.age_bias = nn.Parameter(torch.full((heads,), init))

    def initial_state(self, batch: int, device, dtype) -> SummaryState:
        return SummaryState.zeros(batch, self.heads, self.slots, self.head_dim, self.width, device, dtype)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], t.shape[1], self.heads, self.head_dim)

    def _pooled_heads(self, pooled: torch.Tensor) -> torch.Tensor:
        """``[B, width] -> [B, H, 1, dh]`` for a single pooled summary."""
        batch = pooled.shape[0]
        return pooled.view(batch, 1, self.heads, self.head_dim).transpose(1, 2)

    def _read(self, x: torch.Tensor, state: SummaryState) -> torch.Tensor:
        batch, length, _ = x.shape
        query = F.normalize(self._split(self.query(x)), dim=-1)
        keys = F.normalize(state.keys, dim=-1, eps=1e-6)
        scores = torch.einsum("bnhd,bhsd->bnhs", query, keys) / math.sqrt(self.head_dim)
        age = (state.index - state.pos).clamp_min(0.0)
        bias = F.softplus(self.age_bias)[None, :, None] * age[:, None, :]
        scores = scores - bias[:, None]
        # Unwritten slots are neutralised rather than masked with -inf: values are zero, so an
        # all-invalid ring returns exactly zero instead of NaN.
        scores = scores.masked_fill((state.usage > 0)[:, None, None, :] == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bnhs,bhsd->bnhd", weights, state.values)
        return self.out(attended.reshape(batch, length, self.width))

    def _write(self, pooled: torch.Tensor, state: SummaryState) -> SummaryState:
        slot = state.write_ptr
        new_key = self._pooled_heads(self.key(pooled))  # [B,H,1,dh]
        new_value = self._pooled_heads(self.value(pooled))
        ring = F.one_hot(torch.tensor(slot, device=pooled.device), self.slots).to(state.keys.dtype)
        ring = ring.view(1, 1, self.slots, 1)
        ring_flat = ring.view(1, self.slots)

        keys = F.normalize(state.keys * (1.0 - ring) + new_key * ring, dim=-1, eps=1e-6)
        values = state.values * (1.0 - ring) + new_value * ring
        usage = state.usage * (1.0 - ring_flat) + ring_flat
        pos = state.pos * (1.0 - ring_flat) + float(state.index) * ring_flat
        return SummaryState(
            keys,
            values,
            usage,
            pos,
            torch.zeros_like(state.acc),
            state.index + 1,
            (slot + 1) % self.slots,
            0,
        )

    def forward(self, x: torch.Tensor, state: SummaryState | None) -> tuple[torch.Tensor, SummaryState]:
        batch, length, _ = x.shape
        if state is None:
            state = self.initial_state(batch, x.device, x.dtype)

        pieces = []
        offset = 0
        while offset < length:
            take = min(self.chunk - state.count, length - offset)
            piece = x[:, offset : offset + take]
            pieces.append(self._read(piece, state))
            state.acc = state.acc + piece.sum(dim=1)
            state.count += take
            offset += take
            if state.count == self.chunk:
                state = self._write(state.acc / self.chunk, state)

        out = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
        return out, state
