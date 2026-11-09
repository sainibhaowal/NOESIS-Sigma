"""Chunkwise gated delta rule — the associative-recall operator.

This is the one operator in the stack that can store and retrieve key->value bindings, which is what
separates a workable non-attention model from a plain recurrent one. It is *not* attention: there is
no ``[T, T]`` score matrix and cost is linear in the sequence length.

Recurrence, per head, with ``a_t = exp(g_t)`` and ``b_t = sigmoid(beta_t)``::

    S_t = a_t * S_{t-1} + b_t * k_t * (v_t - a_t * k_t^T S_{t-1})^T
    o_t = q_t^T S_t

The ``(v - k^T S)`` term is the error correction: the state is rewritten whenever a key is seen with
a value it does not already predict. That is the "evidence-gated write" of the Sigma law, made
differentiable.

Chunkwise form, for a chunk with start state ``S0`` and cumulative log-decay ``gc``::

    A[i, j] = exp(gc_i - gc_j) * b_j * (k_i . k_j)          for j < i
    r       = (I + A)^{-1} (V - exp(gc) * K S0)             (unit lower-triangular solve)
    O       = exp(gc) * Q S0 + tril(exp(gc_i - gc_j) * b_j * (q_i . k_j)) @ r
    S_chunk = exp(gc_last) * S0 + (exp(gc_last - gc_j) * b_j * K)^T @ r

Every ratio ``exp(gc_i - gc_j)`` used is <= 1, so no overflow is reachable on the causal region.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps))


def _delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk. Shapes: q/k/v ``[B,H,C,dh]``, beta/g ``[B,H,C]``, state ``[B,H,dh,dh]``."""

    gc = torch.cumsum(g, dim=-1)
    a = torch.exp(gc)
    length = q.shape[2]
    eye = torch.eye(length, device=q.device, dtype=torch.float32)

    # ratio[i, j] = exp(gc_i - gc_j). Only j <= i is consumed, where the exponent is <= 0.
    ratio = torch.exp((gc[..., :, None] - gc[..., None, :]).clamp(max=0.0))

    a_mat = torch.tril(ratio * beta[..., None, :] * torch.matmul(k, k.transpose(-1, -2)), diagonal=-1)
    rhs = v - torch.matmul(a[..., None] * k, state)
    r = torch.linalg.solve_triangular(a_mat.float() + eye, rhs.float(), upper=False, unitriangular=True).to(
        v.dtype
    )

    out = torch.matmul(a[..., None] * q, state) + torch.matmul(
        torch.tril(ratio * beta[..., None, :] * torch.matmul(q, k.transpose(-1, -2))), r
    )

    tail = torch.exp(gc[..., -1:] - gc) * beta
    next_state = a[..., -1, None, None] * state + torch.matmul((tail[..., None] * k).transpose(-1, -2), r)
    return out, next_state


def delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token reference used by the equivalence test. Shapes as in :func:`_delta_chunk`."""

    batch, heads, length, dim = k.shape
    if state is None:
        state = k.new_zeros(batch, heads, dim, v.shape[-1])
    outputs = []
    for t in range(length):
        k_t = k[:, :, t]
        a_t = torch.exp(g[:, :, t]).unsqueeze(-1)
        b_t = beta[:, :, t].unsqueeze(-1)
        residual = v[:, :, t] - a_t * torch.einsum("bhd,bhde->bhe", k_t, state)
        state = a_t.unsqueeze(-1) * state + (b_t * k_t).unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append(torch.einsum("bhd,bhde->bhe", q[:, :, t], state))
    return torch.stack(outputs, dim=2), state


def decay_spectrum(heads: int, min_rate: float, max_rate: float) -> torch.Tensor:
    """Log-spaced initial decay rates, one per head.

    Rates are geometric from ``min_rate`` (slow, alpha ~ 1 - 1e-4) to ``max_rate`` (fast). With
    ``heads`` heads the model starts with a full multiscale memory: some heads hold bindings for the
    whole sequence, others track local context. Every rate remains learnable through the zeroed
    projection, so this is a prior, not a hard constraint.
    """

    if heads < 1:
        raise ValueError("heads must be positive")
    if not 0.0 < min_rate <= max_rate:
        raise ValueError("require 0 < min_rate <= max_rate")
    if heads == 1:
        return torch.tensor([math.sqrt(min_rate * max_rate)], dtype=torch.float32)
    ratio = max_rate / min_rate
    exponents = torch.linspace(0.0, 1.0, heads)
    return min_rate * ratio**exponents


class GatedDeltaRule(nn.Module):
    """Sequence mixer with a bounded, per-head associative state.

    ``expand`` multiplies the number of independent state matrices while leaving each matrix's
    head dimension at ``width // heads``. Capacity is linear in ``expand`` and so is cost; unlike a
    wider head this does not shrink the number of retrieval directions per token.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        chunk_size: int = 64,
        decay_min_rate: float = 1e-4,
        decay_max_rate: float = 3e-2,
        expand: int = 1,
        max_rate: float = 8.0,
        decay_learnable: bool = True,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if expand < 1:
            raise ValueError("expand must be at least 1")
        self.width = width
        self.heads = heads * expand
        self.head_dim = width // heads
        self.inner = width * expand
        self.chunk_size = chunk_size
        self.max_rate = max_rate
        self.q_proj = nn.Linear(width, self.inner, bias=False)
        self.k_proj = nn.Linear(width, self.inner, bias=False)
        self.v_proj = nn.Linear(width, self.inner, bias=False)
        self.beta_proj = nn.Linear(width, self.heads)
        # decay is ``exp(-softplus(.))`` with per-head bias encoding a log-spaced rate spectrum.
        # A gate that starts near a fast decay cannot hold a binding until it has already learned to
        # route, which is the failure mode that keeps pure recurrent recall at chance; a single
        # shared rate cannot cover both local and long-range bindings. The spectrum gives the model
        # both timescales from step zero, and the zeroed weight makes each rate learnable.
        self.decay_proj = nn.Linear(width, self.heads)
        nn.init.zeros_(self.decay_proj.weight)
        rates = decay_spectrum(self.heads, decay_min_rate, decay_max_rate)
        with torch.no_grad():
            self.decay_proj.bias.copy_(torch.log(torch.expm1(rates)))
        # Opt out of the generic initialiser: zeroing this bias would reset alpha to 0.5, which
        # erases every binding within a couple of tokens.
        self.decay_proj._custom_init = True
        if not decay_learnable:
            # The projection is already zero, so freezing both parameters pins the rate per head to
            # the initialised spectrum. This turns the spectrum into a hard prior for ablation.
            self.decay_proj.weight.requires_grad_(False)
            self.decay_proj.bias.requires_grad_(False)
        self.gate_proj = nn.Linear(width, self.inner)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def initial_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.heads, self.head_dim, self.head_dim, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        if state is None:
            state = self.initial_state(batch, x.device, x.dtype)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q = _l2norm(split(self.q_proj(x)))
        # Keys are read one token back. A binding is written when the value arrives (token t) and
        # the key is the token that preceded it (t-1); the same projection then serves as the query
        # when that key reappears. Reading both from the same position forces the projections to
        # separate two blended tokens, which is the difference between chance and recall here.
        if prev is None:
            k_source = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        else:
            k_source = torch.cat([prev.to(x.dtype)[:, None], x[:, :-1]], dim=1)
        k = _l2norm(split(self.k_proj(k_source)))
        v = split(self.v_proj(x))
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        # alpha = exp(-rate) with rate = softplus(.), so alpha can approach 1 and a binding
        # survives the sequence. The upper clamp only prevents a matrix-free underflow of
        # exp(gc) inside a chunk; it is never reached by a trained model.
        rate = torch.nn.functional.softplus(self.decay_proj(x)).clamp(min=1e-5, max=self.max_rate)
        g = -rate.transpose(1, 2)

        pieces = []
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            chunk, state = _delta_chunk(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                beta[:, :, start:end],
                g[:, :, start:end],
                state,
            )
            pieces.append(chunk)
        out = torch.cat(pieces, dim=2).transpose(1, 2).reshape(batch, length, self.inner)
        return self.out_proj(out * torch.sigmoid(self.gate_proj(x))), state
