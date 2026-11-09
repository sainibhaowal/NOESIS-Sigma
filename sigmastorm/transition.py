"""The executable Sigma transition.

One call advances the per-token latent by one step of::

    v = x + dt(u) * [ sum_i alpha_i(x) * d_i(x) - grad E(x) + B u ]
    x <- Project_{||.|| <= R}(v)

``d_i`` are the operator proposals (short convolution, delta rule, slot memory, optional window
attention), ``alpha`` is the controller, ``grad E`` is the learned contraction, and ``Project`` is the
bounded-state constraint. Depth in the stack plays the role of transition time: L blocks = L steps
of the law on the token-local latent, while ``(h_t, M_t)`` carry information across tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import SigmaConfig
from .controller import SigmaController
from .ops import (
    EnergyPotential,
    GatedDeltaRule,
    ShortConv,
    SlotMemory,
    SummaryAttention,
    WindowAttention,
    project_norm_ball,
)
from .ops.norm import RMSNorm
from .state import BlockState


class SigmaTransition(nn.Module):
    def __init__(self, config: SigmaConfig, use_window: bool, use_summary: bool) -> None:
        super().__init__()
        width = config.d_model
        self.config = config
        self.in_norm = RMSNorm(width)
        self.conv = ShortConv(width, config.conv_kernel) if config.use_conv else None
        self.delta = (
            GatedDeltaRule(
                width,
                config.n_heads,
                config.chunk_size,
                decay_min_rate=config.decay_min_rate,
                decay_max_rate=config.decay_max_rate,
                expand=config.state_expand,
                decay_learnable=config.decay_learnable,
            )
            if config.use_delta
            else None
        )
        self.memory = (
            SlotMemory(
                width,
                config.slot_count,
                config.slot_temperature,
                config.slot_block,
                metric=config.memory_metric,
                shrinkage=config.memory_shrinkage,
                novelty_floor=config.memory_novelty_floor,
                read_temperature=config.memory_read_temperature,
                abstain=config.memory_abstain,
            )
            if config.use_memory
            else None
        )
        self.summary = (
            SummaryAttention(
                width,
                config.n_heads,
                config.summary_chunk,
                config.summary_slots,
                config.summary_age_decay,
            )
            if use_summary
            else None
        )
        self.window = WindowAttention(width, config.n_heads, config.window) if use_window else None
        self.energy = EnergyPotential(width, config.energy_rank, config.energy_floor)
        # The law is executed on a bounded latent ``v``; the residual stream stays additive. The
        # first version replaced the stream with the post-normed latent every block, which erased
        # the token identity the associative operator needs (MQAR stayed at chance).
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        names, costs = [], []
        for name, cost in (
            ("conv", 1.0),
            ("delta", 1.5),
            ("memory", 1.0),
            ("summary", 1.25),
            ("window", 2.0),
        ):
            enabled = {
                "conv": self.conv,
                "delta": self.delta,
                "memory": self.memory,
                "summary": self.summary,
                "window": self.window,
            }[name]
            if enabled is not None:
                names.append(name)
                costs.append(cost)
        if not names:
            raise ValueError("at least one operator must be enabled")
        self.operator_names = names
        # Operators have wildly different natural output scales (a retrieval that must be amplified
        # for the LM head can reach norm ~18 while a short conv sits near 1). Without a per-operator
        # normalisation the largest one dominates the mixture regardless of its gate, and the
        # latent saturates the projection.
        self.op_norms = nn.ModuleList(RMSNorm(width) for _ in names)
        self.controller = SigmaController(width, len(costs), costs=costs)

    def initial_block_state(self, batch: int, device, dtype) -> BlockState:
        window = None
        if self.window is not None:
            window = self.window.initial_state(batch, device, dtype)
        return BlockState(
            conv_tail=self.conv.initial_state(batch, device, dtype) if self.conv is not None else None,
            ssm=self.delta.initial_state(batch, device, dtype) if self.delta is not None else None,
            memory=self.memory.initial_state(batch, device, dtype) if self.memory is not None else None,
            window=window,
            win_len=0,
            summary=self.summary.initial_state(batch, device, dtype) if self.summary is not None else None,
        )

    def forward(self, x: torch.Tensor, state: BlockState | None):
        batch, length, width = x.shape
        if state is None:
            state = self.initial_block_state(batch, x.device, x.dtype)

        u = self.in_norm(x)
        if self.conv is not None:
            local, conv_tail = self.conv(u, state.conv_tail)
        else:
            local, conv_tail = u, state.conv_tail

        operators = []
        if self.conv is not None:
            operators.append(local)
        ssm = state.ssm
        if self.delta is not None:
            # The delta rule reads the clean stream, not the convolved one: its key is the token
            # one step back, and a key blended with its neighbours no longer matches itself when the
            # query returns.
            delta_out, ssm = self.delta(u, state.ssm, prev=state.prev)
            operators.append(delta_out)
        memory_state = state.memory
        if self.memory is not None:
            memory_out, memory_state = self.memory(u, x, state.memory)
            operators.append(memory_out)

        summary_state = state.summary
        if self.summary is not None:
            summary_out, summary_state = self.summary(u, state.summary)
            operators.append(summary_out)

        window_state, win_len = state.window, state.win_len
        if self.window is not None:
            window_out, window_state, win_len = self.window(u, state.window, state.win_len)
            operators.append(window_out)

        control = self.controller(u)
        gates = control["gates"]
        # Normalised mixture: gates keep their independent sigmoid meaning (any subset can be on at
        # full strength) while the proposal is a convex combination, so its scale cannot grow with
        # the number of active operators and cannot saturate the bounded-state projection.
        mixture = torch.zeros_like(u)
        gate_total = gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        for index, operator in enumerate(operators):
            mixture = mixture + gates[..., index : index + 1] * self.op_norms[index](operator)
        mixture = mixture / gate_total

        gradient = self.energy.gradient(u)
        step = self.config.dt * (0.5 + control["step"])
        # The law advances the latent once: v = Project(u + dt[mixture - grad E + u]). ``u`` is the
        # normalised stream, so the latent lives at a fixed scale and the projection is a real bound
        # rather than a saturation artefact.
        latent, projected = project_norm_ball(u + step * (mixture - gradient + u), self.config.max_norm)
        x = x + self.dropout(self.out_proj(latent))

        next_state = BlockState(
            conv_tail, ssm, memory_state, window_state, win_len, prev=u[:, -1], summary=summary_state
        )
        telemetry = {
            "energy": float(self.energy.energy(latent).detach().mean()),
            "latent_norm": float(latent.detach().norm(dim=-1).mean()),
            "update_norm": float((step * (mixture - gradient + u)).detach().norm(dim=-1).mean()),
            "budget": float(control["budget"].detach()),
            "projected_fraction": float(projected.detach().float().mean()),
            **{
                f"route_{name}": float(gates[..., index].detach().mean())
                for index, name in enumerate(self.operator_names)
            },
        }
        return x, next_state, telemetry, control["entropy"]
