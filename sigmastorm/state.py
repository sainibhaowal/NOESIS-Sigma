"""Explicit, bounded recurrent state for SigmaStorm.

The state is the ``s_t = (x_t, h_t, M_t, q_t)`` of the Sigma transition law, made concrete:

* ``x_t`` — the residual stream inside the current chunk (activation, not carried);
* ``h_t`` — ``ssm``: the per-head delta-rule associative matrix;
* ``M_t`` — ``memory``: the bounded slot memory;
* ``q_t`` — carried implicitly by the controller, which is a pure function of the input.

Every field has a shape fixed at construction time, so memory does not grow with context length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SlotMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    # Per-token writes are accumulated here and consolidated as a soft update every ``slot_block``
    # tokens. Keeping the accumulators in the state is what makes the operator exactly chunk-invariant:
    # a chunk boundary in the middle of a block changes nothing.
    acc_num: torch.Tensor
    acc_key_num: torch.Tensor
    acc_den: torch.Tensor
    # Causal (within-block) key accumulator used by the novelty gate. It is the running sum of the
    # first-pass writes so that a token sees the keys written before it *inside the current block*,
    # not just the last consolidation. Carried across chunk boundaries and zeroed on consolidation,
    # which keeps the operator chunk-invariant.
    causal_key_num: torch.Tensor
    causal_den: torch.Tensor
    acc_count: int = 0

    @classmethod
    def zeros(cls, batch: int, slots: int, width: int, device, dtype) -> SlotMemoryState:
        shape = (batch, slots, width)
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            usage=torch.zeros((batch, slots), device=device, dtype=dtype),
            acc_num=torch.zeros(shape, device=device, dtype=dtype),
            acc_key_num=torch.zeros(shape, device=device, dtype=dtype),
            acc_den=torch.zeros((batch, slots), device=device, dtype=dtype),
            causal_key_num=torch.zeros(shape, device=device, dtype=dtype),
            causal_den=torch.zeros((batch, slots), device=device, dtype=dtype),
            acc_count=0,
        )

    def detach(self) -> SlotMemoryState:
        return SlotMemoryState(
            self.keys.detach(),
            self.values.detach(),
            self.usage.detach(),
            self.acc_num.detach(),
            self.acc_key_num.detach(),
            self.acc_den.detach(),
            self.causal_key_num.detach(),
            self.causal_den.detach(),
            self.acc_count,
        )


@dataclass
class SummaryState:
    """Fixed ring of pooled chunk summaries (Gap C: the global, exact-distance path).

    ``keys``/``values`` are per-head summary vectors, ``usage`` marks written slots, ``pos`` is the
    summary index at which each slot was written (used for a learned age bias), ``index`` counts
    completed summaries, ``acc`` accumulates the current incomplete chunk and ``write_ptr`` is the
    ring cursor. Every tensor has a shape fixed at construction.
    """

    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    pos: torch.Tensor
    acc: torch.Tensor
    index: int = 0
    write_ptr: int = 0
    count: int = 0

    @classmethod
    def zeros(
        cls, batch: int, heads: int, slots: int, head_dim: int, width: int, device, dtype
    ) -> SummaryState:
        shape = (batch, heads, slots, head_dim)
        return cls(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            usage=torch.zeros((batch, slots), device=device, dtype=dtype),
            pos=torch.zeros((batch, slots), device=device, dtype=dtype),
            acc=torch.zeros((batch, width), device=device, dtype=dtype),
        )

    def detach(self) -> SummaryState:
        return SummaryState(
            self.keys.detach(),
            self.values.detach(),
            self.usage.detach(),
            self.pos.detach(),
            self.acc.detach(),
            self.index,
            self.write_ptr,
            self.count,
        )


@dataclass
class BlockState:
    conv_tail: torch.Tensor
    ssm: torch.Tensor
    memory: SlotMemoryState
    window: tuple[torch.Tensor, torch.Tensor] | None = None
    win_len: int = 0
    # Last input of the previous chunk. The delta rule's key is the token one step back, so the
    # first token of a chunk needs its predecessor to key the same way the unchunked stream does.
    prev: torch.Tensor | None = None
    summary: SummaryState | None = None

    def detach(self) -> BlockState:
        window = None if self.window is None else (self.window[0].detach(), self.window[1].detach())

        def detach(t):
            return None if t is None else t.detach()

        return BlockState(
            detach(self.conv_tail),
            detach(self.ssm),
            None if self.memory is None else self.memory.detach(),
            window,
            self.win_len,
            detach(self.prev),
            None if self.summary is None else self.summary.detach(),
        )


@dataclass
class SigmaState:
    blocks: tuple[BlockState, ...] = field(default_factory=tuple)
    step: int = 0

    def detach(self) -> SigmaState:
        return SigmaState(tuple(block.detach() for block in self.blocks), self.step)

    @property
    def batch_size(self) -> int:
        for block in self.blocks:
            for tensor in (block.ssm, block.conv_tail, block.prev):
                if tensor is not None:
                    return int(tensor.shape[0])
            if block.memory is not None:
                return int(block.memory.keys.shape[0])
        return 0

    def memory_bytes(self) -> int:
        total = 0
        for block in self.blocks:
            for tensor in (block.conv_tail, block.ssm, block.prev):
                if tensor is not None:
                    total += tensor.numel()
            if block.memory is not None:
                total += sum(
                    t.numel()
                    for t in (
                        block.memory.keys,
                        block.memory.values,
                        block.memory.usage,
                        block.memory.acc_num,
                        block.memory.acc_key_num,
                        block.memory.acc_den,
                        block.memory.causal_key_num,
                        block.memory.causal_den,
                    )
                )
            if block.window is not None:
                total += block.window[0].numel() + block.window[1].numel()
            if block.summary is not None:
                total += sum(
                    t.numel()
                    for t in (
                        block.summary.keys,
                        block.summary.values,
                        block.summary.usage,
                        block.summary.pos,
                        block.summary.acc,
                    )
                )
        return total * 2  # bf16/fp16 working assumption
