"""One stacked SigmaStorm block: transition, then gated feed-forward."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import SigmaConfig
from .ops import GatedFFN, SparseMoE
from .ops.norm import RMSNorm
from .state import BlockState
from .transition import SigmaTransition


class SigmaBlock(nn.Module):
    def __init__(self, config: SigmaConfig, index: int) -> None:
        super().__init__()
        self.transition = SigmaTransition(config, config.uses_window(index), config.uses_summary(index))
        self.norm = RMSNorm(config.d_model)
        hidden = int(config.ffn_mult * config.d_model)
        if config.n_experts:
            self.ffn = SparseMoE(
                config.d_model, hidden, config.n_experts, config.top_k, config.n_shared_experts
            )
            self.is_moe = True
        else:
            self.ffn = GatedFFN(config.d_model, hidden)
            self.is_moe = False
        self.grad_checkpoint = False

    def _ffn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.norm(x)
        if self.is_moe:
            out, aux = self.ffn(hidden)
            return out, aux
        return self.ffn(hidden), None

    def forward(self, x: torch.Tensor, state: BlockState | None):
        x, state, telemetry, entropy = self.transition(x, state)
        if self.grad_checkpoint and self.training:
            # Only the feed-forward is checkpointed: it is state-free. Checkpointing the transition
            # would re-run it against the mutated recurrent state.
            out, aux = checkpoint(self._ffn, x, use_reentrant=False)
        else:
            out, aux = self._ffn(x)
        return x + out, state, telemetry, entropy, aux
