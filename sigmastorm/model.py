"""SigmaStorm: stacked Sigma transitions with bracketed language, MTP and telemetry outputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .block import SigmaBlock
from .config import SigmaConfig
from .ops.norm import RMSNorm
from .state import SigmaState


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    token_chunk: int = 256,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy over flat token slices. Never materialises ``[B, T, vocab]``."""

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    if mask is not None:
        flat_targets = flat_targets.masked_fill(~mask.reshape(-1).bool(), ignore_index)

    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, flat_hidden.shape[0], token_chunk):
        target = flat_targets[start : start + token_chunk]
        valid = target != ignore_index
        if not bool(valid.any()):
            continue
        logits = F.linear(flat_hidden[start : start + token_chunk], weight)
        total = total + F.cross_entropy(logits.float(), target, reduction="sum", ignore_index=ignore_index)
        count += int(valid.sum())
    return total, count


class SigmaStorm(nn.Module):
    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(SigmaBlock(config, index) for index in range(config.n_blocks))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.mtp_proj = nn.ModuleDict(
            {
                str(horizon): nn.Linear(config.d_model, config.d_model, bias=False)
                for horizon in config.mtp_horizons
            }
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Modules that encode a calibrated initialisation (e.g. the delta-rule decay bias, which is
        # chosen so alpha starts near 0.98) must not be overwritten by the generic pass.
        if getattr(module, "_custom_init", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ state

    def initial_state(self, batch: int, device=None, dtype=None) -> SigmaState:
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        blocks = tuple(block.transition.initial_block_state(batch, device, dtype) for block in self.blocks)
        return SigmaState(blocks=blocks, step=0)

    def _check_state(self, state: SigmaState, batch: int) -> None:
        if len(state.blocks) != self.config.n_blocks:
            raise ValueError("SigmaState does not match the model's block count")
        if state.batch_size != batch:
            raise ValueError("SigmaState batch size does not match input_ids")

    # --------------------------------------------------------------- forward

    def forward(
        self, input_ids: torch.Tensor, state: SigmaState | None = None
    ) -> tuple[torch.Tensor, SigmaState, dict[str, torch.Tensor], dict[str, float]]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, time], got {tuple(input_ids.shape)}")
        batch = input_ids.shape[0]
        state = state or self.initial_state(batch, input_ids.device, self.embedding.weight.dtype)
        self._check_state(state, batch)

        x = self.embedding(input_ids)
        auxes = []
        telemetry: dict[str, float] = {}
        entropies = []
        new_blocks = []
        for block, block_state in zip(self.blocks, state.blocks, strict=True):
            x, block_state, block_telemetry, entropy, aux = block(x, block_state)
            new_blocks.append(block_state)
            entropies.append(entropy)
            if aux is not None:
                auxes.append(aux)
            for key, value in block_telemetry.items():
                telemetry[key] = telemetry.get(key, 0.0) + value / len(self.blocks)
        hidden = self.final_norm(x)
        new_state = SigmaState(tuple(new_blocks), state.step + input_ids.shape[1])
        extras = {
            "moe_aux": torch.stack(auxes).mean() if auxes else hidden.new_zeros(()),
            "route_entropy": torch.stack(entropies).mean(),
        }
        return hidden, new_state, extras, telemetry

    # ------------------------------------------------------------------ loss

    def loss_from_hidden(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        extras: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Loss terms as sums plus counts, so the caller can average exactly across chunks."""

        length = min(hidden.shape[1], targets.shape[1])
        hidden, targets = hidden[:, :length], targets[:, :length]
        mask = None if loss_mask is None else loss_mask[:, :length]

        lm_sum, lm_count = chunked_cross_entropy(
            hidden, self.lm_head.weight, targets, mask, self.config.loss_token_chunk
        )

        mtp_sum: dict[str, torch.Tensor] = {}
        mtp_count: dict[str, int] = {}
        for horizon in self.config.mtp_horizons:
            if length <= horizon:
                continue
            projected = self.mtp_proj[str(horizon)](hidden[:, :-horizon])
            mtp_sum[str(horizon)], mtp_count[str(horizon)] = chunked_cross_entropy(
                projected,
                self.lm_head.weight,
                targets[:, horizon:],
                None if mask is None else mask[:, horizon:],
                self.config.loss_token_chunk,
            )

        extras = extras or {}
        return {
            "lm_sum": lm_sum,
            "lm_count": lm_count,
            "mtp_sum": mtp_sum,
            "mtp_count": mtp_count,
            "moe_aux": extras.get("moe_aux", hidden.new_zeros(())),
            "route_entropy": extras.get("route_entropy", hidden.new_zeros(())),
        }

    def loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        state: SigmaState | None = None,
    ) -> tuple[dict[str, torch.Tensor], SigmaState]:
        hidden, new_state, extras, _ = self.forward(input_ids, state)
        return self.loss_from_hidden(hidden, targets, loss_mask, extras), new_state

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, state: SigmaState | None = None):
        hidden, state, _, _ = self.forward(input_ids, state)
        return self.lm_head(hidden), state

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if prompt_ids.ndim != 1:
            raise ValueError("generate expects a single 1-D prompt")
        device = next(self.parameters()).device
        prompt = prompt_ids.to(device)
        state = self.initial_state(1, device, self.embedding.weight.dtype)
        hidden, state, _, _ = self.forward(prompt[None], state)
        logits = self.lm_head(hidden[:, -1])[0]

        generated = []
        for _ in range(max_new_tokens):
            next_logits = logits / max(temperature, 1e-5)
            if top_k:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.numel()))
                next_logits = next_logits.masked_fill(next_logits < values[-1], float("-inf"))
            token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            generated.append(int(token))
            if eos_id is not None and int(token) == eos_id:
                break
            hidden, state, _, _ = self.forward(token.reshape(1, 1), state)
            logits = self.lm_head(hidden[:, -1])[0]
        return torch.cat([prompt, torch.tensor(generated, device=device)])

    # ------------------------------------------------------------- utilities

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def state_report(self) -> dict[str, int]:
        zeros = self.initial_state(1)
        return {"state_bytes_per_sample": zeros.memory_bytes(), "parameters": self.num_parameters()}
