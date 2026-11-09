"""Held-out perplexity and MQAR accuracy."""

from __future__ import annotations

import math

import torch

from sigmastorm.model import SigmaStorm

from .data import PackedShards
from .tasks import make_mqar_batch, mqar_accuracy


@torch.no_grad()
def evaluate_loss(
    model: SigmaStorm,
    shards: PackedShards,
    batch_size: int,
    seq_len: int,
    micro_chunk: int,
    device: torch.device,
    max_batches: int = 50,
    precision: str = "fp32",
) -> dict[str, float]:
    from .trainer import autocast_context

    model.eval()
    total, count = 0.0, 0
    for _ in range(max_batches):
        tokens = torch.from_numpy(shards.sample(batch_size, seq_len)).to(device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        state = model.initial_state(batch_size, device)
        for start in range(0, seq_len - 1, micro_chunk):
            with autocast_context(precision, device):
                hidden, state, extras, _ = model(inputs[:, start : start + micro_chunk], state)
                outputs = model.loss_from_hidden(
                    hidden, targets[:, start : start + micro_chunk], None, extras
                )
            total += float(outputs["lm_sum"])
            count += outputs["lm_count"]
    model.train()
    mean = total / max(count, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 20.0)), "tokens": count}


@torch.no_grad()
def evaluate_mqar(
    model: SigmaStorm,
    batch: int = 16,
    length: int = 128,
    n_queries: int = 4,
    iterations: int = 10,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()
    correct, total = 0.0, 0
    for _ in range(iterations):
        data = make_mqar_batch(batch, length, n_queries=n_queries, device=device)
        accuracy, supervised = mqar_accuracy(model, data)
        correct += accuracy * supervised
        total += supervised
    model.train()
    return {"accuracy": correct / max(total, 1), "positions": total}
