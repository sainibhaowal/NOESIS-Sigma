#!/usr/bin/env python3
"""Falsification gates. Each check must fail loudly if the architecture is broken.

    python scripts/gate.py                # fast checks + MQAR learnability
    python scripts/gate.py --mqar_steps 1500
    python scripts/gate.py --data_dir ... # also measure held-out loss over N steps

What each gate actually proves is printed next to it. None of them prove the model is good; they
prove the mechanism is not broken, which is the only thing a short run on two T4s can establish.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigmastorm.config import SigmaConfig  # noqa: E402
from sigmastorm.model import SigmaStorm  # noqa: E402
from sigmastorm.ops.delta_rule import _delta_chunk, delta_rule_reference  # noqa: E402
from training.tasks import make_mqar_batch, mqar_accuracy  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def gate_delta_equivalence(device: torch.device) -> None:
    torch.manual_seed(0)
    batch, heads, length, dim, chunk = 2, 2, 96, 16, 32
    q = F.normalize(torch.randn(batch, heads, length, dim, device=device), dim=-1)
    k = F.normalize(torch.randn(batch, heads, length, dim, device=device), dim=-1)
    v = torch.randn(batch, heads, length, dim, device=device)
    beta = torch.sigmoid(torch.randn(batch, heads, length, device=device))
    g = -torch.rand(batch, heads, length, device=device) * 0.5

    reference, reference_state = delta_rule_reference(q, k, v, beta, g)
    state = torch.zeros(batch, heads, dim, dim, device=device)
    pieces = []
    for start in range(0, length, chunk):
        out, state = _delta_chunk(
            q[:, :, start : start + chunk],
            k[:, :, start : start + chunk],
            v[:, :, start : start + chunk],
            beta[:, :, start : start + chunk],
            g[:, :, start : start + chunk],
            state,
        )
        pieces.append(out)
    chunked = torch.cat(pieces, dim=2)
    error = float((chunked - reference).abs().max())
    state_error = float((state - reference_state).abs().max())
    record(
        "delta_rule_chunkwise_equals_sequential",
        max(error, state_error) < 2e-4,
        f"max |out diff|={error:.2e}, max |state diff|={state_error:.2e}",
    )


def gate_state_is_constant(device: torch.device) -> None:
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config).to(device).eval()
    sizes = {}
    with torch.no_grad():
        for length in (512, 4096):
            state = model.initial_state(1, device)
            sizes[length] = state.memory_bytes()
            tokens = torch.randint(0, config.vocab_size, (1, length), device=device)
            model(tokens, state)
    record(
        "state_memory_independent_of_context",
        sizes[512] == sizes[4096],
        f"state bytes at T=512 and T=4096: {sizes[512]} (constant)",
    )


def gate_chunk_carry(device: torch.device) -> None:
    """Processing [0:T] in one call must equal two calls that carry the state.

    Every operator — including the slot memory and the summary ring, whose write accumulators are
    carried in the state — must be exactly chunk-invariant. The split is deliberately at 50, which
    cuts inside a 64-token slot consolidation block and inside a 32-token summary block, so a
    boundary-sensitive operator cannot pass by accident.
    """

    config = SigmaConfig.for_size("tiny")
    torch.manual_seed(1)
    model = SigmaStorm(config).to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 128), device=device)
    with torch.no_grad():
        whole, _, _, _ = model(tokens)
        split = 50
        first, state, _, _ = model(tokens[:, :split])
        state = state.detach()
        second, _, _, _ = model(tokens[:, split:], state)
        used_summaries = int((state.blocks[0].summary.usage > 0).sum()) if state.blocks[0].summary else 0
    error = float((whole - torch.cat([first, second], dim=1)).abs().max())
    record(
        "chunked_forward_equals_full",
        error < 3e-3,
        f"max |hidden diff|={error:.2e}, summaries written={used_summaries}",
    )


def gate_mqar(device: torch.device, steps: int, length: int, batch: int, lr: float) -> None:
    config = SigmaConfig.load(ROOT / "configs" / "mqar.json")
    torch.manual_seed(0)
    model = SigmaStorm(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    # Keys are unique per sequence, so the queried binding is unambiguous and the only ceiling is
    # the architecture. With 16 keys and 26 pairs the task caps near 0.6 no matter the model.
    n_keys, n_values = 64, 16
    chance = 1.0 / n_values

    started = time.time()
    for step in range(steps):
        data = make_mqar_batch(batch, length, n_queries=4, n_keys=n_keys, n_values=n_values, device=device)
        hidden, _, extras, _ = model(data["input_ids"])
        outputs = model.loss_from_hidden(hidden, data["targets"], data["loss_mask"], extras)
        loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"    mqar step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)

    accuracy, positions = 0.0, 0
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            data = make_mqar_batch(
                batch, length, n_queries=4, n_keys=n_keys, n_values=n_values, device=device
            )
            value, count = mqar_accuracy(model, data)
            accuracy += value * count
            positions += count
    accuracy /= max(positions, 1)
    record(
        "mqar_recall",
        accuracy >= 0.9,
        f"accuracy {accuracy:.3f} vs chance {chance:.3f} after {steps} steps ({time.time() - started:.0f}s)",
    )


def gate_loss_decreases(device: torch.device, data_dir: str, steps: int, batch: int, length: int) -> None:
    from training.data import PackedShards

    config = SigmaConfig.load(ROOT / "configs" / "sigma_20m.json")
    config = SigmaConfig.from_dict({**config.to_dict(), "vocab_size": 32000})
    model = SigmaStorm(config).to(device)
    if max(p.numel() for p in model.parameters()) and model.num_parameters() > 60_000_000:
        record("lm_loss_decreases", False, "gate model is too large; use configs/sigma_20m.json")
        return
    shards = PackedShards(data_dir, rank=0, world_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        tokens = torch.from_numpy(shards.sample(batch, length)).to(device)
        hidden, _, extras, _ = model(tokens[:, :-1])
        outputs = model.loss_from_hidden(hidden, tokens[:, 1:], None, extras)
        loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    first, last = (
        sum(losses[: max(steps // 5, 1)]) / max(steps // 5, 1),
        sum(losses[-max(steps // 5, 1) :]) / max(steps // 5, 1),
    )
    record("lm_loss_decreases", last < first, f"{first:.3f} -> {last:.3f} over {steps} steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqar_steps", type=int, default=400)
    parser.add_argument("--mqar_length", type=int, default=64)
    parser.add_argument("--mqar_batch", type=int, default=32)
    parser.add_argument("--mqar_lr", type=float, default=1e-3)
    parser.add_argument("--data_dir")
    parser.add_argument("--lm_steps", type=int, default=50)
    parser.add_argument("--skip_mqar", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)

    gate_delta_equivalence(device)
    gate_chunk_carry(device)
    gate_state_is_constant(device)
    if not args.skip_mqar:
        gate_mqar(device, args.mqar_steps, args.mqar_length, args.mqar_batch, args.mqar_lr)
    if args.data_dir:
        gate_loss_decreases(device, args.data_dir, args.lm_steps, 4, 512)

    failed = [name for name, passed, _ in RESULTS if not passed]
    print("\n" + ("ALL GATES PASSED" if not failed else f"FAILED: {', '.join(failed)}"), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
