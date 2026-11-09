"""Distributed trainer with truncated BPTT over the state chain.

Why the chunking exists: the recurrent state is carried across segments, and a segment only links
to the next one through that state. Backpropagating through *every* segment of a think-lens 32K
sequence would retain the whole activation graph. ``bptt_chunks`` sets the differentiable horizon
(``bptt_chunks * micro_chunk`` tokens) and the state is detached at the boundary. The forward context
can be arbitrarily long; only the gradient horizon is bounded. This is stated explicitly rather than
hidden behind a "long context" claim.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sigmastorm.config import SigmaConfig
from sigmastorm.model import SigmaStorm

from .data import PackedShards


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "runs/sigma"
    batch_size: int = 4
    seq_len: int = 2048
    micro_chunk: int = 512
    bptt_chunks: int = 4
    grad_accum: int = 1
    total_steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    # WSD (warmup-stable-decay) keeps the full learning rate for most of the run and decays only at
    # the end. It consistently reaches a lower loss than cosine at the same token budget because the
    # model spends the middle of training at high LR instead of already annealing.
    schedule: str = "wsd"
    decay_frac: float = 0.1
    weight_decay: float = 0.1
    clip: float = 1.0
    mtp_weight: float = 0.3
    # The multi-token head is a harder auxiliary objective; weighting it from step zero competes
    # with the LM loss before the backbone has any features. Ramp it in over this many steps.
    mtp_warmup_steps: int = 0
    moe_aux_weight: float = 0.01
    entropy_weight: float = 0.0
    # Short-context warmup: sample 0 (disabled) or a smaller sequence length and grow linearly to
    # ``seq_len`` over ``seq_ramp_steps``. Early steps see dense, easy gradients; long-context
    # structure is learned once the model can already model short windows.
    seq_len_start: int = 0
    seq_ramp_steps: int = 0
    precision: str = "auto"
    grad_checkpoint: bool = False
    compile: bool = False
    log_every: int = 10
    save_every: int = 500
    seed: int = 1234
    resume: bool = True
    # Wall-clock budget for hosted sessions with a hard kill (Kaggle, Colab). 0 disables it. When
    # the budget is reached the loop exits between steps and writes a final checkpoint, so the run
    # resumes from the last completed step instead of the last periodic save.
    max_hours: float = 0.0


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.schedule not in {"wsd", "cosine"}:
        raise ValueError(f"unknown schedule {cfg.schedule!r}")
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    if cfg.schedule == "cosine":
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
    else:
        decay_steps = max(int(cfg.total_steps * cfg.decay_frac), 1)
        decay_start = max(cfg.total_steps - decay_steps, cfg.warmup_steps)
        progress = (step - decay_start) / max(cfg.total_steps - decay_start, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def mtp_weight_at(step: int, cfg: TrainConfig) -> float:
    if cfg.mtp_warmup_steps <= 0:
        return cfg.mtp_weight
    return cfg.mtp_weight * min(1.0, (step + 1) / cfg.mtp_warmup_steps)


def seq_len_at(step: int, cfg: TrainConfig) -> int:
    if cfg.seq_ramp_steps <= 0 or cfg.seq_len_start <= 0:
        return cfg.seq_len
    progress = min(1.0, (step + 1) / cfg.seq_ramp_steps)
    length = cfg.seq_len_start + (cfg.seq_len - cfg.seq_len_start) * progress
    blocks = max(1, int(round(length / cfg.micro_chunk)))
    return max(cfg.micro_chunk, min(cfg.seq_len, blocks * cfg.micro_chunk))


def _loss_terms(outputs: dict, cfg: TrainConfig, local_tokens: int, step: int) -> torch.Tensor:
    """Weighted mean loss for one segment, scaled by its share of the optimizer-step tokens."""

    lm = outputs["lm_sum"] / max(outputs["lm_count"], 1)
    total = lm
    mtp_weight = mtp_weight_at(step, cfg)
    for horizon, value in outputs["mtp_sum"].items():
        total = total + mtp_weight * value / max(outputs["mtp_count"][horizon], 1)
    total = total + cfg.moe_aux_weight * outputs["moe_aux"] - cfg.entropy_weight * outputs["route_entropy"]
    return total * (outputs["lm_count"] / max(local_tokens, 1))


def _train_worker(rank: int, world_size: int, config: SigmaConfig, cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)
    on_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}") if on_cuda else torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if on_cuda else "gloo", rank=rank, world_size=world_size)

    precision = resolve_precision(cfg.precision, device)
    if precision == "bf16" and not (on_cuda and torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 was requested but is not supported on this device")
    if on_cuda:
        # TF32 matmuls are ~2x faster on Ampere+ and have no measurable effect on LM quality.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = SigmaStorm(config).to(device)
    if cfg.grad_checkpoint:
        for block in model.blocks:
            block.grad_checkpoint = True
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if on_cuda else None)
    if cfg.compile:
        if world_size > 1:
            raise RuntimeError("torch.compile is only enabled for single-process runs in this trainer")
        model = torch.compile(model)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=on_cuda,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    except TypeError:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    shards = PackedShards(cfg.data_dir, rank=rank, world_size=world_size, seed=cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.pt"
    start_step = 0
    if cfg.resume and latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("config") != config.to_dict():
            raise RuntimeError(
                f"{latest} was written for a different architecture; re-run with --fresh "
                "or point --out_dir at a new directory"
            )
        target = model.module if world_size > 1 else model
        try:
            target.load_state_dict(payload["model"])
        except RuntimeError as error:
            raise RuntimeError(
                f"cannot load {latest}: {error}. Use --fresh if this checkpoint predates the "
                "current operator set."
            ) from error
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        shards.load_state_dict(payload["data"])
        start_step = int(payload["step"])
        torch.set_rng_state(payload["rng"])

    is_main = rank == 0
    if is_main:
        print(model.module.state_report() if world_size > 1 else model.state_report(), flush=True)
        print(
            f"precision={precision} steps={cfg.total_steps} seq_len={cfg.seq_len} "
            f"micro_chunk={cfg.micro_chunk} bptt_tokens={cfg.bptt_chunks * cfg.micro_chunk}",
            flush=True,
        )

    def save(step_done: int) -> None:
        payload = {
            "version": 2,
            "config": config.to_dict(),
            "train": asdict(cfg),
            "model": (model.module if world_size > 1 else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step_done,
            "data": shards.state_dict(),
            "rng": torch.get_rng_state(),
        }
        tmp = out_dir / "latest.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, latest)

    run_started = time.time()
    last_done = start_step
    for step in range(start_step, cfg.total_steps):
        if cfg.max_hours > 0 and time.time() - run_started >= cfg.max_hours * 3600:
            if is_main:
                print(f"wall-clock budget reached after step {last_done}; saving and exiting", flush=True)
            break
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        loss_accum = 0.0
        loss_weight = 0

        step_len = seq_len_at(step, cfg)
        segments = list(range(0, step_len, cfg.micro_chunk))
        local_tokens = cfg.batch_size * step_len * cfg.grad_accum

        for micro in range(cfg.grad_accum):
            tokens = torch.from_numpy(shards.sample(cfg.batch_size, step_len + 1))
            tokens = tokens.to(device, non_blocking=True)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            state = (model.module if world_size > 1 else model).initial_state(cfg.batch_size, device)

            for window_start in range(0, len(segments), cfg.bptt_chunks):
                window = segments[window_start : window_start + cfg.bptt_chunks]
                window_loss = None
                for index, segment in enumerate(window):
                    end = segment + cfg.micro_chunk
                    sync = (
                        world_size > 1
                        and (micro == cfg.grad_accum - 1)
                        and (index == len(window) - 1)
                        and (window_start + len(window) >= len(segments))
                    )
                    ctx = model.no_sync() if world_size > 1 and not sync else contextlib.nullcontext()
                    with ctx, autocast_context(precision, device):
                        hidden, state, extras, _ = model(inputs[:, segment:end], state)
                        target_model = model.module if world_size > 1 else model
                        outputs = target_model.loss_from_hidden(hidden, targets[:, segment:end], None, extras)
                        segment_loss = _loss_terms(outputs, cfg, local_tokens, step)
                    window_loss = segment_loss if window_loss is None else window_loss + segment_loss
                    loss_accum += float(outputs["lm_sum"].detach()) / max(outputs["lm_count"], 1)
                    loss_weight += 1
                scaler.scale(window_loss).backward()
                # Detach at the BPTT horizon: the forward context continues, the gradient does not.
                state = state.detach()

        # Standard fp16 protocol: unscale, clip, step. If any gradient is non-finite the scaler
        # skips the update and lowers the scale on ``update`` instead of corrupting the weights.
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], cfg.clip)
        scaler.step(optimizer)
        scaler.update()
        last_done = step + 1

        if is_main and (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - started
            throughput = local_tokens * world_size / max(elapsed, 1e-6)
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if on_cuda else 0.0
            print(
                f"step {step + 1:6d} loss {loss_accum / max(loss_weight, 1):.4f} "
                f"gnorm {float(grad_norm):.3f} lr {lr_at(step, cfg):.2e} "
                f"seq {step_len:5d} mtp_w {mtp_weight_at(step, cfg):.3f} "
                f"tok/s {throughput:8.0f} peak_gb {peak:.2f}",
                flush=True,
            )

        if is_main and cfg.save_every and (step + 1) % cfg.save_every == 0:
            save(step + 1)

    # A budget stop or a run shorter than ``save_every`` would otherwise lose the tail.
    if is_main and last_done > start_step and not (cfg.save_every and last_done % cfg.save_every == 0):
        save(last_done)

    if world_size > 1:
        dist.destroy_process_group()


def train(config: SigmaConfig, cfg: TrainConfig, world_size: int = 1) -> None:
    if world_size > 1 and "RANK" not in os.environ:
        # ``mp.spawn`` does not create the rendezvous environment; ``init_process_group`` defaults
        # to env:// and would otherwise fail on MASTER_ADDR.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        mp.spawn(_train_worker, args=(world_size, config, cfg), nprocs=world_size, join=True)
    else:
        rank = int(os.environ.get("RANK", 0))
        _train_worker(rank, world_size, config, cfg)
