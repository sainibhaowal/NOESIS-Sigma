#!/usr/bin/env python3
"""Train SigmaStorm.

    python scripts/train.py --config configs/sigma_100m.json --data_dir /path/to/shards \
        --batch_size 4 --seq_len 2048 --steps 2000 --nproc 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigmastorm.config import SigmaConfig  # noqa: E402
from training.trainer import TrainConfig, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sigma_100m.json", help="path or preset name")
    parser.add_argument("--data_dir", required=True, help="directory holding metadata.json and *.bin")
    parser.add_argument("--out_dir", default="runs/sigma")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--micro_chunk", type=int, default=512)
    parser.add_argument("--bptt_chunks", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    parser.add_argument("--decay_frac", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mtp_weight", type=float, default=0.3)
    parser.add_argument("--mtp_warmup_steps", type=int, default=0)
    parser.add_argument("--seq_len_start", type=int, default=0, help="0 disables the length ramp")
    parser.add_argument("--seq_ramp_steps", type=int, default=0)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt")
    parser.add_argument("--max_hours", type=float, default=0.0, help="0 disables the wall-clock stop")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = SigmaConfig.load(config_path) if config_path.exists() else SigmaConfig.for_size(args.config)

    allowed = {item.name for item in fields(TrainConfig)}
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        micro_chunk=args.micro_chunk,
        bptt_chunks=args.bptt_chunks,
        grad_accum=args.grad_accum,
        total_steps=args.steps,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        schedule=args.schedule,
        decay_frac=args.decay_frac,
        weight_decay=args.weight_decay,
        clip=args.clip,
        mtp_weight=args.mtp_weight,
        mtp_warmup_steps=args.mtp_warmup_steps,
        seq_len_start=args.seq_len_start,
        seq_ramp_steps=args.seq_ramp_steps,
        precision=args.precision,
        grad_checkpoint=args.grad_checkpoint,
        compile=args.compile,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        resume=not args.fresh,
        max_hours=args.max_hours,
    )
    assert set(vars(train_cfg)) <= allowed
    train(config, train_cfg, world_size=args.nproc)


if __name__ == "__main__":
    main()
