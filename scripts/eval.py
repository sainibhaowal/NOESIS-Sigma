#!/usr/bin/env python3
"""Held-out perplexity and MQAR score for a SigmaStorm checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigmastorm.config import SigmaConfig  # noqa: E402
from sigmastorm.model import SigmaStorm  # noqa: E402
from training.data import PackedShards  # noqa: E402
from training.evaluate import evaluate_loss, evaluate_mqar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--micro_chunk", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--mqar_length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SigmaConfig.from_dict(payload["config"])
    model = SigmaStorm(config)
    model.load_state_dict(payload["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    steps = payload.get("step")

    result = {"checkpoint": args.checkpoint, "step": steps, "parameters": model.num_parameters()}
    if args.data_dir:
        shards = PackedShards(args.data_dir, rank=0, world_size=1)
        result["heldout"] = evaluate_loss(
            model, shards, args.batch_size, args.seq_len, args.micro_chunk, device, args.max_batches
        )
    result["mqar"] = evaluate_mqar(model, length=args.mqar_length, device=device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
