#!/usr/bin/env python3
"""The experiment ladder: evidence for Gaps A–D and for the architecture-level claim.

Each suite is a controlled comparison run on CPU in minutes, not a language-modelling benchmark:

* ``baseline``  — SigmaStorm vs a size-matched causal transformer on MQAR. Same data, optimizer,
  steps and loss (chunked CE); only the sequence mixer differs.
* ``distance``  — recall as a function of how far back the binding lies, full stack vs no summary
  ring vs one decay rate for all heads (Gaps A and C).
* ``capacity``  — dense MQAR (31 of 32 keys bound) with ``state_expand`` 1 vs 2 (Gap B).
* ``slots``     — per-token consolidated run/slot writes (64) vs whole-call consolidation (a
  proxy for the previous chunk-mean write) on a 256-token delayed-recall task (Gap D).
* ``cost``      — parameters, state bytes, and forward time versus a transformer at three lengths.

    python scripts/ladder.py --suite all
    python scripts/ladder.py --suite distance --steps 400

Results are written to ``results/ladder.json`` and a Markdown table to ``results/ladder.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigmastorm.config import SigmaConfig  # noqa: E402
from sigmastorm.model import SigmaStorm, chunked_cross_entropy  # noqa: E402
from sigmastorm.ops.norm import RMSNorm  # noqa: E402
from training.tasks import (  # noqa: E402
    make_aggregate_recall_batch,
    make_delayed_recall_batch,
    make_mqar_batch,
    make_presence_recall_batch,
)

RESULTS: dict[str, dict] = {}


# --------------------------------------------------------------------- baseline


def _sinusoidal(length: int, dim: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequencies)
    encoding[:, 1::2] = torch.cos(position * frequencies)
    return encoding.to(dtype)


class TinyTransformer(nn.Module):
    """Size-matched causal transformer: attention + MLP per block, nothing else.

    This is the reference the architecture claim depends on. It uses the same width, heads, depth,
    FFN multiplier and output loss; only the mixer is different. Sinusoidal positions are added to
    the embeddings: without positional information a transformer cannot implement induction
    (match a key, copy the value one position later), so an unpositioned baseline cannot learn
    MQAR at all and the comparison would be against a broken model rather than an equal one.
    """

    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": RMSNorm(config.d_model),
                    "attn": nn.MultiheadAttention(
                        config.d_model, config.n_heads, batch_first=True, bias=False
                    ),
                    "norm2": RMSNorm(config.d_model),
                    "ff": nn.Sequential(
                        nn.Linear(config.d_model, int(config.ffn_mult * config.d_model)),
                        nn.GELU(),
                        nn.Linear(int(config.ffn_mult * config.d_model), config.d_model),
                    ),
                }
            )
            for _ in range(config.n_blocks)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor):
        length = input_ids.shape[1]
        mask = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), 1)
        x = self.embedding(input_ids) + _sinusoidal(
            length, self.config.d_model, input_ids.device, self.embedding.weight.dtype
        )
        for block in self.blocks:
            h = block["norm1"](x)
            h, _ = block["attn"](h, h, h, attn_mask=mask, need_weights=False)
            x = x + h
            x = x + block["ff"](block["norm2"](x))
        hidden = self.final_norm(x)
        return hidden, None, {}, {}

    def loss_from_hidden(self, hidden, targets, loss_mask=None, extras=None):
        total, count = chunked_cross_entropy(hidden, self.lm_head.weight, targets, loss_mask, 256)
        return {
            "lm_sum": total,
            "lm_count": count,
            "mtp_sum": {},
            "mtp_count": {},
            "moe_aux": hidden.new_zeros(()),
            "route_entropy": hidden.new_zeros(()),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------- tasks


def mqar_task(batch: int, length: int, n_keys: int, n_values: int, n_queries: int, device):
    generator = torch.Generator(device=device).manual_seed(7)
    return lambda: make_mqar_batch(
        batch,
        length,
        n_queries=n_queries,
        n_keys=n_keys,
        n_values=n_values,
        device=device,
        generator=generator,
    )


def delayed_task(batch: int, length: int, n_pairs: int, n_values: int, n_queries: int, device):
    generator = torch.Generator(device=device).manual_seed(11)
    return lambda: make_delayed_recall_batch(
        batch,
        length,
        n_pairs=n_pairs,
        n_values=n_values,
        n_queries=n_queries,
        device=device,
        generator=generator,
    )


def aggregate_task(batch: int, length: int, max_count: int, device):
    generator = torch.Generator(device=device).manual_seed(13)
    return lambda: make_aggregate_recall_batch(
        batch,
        length,
        max_count=max_count,
        device=device,
        generator=generator,
    )


def presence_task(batch: int, length: int, n_symbols: int, n_present: int, device):
    generator = torch.Generator(device=device).manual_seed(17)
    return lambda: make_presence_recall_batch(
        batch,
        length,
        n_symbols=n_symbols,
        n_present=n_present,
        device=device,
        generator=generator,
    )


# -------------------------------------------------------------------- training


def train_model(model, task, steps: int, batch: int, lr: float, device, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    started = time.time()
    for _ in range(steps):
        data = task()
        hidden, _, extras, _ = model(data["input_ids"])
        outputs = model.loss_from_hidden(hidden, data["targets"], data["loss_mask"], extras)
        loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return time.time() - started


@torch.no_grad()
def evaluate_recall(model, task, batches: int = 8, by_distance: bool = False) -> dict:
    model.eval()
    correct = total = 0
    buckets: dict[str, list[int]] = {}
    for _ in range(batches):
        data = task()
        hidden, _, _, _ = model(data["input_ids"])
        predictions = model.lm_head(hidden).argmax(dim=-1)
        mask = data["loss_mask"]
        hit = (predictions[mask] == data["targets"][mask]).float()
        correct += float(hit.sum())
        total += int(hit.numel())
        if by_distance:
            distances = data["distances"].reshape(-1)
            for bucket, (low, high) in {
                "<32": (0, 32),
                "32-127": (32, 128),
                "128-255": (128, 256),
                ">=256": (256, 10**9),
            }.items():
                selected = (distances >= low) & (distances < high)
                if bool(selected.any()):
                    buckets.setdefault(bucket, [0, 0])
                    buckets[bucket][0] += float(hit[selected].sum())
                    buckets[bucket][1] += int(selected.sum())
    result = {"accuracy": correct / max(total, 1)}
    if by_distance:
        result["by_distance"] = {key: value[0] / max(value[1], 1) for key, value in buckets.items()}
    return result


def run_variant(
    name: str,
    config: SigmaConfig,
    task,
    steps: int,
    batch: int,
    device,
    by_distance: bool = False,
    transformer: bool = False,
) -> dict:
    model = TinyTransformer(config) if transformer else SigmaStorm(config)
    seconds = train_model(model, task, steps, batch, 1e-3, device)
    metrics = evaluate_recall(model, task, by_distance=by_distance)
    record = {
        "params": model.num_parameters(),
        "seconds": round(seconds, 1),
        **metrics,
    }
    if not transformer:
        record["state_bytes_per_sample"] = model.state_report()["state_bytes_per_sample"]
    print(
        f"  {name}: acc {record['accuracy']:.3f} params {record['params']:,} "
        f"time {record['seconds']}s" + (f" by distance {metrics['by_distance']}" if by_distance else ""),
        flush=True,
    )
    return record


# --------------------------------------------------------------------- suites


def suit_baseline(args, device) -> dict:
    config = SigmaConfig.from_dict(
        {
            **SigmaConfig.for_size("tiny").to_dict(),
            "vocab_size": 64,
            "summary_chunk": 16,
            "summary_slots": 16,
        }
    )
    task = mqar_task(batch=24, length=64, n_keys=32, n_values=16, n_queries=4, device=device)
    return {
        "task": "MQAR 26 bindings / 16 values / 4 queries",
        "sigma": run_variant("sigma", config, task, args.steps, 24, device),
        "transformer": run_variant("transformer", config, task, args.steps, 24, device, transformer=True),
        "chance": 1 / 16,
    }


def suit_distance(args, device) -> dict:
    base = {
        **SigmaConfig.for_size("tiny").to_dict(),
        "vocab_size": 64,
        "summary_every": 1,
        "summary_chunk": 32,
        "summary_slots": 32,
        "use_window": False,
    }
    # Isolate the delta rule: the slot memory has an unbounded horizon and would otherwise solve the
    # task on its own, hiding the effect of the decay spectrum. The decay bias is frozen so the
    # comparison measures the initialised prior: when the bias can move, a single-rate model simply
    # learns its own spectrum and the ablation shows nothing.
    delta_only = {
        "use_memory": False,
        "use_summary": False,
        "use_window": False,
        "decay_learnable": False,
    }
    variants = {
        "full": {},
        "delta_spectrum": delta_only,
        "delta_single_rate": {
            **delta_only,
            "decay_min_rate": 0.018,
            "decay_max_rate": 0.018,
        },
    }
    task = delayed_task(batch=8, length=320, n_pairs=8, n_values=16, n_queries=4, device=device)
    out = {"task": "delayed recall, 8 bindings, length 320", "variants": {}}
    for name, overrides in variants.items():
        out["variants"][name] = run_variant(
            name,
            SigmaConfig.from_dict({**base, **overrides}),
            task,
            args.steps,
            8,
            device,
            by_distance=True,
        )
    return out


def suit_capacity(args, device) -> dict:
    # 66 bindings is above the ~32-binding practical rank of one 32x32 state matrix per head, but
    # still trainable in a few hundred steps. 90 bindings did not train at all in that budget (both
    # variants stayed at chance), which measures optimisation, not capacity.
    base = {
        **SigmaConfig.for_size("tiny").to_dict(),
        "vocab_size": 128,
        "summary_chunk": 32,
        "summary_slots": 16,
    }
    task = mqar_task(batch=16, length=144, n_keys=66, n_values=8, n_queries=4, device=device)
    out = {"task": "dense MQAR, 66 bindings, 4 queries", "variants": {}}
    for expand in (1, 2):
        out["variants"][f"expand_{expand}"] = run_variant(
            f"expand_{expand}",
            SigmaConfig.from_dict({**base, "state_expand": expand}),
            task,
            args.steps,
            16,
            device,
        )
    return out


def suit_slots(args, device) -> dict:
    # Only convolution + slot memory. The task is presence recall, the capability the slot memory
    # actually has: a content-addressed prototype store, not a key->value binding (that is the
    # delta rule). Key->value delayed recall is at chance for any write granularity, so the earlier
    # version of this suite measured nothing.
    base = {
        **SigmaConfig.for_size("tiny").to_dict(),
        "vocab_size": 64,
        "use_summary": False,
        "use_delta": False,
        "use_window": False,
        "summary_chunk": 32,
    }
    task = presence_task(batch=8, length=192, n_symbols=16, n_present=8, device=device)
    out = {"task": "presence recall, length 192, conv + slots only", "variants": {}}
    for block in (64, 4096):
        out["variants"][f"slot_block_{block}"] = run_variant(
            f"slot_block_{block}",
            SigmaConfig.from_dict({**base, "slot_block": block}),
            task,
            args.steps,
            8,
            device,
        )
    return out


def suit_aggregate(args, device) -> dict:
    base = {
        **SigmaConfig.for_size("tiny").to_dict(),
        "vocab_size": 32,
        "use_window": False,
        "summary_every": 1,
        "summary_chunk": 32,
        "summary_slots": 16,
    }
    task = aggregate_task(batch=8, length=192, max_count=8, device=device)
    out = {"task": "count symbols seen ~150 tokens back", "variants": {}}
    for name, overrides in (("full", {}), ("no_summary", {"use_summary": False})):
        out["variants"][name] = run_variant(
            name,
            SigmaConfig.from_dict({**base, **overrides}),
            task,
            args.steps,
            8,
            device,
        )
    return out


def suit_cost(args, device) -> dict:
    out = {"variants": {}}
    for label, config in (
        ("tiny", SigmaConfig.for_size("tiny")),
        ("20m", SigmaConfig.for_size("20m")),
    ):
        for length in (128, 512, 2048):
            row = {}
            for name, build in (
                ("sigma", lambda c=config: SigmaStorm(c)),
                ("transformer", lambda c=config: TinyTransformer(c)),
            ):
                model = build().to(device).eval()
                tokens = torch.randint(0, config.vocab_size, (1, length), device=device)
                with torch.no_grad():
                    model(tokens)
                    started = time.time()
                    for _ in range(5):
                        model(tokens)
                    seconds = (time.time() - started) / 5
                row[name] = {"params": model.num_parameters(), "forward_s": round(seconds, 4)}
                if name == "sigma":
                    row[name]["state_bytes"] = model.state_report()["state_bytes_per_sample"]
            out["variants"][f"{label}_T{length}"] = row
            print(f"  {label} T={length}: {row}", flush=True)
    return out


SUITES = {
    "baseline": suit_baseline,
    "distance": suit_distance,
    "aggregate": suit_aggregate,
    "capacity": suit_capacity,
    "slots": suit_slots,
    "cost": suit_cost,
}


def to_markdown(results: dict) -> str:
    lines = ["# Experiment ladder results", ""]
    if "baseline" in results:
        row = results["baseline"]
        lines += [
            "## MQAR: SigmaStorm vs a size-matched transformer",
            "",
            f"Task: {row['task']}; chance {row['chance']:.3f}.",
            "",
            "| Model | Params | Accuracy | Train s |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for name in ("sigma", "transformer"):
            item = row[name]
            lines.append(f"| {name} | {item['params']:,} | {item['accuracy']:.3f} | {item['seconds']} |")
        lines.append("")
    if "distance" in results:
        row = results["distance"]
        lines += [
            "## Recall vs distance to the binding",
            "",
            f"Task: {row['task']}.",
            "",
            "| Variant | Accuracy | <32 | 32-127 | 128-255 | >=256 | Params |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, item in row["variants"].items():
            buckets = item["by_distance"]
            lines.append(
                f"| {name} | {item['accuracy']:.3f} | {buckets.get('<32', 0):.3f} | "
                f"{buckets.get('32-127', 0):.3f} | {buckets.get('128-255', 0):.3f} | "
                f"{buckets.get('>=256', 0):.3f} | {item['params']:,} |"
            )
        lines.append("")
    if "aggregate" in results:
        row = results["aggregate"]
        lines += [
            "## Global aggregate path (chunk summaries)",
            "",
            f"Task: {row['task']}.",
            "",
            "| Variant | Accuracy | Params |",
            "| :--- | ---: | ---: |",
        ]
        for name, item in row["variants"].items():
            lines.append(f"| {name} | {item['accuracy']:.3f} | {item['params']:,} |")
        lines.append("")
    for key, title in (("capacity", "State capacity: state_expand"), ("slots", "Slot write granularity")):
        if key in results:
            row = results[key]
            lines += [
                f"## {title}",
                "",
                f"Task: {row['task']}.",
                "",
                "| Variant | Accuracy | Params | State bytes |",
                "| :--- | ---: | ---: | ---: |",
            ]
            for name, item in row["variants"].items():
                lines.append(
                    f"| {name} | {item['accuracy']:.3f} | {item['params']:,} | "
                    f"{item['state_bytes_per_sample']:,} |"
                )
            lines.append("")
    if "cost" in results:
        lines += [
            "## Cost at equal width",
            "",
            "| Config | Sigma params | Sigma forward s | Transformer params | Transformer forward s |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for name, item in results["cost"]["variants"].items():
            lines.append(
                f"| {name} | {item['sigma']['params']:,} | {item['sigma']['forward_s']} | "
                f"{item['transformer']['params']:,} | {item['transformer']['forward_s']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", choices=[*SUITES, "all"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    threads = int(os.environ.get("LADDER_THREADS", "2"))
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} steps={args.steps}", flush=True)
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    for name in suites:
        print(f"[{name}]", flush=True)
        RESULTS[name] = SUITES[name](args, device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Suites can be run one at a time; merge with any earlier results.
    combined = {}
    if (out / "ladder.json").exists():
        combined = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
    combined.update(RESULTS)
    (out / "ladder.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    (out / "ladder.md").write_text(to_markdown(combined) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'ladder.json'} and {out / 'ladder.md'}", flush=True)


if __name__ == "__main__":
    main()
