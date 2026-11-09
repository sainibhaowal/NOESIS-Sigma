#!/usr/bin/env python3
"""Gap D proof: is the slot read content-addressed?

Three controlled experiments, three arms — previous cosine metric, the ZCA-whitened metric on its
own, and the shipped gated mechanism (novelty dead-zone + sharp read + match-confidence abstention)
— same data, seeds, optimizer and steps:

1. **Presence recall** (the capability the slot memory actually has): 16 symbols, 8 present in
   the opening quarter, filler afterwards, query at the end. Conv + slot memory only. Token
   embeddings are frozen and the head is untied, so the model cannot solve the task by collapsing
   the token embeddings (the failure mode measured earlier: centered cosine 0.285, accuracy 0.504).
2. **Aggregate recall** (Gap C path): count symbols seen ~150 tokens back with the full tiny model.
   Guards against a regression in the soft/pooling use of the memory.
3. **MQAR** (Gap A path): 32 key->value bindings, full tiny model. Guards against a regression in
   the delta rule.

Also reports, for the presence model: read entropy (nats; ln slots is uniform), the memory
ablation delta in nats (zero the slot read), and the cross-slot key cosine.

    python scripts/gapd_proof.py --device cuda --out_dir results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sigmastorm.config import SigmaConfig  # noqa: E402
from sigmastorm.model import SigmaStorm  # noqa: E402
from training.tasks import (  # noqa: E402
    make_aggregate_recall_batch,
    make_mqar_batch,
    make_presence_recall_batch,
)

# Three arms, same data/optimiser/steps. ``cosine`` is the previous behaviour; ``whitened`` is the
# ZCA metric on its own; ``gated`` is the shipped mechanism: novelty dead-zone on the write, sharp
# read, and match-confidence abstention.
VARIANTS = {
    "cosine": {},
    "whitened": {"memory_metric": "whitened"},
    "gated": {
        "memory_novelty_floor": 0.1,
        "memory_read_temperature": 0.1,
        "memory_abstain": True,
    },
}


def presence_config(variant: str) -> SigmaConfig:
    return SigmaConfig.from_dict(
        {
            **SigmaConfig.for_size("tiny").to_dict(),
            "vocab_size": 20,
            "d_model": 128,
            "n_blocks": 2,
            "use_delta": False,
            "use_summary": False,
            "use_window": False,
            "slot_count": 16,
            "slot_block": 64,
            "tie_embeddings": False,
            **VARIANTS[variant],
        }
    )


def full_config(variant: str, vocab_size: int) -> SigmaConfig:
    return SigmaConfig.from_dict(
        {
            **SigmaConfig.for_size("tiny").to_dict(),
            "vocab_size": vocab_size,
            "use_window": False,
            **VARIANTS[variant],
        }
    )


def train(
    model: SigmaStorm,
    task,
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
    freeze_embedding: bool = False,
) -> float:
    if freeze_embedding:
        model.embedding.weight.requires_grad_(False)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
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
def accuracy(model: SigmaStorm, task, batches: int, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for _ in range(batches):
        data = task()
        hidden, _, _, _ = model(data["input_ids"])
        predictions = model.lm_head(hidden).argmax(dim=-1)
        mask = data["loss_mask"]
        correct += int((predictions[mask] == data["targets"][mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


@torch.no_grad()
def masked_ce(model: SigmaStorm, task, batches: int, device: torch.device) -> float:
    model.eval()
    total = count = 0.0
    for _ in range(batches):
        data = task()
        hidden, _, extras, _ = model(data["input_ids"])
        outputs = model.loss_from_hidden(hidden, data["targets"], data["loss_mask"], extras)
        total += float(outputs["lm_sum"])
        count += float(outputs["lm_count"])
    return total / max(count, 1.0)


@torch.no_grad()
def slot_diagnostics(model: SigmaStorm, task, device: torch.device) -> dict:
    model.eval()
    data = task()
    hidden, state, _, _ = model(data["input_ids"])
    entropies, top_weights, confidences = [], [], []
    for block in model.blocks:
        memory = block.transition.memory
        if memory is None or memory.last_read is None:
            continue
        probabilities = memory.last_read.float().clamp_min(1e-9)
        entropies.append(float(-(probabilities * probabilities.log()).sum(-1).mean()))
        top_weights.append(float(probabilities.max(-1).values.mean()))
        if getattr(memory, "abstain", False):
            confidences.append(memory.last_confidence.float()[:, -1])
    present_conf = absent_conf = None
    if confidences:
        present = data["is_present"]
        confidence = confidences[0]
        present_conf = float(confidence[present].mean())
        absent_conf = float(confidence[~present].mean())
    cosines = []
    for block in state.blocks:
        if block.memory is None:
            continue
        keys = torch.nn.functional.normalize(block.memory.keys.float(), dim=-1)
        gram = keys @ keys.transpose(1, 2)
        slots = gram.shape[-1]
        off = gram - torch.eye(slots, device=gram.device, dtype=gram.dtype)
        cosines.append(float(off.sum() / (slots * (slots - 1))))
    baseline = masked_ce(model, task, 8, device)
    hooks = []
    for block in model.blocks:
        memory = block.transition.memory
        if memory is None:
            continue
        hooks.append(
            memory.register_forward_hook(
                lambda module, inputs, output: (torch.zeros_like(output[0]), output[1])
            )
        )
    ablated = masked_ce(model, task, 8, device)
    for hook in hooks:
        hook.remove()
    payload = {
        "read_entropy_nats": sum(entropies) / max(len(entropies), 1),
        "read_entropy_uniform": float(torch.log(torch.tensor(float(model.config.slot_count)))),
        "mean_top_weight": sum(top_weights) / max(len(top_weights), 1),
        "cross_slot_key_cosine": sum(cosines) / max(len(cosines), 1),
        "ce_nats": baseline,
        "ablation_delta_nats": ablated - baseline,
    }
    if present_conf is not None:
        payload["confidence_present"] = present_conf
        payload["confidence_absent"] = absent_conf
    return payload


def run_presence(device: torch.device, steps: int, seeds: list[int]) -> dict:
    results = {}
    for variant in VARIANTS:
        rows = []
        for seed in seeds:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(1000 + seed)

            def task(generator=generator):
                return make_presence_recall_batch(
                    8, 192, n_symbols=16, n_present=8, device=device, generator=generator
                )

            model = SigmaStorm(presence_config(variant))
            seconds = train(model, task, steps, 8, 1e-3, device, freeze_embedding=True)
            score = accuracy(model, task, 16, device)
            row = {"seed": seed, "accuracy": score, "train_s": round(seconds, 1)}
            if seed == seeds[0]:
                row.update(slot_diagnostics(model, task, device))
            rows.append(row)
            print(
                f"  presence {variant:8s} seed={seed} acc={score:.3f} "
                f"({seconds:.0f}s)" + (f" diag={row.get('read_entropy_nats')}" if seed == seeds[0] else ""),
                flush=True,
            )
        results[variant] = rows
    return results


def run_aggregate(device: torch.device, steps: int, seeds: list[int]) -> dict:
    results = {}
    for variant in VARIANTS:
        rows = []
        for seed in seeds:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(2000 + seed)

            def task(generator=generator):
                return make_aggregate_recall_batch(8, 192, max_count=8, device=device, generator=generator)

            model = SigmaStorm(full_config(variant, 32))
            seconds = train(model, task, steps, 8, 1e-3, device)
            score = accuracy(model, task, 16, device)
            rows.append({"seed": seed, "accuracy": score, "train_s": round(seconds, 1)})
            print(f"  aggregate {variant:8s} seed={seed} acc={score:.3f} ({seconds:.0f}s)", flush=True)
        results[variant] = rows
    return results


def run_mqar(device: torch.device, steps: int, seeds: list[int]) -> dict:
    results = {}
    for variant in VARIANTS:
        rows = []
        for seed in seeds:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(3000 + seed)

            def task(generator=generator):
                return make_mqar_batch(
                    16, 64, n_queries=4, n_keys=32, n_values=16, device=device, generator=generator
                )

            model = SigmaStorm(full_config(variant, 64))
            seconds = train(model, task, steps, 16, 1e-3, device)
            score = accuracy(model, task, 16, device)
            rows.append({"seed": seed, "accuracy": score, "train_s": round(seconds, 1)})
            print(f"  mqar {variant:8s} seed={seed} acc={score:.3f} ({seconds:.0f}s)", flush=True)
        results[variant] = rows
    return results


def to_markdown(payload: dict) -> str:
    lines = ["# Gap D proof: gated content-addressed slot memory", ""]
    lines.append(f"Device: `{payload['device']}` — steps: {payload['steps']}, seeds: {payload['seeds']}")
    lines.append("")

    def table(title: str, section: dict, note: str) -> None:
        lines.extend([f"## {title}", "", note, ""])
        lines.append("| Metric | Seed | Accuracy | Train s |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for metric, rows in section.items():
            for row in rows:
                lines.append(f"| {metric} | {row['seed']} | {row['accuracy']:.3f} | {row['train_s']} |")
        lines.append("")

    for key, title, note in (
        (
            "presence",
            "Presence recall (conv + slots, frozen embeddings)",
            "Chance 0.50. The task the slot memory is for.",
        ),
        ("aggregate", "Aggregate recall (full tiny model)", "Chance 1/9. Gap C path; must not regress."),
        ("mqar", "MQAR (full tiny model)", "Chance 1/16. Gap A path; must not regress."),
    ):
        if key in payload:
            table(title, payload[key], note)
    diagnostics = payload.get("presence_diagnostics")
    if diagnostics:
        lines.extend(["## Presence diagnostics (seed 0)", ""])
        lines.append(
            "| Metric | Read entropy | ln slots | Top weight | Cross-slot cos | CE | Ablation Δ nats | "
            "Confidence present | Confidence absent |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric, row in diagnostics.items():
            present = row.get("confidence_present")
            absent = row.get("confidence_absent")
            conf = f"{present:.3f} | {absent:.3f}" if present is not None else "— | —"
            lines.append(
                f"| {metric} | {row['read_entropy_nats']:.3f} | {row['read_entropy_uniform']:.3f} | "
                f"{row['mean_top_weight']:.3f} | {row['cross_slot_key_cosine']:.3f} | "
                f"{row['ce_nats']:.3f} | {row['ablation_delta_nats']:+.3f} | {conf} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--aggregate_steps", type=int, default=300)
    parser.add_argument("--mqar_steps", type=int, default=400)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sections", default="presence,aggregate,mqar")
    parser.add_argument("--out_dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    sections = [part for part in args.sections.split(",") if part]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(part) for part in args.seeds.split(",") if part]
    print(f"device={device} seeds={seeds} presence_steps={args.steps}", flush=True)

    payload = {"device": str(device), "steps": args.steps, "seeds": seeds}
    if "presence" in sections:
        payload["presence"] = run_presence(device, args.steps, seeds)
        payload["presence_diagnostics"] = {
            metric: {
                key: value
                for key, value in rows[0].items()
                if key
                in (
                    "read_entropy_nats",
                    "read_entropy_uniform",
                    "mean_top_weight",
                    "cross_slot_key_cosine",
                    "ce_nats",
                    "ablation_delta_nats",
                    "confidence_present",
                    "confidence_absent",
                )
            }
            for metric, rows in payload["presence"].items()
        }
    if "aggregate" in sections:
        print("[aggregate]", flush=True)
        payload["aggregate"] = run_aggregate(device, args.aggregate_steps, seeds[:2])
    if "mqar" in sections:
        print("[mqar]", flush=True)
        payload["mqar"] = run_mqar(device, args.mqar_steps, seeds[:2])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gapd_proof.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "gapd_proof.md").write_text(to_markdown(payload) + "\n")
    print(f"wrote {out / 'gapd_proof.json'} and {out / 'gapd_proof.md'}", flush=True)


if __name__ == "__main__":
    main()
