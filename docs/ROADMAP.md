# Roadmap: from a validated mechanism to a model

## Realistic hardware envelope

Planning target: 2 × T4 (16 GB each, no native bf16 → fp16 + gradient scaler). This is a rough
planning envelope for research up to roughly 100–300M parameters with short sequences, plus smoke
training at longer context; it is not a measured capacity guarantee. It does not support a 1.5B run:
parameters alone are ~3 GB in fp16, AdamW fp32 states add ~12 GB, and the activations for an 8K
chunk do not fit alongside.

Approximate per-GPU budget:

| Preset / planning size | Current or planned params | fp16 weights | AdamW states | Fits on 16 GB? |
| :--- | ---: | ---: | ---: | :--- |
| tiny preset | 2.08M | 4.2 MB | 16.6 MB | yes, trivially |
| 20M preset | 20.97M | 41.9 MB | 167.7 MB | yes |
| 100M preset | 142.11M | 284.2 MB | 1.14 GB | yes, small batch |
| 300M | 300M | 600 MB | 2.4 GB | yes, small batch |
| 1.5B | 1.5B | 3 GB | 12 GB | no |

## Stage 0 — gates (done)

`scripts/gate.py`: delta-rule equivalence, chunk invariance (split inside a slot block and a summary
block), constant state, MQAR ≥ 0.9. All pass.

**Establishes:** the mechanism works. **Does not establish:** language quality.

## Stage 0.5 — the experiment ladder (done)

`scripts/ladder.py` runs the controlled comparisons that back the architecture claim:

1. **SigmaStorm vs a size-matched transformer** on MQAR — same data, optimizer, steps and loss,
   only the sequence mixer differs.
2. **Recall vs distance to the binding**, with and without the summary ring, with a multi-timescale
   decay spectrum vs one rate (Gaps A and C).
3. **State capacity**, `state_expand` 1 vs 2 on dense MQAR (Gap B).
4. **Slot-write granularity**, 64-token consolidation vs whole-call (the old chunk-mean behaviour)
   on delayed recall (Gap D).
5. **Cost**: parameters, state bytes and forward time at three lengths.

Results are written to `results/ladder.md`. These are small CPU experiments; they establish
mechanism-level behaviour, not language quality or frontier parity.

## Stage 1 — 20M byte-level-BPE model, 8K context

Goal: a model that produces coherent text at all, and a loss curve that behaves.

1. Build 50–200M tokens of your own corpus into uint32 shards (`metadata.json` + `.bin`).
2. Train `configs/sigma_20m.json` at `seq_len 8192`, `micro_chunk 2048`, `bptt_chunks 4`, with the
   defaults `--schedule wsd --mtp_warmup_steps <10% of steps> --seq_len_start 1024
   --seq_ramp_steps <20% of steps>`.
3. Watch: held-out loss, `route_delta` vs `route_conv` (does the model use recall or only locality?),
   `projected_fraction` (should stay well below 1.0), gradient norm stability under fp16.
4. Measure tokens/s and peak memory per GPU; record them. These numbers decide Stage 2.

**Kill criteria:** if held-out loss stops improving while train loss falls, the model is memorising
through the conv/FFN and the state is not being used; reduce FFN width before scaling.

## Stage 2 — 100M, context scaling

Goal: show that quality does not degrade as the context grows, at a cost that stays linear.

1. Train `configs/sigma_100m.json` at 8K, then evaluate at 8K / 16K / 32K.
2. Compare per-token memory and time to a same-size transformer with an 8K window (if one is
   available) — this is where the fixed state should win.
3. Ablate: `local_attn_every = 0` (no window attention). If quality holds, the recurrence is doing
   the work; if it collapses, the delta rule is not yet strong enough.

**Deliverable:** a measured long-context advantage, or a clear negative result. Both are publishable
and both tell you whether to continue.

## Stage 3 — 300M with MoE

`n_experts > 0`, shared expert, top-k routing. Verify the aux load-balance loss and the expert
utilisation telemetry. This is the last stage that fits comfortably on 2 × T4.

## Stage 4 — beyond the hardware

A 1.5B run needs 8 × 80 GB or equivalent, plus a multi-billion-token corpus. Before that:

- add cross-chunk backpropagation (context parallelism) if long-range learning matters;
- add a token-sparse gather so gate zeros actually skip compute;
- replace the slot memory's novelty gate with a content-addressed write once the gradients are
  understood at 300M;
- grow the summary ring or add a two-level summary hierarchy if 1M-token addressing is needed.

## What to measure at every stage

- Held-out loss (chunked, no grad) on a fixed validation shard.
- MQAR at the same sequence length you train at — recall should scale with context, not degrade.
- `route_*` gates per block: if one operator's gate goes to zero everywhere, the mixture is
  collapsing and the other operators are not sufficient.
- `projected_fraction`: if it approaches 1.0, raise `max_norm` or lower `dt`.
- State bytes per sample: must be constant in `seq_len`.
- Peak GPU memory and tokens/s: record them per configuration.

## Honest expectation

Frontier-adjacent performance requires compute and data measured in thousands of GPU-years and
trillions of tokens. Nothing in this package changes that. What this package can do is establish,
rigorously and cheaply, whether a bounded-state non-transformer is competitive at small scale — and
the first evidence (recall at 97% with a fixed-size state) is positive.
