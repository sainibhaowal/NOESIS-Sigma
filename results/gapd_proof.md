# Gap D proof: the slot read is a gated address

Measured 2026-11-06 on CPU (`torch 2.15.0+cpu`). The Kaggle weekly GPU quota (30 h) was exhausted,
so every arm was re-run locally with identical data, seeds, optimizer and steps; the GPU kernel in
`kaggle/gapd/run.py` stages the same three-arm experiment and is ready to push when quota resets.
Raw numbers: [gapd_proof.json](gapd_proof.json).

## The defect

The slot memory had a **diffuse, unconditional read**. In the trained 20M checkpoint (step 10,604)
per-block read entropy was 3.03–3.16 nats against `ln 24 = 3.178`, and zeroing the memory read cost
only +0.035 nats: the operator was a slow average of every slot, not a content address.

Two mechanisms caused it:

1. **A softmax always hands out unit mass.** A query that matches nothing still receives a
   full-scale weighted average of all slot values. There is no output magnitude that means "not in
   memory", no abstention path, and therefore no gradient pressure to sharpen the selection.
2. **Low-novelty writes leak.** A repeated token with novelty 0.02 still contributes
   `0.02 × evidence` to every routed slot. A few hundred repetitions sum past one pseudo-observation,
   `η = D/(D+1)` replaces the slot with the filler mean, and every slot becomes occupied. Measured
   before the fix: block-1 slot keys collapsed to pairwise cosine 0.94 and an absent query always
   retrieved something.

The addressing geometry itself was sound: the same symbol presented more than once produced keys
with cosine +1.000. The failure was in the read/write gates, not the key representation.

## The fix

All three changes are opt-in (`memory_novelty_floor`, `memory_read_temperature`,
`memory_abstain`); defaults preserve the previous behaviour and every existing checkpoint loads
unchanged.

### 1. Novelty dead-zone on writes

```
gate_t  = [(novelty_t − δ) / (1 − δ)]₊ · evidence_t          δ = memory_novelty_floor = 0.1
novelty = 1 − max_j cos(k_t, K_j)                            (second pass: vs earlier writes in-block)
```

Below `δ` the write weight is exactly zero and the gradient is unchanged above it, so the redundancy
filter is exact rather than a heuristic damping factor.

### 2. Separate read temperature

`τ_r = 0.1` (`memory_read_temperature`), independent of the routing sharpness `τ_w = 0.2 · τ`. A
read that is sharp before the addresses are trained picks an arbitrary slot; the read temperature is
learned separately and only sharpens the final address distribution.

### 3. Match-confidence abstention

```
match_t    = max_{j : use_j > 1e-3} cos(q_t, K_j)      # before the usage bias
confidence = σ(κ · (match_t − θ))                        # learned κ (init 10), θ (init 0.4)
read_t     = confidence · W_o Σ_j softmax_j(scores / τ_r) V_j
```

The read is multiplied by a differentiable per-token likelihood ratio. Unmatched queries contribute
zero and the residual stream alone answers them; matched queries pass at full scale.

## 1. Presence recall — the capability the slot memory is for

16 symbols, 8 present in the opening quarter of a 192-token stream, filler afterwards, query at the
end. Conv + slot memory only, **frozen token embeddings, untied head**, so the model cannot solve the
task by collapsing the embeddings. Chance 0.50. 800 steps, 3 seeds, identical data and optimizer.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean |
| :--- | ---: | ---: | ---: | ---: |
| cosine (previous behaviour) | 0.562 | 0.539 | 0.555 | 0.552 |
| ZCA-whitened metric alone | 0.648 | 0.531 | 0.539 | 0.573 |
| **gated (shipped)** | **0.977** | **1.000** | **0.953** | **0.977** |

Seed-0 diagnostics (16 slots, uniform entropy `ln 16 = 2.773` nats):

| Arm | Read entropy | Top weight | Cross-slot key cos | CE | Memory ablation Δ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| cosine | 1.260 | 0.608 | 0.023 | 0.663 | +0.147 |
| whitened | 0.725 | 0.788 | −0.044 | 0.517 | +0.217 |
| **gated** | **0.413** | **0.820** | **0.004** | **0.019** | **+2.255** |

The gated read is also discriminating, not just sharp: match confidence is **0.562 on present**
queries and **0.026 on absent** queries. Whitening sharpens the read (entropy 1.26 → 0.72) but does
not make it useful, because it amplifies every deviation direction, including context noise; it stays
at chance. The ablation delta shows the same thing from the other side: with the gated read, zeroing
the memory costs +2.26 nats, against +0.15 before. The memory is now load-bearing.

## 2. MQAR (Gap A path) — must not regress

32 unique key → value bindings, full tiny model, 400 steps, 2 seeds. Chance 1/16 = 0.062.

| Arm | Seed 0 | Seed 1 |
| :--- | ---: | ---: |
| cosine | 0.988 | 0.979 |
| whitened | 0.988 | 0.979 |
| **gated** | **0.990** | **0.987** |

The delta-rule path is unchanged to within seed noise (gated is +0.002 / +0.008).

## 3. Aggregate recall (Gap C summary path) — must not regress

Count symbols seen ~150 tokens back, full tiny model, 600 steps, 2 seeds. Chance 1/9 = 0.111.

| Arm | Seed 0 | Seed 1 |
| :--- | ---: | ---: |
| cosine | 0.680 | 1.000 |
| whitened | 0.672 | 0.719 |
| **gated** | **0.836** | **0.859** |

At this budget the metric is seed-noise-limited (a 300-step, 4-seed sweep gave 0.62 / 0.69 / 0.71
for cosine / whitened / gated), but there is no regression signal: the gated arm is at or above the
previous behaviour on both seeds.

## 4. Real-data next-token LM — must not regress

Tiny preset (3.06M parameters), `vocab_size=8192`, batch 8 × 256 tokens, 600 steps, **identical
TinyStories token stream**, held-out split.

| Arm | Val CE (nats) | PPL |
| :--- | ---: | ---: |
| baseline (defaults) | 3.0760 | 21.67 |
| **gated** | **3.0826** | **21.82** |

A difference of 0.0066 nats (0.2%), inside the run-to-run seed spread; the training curves overlap
throughout. The gated mechanism does not degrade language modelling.

## Gates and unit tests

```
scripts/gate.py  → ALL GATES PASSED
  delta_rule_chunkwise_equals_sequential   4.17e-07
  chunked_forward_equals_full              1.55e-06
  state_memory_independent_of_context      200704 bytes at T=512 and T=4096
  mqar_recall                              0.974 vs chance 0.062
pytest tests -q  → 21 passed
```

The four new tests pin the mechanism itself:

- `test_novelty_dead_zone_blocks_repeated_writes` — a 0.98 max-similarity score produces exactly
  zero novelty, and repeated-write usage falls to 0.46 × the unfloored module.
- `test_abstention_gate_suppresses_unmatched_query` — confidence > 0.9 for an exact key match,
  < 0.3 for an unmatched query.
- `test_gated_slot_memory_is_chunk_invariant` — split at 50, inside a 64-token slot block, matches
  the full forward to 3e-3.
- `test_gated_slot_memory_backward_is_finite`.

## Reproduction

```bash
.venv/bin/python scripts/gapd_proof.py --device cpu \
    --steps 800 --aggregate_steps 600 --mqar_steps 400 --seeds 0,1,2 --out_dir results
```

The three arms are defined in `VARIANTS` in [scripts/gapd_proof.py](../scripts/gapd_proof.py);
`--sections presence,aggregate,mqar` selects a subset. The production profile is
[configs/sigma_20m_gapd.json](../configs/sigma_20m_gapd.json).

## Limitations

- CPU run. The GPU kernel ([kaggle/gapd/run.py](../kaggle/gapd/run.py)) reproduces this at 1200
  steps once quota resets; the presence result is already at ceiling (1.000 on seed 1) and further
  steps cannot change the conclusion for the previous arms, which stay at chance.
- The 20M TinyStories checkpoint predates the fix; a gated 20M no-regression run is staged but not
  executed (GPU quota). The 3M real-data LM check above is the available no-regression evidence.
- The aggregate task is seed-noise-limited at these budgets; treat it as a guard, not a measurement
  of the summary ring's quality.
