# Findings: what the legacy code actually does, and what was kept

This is the written answer to "read `Archive_Legacy/Core/OSC` and tell me what you find". It is an
audit, not a wish list.

## 1. `Archive_Legacy/Core/Sigma` — v1 of your law

**Good parts, kept.**

- A clean contract between `config`, `state`, `telemetry` and the transition. The state is explicit
  and bounded; nothing depends on sequence length.
- The operator vocabulary: local proposal, selective recurrent proposal, interaction proposal,
  associative memory, skew-conservative term, energy gradient, projection. This is the right shape
  for a non-transformer.
- The discipline of declaring which state is carried and which is activation.

**Fatal parts, replaced.**

| Finding | Evidence | Consequence |
| :--- | :--- | :--- |
| `model.py::forward` loops **token by token** | one Python iteration per token | untrainable on a GPU; no batch parallelism |
| no FFN at all | `ff_multiplier` in config, never used | the transition alone must do everything |
| the controller computes **all** operators then mixes them | same time with a gate at 0 | "budget" is decorative |
| energy is a linear decay | `diagonal` term only | `-∇E` is not doing associative work |
| 8 slots, no novelty/evidence gate | fixed writes | memory is overwritten by repeated content |
| single-position recurrence, no chunk form | sequential state update | cannot process a 256K context |

SigmaStorm keeps the vocabulary and rebuilds the execution: chunkwise delta rule with a per-head
decay spectrum and expanded state, per-token gated slot writes consolidated every `slot_block`
tokens, a chunk-summary global path, convex gated mixture, real SwiGLU FFN, MTP heads.

## 2. `Archive_Legacy/Core/OSC` — the energy engine

**What it is.** An operator-split integrator: skew rotation, ICNN-proximal dissipation, norm-ball
projection, plus snapshots and telemetry. It is an *engine*, and the README says so: no tokenizer,
no language model, no training loop.

**What the README itself admits (§11, verbatim):**

1. "No trainer updates `K`; the default is a random low-rank skew matrix."
2. "The ICNN enforces strong convexity (`c_min=0.01`), so a fixed input has one equilibrium. Do not
   describe it as a multi-basin static memory."

**Meaning in plain language.** The energy term cannot be a memory that stores many patterns,
because a strongly convex potential has exactly one minimum. It can only pull every state toward
one learned basin — a stabiliser. That is still useful (it keeps the latent well-conditioned and
bounded), but it is not the mechanism that makes a model remember. An untrained random `K` also
means the conservative part was never actually learned.

**What SigmaStorm does with this.** It keeps `-∇E` as a learned anisotropic contraction
(`E = ½xᵀDx + ½‖Lᵀx‖²`) and stops pretending it is the memory. The memory is the delta rule
(bindings) and the slot memory (long horizon). This is documented in
[`ops/energy.py`](../sigmastorm/ops/energy.py).

## 3. Why the first SigmaStorm run stayed at chance — and what fixed it

Five independent defects. Each was isolated with an ablation; all five are documented in
[`DESIGN.md` §4](../DESIGN.md). In order of impact:

1. **The generic initialiser zeroed the delta-rule decay bias.** α was 0.5 per token instead of
   0.982, so every binding was erased within two tokens. This alone explained most of the failure.
2. **The key was read from the same position as the value.** A binding needs the *previous* token
   as the key; a key blended with its own value never matches on return.
3. **The delta rule read the convolved stream.** The short conv smears neighbouring tokens, so the
   key written at the value position no longer matches the key seen at the query position.
4. **The block replaced the residual stream with the latent.** The token identity needed by the
   recall operator was erased every block, and the projection saturated.
5. **Operator outputs had wildly different scales** and were summed raw, so the largest one
   dominated its gate.

After the fixes, MQAR goes from 6.2% to 97.3% with the full stack in 400 CPU steps.

## 4. The two reference papers

Both PDFs sit in the project root. Reading them against SigmaStorm:

### `2607.24653v2.pdf` — Kimi K3 family

- A 3:1 hybrid of KDA (Kimi Delta Attention, a gated delta-rule variant) and Gated MLA.
- Stable LatentMoE and a 1M-token context claim.

**Take-away for us.** The industry is converging on the delta rule as the linear-time recall
operator; our operator is the same family. What they have that we do not is data and compute — not
a different mechanism. The 1-in-4 window-attention block in SigmaStorm mirrors the hybrid idea
without any full attention.

### `2609.19969v1.pdf` — DeepSeek-V4.1-Flash technical report

- 552B total / 16B active parameters, CSA2 compressed sparse attention, FP4 KV cache, Engram,
  DSpark (multi-token prediction).

**Correction of a previous statement.** I earlier said this report did not exist publicly. That was
wrong: the PDF is a real technical report for DeepSeek-V4.1-Flash, and it is the closest frontier
reference point for long-context sparse attention plus MTP.

**Take-away for us.** Their long-context strategy is still attention with compressed KV and a
token-selection indexer. SigmaStorm replaces that with bounded recurrent state. Ours is cheaper per
token at long context; theirs has an exact retrieval path we approximate. MTP (our `mtp_horizons`)
is the same idea as DSpark: train extra heads on future offsets to densify the gradient signal.

## 5. What this means for the ambition

- **Mechanism:** validated. Recall, chunk invariance, constant state, trainable end to end.
- **Frontier performance:** not reachable on 2 × T4. A 1.5B model needs ~3 GB of parameters and
  12 GB of optimizer state *before* activations; a single 16 GB card cannot hold it, and the
  data budget for frontier-adjacent quality is far beyond this hardware. The honest path is
  small-model validation → 100M → 300M, and measuring whether the recurrence scales. See
  [ROADMAP.md](ROADMAP.md).
- **Differential advantage to test:** at equal parameter count, does a fixed-size state give
  better long-context efficiency than a transformer with a KV cache? That is measurable on your
  hardware and is the experiment worth running first.
