# Findings from the earlier prototype

This document records conclusions from an earlier local audit of the Sigma prototype and its energy
engine. The `Archive_Legacy/` snapshots referenced by the original audit are not included in this
checkout, so the historical findings below are context rather than claims that can be reproduced from
the current tree. Current measurements are recorded in `results/` and in the technical report.

## 1. Earlier Sigma prototype — v1 of the law

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
| single-position recurrence, no chunk form | sequential state update | no tested chunkwise long-context contract |

SigmaStorm keeps the vocabulary and rebuilds the execution: chunkwise delta rule with a per-head
decay spectrum and expanded state, per-token gated slot writes consolidated every `slot_block`
tokens, a chunk-summary global path, convex gated mixture, real SwiGLU FFN, MTP heads.

## 2. Earlier OSC energy engine

**What it is.** An operator-split integrator: skew rotation, ICNN-proximal dissipation, norm-ball
projection, plus snapshots and telemetry. It is an *engine*, and the README says so: no tokenizer,
no language model, no training loop.

**The earlier audit recorded these README statements:**

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

The current mechanism gate records MQAR accuracy `0.974` versus `0.062` chance after 400 steps.
The separate ladder record reports `0.977` on its 26-binding task; these are different protocols and
should not be combined into one number.

## 4. The two reference papers

The two frontier reports mentioned by the earlier audit are not bundled in this repository. Their
bibliographic records are in [`docs/paper/references.bib`](paper/references.bib). The comparison below
is related-work context, not a measurement of SigmaStorm.

### `2607.24653v2.pdf` — Kimi K3 family

- A 3:1 hybrid of KDA (Kimi Delta Attention, a gated delta-rule variant) and Gated MLA.
- Stable LatentMoE and a 1M-token context claim.

**Take-away for us.** Kimi K3 is related work using a gated delta-rule family alongside another
sequence-mixing path. SigmaStorm uses its own chunkwise delta-rule implementation and an optional
local window; this is a family-level comparison, not an equivalence or performance claim.

### `2609.19969v1.pdf` — DeepSeek-V4.1-Flash technical report

- 552B total / 16B active parameters, CSA2 compressed sparse attention, FP4 KV cache, Engram,
  DSpark (multi-token prediction).

**Correction of a previous statement.** I earlier said this report did not exist publicly. That was
wrong: the PDF is a real technical report for DeepSeek-V4.1-Flash, and it is the closest frontier
reference point for long-context sparse attention plus MTP.

**Take-away for us.** The report describes compressed/sparse attention and multi-token prediction as
related design points. SigmaStorm instead uses bounded recurrent state with optional local window
attention. No matched cost, quality, or retrieval comparison with that system has been measured here.
The `mtp_horizons` heads are a related multi-token-prediction mechanism, not a claim of identical
implementation.

## 5. What this means for the ambition

- **Mechanism:** mechanism-level gates establish recall on the synthetic tasks, chunk invariance for
  the tested split, constant state-tensor size at the tested lengths, and end-to-end trainability.
- **Frontier performance:** not established by this repository. A 1.5B model needs ~3 GB of parameters and
  12 GB of optimizer state *before* activations; a single 16 GB card cannot hold it, and the
  data budget for frontier-adjacent quality is far beyond this hardware. The honest path is
  small-model validation → 100M → 300M, and measuring whether the recurrence scales. See
  [ROADMAP.md](ROADMAP.md).
- **Differential advantage to test:** at equal parameter count, does a fixed-size state give
  better long-context efficiency than a transformer with a KV cache? That is measurable on your
  hardware and is the experiment worth running first.
