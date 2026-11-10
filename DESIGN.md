# SigmaStorm design

## 1. The law

One block advances a token-local latent once:

```
u  = RMSNorm(x)
d_i = operator_i(u)                       for i in enabled operators
v  = Project_{‖·‖≤R}( u + dt(u) · [ Σ_i α_i(u) d_i − ∇E(u) + u ] )
x  = x + out_proj(v)
```

- `α_i(u)` are **independent sigmoid gates** (not a softmax). A softmax would force heterogeneous
  operators to compete for a fixed total and could suppress the associative-memory path. The
  independent gates let several operators contribute simultaneously; the repository does not claim
  that one universal routing distribution applies to every checkpoint or input.
- The mixture is **normalised by the gate sum**, so it is a convex combination: any subset of gates
  can be open at full strength, but the proposal cannot grow with the number of active operators.
- Each operator output passes a **per-operator RMSNorm** before gating. This keeps the mixture on a
  comparable scale; the repository does not treat a single output-scale ratio as a universal constant.
- `dt(u) = dt · (0.5 + σ(·))`, `dt = 0.5` by default.
- `Project` enforces the bounded-state contract on the latent `v` only, never on the residual stream.

Depth plays the role of transition time: `L` blocks = `L` steps of the law on the token-local latent,
while `(h_t, M_t, window)` carry information across tokens.

## 2. Why this is not a transformer

- The window operator is a sliding window on 1 block in 4, with a ring-buffer state of `W` keys and
  values. Its retained history is bounded by `W`; however, the current implementation materialises
  an attention mask over the current chunk, so its per-call work can include a chunk-length term.
  Therefore `O(T·W)` is the intended bounded-window architecture, not a guaranteed runtime bound for
  arbitrarily large unchunked calls. Remove the operator with `local_attn_every = 0`.
- The global path is a **ring of chunk summaries** with fixed configured capacity, not pairwise
  attention. For fixed capacity and width its intended cost is `O(T · summary_slots · d_model)`.
- The recurrence is the delta rule with `heads × state_expand` independent `d_head × d_head` states;
  its sequence-length work is linear in `T` for fixed model dimensions.
- State tensor capacity is independent of sequence length: the current tiny-preset gate reports
  `200,704` bytes per sample at both T=512 and T=4096 under its 2-byte fp16/bf16-equivalent
  accounting assumption. This is state-tensor storage, not total process memory.

## 3. The delta rule (the recall operator)

Per head, with `a_t = exp(g_t)` and `b_t = σ(beta_t)`:

```
S_t = a_t · S_{t-1} + b_t · k_t · (v_t − a_t · k_tᵀ S_{t-1})ᵀ
o_t = q_tᵀ S_t
```

The `(v − kᵀS)` term is error correction: the state is rewritten when a key is seen with a value it
does not already predict. Chunkwise form (`chunk_size = 64`):

```
A[i,j] = exp(gc_i − gc_j) · b_j · (k_i · k_j)        for j < i
r      = (I + A)⁻¹ (V − exp(gc) ⊙ K S0)             unit-lower-triangular solve
O      = exp(gc) ⊙ Q S0 + tril(exp(gc_i − gc_j) · b_j · (q_i · k_j)) @ r
S_C    = exp(gc_last) · S0 + (exp(gc_last − gc_j) · b_j · K)ᵀ @ r
```

Every ratio used is ≤ 1 on the causal region, so the causal decay ratios do not overflow. The current
gate compares the chunkwise path to a token-by-token reference with maximum absolute error
`4.17e-7` for the tested configuration.

### 3.1 Decay spectrum (Gap A)

`rate = softplus(Linear(x))`, `α = exp(−rate)`. A single rate for every head is one timescale for the
whole model: it must either forget local context quickly or forget long-range bindings slowly, and it
cannot do both. One rate shared across heads also makes all retrieval directions decay together.

Instead each head starts at its own log-spaced rate between `decay_min_rate` and `decay_max_rate`
(default `1e-4 … 3e-2`, i.e. `α ≈ 0.9999 … 0.970`):

```
rate_h = r_min (r_max / r_min)^(h / (H-1))
bias_h = log(expm1(rate_h))
```

With `decay_learnable=True` (the default), the projection weight is zero at initialisation, so these
are the initial rates and later rates remain token-dependent. `_custom_init` protects the bias from
the generic initialiser (defect 4.2). The result is a multiscale memory at step zero: some heads hold
a binding longer than others. The effect is measured in `results/ladder.md` (`delta_spectrum` vs
`delta_single_rate`).

### 3.2 State expansion (Gap B)

`state_expand = k` multiplies the number of independent state matrices by `k` while leaving each
head dimension at `d_head = d_model / n_heads`:

```
q, k, v : d_model -> k · d_model        S : [H·k, d_head, d_head]
output  : k · d_model -> d_model
```

Capacity is linear in `k` and so is cost; unlike making `d_head` wider, this does not reduce the
number of retrieval directions per token. The decay spectrum is laid out over `H·k` heads, so the
extra capacity also adds timescales.

### 3.3 Slot memory: per-token gated writes (Gap D)

A fixed ring of prototypes. Per token:

```
a_tj  = softmax_j( (k_t · K_j − λ · log(1 + W_j)) / τ_w )
gate_t = [(novelty_t − δ) / (1 − δ)]_+ · evidence_t        δ = novelty floor
novelty_t = 1 − max_j cos(k_t, K_j)          # first pass: consolidated prototypes
           1 − max_j cos(k_t, P_j(t))        # second pass: P_j(t) = writes to slot j before t
```

Before the floor was enabled, a repeated token with novelty 0.02 still contributed approximately
`0.02 · evidence` to routed slots; repeated filler could therefore accumulate into a pseudo-
observation. In the pre-floor diagnostic, block-1 slot keys reached pairwise cosine 0.94 and every
slot acquired usage, so an absent query always retrieved something. With `δ > 0`, the redundancy
filter is exactly zero below `δ`; above the floor, novelty is linearly re-scaled by `1 − δ` as shown
in the formula.

The second pass is causal: the first pass's gated writes are folded into a running per-slot
prototype `P_j(t)` under `torch.no_grad()`, so a token inside the block is compared with the
earlier tokens *of the same block*. A repeated filler is therefore suppressed on its second
occurrence instead of flooding the accumulator; the carried `causal_key_num` / `causal_den`
fields keep this exact across call boundaries. `W_j` is the causal write mass in the block and
`λ = 1`; it penalises routing into an already-loaded slot (without it, sharp routing sends every
similar token to the same argmax slot and the ring collapses to one prototype).

Writes accumulate for `slot_block` tokens and are consolidated with a Bayesian weight `η`:

```
D_j   = Σ_t gate_t a_tj               η_j = D_j / (D_j + 1)
V_j  <- (1−η_j) V_j + η_j · (Σ_t gate_t a_tj v_t) / D_j
K_j  <- normalize((1−η_j) K_j + η_j · normalize(Σ_t gate_t a_tj k_t / D_j))
use_j <- (1−η_j) use_j + η_j
```

One confident token moves a slot slightly; a block of agreeing tokens replaces it. Reads happen
against the slots as of the current consolidation block. The accumulator, token counter, and ring
state are carried in `SlotMemoryState`, removing the previous dependency on writing once per call on
the chunk mean. The current equivalence gate splits inside a slot block and confirms the carried-state
path to its reported tolerance; this is evidence for the tested configuration, not a blanket claim
for every backend or configuration.

**The read is a gated address, not an average (Gap D).** The first version of this operator had a
diffuse read: in the trained 20M checkpoint (step 10,604) per-block read entropy was 3.03–3.16 nats
against ln 24 = 3.178, and zeroing the memory output cost only +0.035 nats. A softmax always hands
out unit mass, so even a query that matches nothing receives a full-scale average of every slot;
there is no way for the model to express "not in memory" and no gradient pressure to sharpen the
selection. Three changes fix this:

```
match_t    = max_{j: use_j > 0} cos(q_t, K_j)            # before the usage bias
confidence = σ(κ · (match_t − θ))                        # learned κ, θ
read_t     = confidence · W_o Σ_j softmax_j(scores / τ_r) V_j
```

- a separate **read temperature** `τ_r` (the routing sharpness `τ_w = 0.2 · τ` is not a read
  temperature: a read that is sharp before the addresses are trained picks an arbitrary slot);
- the **match-confidence gate** above, which multiplies the read by a per-token likelihood ratio.
  It is a differentiable abstention path: unmatched queries contribute zero, so the residual stream
  alone answers them;
- the **novelty floor** `δ` on the write path, which stops the slow leak that fills every slot.

These controls are opt-in (`memory_novelty_floor`, `memory_read_temperature`, `memory_abstain`). The
default configuration preserves the prior behavior, while checkpoint compatibility remains governed
by the saved architecture configuration and must be verified for each checkpoint artifact.

**Evidence (`scripts/gapd_proof.py`, frozen-embedding presence task, 16 slots, 8 of 16 symbols
present, chance 0.50, 800 steps, three seeds).** Previous cosine read: 0.562 / 0.539 / 0.555.
ZCA-whitened metric alone: 0.648 / 0.531 / 0.539 — it sharpens the read (entropy 1.26 → 0.72 nats)
but does not make it useful, because whitening amplifies *every* deviation direction, including the
context noise. Gated read: **0.977 / 1.000 / 0.953**, confidence **0.562 on present** vs
**0.026 on absent** queries, and read entropy 0.413 nats.
MQAR (Gap A path) is unchanged to three decimals; the aggregate
count task (Gap C path, unaffected by this operator) stays within its seed variance. The ZCA option
is kept as a documented alternative for address spaces with a dominant shared direction, but it is
not the mechanism that makes the memory content-addressed.

### 3.4 Chunk summaries: the fixed-ring global path (Gap C)

Every `summary_chunk` tokens one pooled summary is written to a fixed ring of `summary_slots`:

```
p_c  = (1/C) Σ_{t in chunk c} u_t
K_c  = normalize(W_k p_c) / per head        V_c = W_v p_c / per head
score_tc = q_t · K_c / sqrt(d_head) − softplus(age) · (index − pos_c)
o_t  = W_o Σ_c softmax_c(score_tc) V_c
```

Cost is `O(T · summary_slots · d_model)` for fixed configured capacity and width; state is constant,
and the age bias is learned per head, so a retained summary can stay addressable or fade. The ring
retains only `summary_slots` completed summaries, not every past token. Only completed chunks are
written and the partial accumulator is carried in `SummaryState`, so the tested implementation has a
bounded global path without a full `T²` attention matrix. Here `u_t = RMSNorm(x_t)` is the normalized
latent used by the block.

## 4. The five learning facts

The first version of this package sat at MQAR chance (6.2%). The documented implementation lessons
were:

**4.1 Key must be the previous token.** A binding is written when the value arrives (token `t`); the
key is the token before it (`t−1`). Reading q, k, v all from position `t` forces the projections to
separate two blended tokens, and they do not. The key is shifted by one, and the previous chunk's
last input is carried in `BlockState.prev`, so the current gate preserves chunk equivalence for its
tested configuration (maximum hidden-state difference `1.55e-6`).

**4.2 The generic initialiser zeroed the decay bias.** `model.apply(_init_weights)` ran after the
modules set their own initialisation and reset `decay_proj.bias` to 0, making α = 0.5 per token:
bindings vanished in two tokens. `_init_weights` now skips modules marked `_custom_init`.

**4.3 Additive residual, not state replacement.** Replacing `x` with the post-normed latent every
block erased the token identity the associative operator needs. The law now runs on a latent `v` and
the block adds `out_proj(v)` to the residual stream.

**4.4 Mixing scale.** A raw sum of operator outputs can amplify retrieval and saturate the bounded
projection, reducing useful gradient through the clamp. Convex mixture plus per-operator RMSNorm
keeps the proposal scale controlled in the current design; the repository does not claim a universal
MQAR threshold across all configurations.

**4.5 The MQAR task itself had a ceiling.** Keys were sampled with replacement, so a queried key often
had several bindings and the best possible score was ≈ 0.6. Keys are now unique per sequence; the
ceiling is 1.0.

## 5. State and chunking

`SigmaState` holds one `BlockState` per block:

| Field | Shape | Carried |
| :--- | :--- | :--- |
| `conv_tail` | `[B, k−1, d]` | yes |
| `ssm` | `[B, H·k, d_head, d_head]` | yes |
| `memory` | `[B, slots, d]` keys/values + usage and write accumulators | yes |
| `summary` | `[B, H, S, d_head]` ×2 + usage/pos + partial-chunk accumulator | yes |
| `window` | `[B, H, W, d_head]` ×2 | yes |
| `win_len` | scalar | yes |
| `prev` | `[B, d]` | yes |

For a fixed model configuration, none of these state tensor shapes depends on `T`. The slot memory
and summary ring carry their write accumulators across calls. In the current equivalence gate, a
sequence is split at position 50 — inside a 64-token slot block and a 32-token summary block — and
the chunked hidden states match the full forward with maximum difference `1.55e-6`. This is a tested
tolerance for that configuration, not a proof for every backend or chunk boundary.

## 6. Training

- `micro_chunk` tokens per forward call (CLI default 512), state carried inside a sequence. The
  tracked Kaggle training profile uses 256.
- `bptt_chunks × micro_chunk` differentiable horizon (CLI default `4 × 512 = 2048` tokens); the
  state is detached at the window boundary. Forward context is not limited by this; gradients are.
- DDP with `no_sync()` on all but the final backward; fp16 scaler protocol is
  `unscale → clip → step → update`.
- **WSD schedule** (warmup → stable → decay over the last `decay_frac` of the run). Cosine is still
  available, but WSD keeps the learning rate high through the middle of training instead of
  annealing early at the same token budget.
- **MTP warmup**, when configured: the multi-token loss is ramped from 0 over `mtp_warmup_steps`,
  so the auxiliary objective does not compete with the LM loss before the backbone has features.
- **Sequence-length ramp**, when configured: `seq_len_start → seq_len` over `seq_ramp_steps`. Short,
  dense windows first; long-context structure later.
- **TF32** is enabled for CUDA matmuls where supported; speed and quality effects depend on hardware
  and run settings and are not separately benchmarked here. `torch.compile` is optional and currently
  restricted to single-process training.
- Checkpoints are atomic and resumable: model, optimizer, scaler, data cursor, step, CPU RNG state,
  and architecture config (a mismatched config refuses to load rather than silently reshaping). CUDA
  RNG streams are not explicitly serialized by the current checkpoint payload.
- Chunked cross-entropy: logits are sliced over flat tokens, never `[B, T, vocab]`.

## 7. What is deliberately absent

- Multi-basin memory in the energy term. `E(x) = ½xᵀDx + ½‖Lᵀx‖²` is strongly convex by design; it
  is a contraction, not an attractor network.
- A compute-skipping controller. Gates are mixture weights; a zero gate does not skip a kernel in
  dense PyTorch. Token-sparse gather is future work.
- Cross-chunk backpropagation through the whole context. The default trainer truncates gradients at
  the configured BPTT horizon; extending that horizon increases activation memory and may exceed a
  given GPU budget.
