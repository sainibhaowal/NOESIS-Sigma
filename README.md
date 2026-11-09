<div align="center">
  <img src="assets/noesis-sigma-banner.svg" alt="NOESIS-Σ bounded-state sequence modeling banner" width="100%">
</div>

> **Naming:** NOESIS-Σ is the project. **SigmaStorm** is the current executable model and package
> implementation.

## At a glance

| Property | Design |
| :--- | :--- |
| Model family | Non-transformer recurrent sequence model |
| Core state | Fixed-size latent, associative matrices, slot memory, and summary rings |
| Global path | Chunk summaries with `O(T · S)` work, never a dense `T × T` attention matrix |
| Context scaling | State memory is independent of sequence length; state is carried between chunks |
| Main recall path | Gated delta rule with a learned multi-timescale decay spectrum |
| Current evidence | Chunk invariance, constant state size, MQAR recall, and controlled ablations |

## What it is

SigmaStorm advances a token-local latent with a bounded, energy-contracted transition. Each block
combines operators that solve different sequence-modeling problems, then adds the result back to the
residual stream:

~~~text
u        = RMSNorm(x)
proposal = normalized_gated_mix(operators(u)) - ∇E(u) + u
v        = Project‖v‖≤R(u + dt(u) · proposal)
x        = x + out_proj(v)
~~~

There is no global `[T, T]` attention. A 256K-token sequence can be processed in chunks while
carrying a bounded state between calls.

## Architecture

| Operator | Role | Work | Carried state |
| :--- | :--- | :--- | :--- |
| Short convolution | Local order and fast proposal | `O(T · k)` | `k − 1` recent tokens |
| Gated delta rule | Associative key → value recall | `O(T)` | `H · k` matrices of size `d_head × d_head` |
| Slot memory | Long-horizon prototypes with gated writes | `O(T · slots)` | Slots, usage, and write accumulators |
| Chunk summaries | Global exact-distance path without `T²` | `O(T · S)` | Ring of `S` summaries |
| Window attention | Optional precise local copying | `O(T · W)` | Ring buffer of `W` keys and values |
| Energy potential | Learned anisotropic contraction | `O(T · d)` | None |
| Bounded projection | Enforces `‖v‖ ≤ R` | `O(T · d)` | None |
| Gated FFN / MoE | Per-token nonlinear compute | `O(T · d · ff)` | None |

### The four architectural upgrades

- **Multi-timescale decay.** Delta-rule heads initialize on a log-spaced spectrum
  (`α ≈ 0.9999 … 0.970`) and remain token-dependent and learned.
- **State expansion.** `state_expand` adds independent associative matrices without shrinking each
  head, so capacity and cost scale linearly.
- **Chunk summaries.** A fixed summary ring supplies a learned global path at `O(T · S)` cost.
- **Per-token slot writes.** Novelty, evidence, causal in-block deduplication, load-aware routing,
  and Bayesian consolidation prevent repeated filler tokens from flooding memory.

The controller uses independent sigmoid gates, per-operator RMSNorm, a normalized convex mixture,
and an additive residual. These details are important: the operators have different natural scales,
and replacing the residual stream with the bounded latent destroys the recall signal.

## Verification and evidence

The repository contains falsification gates and controlled experiment records:

| Check | Recorded result |
| :--- | :--- |
| Chunked delta rule vs. sequential reference | Maximum output/state error below `4.2e−7` in the reference run |
| Full forward vs. carried-state chunks | Maximum hidden-state error around `1.4e−6` in the reference run |
| State size vs. context length | Constant at `T = 512` and `T = 4096` |
| MQAR associative recall | `0.974` accuracy vs. `0.062` chance after 400 steps |
| TinyStories 20M run | Held-out PPL `3.498` at step `10,604` in the recorded experiment |

The MQAR, chunking, and constant-state checks establish that the mechanism is trainable and
chunk-invariant. They do not establish parity with frontier transformers.

Reference records:

- [20M TinyStories run](results/20m_tinystories.md)
- [Gap D memory proof](results/gapd_proof.md)
- [Controlled experiment ladder](results/ladder.md)
- [Design notes and mathematical details](DESIGN.md)
- [Roadmap and scale plan](docs/ROADMAP.md)
- [Technical report](docs/paper/README.md)

## Quickstart

Create an environment and install the dependencies:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Run the CPU gates and tests:

~~~bash
python scripts/gate.py
python -m pytest tests -q
~~~

Train a 20M model on uint32 token shards:

~~~bash
python scripts/train.py \
  --config configs/sigma_20m.json \
  --data_dir /path/to/shards \
  --seq_len 8192 \
  --micro_chunk 2048 \
  --bptt_chunks 4 \
  --batch_size 2 \
  --steps 5000 \
  --nproc 2
~~~

Resume from `runs/sigma/latest.pt` automatically. Add `--fresh` to ignore the checkpoint.

Evaluate a checkpoint:

~~~bash
python scripts/eval.py \
  --config configs/sigma_20m.json \
  --checkpoint runs/sigma/latest.pt \
  --data_dir /path/to/shards \
  --seq_len 8192
~~~

## Build a corpus

The corpus builder trains an 8K byte-level BPE tokenizer and writes EOS-separated uint32 shards:

~~~bash
python scripts/build_corpus.py \
  --source tinystories \
  --out_dir data/tinystories \
  --train_tokens 300_000_000 \
  --val_tokens 5_000_000
~~~

Shard metadata has this shape:

~~~json
{
  "dtype": "uint32",
  "eos_token_id": 2,
  "shards": [{"path": "shard-00000.bin", "tokens": 50000000}]
}
~~~

For Kaggle workflows, see [docs/KAGGLE.md](docs/KAGGLE.md) and the helpers in `kaggle/`.

## Repository layout

~~~text
sigmastorm/         model, transition law, controller, state, and operators
training/           sharded data, MQAR tasks, trainer, and evaluation
scripts/            train, eval, gate, corpus, ladder, and Kaggle utilities
configs/            tiny, MQAR, 20M, and 100M presets
tests/              equivalence, state, boundedness, gradient, and recall tests
results/            recorded gates, ablations, and training runs
docs/               roadmap, Kaggle instructions, findings, and technical report
assets/logo.png     source project mark
assets/noesis-sigma-banner.svg  rounded SVG banner used at the top of this README
~~~

## Three details that make recall learn

These are documented in [DESIGN.md](DESIGN.md):

1. The delta rule uses the **previous token as the key** and the current token as the value.
2. It reads the **clean stream**, and its decay bias is initialized to preserve long-lived state.
3. The transition is an **additive residual** with a normalized convex mixture and per-operator
   RMSNorm.

## What this is not

- Not a transformer with layers removed: recurrence, energy contraction, and bounded projection are
  the architecture.
- Not a claim of frontier quality: the available hardware and data budget are research-scale.
- Not a claim that the energy term is a multi-basin memory: strong convexity makes it a stabilizer;
  the delta rule and slot memory hold information.
- Not a claim that the trained 20M slot memory is already a reliable content-addressed retriever;
  the demonstrated associative recall is primarily the MQAR delta-rule path.

## License and citation

See the repository metadata and technical report for the project’s current publication and licensing
information.
