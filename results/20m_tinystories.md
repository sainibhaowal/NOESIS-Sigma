# 20M TinyStories run — 2 × T4, two sessions

Model: `configs/sigma_20m.json` (20,965,370 parameters in the current configuration; all operators on: conv, decay-spectrum delta
rule with `state_expand=2`, slot memory, window attention, chunk summaries, energy contraction,
MTP horizons 2/4).

Data: TinyStories, 8K byte-level BPE, 300M tokens in six uint32 shards, EOS between documents.
Training: DDP over 2 × T4 (15.36 GB, fp16), batch 16/rank, `seq_len=1024`, `micro_chunk=256`,
`bptt_chunks=4`. Each Kaggle session is capped at 2.6 h; the second session resumed from `latest.pt`.

| Session | Steps | Held-out loss | PPL | MQAR | Peak GPU | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (2026-11-05) | 1 – 5,442 | 1.3225 | 3.753 | 0.028 | 13.49 GB | ~18 k tok/s |
| 2 (2026-11-06) | 5,443 – 10,604 | **1.2523** | **3.498** | 0.008 | 13.49 GB | ~18 k tok/s |

Held-out evaluation is 20 batches of 4 × 1024 tokens from the held-out split (81,840 supervised
tokens), on the checkpoint's own GPU. Total training tokens: ~348M (10604 steps × 16 × 1024 × 2).
MQAR is far out of distribution for a TinyStories language model; it is reported for completeness,
not as a recall claim (the MQAR gate on a model trained for it is 0.974).

## Samples (checkpoint at step 10,604, T = 0.7, top-k 40)

```
Once upon a time, there was a little girl named Lily.
She loved to play with her toys and make them into different things. One day, she found a toy that
was different from the others. It was a big, shiny rock that had a pointy edge. Lily didn't know
what a rock was, but she knew it was interesting. She showed it to her mom and asked, "What is
this?" Her mom replied, "That's...

The little dog ran into the garden and saw the flowers. He was so happy and he couldn't wait to show
all his friends. But then, something bad happened. A big dog came running and scared the little dog
away. The little dog was scared and didn't know what to do. He ran away and never came back. ...

Tom was very sad because he lost his toy. He looked everywhere for it, but it was nowhere to be
found. He started to cry because he missed his toy so much. His mother was worried and asked him
what was wrong. Tom told her about his lost toy. His mother comforted him and said they would find
it together. They searched all around the house, but it was nowhere to be found.
```

The recorded samples are grammatical, on-genre, and locally coherent. They are not evidence of
long-horizon recall; see below.

## Slot memory audit (this checkpoint)

Read routing, over 24 slots (`ln 24 = 3.178` nats is the uniform maximum):

| Block | Read entropy | Mean top weight | Cross-slot key cosine |
| :--- | ---: | ---: | ---: |
| 0 | 3.133 | 0.065 | 0.465 |
| 1 | 3.109 | 0.073 | 0.525 |
| 2 | 3.160 | 0.070 | 0.810 |
| 3 | 3.159 | 0.055 | 0.715 |
| 4 | 3.048 | 0.131 | 0.716 |
| 5 | 3.095 | 0.078 | 0.483 |

Zeroing every slot-memory read costs **+0.035 nats** on held-out data: the read is close to an
average of all slot values, and the memory acts as a slow global summary rather than a
content-addressed store. Sharpening the read temperature (0.5 → 0.05), restricting to top-1/2 slots,
or centering the read addresses at inference all *increase* loss on this checkpoint.

Token embeddings are healthy (`centered` pairwise cosine 0.109 mean-absolute, mean 0.007): the
common-direction component that inflates the raw cosine (0.505) carries no token identity.

## Microbenchmark: presence recall (Gap D)

Task: 16 symbols scattered over the first quarter of a 192-token stream, repeated filler for the
rest, then `[Q, symbol]` and a single supervised yes/no answer. Conv + slot memory only.
Chance 0.5.

| Variant | Accuracy |
| :--- | ---: |
| trainable embeddings, `slot_block` 16 / 32 / 64 (1200 steps) | 0.504 |
| frozen embeddings, `slot_block` 64 | 0.718 |
| frozen embeddings, read τ = 0.1 | 0.681 |
| frozen embeddings, read τ = 0.02 | 0.603 |

With one supervised token per 192, next-token supervision never asks the output head to separate the
symbols, and the token embeddings collapse (centered cosine 0.285) — the memory never receives a
usable address. Freezing the embeddings restores most of the accuracy, which locates the failure in
the representation, not in the write gate. A sharper read does not fix it. The real language-modeling
run has dense supervision and does **not** collapse, but its read routing still stays near uniform,
so the remaining Gap D work is a retrieval mechanism (hard/straight-through addressing, product keys,
or a codebook objective), not a temperature change.

## Gap C aggregate path, current code

Count task, 300 steps, fresh seeds: `full` 0.734 vs `no_summary` 0.672 (a second pair of runs gave
0.703 vs 0.703). The previously recorded 0.797 vs 0.891 did not reproduce, so the summary operator
shows no reliable regression; the metric is noise-limited at this budget. No summary change was
adopted.

## Gap D resolution (2026-11-06)

The audit above is the *before* picture. The diffuse read was fixed in the operator, not in this
checkpoint: a novelty dead-zone on writes, a separate read temperature (τ_r = 0.1), and a learned
match-confidence gate that lets the read abstain. On the frozen-embedding presence task the read goes
from chance (0.562 / 0.539 / 0.555) to **0.977 / 1.000 / 0.953** across three seeds; read entropy
falls to 0.41 nats (uniform 2.773), the memory ablation delta rises from +0.15 to +2.26 nats, and
confidence separates present (0.56) from absent (0.03) queries. MQAR is unchanged (0.990 / 0.987 vs
0.988 / 0.979) and a 600-step real-data LM check on the tiny preset moves val CE from 3.0760 to
3.0826 nats (+0.2%, inside seed spread).

The full measurements, diagnostics and reproduction command are in
[gapd_proof.md](gapd_proof.md) and [gapd_proof.json](gapd_proof.json); the mechanism is described in
§3.3 of [DESIGN.md](../DESIGN.md). This 20M checkpoint predates the fix, so its samples and slot
audit above are unchanged. No gated 20M language-model checkpoint is recorded here; the run is
staged in `configs/sigma_20m_gapd.json` and `kaggle/gapd_train/` for a future controlled rerun.
