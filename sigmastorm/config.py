"""Validated configuration for SigmaStorm."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SigmaConfig:
    vocab_size: int = 32000
    d_model: int = 640
    n_blocks: int = 10
    n_heads: int = 8

    chunk_size: int = 64
    conv_kernel: int = 4
    slot_count: int = 32
    slot_temperature: float = 0.5
    # Slot writes are consolidated in blocks of ``slot_block`` tokens. Small blocks keep the
    # read/write lag short; the accumulators are carried across calls, so the result is exactly
    # chunk-invariant for any boundary that does not cut inside a consolidation block.
    slot_block: int = 64
    # Gap D addressing metric. ``cosine`` is the previous behaviour: cosine similarity between the
    # query and the slot keys. ``whitened`` removes the shared residual-stream direction and
    # equalises the variance of every address direction (ZCA/Mahalanobis whitening with shrinkage),
    # which is what makes the read content-addressed when the slot keys share a dominant component.
    memory_metric: str = "cosine"
    # Shrinkage added to the address covariance: C <- (1-g) C + g (tr C / slots) I. Scale-free and
    # keeps the inverse square root well conditioned when the keys are nearly collinear.
    memory_shrinkage: float = 0.05
    # Write dead-zone. A token whose novelty is below this floor contributes exactly zero to the
    # accumulators. Without it a repeated token with novelty 0.02 still writes 0.02 * evidence into
    # every routed slot; over a few hundred repetitions that sums to more than one pseudo-observation
    # and eta = D/(D+1) overwrites the slot with the filler mean (measured: block-1 slot keys
    # collapse to cosine 0.94 and all slots fill, so an absent query always retrieves something).
    memory_novelty_floor: float = 0.0
    # Read softmax temperature. ``None`` keeps ``slot_temperature``. The write path already routes at
    # ``0.2 * slot_temperature``; the read needs its own value because a temperature sharp enough for
    # routing makes an untrained read pick an arbitrary slot.
    memory_read_temperature: float | None = None
    # A null (background) slot in the read mixture: one extra logit whose value is the zero vector.
    # Probability mass routed to it removes the read, so a query that matches no slot is answered
    # from the residual stream alone. This is the abstention path the presence task needs.
    memory_abstain: bool = False
    window: int = 256
    local_attn_every: int = 4

    # Gap A: one decay rate per head is a single timescale for the whole model. The spectrum below
    # gives each head its own log-spaced initial rate (alpha = exp(-rate)), so the delta rule can
    # hold a binding for ~10 tokens in one head and ~10K in another. The per-token projection is
    # zero-initialised, so these are exactly the initial rates and stay learnable.
    decay_min_rate: float = 1e-4
    decay_max_rate: float = 3e-2
    # When False the per-head decay stays exactly at its initialised spectrum. Used by the ladder to
    # measure the prior itself; a learnable bias can otherwise recover a spectrum from a single
    # initial rate, which hides whether the initialisation helped.
    decay_learnable: bool = True
    # Gap B: independent associative matrices per head, i.e. head count multiplied by this factor.
    # Capacity grows linearly; cost grows linearly; state per head is unchanged.
    state_expand: int = 1

    # Gap C: chunk summaries — a fixed ring of pooled chunks that every token can read globally.
    # Cost is O(T * summary_slots), never O(T^2).
    use_summary: bool = True
    summary_every: int = 2
    summary_chunk: int = 256
    summary_slots: int = 64
    summary_age_decay: float = 1e-2

    ffn_mult: float = 2.67
    n_experts: int = 0
    top_k: int = 1
    n_shared_experts: int = 1

    max_norm: float = 16.0
    dt: float = 0.5
    energy_rank: int = 32
    energy_floor: float = 1e-2

    mtp_horizons: tuple[int, ...] = (2, 4)
    loss_token_chunk: int = 256
    tie_embeddings: bool = True
    dropout: float = 0.0

    use_conv: bool = True
    use_delta: bool = True
    use_memory: bool = True
    use_window: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        for name in (
            "vocab_size",
            "d_model",
            "n_blocks",
            "n_heads",
            "chunk_size",
            "conv_kernel",
            "slot_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.conv_kernel < 2:
            raise ValueError("conv_kernel must be at least 2")
        if self.slot_block < 1:
            raise ValueError("slot_block must be positive")
        if self.local_attn_every and self.window < 2:
            raise ValueError("window must be at least 2")
        if not 0.0 < self.decay_min_rate <= self.decay_max_rate < 8.0:
            raise ValueError("decay rates must satisfy 0 < min_rate <= max_rate < 8")
        if self.state_expand < 1:
            raise ValueError("state_expand must be at least 1")
        if self.summary_every < 0:
            raise ValueError("summary_every must be non-negative")
        if self.summary_chunk < 1 or self.summary_slots < 1:
            raise ValueError("summary_chunk and summary_slots must be positive")
        if self.summary_age_decay < 0.0:
            raise ValueError("summary_age_decay must be non-negative")
        if self.memory_metric not in ("cosine", "whitened"):
            raise ValueError("memory_metric must be 'cosine' or 'whitened'")
        if not 0.0 <= self.memory_shrinkage < 1.0:
            raise ValueError("memory_shrinkage must be in [0, 1)")
        if not 0.0 <= self.memory_novelty_floor < 1.0:
            raise ValueError("memory_novelty_floor must be in [0, 1)")
        if self.memory_read_temperature is not None and self.memory_read_temperature <= 0.0:
            raise ValueError("memory_read_temperature must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.n_experts and not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if any(h < 1 for h in self.mtp_horizons):
            raise ValueError("mtp_horizons must be positive offsets")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def heads_eff(self) -> int:
        return self.n_heads * self.state_expand

    @property
    def inner_dim(self) -> int:
        return self.d_model * self.state_expand

    def uses_window(self, block_index: int) -> bool:
        return (
            self.use_window and self.local_attn_every > 0 and (block_index + 1) % self.local_attn_every == 0
        )

    def uses_summary(self, block_index: int) -> bool:
        return self.use_summary and self.summary_every > 0 and block_index % self.summary_every == 0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["mtp_horizons"] = list(self.mtp_horizons)
        return values

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SigmaConfig:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown SigmaConfig fields: {unknown}")
        clean = dict(values)
        if "mtp_horizons" in clean:
            clean["mtp_horizons"] = tuple(clean["mtp_horizons"])
        return cls(**clean)

    @classmethod
    def load(cls, path: str | Path) -> SigmaConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_size(cls, name: str) -> SigmaConfig:
        presets: dict[str, dict[str, Any]] = {
            "tiny": dict(
                vocab_size=512,
                d_model=128,
                n_blocks=4,
                n_heads=4,
                slot_count=16,
                window=64,
                state_expand=2,
                summary_chunk=32,
                summary_slots=16,
            ),
            "20m": dict(
                vocab_size=8192,
                d_model=320,
                n_blocks=6,
                n_heads=8,
                slot_count=24,
                state_expand=2,
                summary_chunk=128,
                summary_slots=32,
                window=128,
            ),
            "100m": dict(
                vocab_size=32000,
                d_model=640,
                n_blocks=10,
                n_heads=8,
                slot_count=32,
                state_expand=2,
                summary_chunk=256,
                summary_slots=64,
            ),
        }
        key = name.lower()
        if key not in presets:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        return cls.from_dict(presets[key])
