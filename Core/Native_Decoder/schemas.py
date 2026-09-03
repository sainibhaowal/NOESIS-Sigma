from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class DecodeRequest:
    graph_dict: dict
    mode: str
    trace_id: str
    max_tokens: int = 512
    temperature: float = 0.3
    # Road B: optional converged OSC state. When present, the emitter reads the
    # brain's actual final_state directly instead of re-encoding graph text.
    # Kept optional so legacy callers (graph-only) still work unchanged.
    final_state: Optional[Sequence[float]] = None


@dataclass(frozen=True)
class DecodeResponse:
    text: str
    mode: str
    trace_id: str
    grounded: bool
    elapsed_ms: float
