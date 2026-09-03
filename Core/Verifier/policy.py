# Verifier/policy.py
# NOESIS-S -- Verifier Policy (D4)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["fast", "balanced", "strict"]


@dataclass(frozen=True)
class VerifierPolicy:
    mode: Mode
    min_score: float
    require_spans: bool
    max_statements: int


def policy_for(mode: Mode) -> VerifierPolicy:
    if mode == "fast":
        return VerifierPolicy(
            mode=mode, min_score=0.0, require_spans=False, max_statements=16
        )
    if mode == "balanced":
        return VerifierPolicy(
            mode=mode, min_score=0.6, require_spans=True, max_statements=64
        )
    return VerifierPolicy(
        mode="strict", min_score=0.95, require_spans=True, max_statements=128
    )
