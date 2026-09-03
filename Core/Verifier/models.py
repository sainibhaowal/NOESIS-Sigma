# Verifier/models.py
# NOESIS-S -- Verifier Models (D4)

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["SIM", "WKS"]


class SpanRef(BaseModel):
    source: SourceKind
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    sha256: str | None = None


class Citation(BaseModel):
    span: SpanRef
    note: str | None = None


class Witness(BaseModel):
    trace_id: str
    mode: Literal["fast", "balanced", "strict"]
    snapshot_id: str | None = None
    snapshot_root_hash: str | None = None
    statements: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    vars: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())


class VerifyResult(BaseModel):
    verdict: Literal["pass", "fail", "unsure"]
    verifier_score: float = Field(ge=0.0, le=1.0)
    errors: list[dict] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Receipt(BaseModel):
    receipt_id: str
    created_at: datetime
    policy: Literal["fast", "balanced", "strict"]
    snapshot_id: str | None
    answer_sha256: str
    witness_sha256: str
    verifier_score: float
    public_key_id: str
    signature_b64: str
