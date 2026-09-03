from __future__ import annotations

from Core.Verifier.models import Citation, SpanRef, Witness
from Core.Verifier.service import verify
from Core.Verifier.span_store import SpanText


class _StubStore:
    def fetch(self, *, source: str, span_id: str, start: int, end: int, snapshot_id: str | None) -> SpanText:
        return SpanText(text="hello", sha256="goodhash")

    def get_snapshot_root_hash(self, *, snapshot_id: str):
        return "root_ok"


def test_hash_tamper_fails() -> None:
    w = Witness(
        trace_id="t1",
        mode="strict",
        snapshot_id=None,
        statements=["1 == 1"],
        citations=[Citation(span=SpanRef(source="SIM", id="r1", start=0, end=5, sha256="badhash"))],
    )
    res = verify(w, _StubStore())
    assert res.verdict == "fail"
    assert any(e.get("code") == "E-HASH-409" for e in res.errors)
