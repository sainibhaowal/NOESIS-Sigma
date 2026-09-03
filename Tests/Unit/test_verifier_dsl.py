from __future__ import annotations

from Core.Verifier.models import Witness
from Core.Verifier.service import verify
from Core.Verifier.span_store import SpanStore, SpanText


class _StubStore:
    def fetch(self, *, source: str, span_id: str, start: int, end: int, snapshot_id: str | None) -> SpanText:
        if span_id != "s1":
            raise KeyError("missing")
        text = "abcde"
        return SpanText(text=text[start:end], sha256=None)

    def get_snapshot_root_hash(self, *, snapshot_id: str) -> str | None:
        return None


def test_verifier_dsl_passes_basic() -> None:
    w = Witness(
        trace_id="t1",
        mode="strict",
        snapshot_id=None,
        statements=["SPAN('s1',0,3) == 'abc'"],
    )
    res = verify(w, _StubStore())
    assert res.verdict == "pass"
    assert res.verifier_score == 1.0


def test_verifier_dsl_rejects_bad_ast() -> None:
    w = Witness(
        trace_id="t2",
        mode="strict",
        snapshot_id=None,
        statements=["__import__('os').system('echo hi')"],
    )
    res = verify(w, _StubStore())
    assert res.verdict in ("fail", "unsure")
    assert any(e.get("code") == "E-DSL-001" for e in res.errors)
