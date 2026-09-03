from __future__ import annotations

import json
from pathlib import Path

from Core.Verifier.checker import CheckLimits, verify_witness
from Core.Verifier.dsl_ast import compile_expr
from Core.Verifier.models import Witness
from Core.Verifier.rules_engine import _load_rules_cached
from Core.Verifier.service import verify
from Core.Verifier.span_store import SpanLimits, SpanText


class _DummyStore:
    def fetch(self, *, source, span_id, start, end, snapshot_id):
        return SpanText(text="abcdef", sha256=None)


def test_dsl_accepts_arithmetic_binops() -> None:
    compiled, err = compile_expr("1+2>2")
    assert compiled is not None
    assert err is None


def test_max_eval_ops_is_enforced() -> None:
    w = Witness(
        trace_id="t1",
        mode="strict",
        statements=["1+2>2", "3+4>1"],
        citations=[],
        vars={},
    )
    res = verify_witness(w, _DummyStore(), CheckLimits(max_eval_ops=5))
    assert any(e.get("code") == "E-LIM-413" for e in res.errors)


def test_max_spans_is_enforced() -> None:
    w = Witness(
        trace_id="t2",
        mode="balanced",
        statements=["SPAN('a',0,1,'SIM') != '' and SPAN('b',0,1,'SIM') != ''"],
        citations=[],
        vars={},
    )
    limits = CheckLimits(span_limits=SpanLimits(max_bytes=64000, max_spans=1))
    res = verify_witness(w, _DummyStore(), limits)
    assert any(e.get("code") == "E-LIM-413" for e in res.errors)


def test_rule_pack_is_consumed(tmp_path: Path, monkeypatch) -> None:
    rules = {
        "version": 2,
        "notes": "test pack",
        "rules": [
            {
                "id": "need_two_citations",
                "kind": "min_citations",
                "enabled": True,
                "modes": ["fast"],
                "min": 2,
            }
        ],
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    monkeypatch.setenv("NOESIS_VERIFIER_RULES_PATH", str(path))
    _load_rules_cached.cache_clear()

    w = Witness(trace_id="t3", mode="fast", statements=[], citations=[], vars={})
    res = verify(w, _DummyStore())
    assert res.details.get("rule_pack", {}).get("version") == 2
    assert any(e.get("code") == "E-RULE-001" for e in res.errors)

