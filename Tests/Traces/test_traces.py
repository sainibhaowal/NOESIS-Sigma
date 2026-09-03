"""
Tests/Traces/test_traces.py

Tests for the Sprint C1 trace collection pipeline.

Tests:
  - Schema: CognitionTrace to_dict / from_row roundtrip
  - TraceWriter: submit is non-blocking, writes to SQLite
  - TraceWriter: flush waits for background write
  - TraceWriter: queue.Full drops silently (no exception)
  - TraceReader: count, get_trace, list_traces
  - TraceReader: export_for_graph_extractor_training quality gate
  - TraceReader: export_for_decoder_training quality gate
  - TraceReader: export_for_next_state_training quality gate
  - Tenant isolation: traces from different tenants are not mixed
  - Duplicate trace_id: INSERT OR REPLACE updates existing trace
  - Quality update: update_quality stores user feedback
  - OscOrchestrator: _init_trace_writer returns a TraceWriter or None (no crash)
"""

from __future__ import annotations

import tempfile
import time
import uuid
from typing import Optional

import pytest

from Runtime.Traces.trace_schema import CognitionTrace
from Runtime.Traces.trace_writer import TraceWriter
from Runtime.Traces.trace_reader import TraceReader


# ------------------------------------------------------------------ helpers

def _make_trace(
    trace_id: Optional[str] = None,
    tenant_id: str = "tenant-a",
    session_id: str = "session-1",
    verifier_score: float = 0.95,
    graph_grounded: bool = True,
    n_steps: int = 30,
    decode_mode: str = "chat",
    response_text: str = "The answer is 42.",
    request_class: str = "general",
) -> CognitionTrace:
    norms = [float(10 - i * 0.3) for i in range(n_steps)]
    return CognitionTrace(
        trace_id=trace_id or str(uuid.uuid4()),
        tenant_id=tenant_id,
        session_id=session_id,
        request_text="What is the answer?",
        n_steps=n_steps,
        trajectory_norms=norms,
        final_state_norm=norms[-1] if norms else 0.0,
        graph_dict={"graph_id": "g1", "nodes": [], "edges": []},
        graph_grounded=graph_grounded,
        node_type_counts={"INTENT": 1, "FACT": 3, "OUTPUT": 1},
        response_text=response_text,
        decode_mode=decode_mode,
        elapsed_ms_total=250.0,
        verifier_result="verified" if verifier_score >= 0.9 else "unverified",
        verifier_score=verifier_score,
        request_class=request_class,
        sim_facts=2,
        wks_facts=1,
        world_model_facts=0,
        skill_plans=0,
    )


# ------------------------------------------------------------------ schema

def test_trace_to_dict_has_required_keys():
    t = _make_trace()
    d = t.to_dict()
    for key in ("trace_id", "tenant_id", "session_id", "request_text",
                "n_steps", "trajectory_norms", "graph_dict",
                "response_text", "verifier_score"):
        assert key in d


def test_trace_from_row_roundtrip():
    import json
    t = _make_trace()
    row = {
        "trace_id":               t.trace_id,
        "tenant_id":              t.tenant_id,
        "session_id":             t.session_id,
        "request_text":           t.request_text,
        "n_steps":                t.n_steps,
        "trajectory_norms_json":  json.dumps(t.trajectory_norms),
        "final_state_norm":       t.final_state_norm,
        "graph_dict_json":        json.dumps(t.graph_dict),
        "graph_grounded":         int(t.graph_grounded),
        "node_type_counts_json":  json.dumps(t.node_type_counts),
        "response_text":          t.response_text,
        "decode_mode":            t.decode_mode,
        "elapsed_ms_total":       t.elapsed_ms_total,
        "verifier_result":        t.verifier_result,
        "verifier_score":         t.verifier_score,
        "request_class":          t.request_class,
        "sim_facts":              t.sim_facts,
        "wks_facts":              t.wks_facts,
        "world_model_facts":      t.world_model_facts,
        "skill_plans":            t.skill_plans,
        "user_corrected":         0,
        "user_rating":            None,
        "quality_score":          None,
        "created_at":             t.created_at,
    }
    t2 = CognitionTrace.from_row(row)
    assert t2.trace_id == t.trace_id
    assert t2.trajectory_norms == t.trajectory_norms
    assert t2.graph_grounded == t.graph_grounded
    assert t2.verifier_score == t.verifier_score


# ------------------------------------------------------------------ writer

def test_writer_submit_is_nonblocking():
    with tempfile.TemporaryDirectory() as tmp:
        w = TraceWriter(db_path=f"{tmp}/traces.db")
        t0 = time.monotonic()
        for _ in range(50):
            w.submit(_make_trace())
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"submit() too slow: {elapsed:.3f}s for 50 traces"


def test_writer_flush_persists_traces():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        traces = [_make_trace() for _ in range(5)]
        for t in traces:
            w.submit(t)
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        assert r.count() == 5


def test_writer_full_queue_does_not_raise():
    with tempfile.TemporaryDirectory() as tmp:
        w = TraceWriter(db_path=f"{tmp}/traces.db")
        # Fill queue beyond capacity — should silently drop, never raise
        for _ in range(3000):
            w.submit(_make_trace())  # no exception expected


def test_writer_duplicate_trace_id_replaces():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        tid = str(uuid.uuid4())
        t1 = _make_trace(trace_id=tid, response_text="first")
        t2 = _make_trace(trace_id=tid, response_text="second")
        w.submit(t1)
        w.submit(t2)
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        assert r.count() == 1
        fetched = r.get_trace(tid)
        assert fetched is not None
        assert fetched.response_text == "second"


# ------------------------------------------------------------------ reader

def test_reader_count_empty_db():
    with tempfile.TemporaryDirectory() as tmp:
        r = TraceReader(db_path=f"{tmp}/traces.db")
        assert r.count() == 0


def test_reader_get_trace_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        r = TraceReader(db_path=f"{tmp}/traces.db")
        assert r.get_trace("nonexistent") is None


def test_reader_list_traces():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        for i in range(4):
            w.submit(_make_trace(verifier_score=0.7 + i * 0.1))
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        all_traces = r.list_traces(min_verifier_score=0.0)
        assert len(all_traces) == 4

        # Use exact threshold values that avoid floating-point rounding (0.7+2*0.1 may be 0.899...)
        high_quality = r.list_traces(min_verifier_score=0.95)
        assert len(high_quality) == 1  # only score=1.0 (0.7+3*0.1)


# ------------------------------------------------------------------ exports

def test_export_graph_extractor_quality_gate():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        w.submit(_make_trace(verifier_score=0.95, graph_grounded=True))   # included
        w.submit(_make_trace(verifier_score=0.70, graph_grounded=True))   # excluded: low score
        w.submit(_make_trace(verifier_score=0.90, graph_grounded=False))  # excluded: not grounded
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        records = r.export_for_graph_extractor_training(min_verifier_score=0.8)
        assert len(records) == 1
        assert "trajectory_norms" in records[0]
        assert "graph_dict" in records[0]
        assert "node_type_counts" in records[0]


def test_export_decoder_quality_gate():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        w.submit(_make_trace(verifier_score=0.95, graph_grounded=True, response_text="answer"))
        w.submit(_make_trace(verifier_score=0.85, graph_grounded=True, response_text="answer"))  # excluded
        w.submit(_make_trace(verifier_score=0.95, graph_grounded=True, response_text=""))  # excluded: empty
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        records = r.export_for_decoder_training(min_verifier_score=0.9)
        assert len(records) == 1
        assert "graph_dict" in records[0]
        assert "response_text" in records[0]
        assert records[0]["response_text"] == "answer"


def test_export_next_state_quality_gate():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        w.submit(_make_trace(verifier_score=0.90, n_steps=20))   # included
        w.submit(_make_trace(verifier_score=0.70, n_steps=20))   # excluded: low score
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        records = r.export_for_next_state_training(min_verifier_score=0.8)
        assert len(records) == 1
        assert "trajectory_norms" in records[0]
        assert "final_state_norm" in records[0]
        assert len(records[0]["trajectory_norms"]) == 20


# ------------------------------------------------------------------ tenant isolation

def test_tenant_isolation_list():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        w.submit(_make_trace(tenant_id="tenant-a"))
        w.submit(_make_trace(tenant_id="tenant-a"))
        w.submit(_make_trace(tenant_id="tenant-b"))
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        assert r.count(tenant_id="tenant-a") == 2
        assert r.count(tenant_id="tenant-b") == 1
        assert r.count() == 3  # total


def test_tenant_isolation_export():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        w.submit(_make_trace(tenant_id="tenant-a", verifier_score=0.95))
        w.submit(_make_trace(tenant_id="tenant-b", verifier_score=0.95))
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        a_records = r.export_for_decoder_training(tenant_id="tenant-a", min_verifier_score=0.9)
        b_records = r.export_for_decoder_training(tenant_id="tenant-b", min_verifier_score=0.9)
        assert len(a_records) == 1
        assert len(b_records) == 1
        # Cross-check: no bleed
        all_a_ids = {rec["trace_id"] for rec in a_records}
        all_b_ids = {rec["trace_id"] for rec in b_records}
        assert all_a_ids.isdisjoint(all_b_ids)


# ------------------------------------------------------------------ quality update

def test_update_quality():
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/traces.db"
        w = TraceWriter(db_path=db)
        tid = str(uuid.uuid4())
        w.submit(_make_trace(trace_id=tid))
        w.flush(timeout=5.0)

        r = TraceReader(db_path=db)
        ok = r.update_quality(tid, user_rating=4, quality_score=0.92, user_corrected=False)
        assert ok is True

        fetched = r.get_trace(tid)
        assert fetched is not None
        assert fetched.user_rating == 4
        assert abs(fetched.quality_score - 0.92) < 1e-6


def test_update_quality_missing_trace():
    with tempfile.TemporaryDirectory() as tmp:
        r = TraceReader(db_path=f"{tmp}/traces.db")
        ok = r.update_quality("nonexistent", user_rating=3)
        assert ok is False


# ------------------------------------------------------------------ orchestrator wiring

def test_orchestrator_trace_writer_init_no_crash():
    from External.Orchestrator.osc_chat import OscOrchestrator
    orch = OscOrchestrator()
    # _trace_writer is either a TraceWriter or None — never raises
    assert hasattr(orch, "_trace_writer")


def test_submit_trace_no_crash_on_bad_input():
    from External.Orchestrator.osc_chat import OscOrchestrator, OscRequest, OscResponse
    orch = OscOrchestrator()
    # Build minimal OscRequest and OscResponse
    req = OscRequest(text="test", tenant_id="t1", user_id="u1")
    resp = OscResponse(
        answer="test",
        compute_profile="BALANCED",
        verify_mode="OFF",
        graph_summary={},
        trajectory_norms=[],
    )
    # Must not raise regardless of trace_writer state
    orch._submit_trace(req, resp, "session-test", 100.0)
