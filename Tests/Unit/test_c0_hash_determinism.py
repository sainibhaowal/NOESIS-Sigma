from __future__ import annotations

from Core.Continuity.cwm import CWMBuilder, CWMConfig
from Core.Continuity.hashing import apply_hashes
from Core.Continuity.models import (
    ContinuityState,
    FactsTable,
    RecentBuffer,
    SummarySpine,
)


def _make_state() -> ContinuityState:
    cwm = CWMBuilder.bootstrap_empty(cfg=CWMConfig(dim=64, dtype="f16"), policy_mode="BALANCED")
    return ContinuityState(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        turn_id=1,
        policy_mode="BALANCED",
        cwm=cwm,
        summary_spine=SummarySpine(identity="id"),
        facts_table=FactsTable(items=[]),
        recent_buffer=RecentBuffer(tokens=[1, 2, 3]),
        updated_at_ms=1234,
        trace_id="trace",
    )


def test_hash_determinism() -> None:
    st = _make_state()
    st1 = apply_hashes(st)
    st2 = apply_hashes(st)
    assert st1.continuity_root_hash == st2.continuity_root_hash
    assert st1.cwm.cwm_root_hash == st2.cwm.cwm_root_hash

