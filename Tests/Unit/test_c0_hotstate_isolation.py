from __future__ import annotations

from Core.Continuity.cwm import CWMBuilder, CWMConfig
from Core.Continuity.models import ContinuityState, FactsTable, SummarySpine
from Core.Continuity.service import HotStateStore


def _mk_state(tenant: str, user: str, session: str) -> ContinuityState:
    cwm = CWMBuilder.bootstrap_empty(cfg=CWMConfig(dim=32, dtype="f16"), policy_mode="BALANCED")
    return ContinuityState(
        tenant_id=tenant,
        user_id=user,
        session_id=session,
        turn_id=1,
        policy_mode="BALANCED",
        cwm=cwm,
        summary_spine=SummarySpine(identity=tenant),
        facts_table=FactsTable(items=[]),
    )


def test_hotstate_isolation() -> None:
    store = HotStateStore()
    a = _mk_state("t1", "u1", "s1")
    b = _mk_state("t2", "u1", "s1")
    store.save(state=a)
    store.save(state=b)

    ra = store.load(tenant_id="t1", user_id="u1", session_id="s1")
    rb = store.load(tenant_id="t2", user_id="u1", session_id="s1")
    assert ra is not None and rb is not None
    assert ra.tenant_id == "t1"
    assert rb.tenant_id == "t2"

