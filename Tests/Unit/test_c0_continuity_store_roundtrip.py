from __future__ import annotations

import os

import pytest

from Core.Continuity.cwm import CWMBuilder, CWMConfig
from Core.Continuity.hashing import apply_hashes
from Core.Continuity.models import ContinuityState, FactsTable, SummarySpine
from External.Sim.Services.continuity_store import ContinuityStore
from External.Sim.Storage.warm_store import WarmStore


def test_continuity_store_roundtrip() -> None:
    db_url = os.getenv("SIM_DB_URL", "").strip()
    if not db_url:
        pytest.skip("SIM_DB_URL not set; PostgreSQL required for continuity_store_roundtrip.")

    warm = WarmStore(db_url=db_url, echo_sql=False)
    warm.init_db()

    store = ContinuityStore(engine=warm.engine, ledger=None)
    cwm = CWMBuilder.bootstrap_empty(cfg=CWMConfig(dim=32, dtype="f16"), policy_mode="BALANCED")
    st = ContinuityState(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        turn_id=1,
        policy_mode="BALANCED",
        cwm=cwm,
        summary_spine=SummarySpine(identity="id"),
        facts_table=FactsTable(items=[]),
    )
    st = apply_hashes(st)
    store.put_continuity(st)
    loaded = store.get_latest_continuity(tenant_id="t1", user_id="u1", session_id="s1")
    assert loaded is not None
    assert loaded.continuity_root_hash == st.continuity_root_hash
