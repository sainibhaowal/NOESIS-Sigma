from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, select

from External.Sim.Ledger.audit import LedgerAudit
from External.Sim.Ledger.consent_ledger import ConsentLedger
from External.Sim.Models.ledger import SimLedgerRecord
from External.Sim.Storage.warm_store import WarmStore


def test_ledger_append_only_and_tamper(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    warm = WarmStore(db_url=f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}", echo_sql=False)
    warm.init_db()
    ledger = ConsentLedger(engine=warm.engine)

    for i in range(5):
        ledger.append(
            tenant_id="t1",
            user_id="u1",
            actor_role="USER",
            action="WRITE",
            basis="policy",
            resource_id=str(i),
            event={"i": i},
        )

    audit = LedgerAudit(engine=warm.engine)
    vr = audit.verify_tenant_chain(tenant_id="t1")
    assert vr.ok is True

    # tamper with one row
    with Session(warm.engine) as s:
        row = s.exec(select(SimLedgerRecord).where(SimLedgerRecord.tenant_id == "t1").limit(1)).first()
        assert row is not None
        row.event_json = row.event_json + "x"
        s.add(row)
        s.commit()

    vr2 = audit.verify_tenant_chain(tenant_id="t1")
    assert vr2.ok is False
