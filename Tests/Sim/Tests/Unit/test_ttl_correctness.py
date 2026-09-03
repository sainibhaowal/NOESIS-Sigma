from __future__ import annotations

import base64
import time
from pathlib import Path

from sqlmodel import Session, select

from External.Sim.Ledger.audit import LedgerAudit
from External.Sim.Ledger.consent_ledger import ConsentLedger
from External.Sim.Models.records import MemoryRecord
from External.Sim.Services.compactor import SIMCompactor
from External.Sim.Storage.hot_store import HotStoreLRU
from External.Sim.Storage.warm_store import WarmStore


def test_ttl_correctness(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x55" * 32).decode("ascii"))

    warm = WarmStore(db_url=f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}", echo_sql=False)
    warm.init_db()
    hot = HotStoreLRU(max_keys=128, default_max_items_per_key=64)
    ledger = ConsentLedger(engine=warm.engine)
    comp = SIMCompactor(hot=hot, warm=warm, ledger=ledger)

    # expired record
    warm.write_record(
        tenant_id="t",
        user_id="u",
        memory_type="episodic",
        payload_plain="expired",
        embed_b64_plain="",
        tags={},
        ttl_ms=10,
        enc_v=0,
        dek_wrapped_b64="",
        dek_wrap_nonce_b64="",
        payload_ct_b64="",
        payload_nonce_b64="",
        embed_ct_b64="",
        embed_nonce_b64="",
    )
    # unexpired record
    warm.write_record(
        tenant_id="t",
        user_id="u",
        memory_type="episodic",
        payload_plain="keep",
        embed_b64_plain="",
        tags={},
        ttl_ms=10_000,
        enc_v=0,
        dek_wrapped_b64="",
        dek_wrap_nonce_b64="",
        payload_ct_b64="",
        payload_nonce_b64="",
        embed_ct_b64="",
        embed_nonce_b64="",
    )

    time.sleep(0.05)
    deleted = comp.ttl_sweep(tenant_id="t")
    assert deleted >= 1

    with Session(warm.engine) as s:
        rows = list(
            s.exec(select(MemoryRecord).where(MemoryRecord.tenant_id == "t", MemoryRecord.user_id == "u")).all()
        )
    payloads = [r.payload for r in rows]
    assert "expired" not in payloads
    assert "keep" in payloads

    audit = LedgerAudit(engine=warm.engine)
    entries = audit.query(tenant_id="t", action=None, limit=50)
    assert any(e.action in ("TTL_SWEEP", "DELETE") and e.basis == "ttl" for e in entries)
