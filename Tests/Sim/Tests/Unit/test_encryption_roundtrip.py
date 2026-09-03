from __future__ import annotations

import base64

import pytest

from External.Sim.Ledger.consent_ledger import ConsentLedger
from External.Sim.Security.encryption import open_envelope, seal
from External.Sim.Services.sim_core_service import Identity, SIMCoreService
from External.Sim.Storage.hot_store import HotStoreLRU
from External.Sim.Storage.warm_store import WarmStore


def test_encryption_roundtrip_and_aad(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x66" * 32).decode("ascii"))

    env = seal(
        tenant_id="t1",
        user_id="u1",
        memory_type="episodic",
        payload="secret",
        embed_b64_plain="abc",
    )
    payload, emb = open_envelope(tenant_id="t1", user_id="u1", memory_type="episodic", env=env)
    assert payload == "secret"
    assert emb == "abc"

    with pytest.raises(Exception):
        open_envelope(tenant_id="t2", user_id="u1", memory_type="episodic", env=env)

    # DB roundtrip via service (plaintext should not be stored)
    db_url = f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}"
    warm = WarmStore(db_url=db_url, echo_sql=False)
    warm.init_db()
    hot = HotStoreLRU(max_keys=64, default_max_items_per_key=32)
    ledger = ConsentLedger(engine=warm.engine)
    svc = SIMCoreService(hot=hot, warm=warm, ledger=ledger)

    ident = Identity(tenant_id="t1", user_id="u1", role="USER")
    out = svc.write(
        ident=ident,
        memory_type="semantic",
        payload="bank pin 1234",
        tags={},
        ttl_ms=None,
        profile_name="STRICT",
    )
    assert out["ok"] is True

    rows = warm.export_user_records(tenant_id="t1", user_id="u1")
    assert rows
    assert rows[-1].payload == ""
    assert rows[-1].payload_ct_b64 != ""
