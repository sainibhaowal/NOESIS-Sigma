from __future__ import annotations

import base64
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from External.Sim.Api.sim_api import get_sim_state, router
from External.Sim.Models.records import MemoryRecord


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_payload_not_stored_plaintext_when_encrypted(monkeypatch, tmp_path):
    # 32 bytes master key
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x11" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t_enc", "X-User-Id": "u_enc", "X-Role": "USER"}

    payload = "sample payload 1234"
    w = c.post(
        "/memory/write",
        headers=headers,
        json={"memory_type": "semantic", "payload": payload, "profile": "STRICT"},
    )
    assert w.status_code == 200
    rid = w.json()["record_id"]

    # inspect DB row to ensure plaintext payload field is empty and ciphertext exists
    st = get_sim_state()
    eng = st["warm"].engine
    with Session(eng) as s:
        rec = s.exec(select(MemoryRecord).where(MemoryRecord.tenant_id == "t_enc", MemoryRecord.user_id == "u_enc").limit(1)).first()
        assert rec is not None
        assert rec.payload == ""  # not stored in plaintext
        assert rec.payload_ct_b64 != ""

    # read returns decrypted payload
    r = c.post(
        "/memory/read",
        headers=headers,
        json={
            "query": "sample payload",
            "memory_type": "semantic",
            "limit": 5,
            "profile": "STRICT",
        },
    )
    assert r.status_code == 200
    top = r.json()["items"][0]["payload"]
    assert "sample payload" in top
