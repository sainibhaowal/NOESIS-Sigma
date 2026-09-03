from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_ledger_chain_verifies(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x22" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_user = {"X-Tenant-Id": "t_led", "X-User-Id": "u_led", "X-Role": "USER"}
    for i in range(5):
        r = c.post("/memory/write", headers=h_user, json={"memory_type": "episodic", "payload": f"note {i} about coffee", "profile": "STRICT"})
        assert r.status_code == 200

    h_admin = {"X-Tenant-Id": "t_led", "X-User-Id": "u_led", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}
    q = c.get("/sim/ledger?tenant_id=t_led&limit=50", headers=h_admin)
    assert q.status_code == 200
    j = q.json()
    assert j["verify"]["ok"] is True
    assert j["verify"]["checked"] >= 5
