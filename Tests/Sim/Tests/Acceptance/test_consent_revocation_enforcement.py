from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_consent_revocation_enforcement(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x66" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_user = {"X-Tenant-Id": "t_c", "X-User-Id": "u_c", "X-Role": "USER"}
    h_admin = {"X-Tenant-Id": "t_c", "X-User-Id": "u_c", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    r1 = c.post("/sim/consent", headers=h_user, json={"revoked": True})
    assert r1.status_code == 200

    r2 = c.post("/memory/write", headers=h_user, json={"memory_type": "semantic", "payload": "blocked", "profile": "STRICT"})
    assert r2.status_code == 403

    r3 = c.post("/memory/read", headers=h_user, json={"query": "blocked", "limit": 5, "profile": "STRICT"})
    assert r3.status_code == 403

    # Admin can still audit ledger
    q = c.get("/sim/ledger?tenant_id=t_c&limit=20", headers=h_admin)
    assert q.status_code == 200
    items = q.json()["items"]
    assert any(it["action"] == "CONSENT" for it in items)
    assert any(it["action"] == "DENY" for it in items)
