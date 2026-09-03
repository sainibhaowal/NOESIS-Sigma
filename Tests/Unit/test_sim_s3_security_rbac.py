from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_user_cannot_access_ledger_admin_can(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    get_sim_state.cache_clear()
    app = make_app()
    c = TestClient(app)

    # USER role: should be forbidden
    h_user = {"X-Tenant-Id": "t3", "X-User-Id": "u3", "X-Role": "USER"}
    r1 = c.get("/sim/ledger?tenant_id=t3", headers=h_user)
    assert r1.status_code in (401, 403)

    # ADMIN role with key: allowed
    h_admin = {"X-Tenant-Id": "t3", "X-User-Id": "u3", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}
    r2 = c.get("/sim/ledger?tenant_id=t3", headers=h_admin)
    assert r2.status_code == 200
    assert r2.json()["ok"] is True
