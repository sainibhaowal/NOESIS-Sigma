from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_rollback_restores_previous_state(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x44" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_user = {"X-Tenant-Id": "t_rb", "X-User-Id": "u_rb", "X-Role": "USER"}
    h_admin = {"X-Tenant-Id": "t_rb", "X-User-Id": "u_rb", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    # state A
    c.post("/memory/write", headers=h_user, json={"memory_type": "episodic", "payload": "state A memory", "profile": "STRICT"})

    snap = c.post("/sim/snapshot", headers=h_user)
    assert snap.status_code == 200
    snap_id = snap.json()["snapshot_id"]

    # state B (after snapshot)
    c.post("/memory/write", headers=h_user, json={"memory_type": "episodic", "payload": "state B memory", "profile": "STRICT"})

    # rollback to snapshot
    rb = c.post("/sim/rollback", headers=h_admin, params={"tenant_id": "t_rb", "user_id": "u_rb", "snapshot_id": snap_id})
    assert rb.status_code == 200

    # verify B is gone (warm scan)
    r = c.post("/memory/read", headers=h_user, json={"query": "state B", "memory_type": "episodic", "limit": 10, "profile": "STRICT"})
    assert r.status_code == 200
    payloads = [it["payload"] for it in r.json()["items"]]
    assert all("state B" not in p for p in payloads)

    # verify A exists
    r2 = c.post("/memory/read", headers=h_user, json={"query": "state A", "memory_type": "episodic", "limit": 10, "profile": "STRICT"})
    assert r2.status_code == 200
    payloads2 = [it["payload"] for it in r2.json()["items"]]
    assert any("state A" in p for p in payloads2)
