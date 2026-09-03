from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_snapshot_rollback_restores_state(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x44" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    user_headers = {"X-Tenant-Id": "t_rb", "X-User-Id": "u_rb", "X-Role": "USER"}
    admin_headers = {"X-Tenant-Id": "t_rb", "X-User-Id": "u_rb", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    c.post("/memory/write", headers=user_headers, json={"memory_type": "semantic", "payload": "remember alpha", "profile": "STRICT"})
    c.post("/memory/write", headers=user_headers, json={"memory_type": "semantic", "payload": "remember beta", "profile": "STRICT"})

    snap = c.post("/sim/snapshot/create?tenant_id=t_rb&user_id=u_rb", headers=admin_headers)
    assert snap.status_code == 200
    snapshot_id = snap.json()["snapshot_id"]

    c.post("/memory/write", headers=user_headers, json={"memory_type": "semantic", "payload": "gamma after snapshot", "profile": "STRICT"})

    rb = c.post(f"/sim/snapshot/rollback?snapshot_id={snapshot_id}", headers=admin_headers)
    assert rb.status_code == 200

    r = c.post("/memory/read", headers=user_headers, json={"query": "gamma", "limit": 10, "profile": "STRICT"})
    assert r.status_code == 200
    items = [it["payload"] for it in r.json()["items"]]
    assert all("gamma after snapshot" not in p for p in items)

    r2 = c.post("/memory/read", headers=user_headers, json={"query": "remember", "limit": 10, "profile": "STRICT"})
    assert r2.status_code == 200
    payloads = [it["payload"] for it in r2.json()["items"]]
    assert any("remember alpha" in p for p in payloads)
    assert any("remember beta" in p for p in payloads)
