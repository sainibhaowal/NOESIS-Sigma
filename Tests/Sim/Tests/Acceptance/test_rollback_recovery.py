from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_rollback_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x55" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_user = {"X-Tenant-Id": "t_acc", "X-User-Id": "u_acc", "X-Role": "USER"}
    h_admin = {"X-Tenant-Id": "t_acc", "X-User-Id": "u_acc", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    c.post("/memory/write", headers=h_user, json={"memory_type": "episodic", "payload": "A", "profile": "STRICT"})
    snap = c.post("/sim/snapshot", headers=h_user)
    assert snap.status_code == 200
    snap_id = snap.json()["snapshot_id"]

    c.post("/memory/write", headers=h_user, json={"memory_type": "episodic", "payload": "B", "profile": "STRICT"})
    rb = c.post("/sim/rollback", headers=h_admin, params={"tenant_id": "t_acc", "user_id": "u_acc", "snapshot_id": snap_id})
    assert rb.status_code == 200

    r = c.post("/memory/read", headers=h_user, json={"query": "B", "limit": 10, "profile": "STRICT"})
    assert r.status_code == 200
    payloads = [it["payload"] for it in r.json()["items"]]
    assert all("B" not in p for p in payloads)

    # active snapshot pointer updated
    st = get_sim_state()
    active = st["warm"].get_active_snapshot(tenant_id="t_acc", user_id="u_acc")
    assert active == snap_id

    # ledger has ROLLBACK event
    q = c.get("/sim/ledger?tenant_id=t_acc&limit=200", headers=h_admin)
    assert q.status_code == 200
    assert any(it["action"] == "ROLLBACK" for it in q.json()["items"])
