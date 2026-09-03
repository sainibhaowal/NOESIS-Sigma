from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_snapshot_root_hash_determinism(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x33" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    user_headers = {"X-Tenant-Id": "t_snap", "X-User-Id": "u_snap", "X-Role": "USER"}
    admin_headers = {"X-Tenant-Id": "t_snap", "X-User-Id": "u_snap", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    c.post("/memory/write", headers=user_headers, json={"memory_type": "episodic", "payload": "alpha", "profile": "STRICT"})
    c.post("/memory/write", headers=user_headers, json={"memory_type": "episodic", "payload": "beta", "profile": "STRICT"})

    s1 = c.post("/sim/snapshot/create?tenant_id=t_snap&user_id=u_snap", headers=admin_headers)
    s2 = c.post("/sim/snapshot/create?tenant_id=t_snap&user_id=u_snap", headers=admin_headers)

    assert s1.status_code == 200
    assert s2.status_code == 200
    j1 = s1.json()
    j2 = s2.json()

    assert j1["root_hash"] == j2["root_hash"]
    assert j1["snapshot_id"] != j2["snapshot_id"]
