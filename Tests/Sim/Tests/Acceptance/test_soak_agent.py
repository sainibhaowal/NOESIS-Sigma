from __future__ import annotations

import base64
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_soak_agent(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x55" * 32).decode("ascii"))
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_user = {"X-Tenant-Id": "t_soak", "X-User-Id": "u_soak", "X-Role": "USER"}
    h_admin = {"X-Tenant-Id": "t_soak", "X-User-Id": "u_soak", "X-Role": "ADMIN", "X-Api-Key": "adminkey"}

    start = time.time()
    writes = 0
    reads = 0
    while time.time() - start < 2.0:
        r = c.post("/memory/write", headers=h_user, json={"memory_type": "semantic", "payload": f"soak {writes}", "profile": "STRICT"})
        assert r.status_code == 200
        writes += 1

        r2 = c.post("/memory/read", headers=h_user, json={"query": "soak", "limit": 5, "profile": "STRICT"})
        assert r2.status_code == 200
        reads += 1

        if writes % 5 == 0:
            snap = c.post("/sim/snapshot", headers=h_user)
            assert snap.status_code == 200

        if writes % 7 == 0:
            c.post("/sim/maintenance/compact?tenant_id=t_soak&user_id=u_soak&memory_type=semantic&target_n=64", headers=h_admin)

    # ledger verify passes
    q = c.get("/sim/ledger?tenant_id=t_soak&limit=200", headers=h_admin)
    assert q.status_code == 200
    assert q.json()["verify"]["ok"] is True

    # job queue should be bounded
    jobs = c.get("/sim/maintenance/jobs?tenant_id=t_soak&limit=200", headers=h_admin)
    assert jobs.status_code == 200
    assert len(jobs.json().get("jobs", [])) <= 200
