from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_snapshot_root_hash_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x33" * 32).decode("ascii"))
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)
    h = {"X-Tenant-Id": "t_s4", "X-User-Id": "u_s4", "X-Role": "USER"}

    # seed stable state
    for i in range(3):
        r = c.post("/memory/write", headers=h, json={"memory_type": "semantic", "payload": f"alpha {i}", "profile": "STRICT"})
        assert r.status_code == 200

    s1 = c.post("/sim/snapshot", headers=h)
    assert s1.status_code == 200
    s2 = c.post("/sim/snapshot", headers=h)
    assert s2.status_code == 200

    j1, j2 = s1.json(), s2.json()
    assert j1["root_hash"] == j2["root_hash"]
