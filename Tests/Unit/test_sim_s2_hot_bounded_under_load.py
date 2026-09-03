from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_hot_bounded_under_heavy_write_load(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t_s2_load", "X-User-Id": "u_s2_load"}

    # blast writes; hot should remain bounded by profile cap
    for i in range(500):
        payload = f"episodic item {i} coffee work focus {i%9}"
        r = c.post(
            "/memory/write",
            headers=headers,
            json={"memory_type": "episodic", "payload": payload, "profile": "FAST"},
        )
        assert r.status_code == 200

    # read should return max 200 (schema), but practically <= requested
    r2 = c.post(
        "/memory/read",
        headers=headers,
        json={"query": "coffee", "memory_type": "episodic", "limit": 100, "profile": "FAST"},
    )
    assert r2.status_code == 200
    j = r2.json()
    assert j["ok"] is True
    assert len(j["items"]) <= 100
