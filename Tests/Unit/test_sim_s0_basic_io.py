from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_basic_write_and_read(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t1", "X-User-Id": "u1"}

    r1 = c.post(
        "/memory/write",
        headers=headers,
        json={
            "memory_type": "episodic",
            "payload": "hello world",
            "tags": {"a": 1},
            "profile": "FAST",
        },
    )
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1["ok"] is True
    assert "record_id" in j1

    r2 = c.post(
        "/memory/read",
        headers=headers,
        json={"query": "hello", "memory_type": "episodic", "limit": 10, "profile": "FAST"},
    )
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["ok"] is True
    assert len(j2["items"]) >= 1
    assert any("hello world" in it["payload"] for it in j2["items"])
