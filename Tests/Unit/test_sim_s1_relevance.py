from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_relevance_query_returns_best_match(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t_rel", "X-User-Id": "u_rel"}

    c.post(
        "/memory/write",
        headers=headers,
        json={"memory_type": "semantic", "payload": "I love black coffee every morning", "profile": "FAST"},
    )
    c.post(
        "/memory/write",
        headers=headers,
        json={"memory_type": "semantic", "payload": "I play football on weekends", "profile": "FAST"},
    )
    c.post(
        "/memory/write",
        headers=headers,
        json={
            "memory_type": "semantic",
            "payload": "I studied in Germany and build NOESIS-Sigma",
            "profile": "FAST",
        },
    )

    r = c.post(
        "/memory/read",
        headers=headers,
        json={"query": "coffee morning", "memory_type": "semantic", "limit": 5, "profile": "FAST"},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert len(j["items"]) >= 1

    # Top result should mention coffee
    top = j["items"][0]["payload"].lower()
    assert "coffee" in top
