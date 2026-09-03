from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_tenant_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    h_a = {"X-Tenant-Id": "tenantA", "X-User-Id": "user1"}
    h_b = {"X-Tenant-Id": "tenantB", "X-User-Id": "user1"}

    c.post(
        "/memory/write",
        headers=h_a,
        json={"memory_type": "episodic", "payload": "secret A", "profile": "FAST"},
    )
    c.post(
        "/memory/write",
        headers=h_b,
        json={"memory_type": "episodic", "payload": "secret B", "profile": "FAST"},
    )

    ra = c.post(
        "/memory/read",
        headers=h_a,
        json={"memory_type": "episodic", "limit": 50, "profile": "FAST"},
    ).json()
    rb = c.post(
        "/memory/read",
        headers=h_b,
        json={"memory_type": "episodic", "limit": 50, "profile": "FAST"},
    ).json()

    a_payloads = [it["payload"] for it in ra["items"]]
    b_payloads = [it["payload"] for it in rb["items"]]

    assert "secret A" in a_payloads
    assert "secret B" not in a_payloads

    assert "secret B" in b_payloads
    assert "secret A" not in b_payloads
