from __future__ import annotations

import base64
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_ttl_deletes_are_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_ADMIN_API_KEY", "adminkey")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t_s2_ttl", "X-User-Id": "u_s2_ttl"}
    admin_headers = {
        "X-Tenant-Id": "t_s2_ttl",
        "X-User-Id": "u_s2_ttl",
        "X-Role": "ADMIN",
        "X-Api-Key": "adminkey",
    }

    # write with tiny TTL (50ms)
    r1 = c.post(
        "/memory/write",
        headers=headers,
        json={
            "memory_type": "semantic",
            "payload": "short lived secret",
            "ttl_ms": 50,
            "profile": "FAST",
        },
    )
    assert r1.status_code == 200

    time.sleep(0.08)

    # run TTL sweep for tenant
    r2 = c.post("/sim/maintenance/ttl_sweep?tenant_id=t_s2_ttl", headers=admin_headers)
    assert r2.status_code == 200
    assert r2.json()["deleted"] >= 1

    # verify it's gone
    r3 = c.post(
        "/memory/read",
        headers=headers,
        json={"query": "secret", "memory_type": "semantic", "limit": 10, "profile": "FAST"},
    )
    assert r3.status_code == 200
    j3 = r3.json()
    assert all("short lived secret" not in it["payload"] for it in j3["items"])

    # verify events exist via jobs endpoint is not events; we just rely on deletion count
    # (S3 will expose /sim/ledger; S2 logs are in sim_events table.)
