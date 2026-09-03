from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_ot_stability_many_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)

    headers = {"X-Tenant-Id": "t_ot", "X-User-Id": "u_ot"}

    # Many writes should not crash, and reads should remain bounded and finite.
    for i in range(200):
        payload = f"memory item {i} about coffee and work {i%7}"
        r = c.post(
            "/memory/write",
            headers=headers,
            json={"memory_type": "episodic", "payload": payload, "profile": "FAST"},
        )
        assert r.status_code == 200

    r2 = c.post(
        "/memory/read",
        headers=headers,
        json={"query": "coffee", "memory_type": "episodic", "limit": 50, "profile": "FAST"},
    )
    assert r2.status_code == 200
    j = r2.json()
    assert j["ok"] is True
    assert len(j["items"]) <= 50

    # Scores must be finite when present
    for it in j["items"]:
        sc = it.get("score", None)
        if sc is not None:
            assert isinstance(sc, (int, float))
            assert sc == sc  # not NaN
