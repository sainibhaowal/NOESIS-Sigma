from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _set_sim_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    key = base64.b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", key)
    get_sim_state.cache_clear()


def test_long_term_recall_precision(monkeypatch, tmp_path):
    _set_sim_env(monkeypatch, tmp_path)
    app = make_app()
    c = TestClient(app)
    headers = {"X-Tenant-Id": "tenantA", "X-User-Id": "userA"}

    keywords = []
    for i in range(12):
        kw = f"kw{i}x"
        keywords.append(kw)
        payload = f"{kw} {kw} {kw} Project Apollo milestone {i} contains {kw}."
        r = c.post(
            "/memory/write",
            headers=headers,
            json={"memory_type": "episodic", "payload": payload, "tags": {}},
        )
        assert r.status_code == 200

    hits = 0
    for kw in keywords:
        r = c.post(
            "/memory/read",
            headers=headers,
            json={"query": kw, "memory_type": None, "limit": 5},
        )
        assert r.status_code == 200
        items = r.json()["items"]
        if any(kw in (it.get("payload") or "") for it in items):
            hits += 1

    recall = hits / len(keywords)
    assert recall >= 0.9


def test_auto_summary_creates_facts(monkeypatch, tmp_path):
    _set_sim_env(monkeypatch, tmp_path)
    app = make_app()
    c = TestClient(app)
    headers = {"X-Tenant-Id": "tenantB", "X-User-Id": "userB"}

    st = get_sim_state()
    warm = st["warm"]
    for i in range(6):
        warm.add_session_summary(
            tenant_id="tenantB",
            user_id="userB",
            session_id="sess-1",
            turn_id=1000 + i,
            summary_text=f"User: q{i}\nAssistant: The system recorded fact {i}.",
        )

    r = c.post(
        "/summary/auto",
        headers=headers,
        json={"session_id": "sess-1", "max_summaries": 10, "max_facts": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["facts"]) >= 1

    facts = warm.list_memory_items(
        tenant_id="tenantB", user_id="userB", kind="fact", limit=10
    )
    assert any("fact" in (f.text or "").lower() for f in facts)
