from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo = Path(__file__).resolve()
    for _ in range(6):
        if (repo / 'pyproject.toml').exists():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

_ensure_repo_root()


import json
import os

import requests  # type: ignore[import-untyped]

BASE = os.getenv("SIM_API_URL", "http://127.0.0.1:9000")

H = {
    "X-Tenant-Id": "tenant_demo",
    "X-User-Id": "user_demo",
}


def post(path: str, payload: dict):
    r = requests.post(BASE + path, headers=H, json=payload, timeout=10)
    print(path, r.status_code)
    print(json.dumps(r.json(), indent=2))
    return r


if __name__ == "__main__":
    try:
        h = requests.get(BASE + "/health", timeout=2)
        if h.status_code >= 500:
            print("SKIP: server error on /health")
            raise SystemExit(0)
    except Exception:
        print(f"SKIP: server not reachable at {BASE}")
        raise SystemExit(0)
    post(
        "/memory/write",
        {
            "memory_type": "episodic",
            "payload": "I like black coffee.",
            "tags": {"source": "demo"},
            "profile": "BALANCED",
        },
    )
    post(
        "/memory/write",
        {
            "memory_type": "semantic",
            "payload": "NOESIS-Σ uses SIM as authoritative memory.",
            "tags": {"source": "demo"},
        },
    )
    post("/memory/read", {"query": "coffee", "memory_type": "episodic", "limit": 10})
