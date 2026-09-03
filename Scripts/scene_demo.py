
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

import requests  # type: ignore[import-untyped]


def main() -> None:
    try:
        h = requests.get("http://127.0.0.1:9000/health", timeout=2)
        if h.status_code >= 500:
            print("SKIP: server error on /health")
            return
    except Exception:
        print("SKIP: server not reachable at http://127.0.0.1:9000")
        return
    r = requests.post(
        "http://127.0.0.1:9000/scene/build",
        json={
            "profile": "BALANCED",
            "text": "What is SIM?",
            "seed": 1,
            "tenant_id": "default",
            "user_id": "default",
            "session_id": "demo",
            "sim_top_k": 4,
            "wks_top_k": 4,
            "pointer_byte_budget": 2000,
        },
        timeout=30,
    )
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
