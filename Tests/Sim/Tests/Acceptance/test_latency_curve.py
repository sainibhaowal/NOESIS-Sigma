from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    xs = sorted(samples)
    idx = int(0.95 * (len(xs) - 1))
    return xs[idx]


def test_latency_curve(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x44" * 32).decode("ascii"))
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)
    h = {"X-Tenant-Id": "t_lat", "X-User-Id": "u_lat", "X-Role": "USER"}

    stages = [200, 600, 1200]
    latencies: dict[str, list[float]] = {"FAST": [], "BALANCED": [], "STRICT": []}
    run_id = os.getenv("SIM_BENCH_RUN_ID", f"SIMBENCH_{int(time.time())}")
    bench_dir = Path(os.getenv("SIM_BENCH_DIR", str(tmp_path / "Runtime" / "Benchmarks" / run_id)))
    bench_dir.mkdir(parents=True, exist_ok=True)
    telemetry: list[dict] = []
    total = 0
    for target in stages:
        while total < target:
            r = c.post(
                "/memory/write",
                headers=h,
                json={"memory_type": "semantic", "payload": f"lat {total}", "profile": "FAST"},
            )
            assert r.status_code == 200
            total += 1

        for mode in ("FAST", "BALANCED", "STRICT"):
            for _ in range(5):
                t0 = time.perf_counter()
                r = c.post(
                    "/memory/read",
                    headers=h,
                    json={"query": "lat", "limit": 20, "profile": mode},
                )
                assert r.status_code == 200
                lat = (time.perf_counter() - t0) * 1000.0
                latencies[mode].append(lat)
                telemetry.append(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "run_id": run_id,
                        "profile": mode,
                        "phase": "read_mix",
                        "op": "read",
                        "lat_ms": float(lat),
                        "warm_items": total,
                        "hot_items": None,
                    }
                )

    out_path = bench_dir / "telemetry.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in telemetry:
            f.write(json.dumps(row) + "\n")

    # Defaults are tolerant for laptop-class hardware; CI can override via env.
    fast_thr = float(os.getenv("SIM_LAT_P95_FAST_MS", "150"))
    bal_thr = float(os.getenv("SIM_LAT_P95_BAL_MS", "260"))
    strict_thr = float(os.getenv("SIM_LAT_P95_STRICT_MS", "400"))

    assert _p95(latencies["FAST"]) <= fast_thr
    assert _p95(latencies["BALANCED"]) <= bal_thr
    assert _p95(latencies["STRICT"]) <= strict_thr
