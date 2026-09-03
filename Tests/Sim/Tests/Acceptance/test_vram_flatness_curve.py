from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from External.Sim.Api.sim_api import get_sim_state, router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _vram_used_mb() -> int:
    if not shutil.which("nvidia-smi"):
        return -1
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if not out:
        return -1
    return int(out.splitlines()[0].strip())


def test_vram_flatness_curve(monkeypatch, tmp_path):
    if not shutil.which("nvidia-smi"):
        pytest.skip("nvidia-smi not available")

    monkeypatch.setenv("SIM_DB_URL", f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}")
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x33" * 32).decode("ascii"))
    monkeypatch.setenv("NOESIS_REPO_ROOT", tmp_path.as_posix())
    get_sim_state.cache_clear()

    app = make_app()
    c = TestClient(app)
    h = {"X-Tenant-Id": "t_vram", "X-User-Id": "u_vram", "X-Role": "USER"}

    stages = [200, 600, 1200]
    run_id = os.getenv("SIM_BENCH_RUN_ID", f"SIMBENCH_{int(time.time())}")
    bench_dir = Path(os.getenv("SIM_BENCH_DIR", str(tmp_path / "Runtime" / "Benchmarks" / run_id)))
    bench_dir.mkdir(parents=True, exist_ok=True)
    telemetry = []
    total = 0
    for target in stages:
        while total < target:
            payload = f"warm item {total}"
            r = c.post("/memory/write", headers=h, json={"memory_type": "semantic", "payload": payload, "profile": "FAST"})
            assert r.status_code == 200
            total += 1

        # fixed read workload
        for _ in range(5):
            r = c.post("/memory/read", headers=h, json={"query": "warm", "limit": 10, "profile": "FAST"})
            assert r.status_code == 200

        used_mb = _vram_used_mb()
        assert used_mb >= 0
        telemetry.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": run_id,
                "profile": "FAST",
                "phase": "warm_fill",
                "op": "vram_sample",
                "lat_ms": 0.0,
                "warm_items": total,
                "hot_items": None,
                "vram_mb": float(used_mb),
            }
        )

    # write telemetry JSONL
    out_path = bench_dir / "telemetry.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in telemetry:
            f.write(json.dumps(row) + "\n")

    from typing import cast

    vram_vals = [float(cast(float, t["vram_mb"])) for t in telemetry]
    items = [int(cast(int, t["warm_items"])) for t in telemetry]
    dv = max(vram_vals) - min(vram_vals)
    di = max(items) - min(items)
    slope = (dv / (di / 1000.0)) if di > 0 else 0.0

    slope_thr = float(os.getenv("SIM_VRAM_SLOPE_MB_PER_K", "0.02"))
    max_delta_thr = float(os.getenv("SIM_VRAM_MAX_DELTA_MB", "200"))
    assert slope <= slope_thr or dv <= max_delta_thr
