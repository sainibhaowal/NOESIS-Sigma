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


import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests  # type: ignore[import-untyped]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _vram_mb() -> float | None:
    try:
        import shutil
        import subprocess

        if not shutil.which("nvidia-smi"):
            return None
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if not out:
            return None
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def _emit(fh, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def main() -> int:
    p = argparse.ArgumentParser(description="SIM benchmark runner (JSONL)")
    p.add_argument("--api-url", default=os.getenv("SIM_API_URL", "http://127.0.0.1:9000"))
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--fill", type=int, default=1000)
    p.add_argument("--reads", type=int, default=50)
    p.add_argument("--profile", default="BALANCED")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_root = Path(os.getenv("NOESIS_REPO_ROOT", Path.cwd()))
    run_id = os.getenv("SIM_BENCH_RUN_ID", f"SIMBENCH_{uuid.uuid4().hex[:8]}")
    bench_dir = Path(os.getenv("SIM_BENCH_DIR", str(repo_root / "Runtime" / "Benchmarks" / run_id)))
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (bench_dir / "telemetry.jsonl")

    headers = {
        "X-Tenant-Id": os.getenv("SIM_BENCH_TENANT", "bench_tenant"),
        "X-User-Id": os.getenv("SIM_BENCH_USER", "bench_user"),
        "X-Role": "USER",
    }

    if not args.dry_run:
        try:
            h = requests.get(args.api_url + "/health", timeout=2)
            if h.status_code >= 500:
                print("SKIP: server error on /health")
                return 0
        except Exception:
            print(f"SKIP: server not reachable at {args.api_url}")
            return 0

    with out_path.open("w", encoding="utf-8") as f:
        warm_items = 0
        for _ in range(max(1, int(args.runs))):
            # warm fill
            for i in range(int(args.fill)):
                t0 = time.perf_counter()
                if not args.dry_run:
                    r = requests.post(
                        args.api_url + "/memory/write",
                        headers=headers,
                        json={
                            "memory_type": "semantic",
                            "payload": f"bench item {warm_items}",
                            "profile": args.profile,
                        },
                        timeout=10,
                    )
                    r.raise_for_status()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                warm_items += 1
                row = {
                    "ts": _iso_now(),
                    "run_id": run_id,
                    "profile": args.profile,
                    "phase": "warm_fill",
                    "op": "write",
                    "lat_ms": float(lat_ms),
                    "warm_items": warm_items,
                    "hot_items": None,
                }
                rss = _get_rss_mb()
                vram = _vram_mb()
                if rss is not None:
                    row["rss_mb"] = rss
                if vram is not None:
                    row["vram_mb"] = vram
                _emit(f, row)

            # read mix
            for _ in range(int(args.reads)):
                t0 = time.perf_counter()
                if not args.dry_run:
                    r = requests.post(
                        args.api_url + "/memory/read",
                        headers=headers,
                        json={"query": "bench", "limit": 50, "profile": args.profile},
                        timeout=10,
                    )
                    r.raise_for_status()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                row = {
                    "ts": _iso_now(),
                    "run_id": run_id,
                    "profile": args.profile,
                    "phase": "read_mix",
                    "op": "read",
                    "lat_ms": float(lat_ms),
                    "warm_items": warm_items,
                    "hot_items": None,
                    "k": 50,
                }
                rss = _get_rss_mb()
                vram = _vram_mb()
                if rss is not None:
                    row["rss_mb"] = rss
                if vram is not None:
                    row["vram_mb"] = vram
                _emit(f, row)

            # snapshot
            t0 = time.perf_counter()
            snapshot_id = None
            root_hash = None
            if not args.dry_run:
                r = requests.post(args.api_url + "/sim/snapshot", headers=headers, timeout=10)
                r.raise_for_status()
                jr = r.json()
                snapshot_id = jr.get("snapshot_id")
                root_hash = jr.get("root_hash")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            row = {
                "ts": _iso_now(),
                "run_id": run_id,
                "profile": args.profile,
                "phase": "snapshot",
                "op": "snapshot",
                "lat_ms": float(lat_ms),
                "snapshot_id": snapshot_id,
                "root_hash": root_hash,
                "warm_items": warm_items,
            }
            _emit(f, row)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
