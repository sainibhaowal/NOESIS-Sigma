#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import httpx


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser(description="FAST p95 latency benchmark for /osc/chat")
    ap.add_argument("--url", default="http://127.0.0.1:9000/osc/chat")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--user-id", default="bench")
    ap.add_argument("--session-id", default="bench_fast")
    ap.add_argument("--prompt", default="Say hello in one short sentence.")
    ap.add_argument("--out", default="Runtime/Logs/bench_fast_p95.json")
    args = ap.parse_args()

    headers = {
        "Content-Type": "application/json",
        "X-Tenant-Id": args.tenant_id,
        "X-User-Id": args.user_id,
    }
    payload = {
        "text": args.prompt,
        "compute_profile": "FAST",
        "verify_mode": "OFF",
        "session_id": args.session_id,
        "max_tokens": 64,
    }

    latencies: List[float] = []
    with httpx.Client(timeout=args.timeout) as client:
        # warmup
        for _ in range(args.warmup):
            client.post(args.url, headers=headers, json=payload)
        for _ in range(args.requests):
            t0 = time.perf_counter()
            r = client.post(args.url, headers=headers, json=payload)
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    out = {
        "count": len(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
