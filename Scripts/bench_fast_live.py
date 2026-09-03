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


def summarize(latencies_ms: List[float]) -> dict:
    return {
        "count": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "min_ms": min(latencies_ms) if latencies_ms else 0.0,
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous FAST latency monitor for /osc/chat")
    ap.add_argument("--url", default="http://127.0.0.1:9000/osc/chat")
    ap.add_argument("--duration", type=int, default=0, help="Seconds to run; 0=forever")
    ap.add_argument("--window", type=int, default=50, help="Rolling window size")
    ap.add_argument("--interval", type=float, default=1.0, help="Sleep between requests")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--user-id", default="bench")
    ap.add_argument("--session-id", default="bench_fast_live")
    ap.add_argument("--prompt", default="Say hello in one short sentence.")
    ap.add_argument("--out", default="Runtime/Logs/bench_fast_live.json")
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
    start = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout) as client:
        while True:
            t0 = time.perf_counter()
            r = client.post(args.url, headers=headers, json=payload)
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if len(latencies) > args.window:
                latencies = latencies[-args.window :]
            summary = summarize(latencies)
            Path(args.out).write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary))
            if args.duration and (time.time() - start) >= args.duration:
                break
            time.sleep(max(0.0, args.interval))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
