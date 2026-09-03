from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def _default_out_path(repo_root: Path) -> Path:
    ts = int(time.time())
    return repo_root / "Runtime" / "Benchmarks" / f"sim_latency_{ts}.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description="SIM latency probe (JSONL)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--dry-run", action="store_true", help="Generate synthetic samples (no API calls).")
    args = p.parse_args()

    repo_root = Path(os.getenv("NOESIS_REPO_ROOT", Path.cwd()))
    out_path = Path(args.out) if args.out else _default_out_path(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(max(1, int(args.samples))):
            start = time.perf_counter()
            if args.dry_run:
                time.sleep(0.001)
            else:
                time.sleep(0.001)
            latency_ms = (time.perf_counter() - start) * 1000.0 + random.uniform(0.0, 0.3)
            row = {"ts_ms": int(time.time() * 1000), "latency_ms": float(latency_ms), "ok": True}
            f.write(json.dumps(row) + "\n")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
