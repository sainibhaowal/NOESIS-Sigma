from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="SIM soak agent (long-run)")
    p.add_argument("--duration-sec", type=int, default=5)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    repo_root = Path(os.getenv("NOESIS_REPO_ROOT", Path.cwd()))
    out_path = Path(args.out) if args.out else (repo_root / "Runtime" / "Benchmarks" / f"soak_summary_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    ops = 0
    while time.time() - start < max(1, int(args.duration_sec)):
        ops += 1
        time.sleep(0.01)

    summary = {
        "duration_sec": float(time.time() - start),
        "ops": ops,
        "ops_per_sec": ops / max(1e-6, (time.time() - start)),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
