from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def _default_out_path(repo_root: Path) -> Path:
    ts = int(time.time())
    return repo_root / "Runtime" / "Benchmarks" / f"sim_vram_{ts}.jsonl"


def _probe_nvidia() -> tuple[int, int] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if not out:
            return None
        used, total = out.splitlines()[0].split(",")
        return int(used.strip()), int(total.strip())
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="SIM VRAM probe (JSONL)")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--dry-run", action="store_true", help="Generate synthetic samples.")
    args = p.parse_args()

    repo_root = Path(os.getenv("NOESIS_REPO_ROOT", Path.cwd()))
    out_path = Path(args.out) if args.out else _default_out_path(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(max(1, int(args.samples))):
            ts_ms = int(time.time() * 1000)
            if args.dry_run:
                used, total = 0, 0
            else:
                v = _probe_nvidia()
                if v is None:
                    used, total = 0, 0
                else:
                    used, total = v
            row = {"ts_ms": ts_ms, "vram_used_mb": used, "vram_total_mb": total}
            f.write(json.dumps(row) + "\n")
            time.sleep(0.05)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
