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
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _group_by_warm(rows: list[dict], key: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for r in rows:
        if "warm_items" not in r:
            continue
        try:
            w = int(r["warm_items"])
        except Exception:
            continue
        try:
            val = float(r.get(key, 0.0))
        except Exception:
            continue
        out.setdefault(w, []).append(val)
    return out


def _p50_p95(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    xs = sorted(vals)
    p50 = xs[int(0.50 * (len(xs) - 1))]
    p95 = xs[int(0.95 * (len(xs) - 1))]
    return float(p50), float(p95)


def main() -> int:
    p = argparse.ArgumentParser(description="SIM plot helper (JSONL -> PNG)")
    p.add_argument("--input", required=True, help="Path to JSONL file")
    p.add_argument("--out-dir", default="", help="Output plots directory")
    if len(sys.argv) == 1:
        p.print_help()
        return 0
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    repo_root = Path(os.getenv("NOESIS_REPO_ROOT", Path.cwd()))
    out_dir = Path(args.out_dir) if args.out_dir else (in_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(in_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit("matplotlib required for plots") from e

    # vram_curve.png
    vram_rows = [r for r in rows if r.get("vram_mb") is not None]
    vram_by_w = _group_by_warm(vram_rows, "vram_mb")
    if vram_by_w:
        xs = sorted(vram_by_w.keys())
        ys = [sum(vram_by_w[w]) / len(vram_by_w[w]) for w in xs]
        plt.figure(figsize=(6, 3))
        plt.plot(xs, ys, marker="o", linewidth=1)
        plt.title("VRAM vs Warm Items")
        plt.xlabel("warm_items")
        plt.ylabel("vram_mb")
        plt.tight_layout()
        plt.savefig(out_dir / "vram_curve.png")
        plt.close()

    # latency_curve.png
    read_rows = [r for r in rows if r.get("op") == "read"]
    lat_by_w = _group_by_warm(read_rows, "lat_ms")
    if lat_by_w:
        xs = sorted(lat_by_w.keys())
        p50s = []
        p95s = []
        for w in xs:
            p50, p95 = _p50_p95(lat_by_w[w])
            p50s.append(p50)
            p95s.append(p95)
        plt.figure(figsize=(6, 3))
        plt.plot(xs, p50s, label="p50", marker="o", linewidth=1)
        plt.plot(xs, p95s, label="p95", marker="o", linewidth=1)
        plt.title("Latency vs Warm Items")
        plt.xlabel("warm_items")
        plt.ylabel("lat_ms")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "latency_curve.png")
        plt.close()

    # soak_summary.png (rss over time if present)
    rss_rows = [r for r in rows if r.get("rss_mb") is not None]
    if rss_rows:
        xs = list(range(len(rss_rows)))
        ys = [float(r["rss_mb"]) for r in rss_rows]
        plt.figure(figsize=(6, 3))
        plt.plot(xs, ys, marker="o", linewidth=1)
        plt.title("Soak Summary (RSS over time)")
        plt.xlabel("sample")
        plt.ylabel("rss_mb")
        plt.tight_layout()
        plt.savefig(out_dir / "soak_summary.png")
        plt.close()

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
