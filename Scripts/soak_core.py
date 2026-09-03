
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

#!/usr/bin/env python3
# ================================================================
# NOESIS-Σ — Golden Edition
# Script: Scripts/soak_core.py
# Purpose: Long-run soak to verify memory stability & latency p95.
# Profiles: AUTO | FAST | BALANCED | STRICT
# Notes:
#   - Uses Core/Exec profiles (dtypes, TF32, determinism).
#   - Optional CUDA-graph replay over the inner unroll (S steps).
#   - Uses GraphBucketManager to reuse captures per (B,S,dt,dtype,profile).
# ================================================================

import argparse
import os
import pathlib
import signal
import sys
import time

import torch

from Core.OSC.dynamics import HotLoopFused, HotLoopGraph
from Core.OSC.Exec.graph_cache import GraphBucketManager
from Core.OSC.Exec.profiles import ProfileConfig, apply_profile, resolve_profile
from Core.OSC.icnn import ICNNDirectGrad

# Ensure repo root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------- Helpers ---------------------------

def percentiles(xs):
    xs = sorted(xs)
    if not xs:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    def p(q):
        k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[k]
    return {"p50": p(0.50), "p95": p(0.95), "p99": p(0.99)}


def lin_reg_slope(xs, ys):
    """Least-squares slope for (time_minutes, MB)."""
    n = float(len(xs))
    if n < 2:
        return float("nan")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def p95_target_ms(profile_name: str) -> float:
    """Latency p95 target by profile (tunable)."""
    name = profile_name.upper()
    if name == "FAST":
        return 5.0
    if name == "BALANCED":
        return 6.0
    if name == "STRICT":
        return 8.0
    return 6.0  # default for AUTO (resolved later)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="NOESIS-Σ soak: memory slope + p95 latency")
    ap.add_argument("--d", type=int, required=True, help="state dimension")
    ap.add_argument("--m", type=int, default=512, help="ICNN hidden size")
    ap.add_argument("--r", type=int, default=64, help="low-rank K rank")
    ap.add_argument("--B", type=int, default=32, help="batch size (states)")
    ap.add_argument("--S", type=int, default=32, help="inner unroll steps")
    ap.add_argument("--dt", type=float, default=0.02, help="time step for each inner step")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"],
                    help="main compute dtype (hint; profile may override)")
    ap.add_argument("--duration", type=int, default=3600, help="seconds to run")
    ap.add_argument("--sample_every", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--profile", type=str, default="AUTO", choices=["AUTO", "FAST", "BALANCED", "STRICT"])
    ap.add_argument("--graph", action="store_true", help="Use CUDA graph for fused mode")
    ap.add_argument("--metrics-file", type=str, default="", help="Write Prometheus textfile metrics here")
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    args = ap.parse_args()

    # Env-driven graph toggle
    USE_GRAPH = bool(args.graph or os.getenv("NOESIS_USE_CUDA_GRAPH") == "1")

    # Resolve & apply numeric profile (sets TF32 & determinism)
    prof: ProfileConfig = apply_profile(resolve_profile(args.profile))

    # Final dtype: prefer profile's main dtype; allow CLI hint if it matches
    cli_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    dtype = prof.main_dtype if cli_dtype != prof.main_dtype else cli_dtype

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Report resolved settings up-front
    print("== SOAK CONFIG ==")
    print(f"Device: {dev}")
    print(f"Profile: {prof.name}  (TF32={torch.backends.cuda.matmul.allow_tf32}, "
          f"Deterministic={torch.are_deterministic_algorithms_enabled()})")
    print(f"Dtypes: main={dtype}, ICNN.ws={prof.icnn_ws}, K.ws={prof.k_ws}")
    print(f"Shapes: d={args.d}, m={args.m}, r={args.r}, B={args.B}, S={args.S}, dt={args.dt}")
    if prof.deterministic and torch.cuda.is_available() and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        print("Note: STRICT mode typically requires CUBLAS_WORKSPACE_CONFIG=:4096:8 for full determinism.")

    # Build modules (disable runtime checks during capture)
    icnn = ICNNDirectGrad(
        d=args.d, m=args.m, dtype=dtype, device=dev,
        ws_dtype=prof.icnn_ws, enforce_convex_runtime=not USE_GRAPH
    )
    KU = torch.randn(args.d, args.r, device=dev, dtype=dtype) * 0.02
    KV = torch.randn(args.d, args.r, device=dev, dtype=dtype) * 0.02
    fused = HotLoopFused(d=args.d, icnn=icnn, KU=KU, KV=KV, device=dev, dtype=dtype, ws_dtype=prof.k_ws)

    # Graph capture (optional)
    gman = GraphBucketManager(capacity=8) if USE_GRAPH else None
    g: HotLoopGraph | None = None
    if gman is not None and torch.cuda.is_available():
        g = gman.get(fused, B=args.B, S=args.S, dt=args.dt, dtype=dtype, prof=prof)

    # Allocate input & warm-up
    x = torch.randn(args.B, args.d, device=dev, dtype=dtype)
    for _ in range(5):
        x = (g.run(x) if g is not None else fused.step_unrolled(x, S=args.S, dt=args.dt))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Soak loop
    p95_goal = p95_target_ms(prof.name)
    ts_min, mem_mb_series, lat_ms = [], [], []

    interrupted = False
    def _sigint(_s, _f):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    t0 = time.perf_counter()
    while not interrupted and (time.perf_counter() - t0) < args.duration:
        iter_start = time.perf_counter()
        x = (g.run(x) if g is not None else fused.step_unrolled(x, S=args.S, dt=args.dt))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - iter_start
        lat_ms.append((elapsed / args.S) * 1000.0)

        now_min = (time.perf_counter() - t0) / 60.0
        mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        ts_min.append(now_min)
        mem_mb_series.append(mb)

        # pacing
        sleep_left = args.sample_every - (time.perf_counter() - iter_start)
        if sleep_left > 0:
            time.sleep(max(0.0, sleep_left))

    # Results
    lat_stats = percentiles(lat_ms)  # ms
    p50_ms = float(lat_stats["p50"])
    p95_ms = float(lat_stats["p95"])
    p99_ms = float(lat_stats["p99"])
    slope = lin_reg_slope(ts_min, mem_mb_series)  # MB/min

    print("== SOAK RESULTS ==")
    if torch.cuda.is_available() and mem_mb_series:
        print(f"VRAM: start={mem_mb_series[0]:.1f}MB end={mem_mb_series[-1]:.1f}MB slope={slope:.3f} MB/min")
    else:
        print("CPU-only or no samples: VRAM slope not applicable.")
    print(f"Per-step latency: p50={p50_ms:.2f} ms  p95={p95_ms:.2f} ms  p99={p99_ms:.2f} ms")

    # Gates
    if torch.cuda.is_available() and mem_mb_series and abs(slope) <= 0.5:
        print("PASS: G3 memory stability (slope ~ 0)")
    if not (p95_ms != p95_ms):  # not NaN
        goal = p95_goal
        if p95_ms <= goal:
            print(f"PASS: latency stability for profile {prof.name} (p95 ≤ {goal:.2f} ms)")
        else:
            print(f"WARN: latency p95 {p95_ms:.2f} ms exceeds target {goal:.2f} ms for {prof.name}")

    # Metrics (Prometheus textfile)
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            f.write("# HELP noesis_core_active_profile Active profile name\n")
            f.write("# TYPE noesis_core_active_profile gauge\n")
            f.write(f'noesis_core_active_profile{{name="{prof.name}"}} 1\n')

            f.write("# HELP noesis_core_graph_used Whether CUDA graph was used\n")
            f.write("# TYPE noesis_core_graph_used gauge\n")
            f.write(f'noesis_core_graph_used {1 if g is not None else 0}\n')

            f.write("# HELP noesis_core_latency_p95_s Per-step latency p95 (seconds)\n")
            f.write("# TYPE noesis_core_latency_p95_s gauge\n")
            f.write(f'noesis_core_latency_p95_s {p95_ms/1000.0}\n')

            f.write("# HELP noesis_core_vram_slope_mb_per_min VRAM slope MB/min\n")
            f.write("# TYPE noesis_core_vram_slope_mb_per_min gauge\n")
            f.write(f'noesis_core_vram_slope_mb_per_min {0.0 if (slope != slope) else slope}\n')


if __name__ == "__main__":
    main()
