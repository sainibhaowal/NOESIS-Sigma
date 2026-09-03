
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
# Script: Scripts/bench_core.py
# Purpose: Measure throughput & latency for:
#   - ICNN direct-grad (grad_only)
#   - Unfused low-rank loop (unfused)
#   - Fused hot loop (fused)
# Profiles: AUTO | FAST | BALANCED | STRICT
# Optional: --graph to CUDA-graph the fused inner unroll
# ================================================================

import argparse
import os
import pathlib
import sys
import time

import torch

from Core.OSC.dynamics import HotLoopFused  # HotLoopGraph must exist
from Core.OSC.Exec.graph_cache import GraphBucketManager
from Core.OSC.Exec.profiles import ProfileConfig, apply_profile, resolve_profile
from Core.OSC.icnn import ICNNDirectGrad

# Ensure repo root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Module-global toggle set by main() for fused runner
USE_GRAPH: bool = False

# -------------------------- Utilities --------------------------

def percentiles(xs):
    xs = sorted(xs)
    def p(q):
        if not xs:
            return float('nan')
        k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[k]
    return {"p50": p(0.50), "p95": p(0.95), "p99": p(0.99)}

def timer_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def echo_profile(tag: str, prof: ProfileConfig, icnn_ws: torch.dtype, k_ws: torch.dtype):
    print(f"[{tag}] dtype={prof.main_dtype}, ICNN.ws={icnn_ws}, K.ws={k_ws}, "
          f"TF32={torch.backends.cuda.matmul.allow_tf32}, "
          f"Deterministic={torch.are_deterministic_algorithms_enabled()}")


# --------------------------- Runners ---------------------------

@torch.inference_mode()
def run_grad_only(d, m, B, iters, dtype, icnn_ws):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=d, m=m, dtype=dtype, device=dev, ws_dtype=icnn_ws)
    x = torch.randn(B, d, device=dev, dtype=dtype)

    # warmup
    for _ in range(20):
        _ = icnn.grad(x)
    timer_sync()

    times = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t = time.perf_counter()
        _ = icnn.grad(x)
        timer_sync()
        times.append(time.perf_counter() - t)
    total = time.perf_counter() - t0
    return {"calls_per_s": iters / total, "latency_s": percentiles(times)}

@torch.inference_mode()
def run_unfused_ref(d, r, m, B, S, iters, dtype, icnn_ws):
    """
    Reference unfused step:
      tmp = x @ KV;  x_half = x + 0.5*dt*(tmp @ KUᵀ);  x = x_half - dt*grad(x_half)
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=d, m=m, dtype=dtype, device=dev, ws_dtype=icnn_ws)
    KU = torch.randn(d, r, device=dev, dtype=dtype) * 0.02
    KV = torch.randn(d, r, device=dev, dtype=dtype) * 0.02
    x = torch.randn(B, d, device=dev, dtype=dtype)

    def step_once(x, dt=0.02):
        tmp_r = x @ KV
        x_half = x + 0.5 * dt * (tmp_r @ KU.t())
        g = icnn.grad(x_half)
        return x_half - dt * g

    # warmup
    for _ in range(5):
        for _ in range(S):
            x = step_once(x)
    timer_sync()

    per_step = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t = time.perf_counter()
        for _ in range(S):
            x = step_once(x)
        timer_sync()
        per_step.append((time.perf_counter() - t) / S)
    total = time.perf_counter() - t0
    steps = iters * S
    return {"steps_per_s": steps / total, "latency_s": percentiles(per_step)}

@torch.inference_mode()
def run_fused(d, r, m, B, S, iters, dtype, icnn_ws, k_ws, prof: ProfileConfig):
    """
    Fused operator-split step using HotLoopFused; optionally CUDA-graph captured.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=d, m=m, dtype=dtype, device=dev, ws_dtype=icnn_ws)
    KU = torch.randn(d, r, device=dev, dtype=dtype) * 0.02
    KV = torch.randn(d, r, device=dev, dtype=dtype) * 0.02
    fused = HotLoopFused(d=d, icnn=icnn, KU=KU, KV=KV, device=dev, dtype=dtype, ws_dtype=k_ws)

    x = torch.randn(B, d, device=dev, dtype=dtype)

    # warmup to stabilize kernels & memory pools
    for _ in range(5):
        x = fused.step_unrolled(x, S=S, dt=0.02)
    timer_sync()

    per_step = []
    t0 = time.perf_counter()

    if USE_GRAPH and torch.cuda.is_available():
        gman = GraphBucketManager(capacity=8)
        g = gman.get(fused, B=B, S=S, dt=0.02, dtype=dtype, prof=prof)
        _ = g.run(x)  # warm replay
        timer_sync()

        for _ in range(iters):
            t = time.perf_counter()
            x = g.run(x)
            timer_sync()
            per_step.append((time.perf_counter() - t) / S)
    else:
        for _ in range(iters):
            t = time.perf_counter()
            x = fused.step_unrolled(x, S=S, dt=0.02)
            timer_sync()
            per_step.append((time.perf_counter() - t) / S)

    total = time.perf_counter() - t0
    steps = iters * S
    return {"steps_per_s": steps / total, "latency_s": percentiles(per_step)}


# ---------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--mode", type=str, default="fused", choices=["grad_only", "unfused", "fused"])
    ap.add_argument("--profile", type=str, default="AUTO", choices=["AUTO","FAST","BALANCED","STRICT"])
    ap.add_argument("--graph", action="store_true", help="Use CUDA graph for fused mode")
    ap.add_argument("--metrics-file", type=str, default="", help="Write Prometheus textfile metrics here")
    ap.add_argument("--echo-profile", action="store_true", help="Print resolved profile settings")
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    args = ap.parse_args()

    global USE_GRAPH
    USE_GRAPH = bool(args.graph or os.getenv("NOESIS_USE_CUDA_GRAPH") == "1")

    # Single source of truth: resolve+apply profile (AUTO allowed)
    prof = apply_profile(resolve_profile(args.profile))

    # For STRICT determinism, seed if you want bitwise repeatability across runs
    if prof.deterministic:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    # Disable autograd globally for bench
    torch.set_grad_enabled(False)

    # dtype & workspaces from profile (CLI --dtype is informative only)
    dtype = prof.main_dtype
    icnn_ws = prof.icnn_ws
    k_ws = prof.k_ws

    if args.echo_profile:
        echo_profile("ACTIVE", prof, icnn_ws, k_ws)

    if args.mode == "grad_only":
        res = run_grad_only(args.d, args.m, args.B, args.iters, dtype, icnn_ws)
        print("== ICNN grad-only ==")
        print(f"calls/s: {res['calls_per_s']:.2f}")
        print(f"latency per call (s): {res['latency_s']}")
        # Optional metrics for grad-only
        if args.metrics_file:
            os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
            with open(args.metrics_file, "w") as f:
                f.write(f'noesis_core_active_profile{{name="{prof.name}"}} 1\n')
                f.write(f'noesis_core_graph_used {1 if USE_GRAPH else 0}\n')
                f.write(f'noesis_core_calls_per_s {res["calls_per_s"]}\n')
                f.write(f'noesis_core_latency_p95_s {res["latency_s"]["p95"]}\n')
        return

    if args.mode == "unfused":
        res = run_unfused_ref(args.d, args.r, args.m, args.B, args.S, args.iters, dtype, icnn_ws)
        print("== Unfused reference ==")
        print(f"steps/s: {res['steps_per_s']:.2f}")
        print(f"per-step latency (s): {res['latency_s']}")
        # Optional metrics for unfused
        if args.metrics_file:
            os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
            with open(args.metrics_file, "w") as f:
                f.write(f'noesis_core_active_profile{{name="{prof.name}"}} 1\n')
                f.write(f'noesis_core_graph_used {1 if USE_GRAPH else 0}\n')
                f.write(f'noesis_core_steps_per_s {res["steps_per_s"]}\n')
                f.write(f'noesis_core_latency_p95_s {res["latency_s"]["p95"]}\n')
        return

    # fused
    res_f = run_fused(args.d, args.r, args.m, args.B, args.S, args.iters, dtype, icnn_ws, k_ws, prof)
    print("== Fused ==")
    print(f"steps/s: {res_f['steps_per_s']:.2f}")
    print(f"per-step latency (s): {res_f['latency_s']}")

    # unfused baseline (shorter run is fine)
    res_u = run_unfused_ref(args.d, args.r, args.m, args.B, args.S, max(50, args.iters // 3), dtype, icnn_ws)
    speedup = res_f["steps_per_s"] / max(1e-9, res_u["steps_per_s"])
    print("== Speedup vs unfused ==")
    print(f"speedup: {speedup:.2f}x")

    # ----- Metrics file (optional) -----
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            f.write("# HELP noesis_core_active_profile Active profile name\n")
            f.write("# TYPE noesis_core_active_profile gauge\n")
            f.write(f'noesis_core_active_profile{{name="{prof.name}"}} 1\n')

            f.write("# HELP noesis_core_graph_used Whether CUDA graph was used\n")
            f.write("# TYPE noesis_core_graph_used gauge\n")
            f.write(f'noesis_core_graph_used {1 if USE_GRAPH else 0}\n')

            f.write("# HELP noesis_core_steps_per_s Steps per second\n")
            f.write("# TYPE noesis_core_steps_per_s gauge\n")
            f.write(f'noesis_core_steps_per_s {res_f["steps_per_s"]}\n')

            f.write("# HELP noesis_core_latency_p95_s Per-step latency p95 in seconds\n")
            f.write("# TYPE noesis_core_latency_p95_s gauge\n")
            f.write(f'noesis_core_latency_p95_s {res_f["latency_s"]["p95"]}\n')

            f.write("# HELP noesis_core_speedup_vs_unfused Speedup against unfused\n")
            f.write("# TYPE noesis_core_speedup_vs_unfused gauge\n")
            f.write(f'noesis_core_speedup_vs_unfused {speedup}\n')


if __name__ == "__main__":
    main()
