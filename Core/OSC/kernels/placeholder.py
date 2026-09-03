# Core/kernels/placeholder.py
# NOESIS-Σ — Kernel interface + PyTorch fallbacks (Golden Edition)
#
# Purpose
# -------
# Provide stable, well-documented kernel entrypoints that the Core engine can call.
# These fallbacks are correct and reasonably fast in pure PyTorch, and the file
# doubles as an interface contract for future Triton/CUDA replacements.
#
# Interface (batch-first, right-multiply)
# --------------------------------------
# Given x ∈ R^{B×d}, K ∈ R^{d×d}, U,V ∈ R^{d×r} with K ≈ U @ V^T:
#   y = apply_K_dense(K, x)     -> computes x @ K
#   y = apply_K_lowrank(U, V, x)-> computes x @ (U @ V^T) = (x @ U) @ V^T
#
# This convention matches a common pattern in NOESIS Core: row-batched states and
# right-multiplication to keep shapes [B, d] → [B, d].
#
# Triton/CUDA Replacement
# -----------------------
# You can drop-in optimized kernels by re-implementing these two functions with
# the exact same signatures and semantics. Maintain numerical parity within atol/rtol
# used in the tests (see Tests/core/test_kernels_placeholder.py).
#
# Bench Harness
# -------------
# Run:  python -m Core.kernels.placeholder --d 4096 --r 64 --B 1024 --device cuda
# Prints throughput (GB/s-ish proxy) and timing for both dense and low-rank paths.
#
# Golden-Edition notes
# --------------------
# • Strict shape/type checks with clear error messages
# • Device/dtype propagation follows input x
# • Deterministic math (no fused random ops)
# • No global state; safe to import under multiple processes/threads
# ------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

# ---------------------------
# Core public entrypoints
# ---------------------------


def apply_K_dense(K: Tensor, x: Tensor) -> Tensor:
    """
    Dense Kernel application (right-multiply).

    Args:
        K: [d, d] tensor (device/dtype agnostic). Will be cast/moved to x.device/dtype.
        x: [B, d] or [d] batch of row vectors.

    Returns:
        y: same batch shape as x, with last-dim d. Computes x @ K.

    Raises:
        ValueError on shape mismatch.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False

    if K.dim() != 2:
        raise ValueError(
            f"apply_K_dense: K must be 2D [d,d], got shape {tuple(K.shape)}"
        )
    if x.dim() != 2:
        raise ValueError(
            f"apply_K_dense: x must be 2D [B,d], got shape {tuple(x.shape)}"
        )
    B, d = x.shape
    if K.shape != (d, d):
        raise ValueError(
            f"apply_K_dense: K shape {tuple(K.shape)} incompatible with x {tuple(x.shape)}"
        )

    # Device/dtype align to x
    K = K.to(device=x.device, dtype=x.dtype)
    y = x @ K
    return y.squeeze(0) if squeezed else y


def apply_K_lowrank(U: Tensor, V: Tensor, x: Tensor) -> Tensor:
    """
    Low-rank Kernel application (right-multiply) with K ≈ U @ V^T.

    Args:
        U: [d, r]
        V: [d, r]
        x: [B, d] or [d] batch of row vectors.

    Returns:
        y: same batch shape as x, last dim d. Computes x @ (U @ V^T).
           Efficiently: (x @ U) @ V^T

    Raises:
        ValueError on shape mismatch.

    Notes:
        This mirrors the Core/dynamics `_apply_K` low-rank path with the convention K = U @ V^T.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeezed = True
    else:
        squeezed = False

    if U.dim() != 2 or V.dim() != 2:
        raise ValueError(
            f"apply_K_lowrank: U,V must be 2D, got U={tuple(U.shape)} V={tuple(V.shape)}"
        )
    if x.dim() != 2:
        raise ValueError(f"apply_K_lowrank: x must be 2D [B,d], got {tuple(x.shape)}")

    B, d = x.shape
    dU, rU = U.shape
    dV, rV = V.shape
    if not (dU == d and dV == d and rU == rV):
        raise ValueError(
            f"apply_K_lowrank: shapes incompatible: x={tuple(x.shape)} U={tuple(U.shape)} V={tuple(V.shape)}"
        )

    # Respect x device/dtype
    U = U.to(device=x.device, dtype=x.dtype)
    V = V.to(device=x.device, dtype=x.dtype)

    # (x @ U) -> [B, r], then @ V^T -> [B, d]
    tmp = x @ U
    y = tmp @ V.transpose(0, 1)
    return y.squeeze(0) if squeezed else y


# --------------------------------
# Bench harness (developer tool)
# --------------------------------


def _throughput_bytes(nbytes: int, seconds: float) -> float:
    # Return GB/s-like proxy
    if seconds <= 0:
        return float("inf")
    return (nbytes / seconds) / (1024**3)


@torch.no_grad()
def bench_apply_K(
    d: int = 4096,
    r: int = 64,
    B: int = 1024,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    warmup: int = 5,
    iters: int = 20,
) -> None:
    """
    Micro-benchmark both dense and low-rank paths on the chosen device/dtype.

    Prints:
      - Mean time per call
      - A crude GB/s proxy based on bytes moved (x, K/U/V, result)

    Usage:
      python -m Core.kernels.placeholder --d 4096 --r 64 --B 1024 --device cuda
    """
    dev = torch.device(device)
    x = torch.randn(B, d, device=dev, dtype=dtype)
    U = torch.randn(d, r, device=dev, dtype=dtype)
    V = torch.randn(d, r, device=dev, dtype=dtype)
    K = U @ V.T  # build dense equivalent

    # Warmups
    for _ in range(warmup):
        _ = apply_K_lowrank(U, V, x)
        _ = apply_K_dense(K, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Measure lowrank
    t0 = time.perf_counter()
    for _ in range(iters):
        y1 = apply_K_lowrank(U, V, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    lowrank_s = (t1 - t0) / iters

    # Measure dense
    t0 = time.perf_counter()
    for _ in range(iters):
        y2 = apply_K_dense(K, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    dense_s = (t1 - t0) / iters

    # Size accounting (very rough)
    nbytes_lowrank = (
        x.numel() * x.element_size()
        + U.numel() * U.element_size()
        + V.numel() * V.element_size()
    )
    nbytes_dense = x.numel() * x.element_size() + K.numel() * K.element_size()

    print("=== NOESIS-Σ Kernel Bench ===")
    print(f"device={dev}, dtype={dtype}, B={B}, d={d}, r={r}, iters={iters}")
    print(
        f"lowrank: {lowrank_s*1e3:.3f} ms/call | ~{_throughput_bytes(nbytes_lowrank, lowrank_s):.2f} GB/s proxy"
    )
    print(
        f"dense  : {dense_s*1e3:.3f} ms/call | ~{_throughput_bytes(nbytes_dense, dense_s):.2f} GB/s proxy"
    )
    # Correctness spot-check
    max_abs = (y1 - y2).abs().max().item()
    print(f"max |lowrank - dense| = {max_abs:.3e}")


# --------------------------------
# CLI
# --------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NOESIS-Σ Kernel placeholder bench")
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    return ap.parse_args()


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


if __name__ == "__main__":
    ns = _parse_args()
    bench_apply_K(
        d=ns.d,
        r=ns.r,
        B=ns.B,
        device=ns.device,
        dtype=_dtype_from_str(ns.dtype),
        iters=ns.iters,
        warmup=ns.warmup,
    )
