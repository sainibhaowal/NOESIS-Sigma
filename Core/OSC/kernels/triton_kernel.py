# Core/kernels/triton_kernel.py
# NOESIS-Σ — Triton fast-path for batched matvec (Golden Edition)
# Provides:
#   apply_K_dense(K, x, *, enable_triton=True)   -> y
#   apply_K_lowrank(U, V, x, *, enable_triton=True) -> y
#
# Shapes:
#   K: [m, n], x: [B, n] -> y: [B, m]
#   U: [m, r], V: [r, n], x: [B, n] -> y: [B, m]
#
# Notes:
#   - Computes in fp32 accumulators; returns x.dtype by default.
#   - Falls back to torch.matmul if Triton not present or disabled.
#   - Includes a micro-bench harness (python -m Core.kernels.triton_kernel).

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

try:  # Triton is optional
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

# ------------------------------
# Triton dense batched matvec
# ------------------------------
# Computes for each batch b: y[b, :] = K @ x[b, :]


@triton.jit
def _dense_mv_kernel(
    K_ptr,
    X_ptr,
    Y_ptr,
    M,
    N,
    stride_k_m,
    stride_k_n,
    stride_x_b,
    stride_x_n,
    stride_y_b,
    stride_y_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row tile id
    pid_b = tl.program_id(1)  # batch id

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_rows = rows < M

    # Pointers to batch slice
    K_batch = K_ptr  # K has no batch dim
    X_batch = X_ptr + pid_b * stride_x_b
    Y_batch = Y_ptr + pid_b * stride_y_b

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop across columns in tiles of BLOCK_N
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask_cols = cols < N

        # Load K tile [BLOCK_M, BLOCK_N]
        k_tile = tl.load(
            K_batch + rows[:, None] * stride_k_m + cols[None, :] * stride_k_n,
            mask=mask_rows[:, None] & mask_cols[None, :],
            other=0.0,
        )
        # Load x slice [BLOCK_N]
        x_slice = tl.load(
            X_batch + cols * stride_x_n,
            mask=mask_cols,
            other=0.0,
        ).to(tl.float32)

        # acc += sum_j K[i, j] * x[j]
        acc += tl.sum(k_tile.to(tl.float32) * x_slice[None, :], axis=1)

    # Store results
    tl.store(
        Y_batch + rows * stride_y_m,
        acc,
        mask=mask_rows,
    )


def _launch_dense_mv_triton(
    K: torch.Tensor, X: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """
    K: [m, n] (contiguous), X: [B, n] (contiguous), returns Y: [B, m]
    Accumulates in fp32, outputs in out_dtype.
    """
    assert K.ndim == 2 and X.ndim == 2, "K [m,n], X [B,n]"
    m, n = K.shape
    B = X.shape[0]
    device = X.device
    Y = torch.empty((B, m), device=device, dtype=torch.float32)

    # Choose reasonable tile sizes (can autotune later)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(m, BLOCK_M), B)

    _dense_mv_kernel[grid](
        K,
        X,
        Y,
        m,
        n,
        K.stride(0),
        K.stride(1),
        X.stride(0),
        X.stride(1),
        Y.stride(0),
        Y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return Y.to(out_dtype)


# ------------------------------
# Public API
# ------------------------------


def apply_K_dense(
    K: torch.Tensor, x: torch.Tensor, *, enable_triton: bool = True
) -> torch.Tensor:
    """
    Batched dense matvec: y = K @ x for each batch row in x.
    K: [m, n], x: [B, n] -> y: [B, m]
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    assert K.ndim == 2 and x.ndim == 2, "K [m,n], x [B,n]"
    m, n = K.shape
    B, n2 = x.shape
    if n2 != n:
        raise ValueError(f"x dim {x.shape} not compatible with K {K.shape}")

    # Device & layout
    device = x.device
    dtype_out = x.dtype
    Kc = K.contiguous().to(device=device)
    Xc = x.contiguous()

    use_triton = bool(enable_triton and _TRITON_OK and device.type == "cuda")

    if use_triton:
        y = _launch_dense_mv_triton(Kc, Xc, out_dtype=dtype_out)
    else:
        # fallback: torch
        y = torch.matmul(Xc, Kc.transpose(0, 1)).to(dtype_out)

    return y.squeeze(0) if squeeze else y


def apply_K_lowrank(
    U: torch.Tensor, V: torch.Tensor, x: torch.Tensor, *, enable_triton: bool = True
) -> torch.Tensor:
    """
    Low-rank K = U @ V:
      U: [m, r], V: [r, n], x: [B, n] -> y: [B, m]
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    m, r = U.shape
    r2, n = V.shape
    B, n2 = x.shape
    if r2 != r or n2 != n:
        raise ValueError(f"Incompatible shapes: U={U.shape}, V={V.shape}, x={x.shape}")

    # temp = V @ x  => [B, r]
    temp = apply_K_dense(
        V, x, enable_triton=enable_triton
    )  # note: K is [r,n], so matvec OK
    # y = U @ temp => [B, m]
    y = apply_K_dense(U, temp, enable_triton=enable_triton)

    return y.squeeze(0) if squeeze else y


# ------------------------------
# Micro-bench / self-check
# ------------------------------
def _check():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, m, n, r = 32, 1024, 1024, 32
    K = torch.randn(m, n, device=device, dtype=torch.float32)
    U = torch.randn(m, r, device=device, dtype=torch.float32)
    V = torch.randn(r, n, device=device, dtype=torch.float32)
    x = torch.randn(B, n, device=device, dtype=torch.float32)

    y_ref = x @ K.t()
    y = apply_K_dense(K, x, enable_triton=True)
    err = (y - y_ref).abs().max().item()
    print(f"[dense] max abs err = {err:.3e}")

    ylr_ref = (x @ V.t()) @ U.t()
    ylr = apply_K_lowrank(U, V, x, enable_triton=True)
    err2 = (ylr - ylr_ref).abs().max().item()
    print(f"[lowrank] max abs err = {err2:.3e}")


def _bench():
    import time

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, m, n, r = 64, 2048, 2048, 64
    K = torch.randn(
        m, n, device=device, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    U = torch.randn(m, r, device=device, dtype=K.dtype)
    V = torch.randn(r, n, device=device, dtype=K.dtype)
    x = torch.randn(B, n, device=device, dtype=K.dtype)

    def bench(fn, label):
        # warmup
        for _ in range(3):
            _ = fn()
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        iters = 30
        for _ in range(iters):
            _ = fn()
        torch.cuda.synchronize() if device == "cuda" else None
        dt = (time.time() - t0) / iters
        print(f"{label}: {dt*1e3:.2f} ms / iter")

    bench(lambda: apply_K_dense(K, x, enable_triton=False), "dense[torch]")
    bench(lambda: apply_K_dense(K, x, enable_triton=True), "dense[triton]")
    bench(lambda: apply_K_lowrank(U, V, x, enable_triton=False), "lowrank[torch]")
    bench(lambda: apply_K_lowrank(U, V, x, enable_triton=True), "lowrank[triton]")


if __name__ == "__main__":
    _check()
    _bench()
