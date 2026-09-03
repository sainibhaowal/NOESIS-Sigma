NOESIS-Σ Kernel Guide (Triton/CUDA Integration)

This document explains how to replace the PyTorch fallback kernels with Triton/CUDA implementations while keeping API and numerical behavior identical to the engine’s expectations.

TL;DR

You must implement exactly these two functions (batch-first, right-multiply):

apply_K_dense(K: [d,d], x: [B,d]) -> [B,d] computes x @ K

apply_K_lowrank(U: [d,r], V: [d,r], x: [B,d]) -> [B,d] computes (x @ U) @ V.T

Keep shape/device/dtype semantics identical to the PyTorch fallback.

Pass all equivalence tests (within tolerances) on CPU (reference) and GPU (optimized).

Use the included bench harness (python -m Core.kernels.placeholder …) to profile.

Here’s a complete, production-grade `Core/kernels/README.md`. Drop it at `NOESIS-Σ/Core/kernels/README.md`.

---

# NOESIS-Σ Core Kernels — Developer Guide

This guide explains how to **replace the PyTorch fallbacks** in `Core/kernels/placeholder.py` with **Triton or CUDA** kernels while preserving **API, shapes, determinism, and tests**.

## 1) API Contract (MUST keep exactly)

Batch-first, right-multiply convention. `x ∈ R^{B×d}`, `K ∈ R^{d×d}`, `U,V ∈ R^{d×r}` with `K ≈ U @ Vᵀ`.

```python
# Low-rank path
def apply_K_lowrank(U: Tensor, V: Tensor, x: Tensor) -> Tensor:
    """
    Args:  U:[d,r], V:[d,r], x:[B,d] or [d]
    Returns: y:[B,d] or [d]   # same batch shape as x
    Semantics: y = (x @ U) @ V.T
    """

# Dense path
def apply_K_dense(K: Tensor, x: Tensor) -> Tensor:
    """
    Args:  K:[d,d], x:[B,d] or [d]
    Returns: y:[B,d] or [d]
    Semantics: y = x @ K
    """
```

**Batching:** If `x` is 1-D `[d]`, return `[d]`. If `[B,d]`, return `[B,d]`.
**Device & dtype:** Cast kernel weights (`K` or `U,V`) to `x.device` and `x.dtype`.
**Errors:** Raise `ValueError` on shape mismatch. No silent broadcasting.

## 2) Shapes & dtype rules

* `x`: `float16|bfloat16|float32|float64` (engine typically uses `float32`; mixed precision allowed).
* Accumulation: Prefer **FP32 accumulation** internally for FP16/BF16 (see §6).
* Contiguity: Require `x.is_contiguous()`; if not, **make contiguous** inside kernel path.
* No RNG; kernel ops must be **deterministic** given the same inputs (see §7).

## 3) Numerical equivalence (acceptance)

Your kernels must match the PyTorch fallbacks within:

* `float32`: `rtol=1e-5`, `atol=1e-6`
* `float16|bfloat16`: `rtol=2e-3`, `atol=2e-3`
* `float64`: `rtol=1e-7`, `atol=1e-8`

Equivalence is tested against `placeholder.apply_K_*` using randomized inputs and multiple sizes.

## 4) Integration options

**Option A (recommended):** Implement in `Core/kernels/triton_kernel.py` (or `cuda_kernel.py`) **with the same two functions**, then in `Core/dynamics.py` swap the import:

```python
# from Core.kernels.placeholder import apply_K_lowrank, apply_K_dense
from Core.kernels.triton_kernel import apply_K_lowrank, apply_K_dense
```

**Option B (runtime switch):** Add a small factory that imports a module path from `NOESIS_KERNEL_IMPL` env var and binds its `apply_K_*` into the engine.

## 5) Test plan (must pass)

Create/extend tests in `Tests/core/test_kernels_placeholder.py`:

* **Parity tests** (fast/PR): small `B,d,r` on CPU and (if available) CUDA; compare Triton/CUDA vs placeholder.
* **Large tests** (nightly): bigger `B,d,r` (e.g., `B=1024,d=4096,r=64`) on GPU.
* **Dtype sweep**: `float32` (always), plus `float16` or `bfloat16` if CUDA is available.
* **1-D x case**: verify `[d] → [d]` path.
* **Non-contiguous x**: transpose once, then ensure kernel handles `x.contiguous()` internally.

Example (add alongside existing tests):

```python
import torch, pytest
from Core.kernels import placeholder as ref
from Core.kernels.triton_kernel import apply_K_lowrank, apply_K_dense  # your impl

@pytest.mark.fast
def test_parity_small_cpu():
    B,d,r = 4, 32, 8
    x = torch.randn(B,d)
    U = torch.randn(d,r); V = torch.randn(d,r)
    K = U @ V.T
    y_ref_lr = ref.apply_K_lowrank(U,V,x)
    y_ref_dn = ref.apply_K_dense(K,x)
    y_lr = apply_K_lowrank(U,V,x)
    y_dn = apply_K_dense(K,x)
    assert torch.allclose(y_lr, y_ref_lr, rtol=1e-5, atol=1e-6)
    assert torch.allclose(y_dn, y_ref_dn, rtol=1e-5, atol=1e-6)
```

Mark heavier tests with `@pytest.mark.nightly`.

## 6) Mixed precision policy

When `x.dtype` is `float16`/`bfloat16`:

* **Accumulate in FP32**, then cast to `x.dtype` for the return value.
* In Triton/CUDA, explicitly upcast to `float32` before `dot/matmul` if needed.
* Avoid overflow: scale inputs only if you prove a benefit; otherwise rely on FP32 accumulation.

## 7) Determinism

* Do not use stochastic kernels.
* Avoid atomics with unordered reduction unless the order is fixed (or numerical drift stays within tolerances).
* When targeting CUDA, prefer **cublasLt** deterministic modes or custom kernels with fixed reduction order.

## 8) Triton implementation notes (sketch)

For **low-rank**: compute `tmp = x @ U` then `y = tmp @ Vᵀ`. You can fuse into one kernel that:

1. Loads a tile of `x` and `U`, produces a tile of `tmp`.
2. Immediately multiplies that tile by a tile of `Vᵀ` into `y`.
3. Writes `y` to memory.

Skeleton (illustrative only):

```python
# Core/kernels/triton_kernel.py
import torch, triton, triton.language as tl

def apply_K_lowrank(U, V, x):
    # validate shapes/devices/dtypes; ensure contiguous
    # dispatch to one or two Triton kernels (XU, then tmpVT) or a fused kernel
    # fall back to PyTorch if device!=cuda
    raise NotImplementedError  # replace with your kernels

def apply_K_dense(K, x):
    # either call cublas (torch.mm) if that’s fast enough, or write custom Triton matmul
    raise NotImplementedError
```

**Tiling suggestions:**

* Blocks: `BM` in batch (`B` rows), `BN` in features (`d` cols), `BK` in inner dimension (`r` or `d`).
* Choose `BM=64/128`, `BN=64/128`, `BK=32/64` based on GPU SM and shared memory.
* Ensure coalesced loads: store matrices row-major, align `ld` strides.
* Use `num_warps`/`num_stages` tuned per device.

## 9) CUDA/C++ alternative

If you prefer CUDA/C++ extensions:

* Use `torch.utils.cpp_extension.load()` to JIT build.
* Implement kernels for `x·U` and `(xU)·Vᵀ` with **row-major** assumptions.
* Provide Python bindings `apply_K_lowrank/apply_K_dense` that mirror the signatures.

## 10) Performance targets (guidance)

* **Low-rank:** aim for ≥ **2–5×** speedup vs fallback for typical `B=1k..4k, d=2k..8k, r=32..128`.
* **Dense:** usually cuBLAS is excellent; you may simply call `torch.mm` (which uses cuBLAS). Custom kernels make sense if you can fuse pre/post ops or exploit structure.

Use the built-in bench in `placeholder.py`:

```bash
python -m Core.kernels.placeholder --d 4096 --r 64 --B 1024 --device cuda
```

Create a similar bench for your kernels and compare.

## 11) Memory & layout

* Expect row-major (`contiguous`) inputs; if not contiguous, make a contiguous copy at the top.
* Keep intermediate buffers (`tmp = x@U`) on the same device/dtype policy.
* Reuse buffers when called repeatedly to reduce allocations (optional optimization).

## 12) Error handling

* Validate shapes; raise `ValueError` with clear messages.
* If device is not CUDA and your implementation requires CUDA, **fall back** to placeholder or raise a precise error (decide one policy; we recommend **fallback** for portability).

## 13) CI & markers

* **PR job:** runs `-m fast` (small sizes CPU and optional CUDA).
* **Nightly job:** runs `-m nightly` (large sizes, CUDA, dtype sweep).
* Skip GPU tests gracefully if CUDA not available.

## 14) Checklist (copy for PRs)

* [ ] Functions implemented: `apply_K_lowrank`, `apply_K_dense` (signatures unchanged).
* [ ] Shape/device/dtype validation & casts.
* [ ] FP32 accumulation for FP16/BF16.
* [ ] Deterministic math (no randomness).
* [ ] Parity tests vs placeholder pass with stated tolerances.
* [ ] Bench numbers documented in PR description.
* [ ] CPU fallback or clear error when CUDA absent.
* [ ] Code documented (docstrings + comments on tiling/launch params).

---

**Where to ask questions:** put kernel-specific notes (SM occupancy, tiling choices, precision trade-offs) in this folder as additional markdown files (`NOTES_<gpu>.md`).
