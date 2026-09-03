# ================================================================
#  NOESIS-Σ — Golden Edition
#  Module: Core/kernels/lowrank.py
#  Component: Low-rank Kᵀ application for batched hot-loop
#  Author: SephiRax Team
# ---------------------------------------------------------------
#  Contract:
#    We represent K ≈ KU @ KVᵀ with KU, KV ∈ ℝ^{d×r}.
#    We need x @ Kᵀ for row-major batches (x ∈ ℝ^{B×d}).
#    Since Kᵀ = KV @ KUᵀ:
#      x @ Kᵀ = (x @ KV) @ KUᵀ
#
#  Implementation:
#    - Two GEMMs with FP32 accumulations.
#    - Uses torch.addmm to fuse the second GEMM with bias/accum in hot loop.
#    - Optional Triton path could be added later (fallback is torch).
# ================================================================

from __future__ import annotations

from typing import Optional

import torch

_TRITON_AVAILABLE = False  # set True if you wire a custom triton kernel later


class LowRankK:
    """Stateful low-rank Kᵀ applier with reusable temporary buffers."""

    def __init__(
        self,
        KU: torch.Tensor,
        KV: torch.Tensor,
        *,
        ws_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        KU, KV: [d, r] (same device/dtype as model); no grads required at inference.
        ws_dtype: accumulation dtype.
        """
        assert KU.ndim == 2 and KV.ndim == 2 and KU.shape == KV.shape
        self.d, self.r = KU.shape
        self.KU = KU
        self.KV = KV
        self.ws_dtype = ws_dtype

        self._tmp_r: Optional[torch.Tensor] = None
        self._last_B: int = -1

    def _ensure_ws(self, B: int, device: torch.device) -> None:
        if self._last_B != B or self._tmp_r is None or self._tmp_r.device != device:
            self._tmp_r = torch.empty(B, self.r, device=device, dtype=self.ws_dtype)
            self._last_B = B

    @torch.inference_mode()
    def addmm_Kt(
        self, x: torch.Tensor, *, alpha: float, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute out = out + alpha * (x @ Kᵀ) with K low-rank.
        Shapes:
          x:   [B, d]
          out: [B, d] or None (if None, a new tensor is returned initialized to zeros)
        Returns: out
        """
        assert x.ndim == 2 and x.shape[1] == self.d
        B = x.shape[0]
        dev = x.device

        # Workspace
        self._ensure_ws(B, dev)
        assert self._tmp_r is not None

        # Promote to FP32 for math
        x32 = x.to(self.ws_dtype)
        KV32 = self.KV.to(self.ws_dtype)
        KU32 = self.KU.to(self.ws_dtype)

        # tmp_r = x @ KV  -> [B, r]
        torch.matmul(x32, KV32, out=self._tmp_r)

        # out = out + alpha * (tmp_r @ KUᵀ) via fused addmm
        if out is None:
            out = torch.zeros(B, self.d, device=dev, dtype=self.ws_dtype)
        torch.addmm(out, self._tmp_r, KU32.t(), out=out, beta=1.0, alpha=alpha)

        return out

    @torch.inference_mode()
    def addmm_Kt_base(
        self,
        x: torch.Tensor,  # [B, d]
        base: torch.Tensor,  # [B, d]   (used as beta*base + alpha*(x@Kᵀ))
        *,
        alpha: float,
        out: torch.Tensor,  # [B, d]   (destination buffer)
    ) -> torch.Tensor:
        """
        Compute out = base + alpha * (x @ Kᵀ) without making an explicit copy.
        Shapes: x/base/out are [B, d].
        """
        assert x.ndim == 2 and base.ndim == 2 and out.ndim == 2
        assert x.shape == base.shape == out.shape and x.shape[1] == self.d
        B = x.shape[0]
        dev = x.device
        self._ensure_ws(B, dev)
        assert self._tmp_r is not None
        x32 = x.to(self.ws_dtype)
        base32 = base.to(self.ws_dtype)
        KV32 = self.KV.to(self.ws_dtype)
        KU32 = self.KU.to(self.ws_dtype)
        # tmp_r = x @ KV
        torch.matmul(x32, KV32, out=self._tmp_r)
        # out = base + alpha * (tmp_r @ KUᵀ)
        torch.addmm(base32, self._tmp_r, KU32.t(), out=out, beta=1.0, alpha=alpha)
        return out
