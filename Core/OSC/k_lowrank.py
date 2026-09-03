# NOESIS-Σ Core — Low-rank conservative operator K (Golden Edition)
# ---------------------------------------------------------------
# This module implements a spectrally-capped low-rank operator
#     K = U diag(ω) U^T
# with an integrated projector Π_L and a small API surface:
#
#   - LowRankK(d, rank, lambda_cap, projector_radius, ...)
#   - forward(x, dt) == apply_and_project(x, dt)
#   - apply_step(x, dt): conservative step (no projection)
#   - apply_and_project(x, dt): conservative + Π_L in one call
#   - spectral_norm_estimate(...): offline safety/diagnostics
#
# The hot-path (apply/apply_and_project) has:
#   - no logging
#   - only batched matmuls + clamp, GPU-friendly and graph-safe.


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn


@dataclass(frozen=True)
class LowRankKConfig:
    """Configuration bundle for LowRankK."""

    d: int
    rank: int
    lambda_cap: float = 1.0
    projector_radius: Optional[float] = None
    enable_projector: bool = True
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


class LowRankK(nn.Module):
    """
    Low-rank conservative operator K with spectral cap and optional projector Π_L.

    Parameterisation
    ----------------
    We represent K as

        K = U diag(ω) U^T

    where U ∈ R^{d×r}, ω ∈ R^r. To enforce a spectral cap λ_max, we use

        ω = lambda_cap * tanh(ω_raw),

    which guarantees |ω_i| <= lambda_cap. If U's columns are approximately
    orthonormal (as they should be after training / SVD), this yields
    ||K||_2 ≈ max_i |ω_i| <= lambda_cap.

    Projector Π_L
    -------------
    We use a simple elementwise clamp projector:

        Π_L(y) = clamp(y, -L, L)

    which is cheap, GPU-friendly and easy to reason about. If projector_radius
    is None or <= 0, projection is a no-op.
    """

    def __init__(self, cfg: LowRankKConfig) -> None:
        super().__init__()

        if cfg.rank <= 0 or cfg.rank > cfg.d:
            raise ValueError(f"rank must be in [1, d], got rank={cfg.rank}, d={cfg.d}")
        if cfg.lambda_cap <= 0.0:
            raise ValueError(f"lambda_cap must be > 0, got {cfg.lambda_cap}")

        self.d = int(cfg.d)
        self.rank = int(cfg.rank)
        self.lambda_cap = float(cfg.lambda_cap)
        self.enable_projector = bool(cfg.enable_projector)
        self.projector_radius = (
            float(cfg.projector_radius) if cfg.projector_radius is not None else None
        )

        # U: [d, r]
        basis = torch.empty(self.d, self.rank, device=cfg.device, dtype=cfg.dtype)
        nn.init.orthogonal_(basis)
        self.basis = nn.Parameter(basis)

        # ω_raw: [r]  → ω = lambda_cap * tanh(ω_raw)
        omega_raw = torch.zeros(self.rank, device=cfg.device, dtype=cfg.dtype)
        self.omega_raw = nn.Parameter(omega_raw)

        # Bind a logger for non-hot-path diagnostics (never used in apply()).
        self._log = logger.bind(component="LowRankK")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def omega(self) -> Tensor:
        """
        Spectrally capped diagonal entries ω.

        |ω_i| <= lambda_cap via tanh reparameterisation.
        """
        return self.lambda_cap * torch.tanh(self.omega_raw)

    # ------------------------------------------------------------------
    # Hot-path operations (no logging, no conditionals beyond projector)
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, dt: float) -> Tensor:  # type: ignore[override]
        """
        Alias to apply_and_project for nn.Module compatibility.
        """
        return self.apply_and_project(x, dt)

    def apply_step(self, x: Tensor, dt: float) -> Tensor:
        """
        Conservative update:

            y = x + dt * (Kx),

        without projection. This is the matmul-heavy part, designed to be
        captured inside a CUDA graph and run in FP16/FP32 as needed.

        Args:
            x:  Tensor[..., d]
            dt: Time step (scalar float)

        Returns:
            Tensor of same shape as x.
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected x.shape[-1] == {self.d}, got {x.shape[-1]}")

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d)

        U = self.basis
        omega = self.omega  # [r]

        # tmp = x U              # [N, r]
        tmp = x_flat @ U

        # tmp = tmp * ω          # broadcast over columns
        tmp = tmp * omega

        # kx = tmp U^T           # [N, d]
        kx = tmp @ U.transpose(0, 1)

        y_flat = x_flat + dt * kx
        return y_flat.reshape(orig_shape)

    def project(self, y: Tensor) -> Tensor:
        """
        Apply Π_L to y if enabled.

        Π_L(y) = clamp(y, -L, L) with L = projector_radius.
        """
        if (not self.enable_projector) or (self.projector_radius is None):
            return y

        if self.projector_radius <= 0.0:
            return y

        L = self.projector_radius
        # Elementwise clamp is a single GPU kernel, graph-safe.
        return torch.clamp(y, -L, L)

    def apply_and_project(self, x: Tensor, dt: float) -> Tensor:
        """
        Combined conservative update + projector:

            y = Π_L(x + dt * Kx)

        This is the call you want on the conservative half-step.
        """
        y = self.apply_step(x, dt)
        return self.project(y)

    # ------------------------------------------------------------------
    # Offline diagnostics: spectral estimate (not on hot path)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def spectral_norm_estimate(
        self,
        n_power_iters: int = 8,
        batch_size: int = 1,
        seed: int = 0,
    ) -> float:
        """
        Estimate ||K||_2 via power iteration using the low-rank form.

        This is intended for tests / STRICT profile checks, not for
        per-step usage.

        Returns:
            float: spectral norm estimate.
        """
        device = self.basis.device
        dtype = self.basis.dtype

        if n_power_iters <= 0:
            n_power_iters = 1
        if batch_size <= 0:
            batch_size = 1

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        v = torch.randn(batch_size, self.d, device=device, dtype=dtype, generator=g)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)

        for _ in range(n_power_iters):
            v = self._apply_K_only(v)
            v_norm = v.norm(dim=-1, keepdim=True) + 1e-9
            v = v / v_norm

        Kv = self._apply_K_only(v)
        num = Kv.norm(dim=-1)
        den = v.norm(dim=-1) + 1e-9
        est = (num / den).max().item()

        self._log.debug(
            "LowRankK spectral_norm_estimate: est={} (lambda_cap={})",
            est,
            self.lambda_cap,
        )
        return float(est)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_K_only(self, x: Tensor) -> Tensor:
        """
        Apply Kx without residual, used only for spectral estimation.
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected x.shape[-1] == {self.d}, got {x.shape[-1]}")

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d)

        U = self.basis
        omega = self.omega

        tmp = x_flat @ U
        tmp = tmp * omega
        kx = tmp @ U.transpose(0, 1)

        return kx.reshape(orig_shape)

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, rank={self.rank}, "
            f"lambda_cap={self.lambda_cap}, "
            f"projector_radius={self.projector_radius}, "
            f"enable_projector={self.enable_projector}"
        )
