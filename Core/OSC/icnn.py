# Core/icnn.py
# ================================================================
# NOESIS-Σ — Production ICNN module (Golden Edition)
# Purpose:
#   - Direct-gradient ICNN (tape-free) for ultra-low-latency inference
#   - Hybrid potential: diag PSD + low-rank PSD + softplus stack + linear
#   - .grad(x) builds NO autograd tape (ICNNDirectGrad)
#   - Snapshot-friendly (state_dict / load_state_dict)
#   - Keeps a separate configurable ICNN (autograd) for training/compat only
# ================================================================

from __future__ import annotations

import typing as t
from typing import TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor: TypeAlias = torch.Tensor
__all__ = ["ICNNDirectGrad", "ICNN"]

# ----------------------------- utils -----------------------------


def _softplus_pos(x: Tensor, beta: float = 1.0, eps: float = 1e-6) -> Tensor:
    """Smooth strictly-positive mapping for parameters (>= eps)."""
    return F.softplus(x, beta=beta) + eps


# ----------------------- Direct-gradient ICNN -----------------------


class ICNNDirectGrad(nn.Module):
    r"""
    Direct-gradient ICNN potential with closed-form ∇φ (no autograd tape).

    Hybrid potential:
        φ(x) = Σ_i a_i * softplus((W x + b)_i)
             + 0.5 * ⟨D, x⊙x⟩
             + 0.5 * || Rᵀ x ||²
             + c·x,

        with a_i >= 0, D_j >= 0 (via softplus parameterization). R is unconstrained;
        the quadratic RRᵀ is PSD by construction.

    Gradient:
        ∇φ(x) = Wᵀ [ a ⊙ sigmoid(Wx + b) ] + (D ⊙ x) + R (Rᵀ x) + c
    """

    def __init__(
        self,
        d: int,
        m: int = 512,
        r: int = 0,  # low-rank factor columns; r=0 disables low-rank PSD term
        *,
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
        init_scale: float = 0.5,
        enforce_convex_runtime: bool = True,
        ws_dtype: torch.dtype | None = None,
        eps: float = 1e-6,
        c_min: float = 0.01,  # Strong convexity floor: ∇²Φ(x) ≥ c_min·I
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.m = int(m)
        self.r = int(r)
        self._main_dtype = dtype
        self._device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self._eps = float(eps)
        self._c_min = float(c_min)

        # ---- Parameters (kept in main dtype) ----
        # Softplus stack
        self.W = nn.Parameter(
            torch.empty(self.m, self.d, device=self._device, dtype=self._main_dtype)
        )
        self.b = nn.Parameter(
            torch.empty(self.m, device=self._device, dtype=self._main_dtype)
        )
        self.a_raw = nn.Parameter(
            torch.empty(self.m, device=self._device, dtype=self._main_dtype)
        )

        # Quadratic (diag PSD + optional low-rank PSD)
        self.D_raw = nn.Parameter(
            torch.empty(self.d, device=self._device, dtype=self._main_dtype)
        )
        self.R = (
            nn.Parameter(
                torch.empty(self.d, self.r, device=self._device, dtype=self._main_dtype)
            )
            if self.r > 0
            else None
        )

        # Linear term
        self.c = nn.Parameter(
            torch.empty(self.d, device=self._device, dtype=self._main_dtype)
        )

        self.reset_parameters(init_scale=init_scale)

        # ---- Workspace precision (internal math) ----
        # Default to FP32 for numerical stability; profiles may override.
        self._ws_dtype: torch.dtype = (
            ws_dtype if ws_dtype is not None else torch.float32
        )

        # ---- Runtime convexity guard (capture-safe) ----
        self._enforce_convex = bool(enforce_convex_runtime)

        # ---- Reusable buffers (allocated on first call per batch-shape) ----
        self._buf_Wx: Tensor | None = None  # [B, m]  : Wx + b
        self._buf_sig: Tensor | None = None  # [B, m]  : sigmoid(Wx+b)
        self._buf_asc: Tensor | None = None  # [B, m]  : a ⊙ sigmoid
        self._buf_Wt: Tensor | None = None  # [B, d]  : Wᵀ(...)
        self._buf_zr: Tensor | None = None  # [B, r]  : x @ R
        self._buf_RRt: Tensor | None = None  # [B, d]  : (xR) @ Rᵀ
        self._last_shapes: tuple[int, int, int] = (-1, -1, -1)  # (B, m, r)

    # ---------------- init & guards ----------------

    def reset_parameters(self, init_scale: float = 0.5) -> None:
        with torch.no_grad():
            # Softplus path
            nn.init.normal_(self.W, mean=0.0, std=init_scale / (self.d**0.5))
            nn.init.normal_(self.b, mean=0.0, std=0.02)
            nn.init.normal_(self.a_raw, mean=0.0, std=0.02)
            # Diagonal PSD (bias positive-ish after softplus)
            nn.init.normal_(self.D_raw, mean=0.0, std=0.02)
            self.D_raw.data.add_(2.0)
            # Low-rank PSD factor (small scale)
            if self.R is not None:
                nn.init.normal_(self.R, mean=0.0, std=0.02)
            # Linear term
            nn.init.normal_(self.c, mean=0.0, std=0.02)

    def _positive_a(self) -> Tensor:
        return _softplus_pos(self.a_raw, eps=self._eps)

    def _positive_D(self) -> Tensor:
        return torch.clamp(_softplus_pos(self.D_raw, eps=self._eps), min=self._c_min)

    @torch.inference_mode()
    def enforce_convexity_runtime(self) -> None:
        """
        Basic convexity safety: ensure no-NaN in positive-mapped params.
        Skips checks while CUDA is capturing (host sync not allowed then).
        """
        if not self._enforce_convex:
            return
        try:
            if torch.cuda.is_available() and hasattr(
                torch.cuda, "is_current_stream_capturing"
            ):
                if torch.cuda.is_current_stream_capturing():
                    return
        except Exception:
            return
        a = self._positive_a()
        D = self._positive_D()
        if torch.isnan(a).any() or torch.isnan(D).any():
            raise RuntimeError(
                "ICNNDirectGrad: NaN detected in convex parameters (a/D)."
            )

    # ---------------- hot-path analytic grad ----------------

    @torch.inference_mode()
    def grad(self, x: Tensor) -> Tensor:
        """
        Compute ∇φ(x) analytically, without building an autograd graph.
        x: [B, d] or [d]  -> returns [B, d], dtype = x.dtype
        """
        self.enforce_convexity_runtime()

        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert (
            x.shape[-1] == self.d
        ), f"ICNNDirectGrad.grad: expected last dim {self.d}, got {x.shape[-1]}"
        B = x.shape[0]
        dev = x.device

        # Cast inputs/params to workspace precision
        x_ws = x.to(self._ws_dtype)
        W_ws = self.W.to(self._ws_dtype)
        b_ws = self.b.to(self._ws_dtype)
        a_ws = self._positive_a().to(self._ws_dtype)
        D_ws = self._positive_D().to(self._ws_dtype)
        c_ws = self.c.to(self._ws_dtype)
        R_ws = self.R.to(self._ws_dtype) if self.R is not None else None

        # (Re)allocate buffers for current shapes (B, m, r)
        cur_sig = (B, self.m, self.r if R_ws is not None else 0)
        if self._last_shapes != cur_sig or self._buf_Wx is None:
            self._buf_Wx = torch.empty(
                B, self.m, device=dev, dtype=self._ws_dtype
            )  # Wx+b
            self._buf_sig = torch.empty(
                B, self.m, device=dev, dtype=self._ws_dtype
            )  # sigmoid
            self._buf_asc = torch.empty(
                B, self.m, device=dev, dtype=self._ws_dtype
            )  # a ⊙ sigmoid
            self._buf_Wt = torch.empty(
                B, self.d, device=dev, dtype=self._ws_dtype
            )  # Wᵀ(...)
            if R_ws is not None and self.r > 0:
                self._buf_zr = torch.empty(
                    B, self.r, device=dev, dtype=self._ws_dtype
                )  # xR
                self._buf_RRt = torch.empty(
                    B, self.d, device=dev, dtype=self._ws_dtype
                )  # (xR)Rᵀ
            else:
                self._buf_zr = None
                self._buf_RRt = None
            self._last_shapes = cur_sig

        # ---- softplus stack contribution: Wᵀ [ a ⊙ sigmoid(Wx + b) ] ----
        buf_Wx = cast(Tensor, self._buf_Wx)
        buf_sig = cast(Tensor, self._buf_sig)
        buf_asc = cast(Tensor, self._buf_asc)
        buf_Wt = cast(Tensor, self._buf_Wt)
        # buf_Wx = Wx + b
        torch.matmul(x_ws, W_ws.t(), out=buf_Wx)
        buf_Wx.add_(b_ws)

        # buf_sig = sigmoid(buf_Wx)
        torch.sigmoid(buf_Wx, out=buf_sig)

        # buf_asc = a ⊙ buf_sig
        torch.mul(buf_sig, a_ws, out=buf_asc)

        # buf_Wt = Wᵀ * buf_asc
        torch.matmul(buf_asc, W_ws, out=buf_Wt)

        # ---- diagonal PSD contribution: D ⊙ x ----
        grad_ws = buf_Wt
        grad_ws.add_(x_ws * D_ws)

        # ---- low-rank PSD contribution: R (Rᵀ x) = (xR)Rᵀ ----
        if R_ws is not None and self.r > 0:
            # zr = x @ R  -> [B, r]
            buf_zr = cast(Tensor, self._buf_zr)
            buf_RRt = cast(Tensor, self._buf_RRt)
            torch.matmul(x_ws, R_ws, out=buf_zr)
            # RRt = zr @ Rᵀ -> [B, d]
            torch.matmul(buf_zr, R_ws.t(), out=buf_RRt)
            grad_ws.add_(buf_RRt)

        # ---- linear term ----
        grad_ws.add_(c_ws)

        return grad_ws.to(x.dtype)

    # ---------------- parameter gradients (for Equilibrium Propagation) ----------------

    def param_grad_at(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute ∂φ/∂θ at state x analytically (no autograd tape).

        Returns a dict mapping parameter name → gradient tensor (same shape as param).
        Used by EquilibriumPropagation to compute free/nudged phase differences.

        x: [B, d] or [d] — state at which to evaluate parameter gradients.
        Gradients are averaged over the batch dimension B.

        R is only included when self.r > 0 (i.e. self.R is not None).
        """
        with torch.no_grad():
            squeezed = x.ndim == 1
            if squeezed:
                x = x.unsqueeze(0)
            B = x.shape[0]

            x_ws  = x.to(self._ws_dtype)
            W_ws  = self.W.to(self._ws_dtype)
            b_ws  = self.b.to(self._ws_dtype)
            a_raw_ws = self.a_raw.to(self._ws_dtype)
            D_raw_ws = self.D_raw.to(self._ws_dtype)
            c_ws  = self.c.to(self._ws_dtype)

            # z = Wx + b  [B, m]
            z = torch.matmul(x_ws, W_ws.t()) + b_ws

            # sigmoid(z)  [B, m]
            sig_z = torch.sigmoid(z)

            # a = softplus(a_raw) + eps  [m]
            a_ws = self._positive_a().to(self._ws_dtype)

            # ∂φ/∂W_ij = a_i · σ(z_i) · x_j  → mean over B → [m, d]
            # a_sig = a ⊙ σ(z)  [B, m]
            a_sig = a_ws.unsqueeze(0) * sig_z
            # outer product per sample, averaged: (a_sig)^T @ x / B  [m, d]
            grad_W = torch.matmul(a_sig.t(), x_ws) / B

            # ∂φ/∂b_i = a_i · σ(z_i)  → mean over B → [m]
            grad_b = a_sig.mean(dim=0)

            # ∂φ/∂a_raw_i = softplus(z_i) · σ(a_raw_i) (chain rule)
            # softplus(z)  [B, m], σ(a_raw) [m]
            sp_z = F.softplus(z, beta=1.0)
            sig_a_raw = torch.sigmoid(a_raw_ws)
            # mean over B  [m]
            grad_a_raw = (sp_z.mean(dim=0)) * sig_a_raw

            # ∂φ/∂D_raw_j = 0.5 · x_j² · σ(D_raw_j)  [d]
            sig_D_raw = torch.sigmoid(D_raw_ws)
            grad_D_raw = 0.5 * (x_ws * x_ws).mean(dim=0) * sig_D_raw

            # ∂φ/∂c_j = x_j  [d]
            grad_c = x_ws.mean(dim=0)

            out: dict[str, Tensor] = {
                "W":     grad_W.to(self.W.dtype),
                "b":     grad_b.to(self.b.dtype),
                "a_raw": grad_a_raw.to(self.a_raw.dtype),
                "D_raw": grad_D_raw.to(self.D_raw.dtype),
                "c":     grad_c.to(self.c.dtype),
            }

            # ∂φ/∂R_jk = x_j · (Rᵀx)_k  → mean over B → [d, r]
            if self.R is not None and self.r > 0:
                R_ws = self.R.to(self._ws_dtype)
                zr = torch.matmul(x_ws, R_ws)          # [B, r]
                grad_R = torch.matmul(x_ws.t(), zr) / B  # [d, r]
                out["R"] = grad_R.to(self.R.dtype)

            return out

    # ---------------- value (kept autograd-connected for training if needed) ----------------

    def forward(self, x: Tensor) -> Tensor:
        """
        φ(x) value computed in workspace precision for stability,
        returned in x.dtype (autograd-connected for optional training/compat).
        """
        squeezed = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True

        x_ws = x.to(self._ws_dtype)
        W_ws = self.W.to(self._ws_dtype)
        b_ws = self.b.to(self._ws_dtype)
        a_ws = self._positive_a().to(self._ws_dtype)
        D_ws = self._positive_D().to(self._ws_dtype)
        c_ws = self.c.to(self._ws_dtype)
        R_ws = self.R.to(self._ws_dtype) if self.R is not None else None

        # softplus stack term
        Wxb = F.linear(x_ws, W_ws, b_ws)  # [B, m]
        term1 = torch.sum(a_ws * F.softplus(Wxb, beta=1.0), dim=-1)

        # diag PSD term
        term2 = 0.5 * torch.sum(D_ws * (x_ws * x_ws), dim=-1)

        # low-rank PSD term: 0.5 * || Rᵀ x ||²
        if R_ws is not None and self.r > 0:
            zr = torch.matmul(x_ws, R_ws)  # [B, r]
            term3 = 0.5 * torch.sum(zr * zr, dim=-1)
        else:
            term3 = torch.zeros_like(term2)

        # linear term
        term4 = torch.sum(c_ws * x_ws, dim=-1)

        out = term1 + term2 + term3 + term4
        out = out.to(x.dtype)
        return out.squeeze(0) if squeezed else out

    # ---- utilities / admin ----
    def set_workspace_dtype(self, ws_dtype: torch.dtype) -> None:
        self._ws_dtype = ws_dtype

    # Snapshot helpers
    def export_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def load_state(self, state: dict) -> None:
        self.load_state_dict(state)


# -------------------- Configurable multi-layer ICNN (training/compat) --------------------


class ICNN(nn.Module):
    """
    Configurable ICNN for training/compat (its .grad uses autograd; NOT for the hot loop).
    Kept to satisfy existing tests and for off-line training only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: t.Sequence[int] = (128, 64),
        nonneg_softplus: bool = True,
        bias: bool = True,
        activation: t.Callable[[Tensor], Tensor] = F.relu,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive integer")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty list of ints")

        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.nonneg_softplus = bool(nonneg_softplus)
        self.bias = bool(bias)
        self.activation = activation
        self.num_layers = len(self.hidden_sizes)

        # U: input→hidden (free); W_raw: hidden→hidden (mapped to ≥0); B: biases
        self.U = nn.ModuleList()
        self.W_raw = nn.ParameterList()
        self.B = nn.ParameterList()

        # First layer
        self.U.append(nn.Linear(self.input_dim, self.hidden_sizes[0], bias=False))

        for i in range(self.num_layers):
            if i > 0:
                w = nn.Parameter(
                    torch.randn(self.hidden_sizes[i], self.hidden_sizes[i - 1]) * 1e-3
                )
                self.W_raw.append(w)
            else:
                w0 = nn.Parameter(
                    torch.zeros(self.hidden_sizes[0], self.hidden_sizes[0]),
                    requires_grad=False,
                )
                self.W_raw.append(w0)

            if i > 0:
                self.U.append(
                    nn.Linear(self.input_dim, self.hidden_sizes[i], bias=False)
                )

            b = (
                nn.Parameter(torch.zeros(self.hidden_sizes[i]))
                if self.bias
                else nn.Parameter(
                    torch.zeros(self.hidden_sizes[i]), requires_grad=False
                )
            )
            self.B.append(b)

        # Final output linear (non-negative for convexity)
        self.final_raw = nn.Parameter(torch.randn(1, self.hidden_sizes[-1]) * 1e-3)
        self.final_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for lin in self.U:
            if isinstance(lin, nn.Linear):
                nn.init.uniform_(lin.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.final_raw, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.final_bias)

    def _positive(self, raw: Tensor) -> Tensor:
        return F.softplus(raw, beta=1.0, threshold=20.0)

    def _get_W(self, i: int) -> Tensor:
        raw = self.W_raw[i]
        return self._positive(raw) if self.nonneg_softplus else raw

    def _get_final_weight(self) -> Tensor:
        return (
            self._positive(self.final_raw) if self.nonneg_softplus else self.final_raw
        )

    def forward(self, x: Tensor) -> Tensor:
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input has wrong dim {x.shape}; expected second dim {self.input_dim}"
            )

        dev = next(self.parameters()).device
        x = x.to(device=dev)

        z = self.activation(self.U[0](x) + self.B[0].view(1, -1))
        for i in range(1, self.num_layers):
            W = self._get_W(i).to(device=dev, dtype=z.dtype)
            z = self.activation(
                torch.matmul(z, W.T) + self.U[i](x) + self.B[i].view(1, -1)
            )

        final_w = self._get_final_weight().to(device=dev, dtype=z.dtype)  # [1, last]
        phi = torch.matmul(z, final_w.T).squeeze(-1) + self.final_bias.to(
            device=dev, dtype=z.dtype
        )
        return phi if not squeezed else phi.squeeze(0)

    def grad(self, x: Tensor) -> Tensor:
        """Autograd-based gradient (NOT for the hot loop).
        Safe under global no_grad/inference_mode for CPU tests by re-enabling grad."""
        was_1d = False
        with torch.enable_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
                was_1d = True
            dev = next(self.parameters()).device
            x = x.to(device=dev).detach().requires_grad_(True)
            phi = self.forward(x)
            total = phi.sum()
            grads = torch.autograd.grad(
                total, x, create_graph=False, retain_graph=False, allow_unused=False
            )[0]
        grads = grads.detach()
        return grads.squeeze(0) if was_1d else grads

    # Snapshot helpers
    def export_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def load_state(self, state: dict) -> None:
        self.load_state_dict(state)

    def project_nonnegativity(self) -> None:
        # Kept for API completeness; softplus already enforces ≥0.
        return

    def __repr__(self) -> str:
        return f"ICNN(input_dim={self.input_dim}, hidden_sizes={self.hidden_sizes}, nonneg_softplus={self.nonneg_softplus})"
