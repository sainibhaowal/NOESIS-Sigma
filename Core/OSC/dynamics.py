# ---------------------------------------------------------------
# Core/dynamics.py
# NOESIS-Σ — Operator-Split Dynamics Engine (GOLDEN, PRODUCTION)
# Author: Generated for Ravinder Singh (SephiRax / NOESIS-Σ)
# Purpose: Deterministic operator-split core with full production hygiene:
#          - Conservative (skew) operator K
#          - Dissipative/implicit proximal step using ICNN.grad (or fallback)
#          - Projection & spectral cap
#          - Deterministic snapshotting (manifest + snapshot.bin + sig) with AES-GCM + Ed25519
#          - RNG save/restore (torch + cuda + numpy + python) for deterministic replay
#          - Batched inputs, shape contracts, concurrency safety
#          - Telemetry metric schema + simple TelemetryClient protocol
#          - Mixed precision safe wrappers + optional dynamic scaling (inference-friendly)
#          - JIT / profiling compile hooks (best-effort, non-invasive)
#          - Diagnostics (spectral estimator, condition estimate, projection counters)
#          - Watchdog & thermostat hook callback
#          - ICNN state_dict serialization support in snapshots
#
# Golden Edition conventions:
# - explicit typing and docstrings
# - deterministic-by-default mode with set_seed()
# - runtime config loader from .env supported (Runtime/Config/.env)
# - telemetry metric names defined in METRIC_* constants
#
# Security: Do NOT commit private keys. When enabling crypto, store keys in Runtime/Config (encrypted).
# -------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import io
import json

# Standard library
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, cast

if TYPE_CHECKING:
    from Core.OSC.Exec.graph_cache import GraphBucketManager

import numpy as np

# Third-party
import torch
from dotenv import load_dotenv
from loguru import logger

from Core.OSC.Control.thermostat import Thermostat, ThermostatConfig
from Core.OSC.Exec.profiles import resolve_profile
from Core.OSC.icnn import ICNNDirectGrad  # must provide .grad(x)
from Core.OSC.kernels.lowrank import LowRankK as KernelLowRankK
from Core.OSC.Utils.thermostat_telemetry import ThermostatStats

from .k_lowrank import LowRankK as CoreLowRankK
from .k_lowrank import LowRankKConfig

# ---------------------------
# Optional crypto
# ---------------------------
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# ---------------------------
# Optional receipts (best-effort)
# ---------------------------
try:
    from External.Output.receipts import create_receipt  # noqa: F401
except Exception:
    create_receipt = None  # type: ignore

# ---------------------------
# Optional Triton fast-path placeholders
# ---------------------------
_HAS_TRITON: bool = False
_apply_K_dense_triton: Optional[Callable[..., torch.Tensor]] = None
_apply_K_lowrank_triton: Optional[Callable[..., torch.Tensor]] = None
try:
    from Core.OSC.kernels.triton_kernel import (
        apply_K_dense as _apply_K_dense_triton,  # pragma: no cover
    )
    from Core.OSC.kernels.triton_kernel import (
        apply_K_lowrank as _apply_K_lowrank_triton,
    )

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
    _apply_K_dense_triton = None
    _apply_K_lowrank_triton = None

# ---------------------------
# Telemetry metric names (schema)
# ---------------------------
METRIC_STEP = "noesis.core.step"  # payload: {step_count, elapsed_ms, energy_before, energy_after, norm_after, projected, projection_count}
METRIC_SPECTRAL = "noesis.core.spectral"  # payload: {est, cap}
METRIC_NAN_EVENT = "noesis.core.nan_event"  # payload: {policy, trace_id}
METRIC_WATCHDOG = "noesis.core.watchdog"  # payload: {elapsed_ms, max_ms}
METRIC_PROFILING = "noesis.core.profile"  # payload: {fn, time_ms}


# ---------------------------
# Telemetry client minimal protocol
# ---------------------------
class TelemetryClient:
    """
    Minimal telemetry client you can swap with a real implementation.
    The record method receives a metric name and a JSON-serializable payload.
    Replace with Prometheus exporter or instrumentator caller in production.
    """

    def record(self, name: str, payload: dict) -> None:
        logger.debug("Telemetry record {}: {}", name, payload)


# ---------------------------
# Engine params
# ---------------------------
@dataclass
class EngineParams:
    """Parameters for OperatorSplitEngine. Use runtime loader to set from env if desired."""

    state_dim: int
    dt: float = 0.005
    max_norm: float = 16.0
    spectral_cap: float = 1.0
    implicit_iters: int = 1
    implicit_tol: float = 1e-6
    rank_r: int = 32
    proj_eps: float = 1e-6
    clip_nan_policy: str = "raise"  # options 'raise','clamp','rewind'
    max_step_time_ms: float = 200.0
    deterministic: bool = True
    enable_triton: bool = False
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cpu")

    # K operator: either full matrix K_matrix (dxd) or low-rank factors K_U (d x r), K_V (d x r)
    K_matrix: Optional[torch.Tensor] = None
    K_U: Optional[torch.Tensor] = None
    K_V: Optional[torch.Tensor] = None

    # Optional ICNN-like module that implements `.grad(x: Tensor) -> Tensor`
    icnn: Optional[object] = None

    # Crypto keys: raw bytes (PEM) or None. Use builder helpers to load from file.
    snapshot_aes_key: Optional[bytes] = None  # 32 bytes AES-GCM key
    snapshot_signing_private_pem: Optional[bytes] = None  # Ed25519 private key PEM
    snapshot_signing_public_pem: Optional[bytes] = None  # Ed25519 public key PEM

    # Profiling & compile options
    enable_jit: bool = False
    enable_profiling: bool = False

    # Mixed precision guard for inference
    allow_mixed_precision: bool = False

    # Telemetry & hooks
    telemetry: Optional[TelemetryClient] = None

    # Thermostat callback: Callable[[Engine, metrics_dict], None]
    thermostat_hook: Optional[Callable[["OperatorSplitEngine", dict], None]] = None

    # Runtime override loader
    @staticmethod
    def from_env(defaults: dict) -> "EngineParams":
        """
        Load runtime overrides from environment variables. Accepts defaults dict with keys matching EngineParams.
        Example: set NOESIS_DT=0.01 to override dt.
        """
        load_dotenv()  # loads .env if present

        # Helper to load float/env
        def _env_or(key, type_, default):
            env_key = f"NOESIS_{key.upper()}"
            val = os.getenv(env_key)
            if val is None:
                return defaults.get(key, default)
            try:
                if type_ is float:
                    return float(val)
                if type_ is int:
                    return int(val)
                if type_ is bool:
                    return val.lower() in ("1", "true", "yes", "on")
                return val
            except Exception:
                return defaults.get(key, default)

        # Build params
        p = EngineParams(
            state_dim=int(_env_or("state_dim", int, defaults.get("state_dim", 128))),
            dt=_env_or("dt", float, defaults.get("dt", 0.005)),
            max_norm=_env_or("max_norm", float, defaults.get("max_norm", 16.0)),
            spectral_cap=_env_or(
                "spectral_cap", float, defaults.get("spectral_cap", 1.0)
            ),
            implicit_iters=_env_or(
                "implicit_iters", int, defaults.get("implicit_iters", 1)
            ),
            implicit_tol=_env_or(
                "implicit_tol", float, defaults.get("implicit_tol", 1e-6)
            ),
            rank_r=_env_or("rank_r", int, defaults.get("rank_r", 32)),
            proj_eps=_env_or("proj_eps", float, defaults.get("proj_eps", 1e-6)),
            clip_nan_policy=_env_or(
                "clip_nan_policy", str, defaults.get("clip_nan_policy", "raise")
            ),
            max_step_time_ms=_env_or(
                "max_step_time_ms", float, defaults.get("max_step_time_ms", 200.0)
            ),
            deterministic=_env_or(
                "deterministic", bool, defaults.get("deterministic", True)
            ),
            dtype=torch.get_default_dtype(),  # may be overridden later
            device=torch.device(
                os.getenv("NOESIS_DEVICE", str(defaults.get("device", "cpu")))
            ),
        )
        # Telemetry left None by default
        return p


# ---------------------------
# Helper utilities
# ---------------------------
def _to_device(
    t: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t.to(device=device, dtype=dtype)


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _safe_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x).item())


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# Crypto helpers -----------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aesgcm_encrypt(key: bytes, plaintext: bytes, associated: bytes = b"") -> bytes:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography required for encryption")
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must be 16/24/32 bytes")
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated)
    return nonce + ct


def _aesgcm_decrypt(key: bytes, blob: bytes, associated: bytes = b"") -> bytes:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography required for decryption")
    aesgcm = AESGCM(key)
    nonce = blob[:12]
    ct = blob[12:]
    return aesgcm.decrypt(nonce, ct, associated)


def _sign_ed25519(private_pem: bytes, data: bytes) -> bytes:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography required for signing")
    priv = serialization.load_pem_private_key(private_pem, password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519 PEM")
    return priv.sign(data)


def _verify_ed25519(public_pem: bytes, data: bytes, sig: bytes) -> bool:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography required for signature verification")
    pub = serialization.load_pem_public_key(public_pem)
    if not isinstance(pub, Ed25519PublicKey):
        raise ValueError("public key must be Ed25519 PEM")
    try:
        pub.verify(sig, data)
        return True
    except Exception:
        return False


# ---------------------------
# Exceptions
# ---------------------------
class DynamicsError(Exception):
    """Base exception for dynamics engine."""


class NumericalStabilityError(DynamicsError):
    """Raised when NaN/Inf or catastrophic norm blow-up occurs and policy is 'raise'."""


class SnapshotSignatureError(DynamicsError):
    """Raised when snapshot signature verification fails."""


# ---------------------------
# CUDA Graph - Hot Loop Graph
# ---------------------------
class HotLoopGraph:
    """
    Wraps a HotLoopFused with torch.cuda.CUDAGraph to eliminate
    per-step kernel launch overhead across the entire inner unroll.
    """

    def __init__(self, fused: HotLoopFused, B: int, S: int, dt: float):
        if not torch.cuda.is_available():
            raise RuntimeError("HotLoopGraph requires CUDA.")
        self.fused = fused
        self.B = int(B)
        self.S = int(S)
        self.dt = float(dt)

        dev = self.fused.device
        dtype = self.fused.dtype

        # Static IO buffers (must not change address between replays)
        self._x_static = torch.empty(self.B, self.fused.d, device=dev, dtype=dtype)
        self._y_static: Optional[torch.Tensor] = None  # filled during capture

        # Ensure all internal work buffers exist BEFORE capture (no alloc inside graph)
        with torch.inference_mode():
            _ = self.fused.step_unrolled(self._x_static, S=1, dt=self.dt)

        # Warm the CUDA memory pool so captures reuse it
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Capture the whole unroll once
        self._g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self._g):
            # Important: call the fused unroll using only preallocated buffers
            self._y_static = self.fused.step_unrolled(
                self._x_static, S=self.S, dt=self.dt
            )

        if self._y_static is None:
            raise RuntimeError("Graph capture failed to produce an output tensor.")

    @torch.inference_mode()
    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Copy x -> static input, replay the graph, return the static output view.
        Shapes/dtypes must match the capture (B,d,dtype,device).
        """
        assert (
            x.shape == self._x_static.shape
        ), f"Expected {tuple(self._x_static.shape)}, got {tuple(x.shape)}"
        assert (
            x.dtype == self._x_static.dtype and x.device == self._x_static.device
        ), "dtype/device mismatch for graph input"
        self._x_static.copy_(x, non_blocking=True)
        self._g.replay()
        # Return a tensor that shares memory with the captured output (no alloc on replay)
        assert self._y_static is not None
        return self._y_static


class HotLoopFused:
    """
    Fused operator-split inner loop with low-rank K and direct-grad ICNN.
    """

    def __init__(
        self,
        d: int,
        icnn: ICNNDirectGrad,
        KU: torch.Tensor,  # [d, r]
        KV: torch.Tensor,  # [d, r]
        *,
        projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ws_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.d = int(d)
        self.icnn = icnn
        # Default workspace dtype: follow model dtype unless overridden
        if ws_dtype is None:
            ws_dtype = dtype or KU.dtype
        self.lrK = KernelLowRankK(KU, KV, ws_dtype=ws_dtype)
        self.projector = projector
        self.device = device or KU.device
        self.dtype = dtype or KU.dtype

        # Scratch buffers
        self._buf_half: Optional[torch.Tensor] = None
        self._buf_out: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._last_B: int = -1
        self._graph_enabled: bool = False
        self._g: Optional[torch.cuda.CUDAGraph] = None
        self._static_x: Optional[torch.Tensor] = None
        self._static_out: Optional[torch.Tensor] = None
        self._captured: bool = False
        self._captured_S: int = -1
        self._captured_B: int = -1
        self._captured_dt: float = 0.0

    def enable_graphs(self, enable: bool = True) -> None:
        self._graph_enabled = enable and torch.cuda.is_available()

    def _maybe_capture(self, B: int, S: int, dt: float):
        if not self._graph_enabled or self._captured:
            return
        self._ensure_ws(B)
        dev = self.device
        ws = self.lrK.ws_dtype
        self._static_x = torch.empty(B, self.d, device=dev, dtype=ws)
        self._static_out = torch.empty(B, self.d, device=dev, dtype=ws)
        torch.cuda.synchronize()
        self._g = torch.cuda.CUDAGraph()
        assert self._buf_half is not None
        assert self._buf_out is not None
        alpha = 0.5 * dt
        # capture: read from _static_x, write to _static_out
        self._g.capture_begin()
        assert self._static_x is not None
        assert self._static_out is not None
        x_ws = self._static_x
        for _ in range(S):
            self.lrK.addmm_Kt_base(x_ws, base=x_ws, alpha=alpha, out=self._buf_half)
            _g = self.icnn.grad(self._buf_half).to(ws)
            torch.add(self._buf_half, _g, alpha=-dt, out=self._buf_out)
            if self.projector is not None:
                self._buf_out = self.projector(self._buf_out)
            x_ws, self._buf_out = self._buf_out, x_ws
        self._static_out.copy_(x_ws)
        self._g.capture_end()
        self._captured = True
        self._captured_S, self._captured_B, self._captured_dt = S, B, dt

    def _ensure_ws(self, B: int) -> None:
        if (
            self._last_B != B
            or self._buf_half is None
            or self._buf_out is None
            or self._grad is None
        ):
            dev = self.device
            ws = self.lrK.ws_dtype
            self._buf_half = torch.empty(
                B, self.d, device=dev, dtype=ws
            )  # x_half (ws dtype)
            self._buf_out = torch.empty(
                B, self.d, device=dev, dtype=ws
            )  # accumulator/out (ws dtype)
            self._grad = torch.empty(
                B, self.d, device=dev, dtype=ws
            )  # grad buffer (ws dtype)
            self._last_B = B

    @torch.inference_mode()
    def step_unrolled(
        self,
        x: torch.Tensor,  # [B, d], must be on device/dtype
        S: int,
        dt: float,
        *,
        drive: Optional[
            torch.Tensor
        ] = None,  # e.g., B×d control input; defaults to None
    ) -> torch.Tensor:
        """
        Run S fused steps in-place-style (without autograd).
        Returns new state tensor [B, d] (same dtype as x).
        """
        assert x.ndim == 2 and x.shape[1] == self.d, "x must be [B, d]"
        B = x.shape[0]
        self._ensure_ws(B)
        assert self._buf_half is not None
        assert self._buf_out is not None

        # Use workspace dtype for math, cast back at the end.
        x_ws = x.to(self.lrK.ws_dtype)

        # Graph fast-path
        if self._graph_enabled and torch.cuda.is_available():
            if (
                not self._captured
                or self._captured_B != B
                or self._captured_S != S
                or self._captured_dt != dt
            ):
                self._captured = False
                self._maybe_capture(B, S, dt)
            assert self._static_x is not None
            assert self._static_out is not None
            assert self._g is not None
            self._static_x.copy_(x_ws)
            self._g.replay()
            return self._static_out.to(x.dtype)

        alpha = 0.5 * dt
        for _ in range(S):

            # x_half = x + alpha*(x @ Kᵀ)  -> write directly into _buf_half
            self.lrK.addmm_Kt_base(x_ws, base=x_ws, alpha=alpha, out=self._buf_half)

            # Dissipative step: x = x_half - dt * ∇φ(x_half)
            _g = self.icnn.grad(self._buf_half).to(self.lrK.ws_dtype)
            # x = x_half - dt * g  -> write into _buf_out
            torch.add(self._buf_half, _g, alpha=-dt, out=self._buf_out)

            # Optional drive term
            if drive is not None:
                self._buf_out.add_(drive.to(self.lrK.ws_dtype))

            # Optional projector Π_L
            if self.projector is not None:
                self._buf_out = self.projector(self._buf_out)

            x_ws, self._buf_out = self._buf_out, x_ws  # swap buffers (avoid alloc)

        return x_ws.to(x.dtype)


# ---------------------------
# OperatorSplitEngine
# ---------------------------
class OperatorSplitEngine:
    """
    Production Operator-Split Engine.

    Public API:
      - step(x, trace_id=None, sim_graft=None) -> Tensor
      - step_many(x, n_steps, trace_id=None, sim_graft=None) -> Tensor
      - energy(x) -> float
      - snapshot(dirpath, sign=True, encrypt=True) -> writes manifest.json + snapshot.bin [+ snapshot.sig]
      - load_snapshot(dirpath, verify=True, decrypt=True)
      - set_seed(seed)
      - set_device(device)
      - compile_jit()  # optional to JIT-compile hot paths
      - enable_deterministic_algos(flag)
    """

    def __init__(self, params: EngineParams, icnn: torch.nn.Module | None = None):
        super().__init__()
        self.params = params

        # -------- Device / dtype --------
        # Prefer CUDA if available; fall back to CPU.
        self.device = (
            torch.device(params.device)
            if isinstance(params.device, torch.device)
            else (
                torch.device(params.device)
                if isinstance(params.device, str)
                else (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            )
        )

        # Normalize dtype (string or torch.dtype)
        if isinstance(params.dtype, str):
            dt_map = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float64": torch.float64,
                "fp64": torch.float64,
                "double": torch.float64,
            }
            self.dtype = dt_map.get(params.dtype.lower(), torch.float32)
        elif isinstance(params.dtype, torch.dtype):
            self.dtype = params.dtype
        else:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # fp16 is not supported on CPU tensors
        if self.device.type == "cpu" and self.dtype == torch.float16:
            self.dtype = torch.float32

        # -------- State dimension --------
        self.state_dim = int(self.params.state_dim)

        # -------- ICNN (direct-gradient; no autograd in hot loop) --------
        # If the caller injected an ICNN, use it; otherwise create ICNNDirectGrad.
        icnn = icnn if icnn is not None else getattr(self.params, "icnn", None)
        if icnn is not None:
            self.icnn = icnn.to(device=self.device, dtype=self.dtype)
        else:
            m_hidden = getattr(self.params, "icnn_m", 512)
            self.icnn = ICNNDirectGrad(
                d=self.state_dim,
                m=m_hidden,
                dtype=self.dtype,
                device=self.device,
            )
        self.icnn.eval()  # inference-oriented; .grad() is tape-free

        # -------- Auto-load trained ICNN checkpoint if present --------
        self._icnn_checkpoint_path = getattr(
            self.params, "icnn_checkpoint",
            "Runtime/Models/osc_training/checkpoints/icnn_ep.pt"
        )
        self._try_load_icnn_checkpoint()

        # -------- Optional CUDA warm-up (no autograd/backward needed) --------
        if self.device.type == "cuda":
            try:
                # Create primary CUDA context & cuBLAS handles
                a = torch.randn(32, 32, device=self.device, dtype=self.dtype)
                b = torch.randn(32, 32, device=self.device, dtype=self.dtype)
                _ = a @ b  # one forward GEMM is enough to warm cuBLAS
                # Warm ICNN workspace so first call doesn't pay allocation cost
                with torch.inference_mode():
                    v = torch.randn(
                        1, self.state_dim, device=self.device, dtype=self.dtype
                    )
                    _ = cast(ICNNDirectGrad, self.icnn).grad(v)
                torch.cuda.synchronize()
            except Exception:
                pass  # best-effort; never fail init because of warm-up

        # -------- Telemetry / hooks / locks --------
        self.telemetry = getattr(self.params, "telemetry", None) or TelemetryClient()
        self.thermostat_hook = getattr(self.params, "thermostat_hook", None)
        self._lock = threading.RLock()
        self.step_count = 0

        # Diagnostics
        self._projection_count = 0
        self._nan_count = 0
        self._last_step_time_ms = 0.0
        self._triton_logged = False
        self._spectral_est: Optional[float] = None

        # -------- Low-rank K + projector operator (fused) --------
        self._k_operator = None
        try:
            d = int(self.state_dim)

            # rank must be in [1, d]; default to min(64, d)
            k_rank = int(getattr(self.params, "rank_r", min(64, d)))
            if k_rank <= 0 or k_rank > d:
                k_rank = min(64, d)

            # spectral cap must be > 0
            k_lambda_cap = float(getattr(self.params, "spectral_cap", 1.0))
            if k_lambda_cap <= 0.0:
                k_lambda_cap = 1.0

            # projector radius: use max_norm if valid, else fallback
            proj_radius = getattr(self.params, "max_norm", None)
            if proj_radius is None or proj_radius <= 0.0:
                proj_radius = float(4.0)

            k_cfg = LowRankKConfig(
                d=d,
                rank=k_rank,
                lambda_cap=k_lambda_cap,
                projector_radius=float(proj_radius),
                enable_projector=True,
                device=self.device,
                dtype=self.dtype,
            )
            self._k_operator = CoreLowRankK(k_cfg)
            logger.debug(
                "LowRankK fused operator initialised: d=%d rank=%d cap=%s radius=%s",
                d,
                k_rank,
                k_lambda_cap,
                proj_radius,
            )
        except Exception:
            logger.warning(
                "LowRankK fused operator could not be initialised; falling back to legacy _apply_K path.",
                exc_info=True,
            )
            self._k_operator = None

        # -------- Prepare linear operator K, etc. --------
        self._K_matrix: Optional[torch.Tensor] = None
        self._K_U: Optional[torch.Tensor] = None
        self._K_V: Optional[torch.Tensor] = None
        self._prepare_K()

        # -------- Optional CUDA graph cache (FAST path) --------
        graph_env = os.getenv("NOESIS_GRAPH_FAST_PATH", "").strip().lower()
        if graph_env in {"0", "false", "no"}:
            self._graph_cache_enabled = False
        elif graph_env in {"1", "true", "yes"}:
            self._graph_cache_enabled = True
        else:
            # auto: enable if CUDA is available
            self._graph_cache_enabled = torch.cuda.is_available()
        self._graph_cache: Optional["GraphBucketManager"] = None
        self._graph_fused: Optional[HotLoopFused] = None
        if self._graph_cache_enabled and self.device.type == "cuda":
            try:
                if self._K_U is not None and self._K_V is not None:
                    self._graph_fused = HotLoopFused(
                        d=self.state_dim,
                        icnn=cast(ICNNDirectGrad, self.icnn),
                        KU=self._K_U,
                        KV=self._K_V,
                        projector=self.project,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    cap = int(os.getenv("NOESIS_GRAPH_CACHE_SIZE", "8"))
                    from Core.OSC.Exec.graph_cache import GraphBucketManager

                    self._graph_cache = GraphBucketManager(capacity=max(1, cap))
            except Exception:
                logger.warning(
                    "CUDA graph cache init failed; falling back to normal path.",
                    exc_info=True,
                )

        # Runtime book-keeping
        self._last_good_state: Optional[torch.Tensor] = None
        self._seed: Optional[int] = None
        self._rng_saved: Optional[dict] = None
        self._inner_step_idx: int = 0
        self._cool_start_steps: int = 0

        # Deterministic algorithms (best-effort)
        if getattr(self.params, "deterministic", False):
            self.enable_deterministic_algorithms(True)

        # Optional JIT compile
        self._jit_compiled = False
        if getattr(self.params, "enable_jit", False):
            try:
                self.compile_jit()
            except Exception as e:
                logger.warning("JIT compile failed: %s", str(e))

        logger.info(
            "OperatorSplitEngine initialized: dim=%d device=%s dt=%s",
            self.state_dim,
            str(self.device),
            str(self.params.dt),
        )

        # Thermostat: default config (tweak bands if you want it more/less aggressive)
        self._thermostat = Thermostat(
            ThermostatConfig(
                s_min=8,
                s_max=256,
                dt_min=1e-4,
                dt_max=5e-2,
                lower_band=1e-3,
                upper_band=5e-2,
                check_interval=4,
                window_M=3,
                warmup_checks=2,
                downscale_S=0.5,
                upscale_S=1.5,
                up_dt_factor=1.25,
                down_dt_factor=0.75,
            )
        )

        # ── Runtime knobs (S, Δt) ───────────────────────────────────────────
        # Δt follows params.dt at start; S defaults to a sane mid value within clamps.
        self._dt: float = float(self.params.dt)
        try:
            init_S = int(getattr(self.params, "inner_S", 64))
        except Exception:
            init_S = 64
        cfg = self._thermostat.cfg
        self._S: int = max(cfg.s_min, min(cfg.s_max, init_S))

        def _thermo_adapter(step_idx: int, x: torch.Tensor, metrics: dict) -> None:
            # Use the engine's own energy proxy if available; else default in Thermostat.
            S_new, dt_new = self._thermostat.maybe_update(
                step_idx=step_idx,
                x=x,
                S=self._S,
                dt=self._dt,
                energy_fn=None,  # or lambda t: self.energy(t)
            )
            if S_new != self._S:
                self._S = int(S_new)
            if dt_new != self._dt:
                self._dt = float(dt_new)
                self.params.dt = self._dt  # keep manifest/metrics consistent

        self.thermostat_hook = _thermo_adapter

    # ----------------------------
    # K operator setup & utilities
    # ----------------------------
    def _prepare_K(self) -> None:
        d = self.params.state_dim
        # Accept provided tensors or create default low-rank random K
        if self.params.K_matrix is not None:
            K = self.params.K_matrix.to(device=self.device, dtype=self.dtype)
            if K.shape != (d, d):
                raise ValueError("K_matrix must be shape (state_dim, state_dim)")
            # enforce skew symmetry numerically
            K = 0.5 * (K - K.T)
            self._K_matrix = K
            self._K_U = None
            self._K_V = None
        elif self.params.K_U is not None and self.params.K_V is not None:
            U = self.params.K_U.to(device=self.device, dtype=self.dtype)
            V = self.params.K_V.to(device=self.device, dtype=self.dtype)
            if U.shape[0] != d or V.shape[0] != d:
                raise ValueError("K_U/K_V row dim must equal state_dim")
            self._K_matrix = None
            self._K_U = U
            self._K_V = V
        else:
            # default small random low-rank operator for tests/prototyping
            r = min(self.params.rank_r, max(4, d // 8))
            U = torch.randn((d, r), dtype=self.dtype, device=self.device) * 1e-3
            V = torch.randn((d, r), dtype=self.dtype, device=self.device) * 1e-3
            self._K_matrix = None
            self._K_U = U
            self._K_V = V
            logger.debug("Default low-rank K created rank={}", r)

        # initial spectral estimate
        try:
            self._spectral_est = self._estimate_spectral_norm(power_iters=6)
        except Exception:
            self._spectral_est = None

    def _validate_params(self) -> None:
        """Check param sanity and set defaults."""
        p = self.params
        if p.clip_nan_policy not in ("raise", "clamp", "rewind"):
            raise ValueError("clip_nan_policy must be in {'raise','clamp','rewind'}")
        if p.max_norm <= 0:
            raise ValueError("max_norm must be positive")
        if p.dt <= 0:
            raise ValueError("dt must be positive")

    def _form_K_matrix(self) -> torch.Tensor:
        """Materialize full K matrix (may be costly for large dims)."""
        if self._K_matrix is not None:
            return self._K_matrix
        if self._K_U is None or self._K_V is None:
            raise RuntimeError("Low-rank K factors are missing")
        A = self._K_U @ self._K_V.T  # (d,d)
        K = A - A.T
        return K

    def _apply_K(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear operator K to row-wise batched input x: [B, d] -> [B, d].

        Dense path:
            if self._K_matrix is not None:
                y = x @ self._K_matrix.T

        Low-rank skew-symmetric path (your design):
            K = U V^T - V U^T  with U,V in R^{d x r}
            y = (x @ V) @ U^T - (x @ U) @ V^T

        Triton fast-path:
            - If enabled, CUDA, and Triton present, we compose the skew form using
              two dense Triton matvecs per term (xV then (xV)U, etc).
            - Falls back to PyTorch on any exception.

        Returns:
            y with the same batch semantics as input (squeeze back if input was 1D).
        """
        # Accept [d] or [B,d]
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True

        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)  # defensive

        # Ensure device/dtype are consistent with engine
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype)

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Decide once and log once
        use_triton = bool(
            _HAS_TRITON
            and _apply_K_dense_triton is not None
            and getattr(self.params, "enable_triton", False)
            and self.device.type == "cuda"
        )
        triton_apply = _apply_K_dense_triton
        if use_triton and not getattr(self, "_triton_logged", False):
            kind = "lowrank" if self._K_matrix is None else "dense"
            logger.info("Using Triton fast-path for K (%s)", kind)
            self._triton_logged = True

        # Dense matrix K available
        K = self._K_matrix
        if K is not None:
            try:
                if use_triton:
                    assert triton_apply is not None
                    y = triton_apply(K, x)
                else:
                    y = x @ K.T
            except Exception:
                y = x @ K.T
            return y.squeeze(0) if squeezed else y

        # Low-rank skew-symmetric path
        U = self._K_U
        V = self._K_V
        if U is None or V is None:
            raise RuntimeError("Low-rank K factors are missing")
        try:
            if use_triton:
                assert triton_apply is not None
                # Compose skew via two Triton dense calls per term:
                # term1 = (x @ V) @ U^T
                #   xV    : K=V^T => x @ V   (because apply_K_dense returns x @ K^T)
                #   (xV)U : K=U^T => (x @ V) @ U
                xV = triton_apply(V.T.contiguous(), x)  # [B, r] == x @ V
                term1 = triton_apply(U.T.contiguous(), xV)  # [B, d] == (x @ V) @ U

                # term2 = (x @ U) @ V^T
                xU = triton_apply(U.T.contiguous(), x)  # [B, r] == x @ U
                term2 = triton_apply(V.T.contiguous(), xU)  # [B, d] == (x @ U) @ V

                y = term1 - term2
            else:
                # PyTorch fallback (original math; efficient and clear)
                term1 = (x @ V) @ U.T
                term2 = (x @ U) @ V.T
                y = term1 - term2
        except Exception:
            # Any Triton/runtime error → safe Torch path
            term1 = (x @ V) @ U.T
            term2 = (x @ U) @ V.T
            y = term1 - term2

        return y.squeeze(0) if squeezed else y

    def _estimate_spectral_norm(self, power_iters: int = 8) -> float:
        """
        Power iteration estimate of operator norm ||K||_2.
        """
        d = self.params.state_dim
        v = torch.randn((d,), device=self.device, dtype=self.dtype)
        v = v / (torch.linalg.norm(v) + 1e-12)
        for _ in range(max(2, power_iters)):
            w = self._apply_K(v.unsqueeze(0))[0]
            w_norm = torch.linalg.norm(w)
            if w_norm.item() == 0.0:
                return 0.0
            v = w / (w_norm + 1e-12)
        Kw = self._apply_K(v.unsqueeze(0))[0]
        est = float(abs(torch.dot(v, Kw).item()))
        return est

    def enforce_spectral_cap(self) -> None:
        """Scale K factors/matrix so operator norm <= spectral_cap."""
        cap = float(self.params.spectral_cap)
        if cap <= 0:
            return
        est = self._estimate_spectral_norm(power_iters=6)
        self._spectral_est = est
        if est > cap * (1.0 + 1e-9):
            scale = cap / (est + 1e-12)
            logger.info(
                "Scaling K by {:.6e} to enforce spectral cap {:.6e} (est {:.6e})",
                scale,
                cap,
                est,
            )
            if self._K_matrix is not None:
                self._K_matrix = (self._K_matrix * scale).to(device=self.device)
            else:
                # scale low-rank factors preserving A = U V^T by sqrt scaling
                s = scale**0.5
                self._K_U = (self._K_U * s).to(device=self.device)
                self._K_V = (self._K_V * s).to(device=self.device)
            # refresh estimate
            self._spectral_est = self._estimate_spectral_norm(power_iters=4)
        # emit telemetry
        try:
            self.telemetry.record(
                METRIC_SPECTRAL, {"est": self._spectral_est, "cap": cap}
            )
        except Exception:
            logger.debug("telemetry metric failed for spectral")

    # ----------------------------
    # Conservative + Dissipative + Projection steps
    # ----------------------------
    def conservative_update(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conservative skew update.

        Prefer the fused low-rank K + Π_L operator if available;
        otherwise fall back to the legacy _apply_K path.
        """
        dt = float(self._dt)  # runtime-adjustable

        # Fused low-rank K + projector (Step 6 path)
        k_op = getattr(self, "_k_operator", None)
        if k_op is not None:
            # LowRankK.apply_and_project is responsible for:
            #   x_next = x + dt * K(x), then Π_L(x_next)
            return k_op.apply_and_project(x, dt=dt)

        # Legacy path: separate K, projection handled by self.project()
        kx = self._apply_K(x)
        return x + dt * kx

    def dissipative_prox(self, x_half: torch.Tensor) -> torch.Tensor:
        """
        Dissipative / implicit proximal step.
        Uses ICNNDirectGrad.grad(y) (analytic; no autograd tape) inside the implicit iteration.
        Fallback is linear damping if no ICNN is attached.
        """
        dt = float(self._dt)  # runtime-adjustable
        iters = int(self.params.implicit_iters)
        tol = float(self.params.implicit_tol)

        icnn = self.icnn

        # Ensure state is on engine space
        y = x_half.to(device=self.device, dtype=self.dtype)

        if icnn is None:
            # Simple stable fallback
            gamma = 1e-1
            return y / (1.0 + dt * gamma)
        icnn = cast(ICNNDirectGrad, icnn)

        # Resolve ICNN param space once
        try:
            p = next(icnn.parameters())
            icnn_device, icnn_dtype = p.device, p.dtype
        except StopIteration:
            icnn_device, icnn_dtype = self.device, self.dtype

        # Avoid redundant copies if spaces already match
        same_space = (icnn_device == self.device) and (icnn_dtype == self.dtype)
        y_work = y if same_space else y.to(device=icnn_device, dtype=icnn_dtype)

        # Tape-free hot loop
        with torch.inference_mode():
            for i in range(max(1, iters)):
                # Analytic ∇Φ(y) from ICNNDirectGrad (batched)
                g_icnn = icnn.grad(y_work)  # on icnn_device/icnn_dtype

                if same_space:
                    # No device/dtype flips needed
                    y_next_work = y_work - dt * g_icnn
                    y_next_engine = y_next_work
                else:
                    # One cast back to engine space for the update
                    g_eng = g_icnn.to(device=self.device, dtype=self.dtype)
                    y_next_engine = y - dt * g_eng
                    # Keep ICNN workspace in sync for the next iteration + convergence check
                    y_next_work = y_next_engine.to(device=icnn_device, dtype=icnn_dtype)

                # Convergence test in ICNN space (expects [B, d])
                if y_work.dim() == 2:
                    delta_vec = (y_next_work - y_work).reshape(y_work.shape[0], -1)
                    max_delta = float(torch.linalg.norm(delta_vec, dim=1).max().item())
                else:
                    max_delta = float(
                        torch.linalg.vector_norm(y_next_work - y_work).item()
                    )

                y_work = y_next_work
                y = y_next_engine  # keep engine-space y in sync

                if max_delta < tol:
                    try:
                        logger.debug(
                            "dissipative_prox converged iter=%d max_delta=%.6e",
                            i + 1,
                            max_delta,
                        )
                    except NameError:
                        pass
                    break

        return y

    def _try_load_icnn_checkpoint(self) -> None:
        """Load trained ICNN weights from checkpoint if the file exists."""
        import os
        path = self._icnn_checkpoint_path
        if not os.path.isfile(path):
            return
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = (
                ckpt.get("icnn_state_dict") or ckpt.get("icnn_state")
                if isinstance(ckpt, dict) else ckpt
            )
            if state_dict is not None:
                self.icnn.load_state_dict(state_dict, strict=True)
                self.icnn.eval()
        except Exception:
            pass  # bad checkpoint or dim mismatch — keep random init

    def save_icnn_checkpoint(self, path: str | None = None) -> str:
        """Persist current ICNN weights. Returns the path written."""
        import os
        out = path or self._icnn_checkpoint_path
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"icnn_state_dict": self.icnn.state_dict()}, out)
        return out

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project per-sample onto norm ball ||x||_2 <= max_norm."""
        max_norm = float(self.params.max_norm)
        flat = x.reshape(x.shape[0], -1)
        norms = torch.linalg.norm(flat, dim=1, keepdim=True)
        need_proj = norms > (max_norm + 1e-12)
        if need_proj.any():
            scale = (max_norm / (norms + 1e-12)).to(device=x.device, dtype=x.dtype)
            x = x * scale.view(-1, *([1] * (x.dim() - 1)))
            self._projection_count += int(need_proj.sum().item())
        return x

    # ----------------------------
    # Energy & diagnostics
    # ----------------------------
    def energy(self, x: torch.Tensor) -> float:
        """Simple quadratic energy 0.5 * sum(x^2)."""
        with torch.no_grad():
            return float(0.5 * torch.sum(x * x).item())

    @torch.no_grad()
    def _energy_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar energy proxy on x.device/FP32 for thermostat metrics."""
        return 0.5 * (x.float() * x.float()).mean()

    def debug_dump(self) -> dict:
        """Return small diagnostic snapshot useful in crash reports."""
        return {
            "step_count": int(self.step_count),
            "state_dim": int(self.params.state_dim),
            "spectral_est": (
                float(self._spectral_est) if self._spectral_est is not None else None
            ),
            "projection_count": int(self._projection_count),
            "nan_count": int(self._nan_count),
        }

    # ----------------------------
    # High-level step API
    # ----------------------------
    def step(
        self,
        x: torch.Tensor,
        trace_id: Optional[str] = None,
        sim_graft: Optional[torch.Tensor] = None,
        *,
        telemetry_tag: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Advance x by one operator-split step (tape-free hot path).
        """
        with self._lock:
            t0 = _now_ms()
            batched_in = True
            if x.dim() == 1:
                x = x.unsqueeze(0)
                batched_in = False
            if x.shape[1] != self.params.state_dim:
                raise ValueError(
                    f"Input state dim {x.shape} != params.state_dim {self.params.state_dim}"
                )

            # move to device/dtype
            x = _to_device(x, self.device, self.dtype)

            # Optional graft
            if sim_graft is not None:
                sg = sim_graft.to(device=self.device, dtype=self.dtype)
                if sg.dim() == 1:
                    sg = sg.unsqueeze(0).expand(x.shape[0], -1)
                if sg.shape != x.shape:
                    raise ValueError(
                        "sim_graft shape must broadcast to [B,d] or be [d]"
                    )
                x = x + sg

            # Save last good for rewind
            self._last_good_state = x.clone()

            # ---- Hot path (tape-free) ----
            with torch.inference_mode():
                x_half = self.conservative_update(x)  # uses self._dt
                x_next = self.dissipative_prox(x_half)  # uses self._dt
                x_proj = self.project(x_next)

            # Numerical checks
            if not _is_finite(x_proj):
                try:
                    self.telemetry.record(
                        METRIC_NAN_EVENT,
                        {"policy": self.params.clip_nan_policy, "trace_id": trace_id},
                    )
                except Exception:
                    logger.debug("telemetry nan_event failed")
                self._nan_count += 1
                x_proj = self._handle_numerical_failure(x_proj, trace_id=trace_id)

            # Book-keeping & telemetry
            self.step_count += 1
            elapsed = _now_ms() - t0
            self._last_step_time_ms = elapsed
            metrics = {
                "trace_id": trace_id,
                "step_count": self.step_count,
                "elapsed_ms": elapsed,
                "energy_before": self.energy(x),
                "energy_after": self.energy(x_proj),
                "norm_after": _safe_norm(x_proj),
                "projection_count": self._projection_count,
                "S": int(self._S),
                "dt": float(self._dt),
            }
            if telemetry_tag is not None:
                metrics["tag"] = telemetry_tag

            try:
                self.telemetry.record(METRIC_STEP, metrics)
            except Exception:
                logger.debug("telemetry.record failed for step")

            # Inner-loop index (best-effort for policies)
            try:
                metrics["inner_idx"] = self._inner_step_idx
            except Exception:
                pass

            # Watchdog
            if elapsed > self.params.max_step_time_ms:
                try:
                    self.telemetry.record(
                        METRIC_WATCHDOG,
                        {"elapsed_ms": elapsed, "max_ms": self.params.max_step_time_ms},
                    )
                except Exception:
                    logger.debug("telemetry watchdog record failed")

            # ---- Cool-start (first couple inner steps after a token boundary) ----
            if getattr(self, "_cool_start_steps", 0) > 0:
                if self._thermostat is not None:
                    dt_min = float(getattr(self._thermostat.cfg, "dt_min", 1e-6))
                    self._dt = max(self._dt * 0.75, dt_min)
                    self.params.dt = self._dt
                else:
                    self._dt = max(self._dt * 0.75, float(self.params.dt) * 0.1)
                self._cool_start_steps -= 1

            # ---- Thermostat adaptive update (cheap) ----
            if self._thermostat is not None:
                try:
                    S_new, dt_new = self._thermostat.maybe_update(
                        step_idx=(
                            self._inner_step_idx
                            if hasattr(self, "_inner_step_idx")
                            else 0
                        ),
                        x=x_proj,
                        S=self._S,
                        dt=self._dt,
                        energy_fn=lambda t: (
                            self._energy_proxy(t) if t is not None else 0.0
                        ),
                    )
                    self._S, self._dt = int(S_new), float(dt_new)
                    self.params.dt = self._dt
                except Exception:
                    logger.debug("thermostat update skipped (error)")
                self._inner_step_idx = (
                    (self._inner_step_idx + 1)
                    if hasattr(self, "_inner_step_idx")
                    else 1
                )

            # Receipts (best-effort; unchanged)
            try:
                if getattr(
                    self.params, "snapshot_signing_private_pem", None
                ) and getattr(self.params, "snapshot_signing_public_pem", None):
                    from External.Output.receipts import create_receipt

                    x_before = self._last_good_state
                    x_after = x_proj
                    if not batched_in:
                        x_before = x_before.squeeze(0)
                        x_after = x_after.squeeze(0)
                    receipt = create_receipt(self, x_before, x_after, trace_id=trace_id)
                    os.makedirs("Runtime/Logs", exist_ok=True)
                    with open("Runtime/Logs/last_receipt.json", "wb") as f:
                        f.write(
                            json.dumps(
                                receipt, sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")
                        )
            except Exception:
                logger.exception("receipt emission failed")

            return x_proj.squeeze(0) if not batched_in else x_proj

    # ──────────────────────────────────────────────────────────────────────
    # New: advance one "token" via S inner steps, with adaptive S, Δt.
    # ──────────────────────────────────────────────────────────────────────
    def advance_token(
        self,
        x: torch.Tensor,
        *,
        trace_id: Optional[str] = None,
        sim_graft: Optional[torch.Tensor] = None,
        telemetry_tag: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Advance exactly self._S inner steps as a token; resets thermostat and cool-starts.
        """
        if getattr(self, "_thermostat", None) is not None:
            try:
                self._thermostat.reset()
            except Exception:
                pass
        self._inner_step_idx = 0
        self._cool_start_steps = 2
        cur = x
        S_local = int(self._S)
        for _ in range(S_local):
            cur = self.step(
                cur, trace_id=trace_id, sim_graft=sim_graft, telemetry_tag=telemetry_tag
            )
        return cur

    def step_many(
        self,
        x: torch.Tensor,
        n_steps: Optional[int] = None,
        *,
        steps: Optional[int] = None,
        token_boundary: bool = True,
        trace_id: Optional[str] = None,
        sim_graft: Optional[torch.Tensor] = None,
        telemetry_tag: Optional[str] = None,
        thermo_stats: Optional[ThermostatStats] = None,
        profile_name: Optional[str] = None,
        use_graph: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Run multiple inner steps.

        If neither ``n_steps`` nor ``steps`` is given, run exactly ``self._S``,
        treating this call as a token boundary (reset thermostat + cool-start).

        IMPORTANT:
        - No logging inside the per-step loop (hot path).
        - If telemetry is needed, pass a ``ThermostatStats`` instance via
          ``thermo_stats`` and/or use an AsyncThermostatSampler from outside
          this method.
        """
        # ------------------------------------------------------------------ #
        # Resolve total step count
        # ------------------------------------------------------------------ #
        total: int
        if n_steps is None and steps is None:
            total = int(self._S)
        elif n_steps is not None and steps is not None and int(n_steps) != int(steps):
            raise TypeError(
                f"Conflicting step counts: n_steps={n_steps} vs steps={steps}"
            )
        else:
            val = n_steps if n_steps is not None else steps
            assert val is not None
            total = int(val)

        # ------------------------------------------------------------------ #
        # Input shape normalisation (vector vs batch)
        # ------------------------------------------------------------------ #
        vector_in = x.ndim == 1
        cur = x.unsqueeze(0) if vector_in else x

        # ------------------------------------------------------------------ #
        # Optional CUDA graph fast-path (explicit, FAST profile only)
        # ------------------------------------------------------------------ #
        if use_graph is None:
            use_graph = self._graph_cache_enabled
        if (
            use_graph
            and self._graph_cache is not None
            and self._graph_fused is not None
            and self.device.type == "cuda"
            and total == 1
        ):
            prof = resolve_profile(profile_name or os.getenv("NOESIS_PROFILE", "AUTO"))
            # Ensure device/dtype
            cur = _to_device(cur, self.device, self.dtype)
            if sim_graft is not None:
                sg = sim_graft.to(device=self.device, dtype=self.dtype)
                if sg.dim() == 1:
                    sg = sg.unsqueeze(0).expand(cur.shape[0], -1)
                if sg.shape != cur.shape:
                    raise ValueError(
                        "sim_graft shape must broadcast to [B,d] or be [d]"
                    )
                cur = cur + sg
            try:
                g = self._graph_cache.get(
                    self._graph_fused,
                    B=cur.shape[0],
                    S=int(total),
                    dt=self._dt,
                    dtype=self.dtype,
                    prof=prof,
                )
                out = g.run(cur)
                return out.squeeze(0) if vector_in else out
            except Exception:
                # fall back to normal path
                pass

        # ------------------------------------------------------------------ #
        # Token boundary hygiene (thermostat + cool start)
        # ------------------------------------------------------------------ #
        if token_boundary and getattr(self, "_thermostat", None) is not None:
            try:
                self._thermostat.reset()
            except Exception:
                # Best-effort; thermostat issues must not crash the core
                pass
            self._inner_step_idx = 0
            self._cool_start_steps = 2

        # ------------------------------------------------------------------ #
        # Deterministic replay guard (STRICT mode)
        # ------------------------------------------------------------------ #
        if getattr(self.params, "deterministic", False) and getattr(
            self, "_rng_saved", None
        ):
            try:
                assert self._rng_saved is not None
                rng = self._rng_saved
                torch_cpu_state = rng.get("torch_cpu")
                torch_cuda_state = rng.get("torch_cuda")
                python_state = rng.get("python")
                numpy_state = rng.get("numpy")

                if torch_cpu_state is not None:
                    torch.set_rng_state(torch_cpu_state)
                if torch.cuda.is_available() and torch_cuda_state is not None:
                    torch.cuda.set_rng_state_all(torch_cuda_state)
                if python_state is not None:
                    random.setstate(python_state)
                if numpy_state is not None:
                    # Support both tuple state and our dict-shaped export
                    if isinstance(numpy_state, dict) and "keys" in numpy_state:
                        algo = numpy_state.get("algo", "MT19937")
                        keys = np.array(numpy_state["keys"], dtype=np.uint32)
                        pos = int(numpy_state.get("pos", 0))
                        has_gauss = bool(numpy_state.get("has_gauss", 0))
                        cached_gaussian = float(numpy_state.get("cached_gaussian", 0.0))
                        np.random.set_state(
                            (algo, keys, pos, has_gauss, cached_gaussian)
                        )
                    else:
                        np.random.set_state(numpy_state)
            except Exception:
                # Determinism is best-effort; do not kill the run if restore fails
                pass

        # ------------------------------------------------------------------ #
        # Inner loop: pure math, no logging. Optional cheap stats collection.
        # ------------------------------------------------------------------ #
        with torch.inference_mode():
            for _ in range(total):
                # One inner physics + thermostat step.
                # `self.step` encapsulates the operator-split dynamics and
                # thermostat logic for a single inner step.
                cur = self.step(
                    cur,
                    trace_id=trace_id,
                    sim_graft=sim_graft,
                    telemetry_tag=telemetry_tag,
                )

                # Optional hot-path-safe thermostat stats (no logging here).
                if (
                    thermo_stats is not None
                    and getattr(self, "_thermostat", None) is not None
                ):
                    try:
                        t = self._thermostat

                        # Best-effort extraction of current thermostat state.
                        # These attribute names should match your thermostat impl;
                        # adjust if you use different ones.
                        energy = getattr(t, "last_energy", None)
                        S_cur = getattr(t, "current_S", None)
                        dt_cur = getattr(t, "current_dt", None)

                        # Fallbacks if thermostat does not expose everything
                        if energy is None and hasattr(t, "energy"):
                            energy = t.energy
                        if S_cur is None and hasattr(self, "_S"):
                            S_cur = self._S
                        if dt_cur is None and hasattr(self, "_dt"):
                            dt_cur = self._dt

                        if (
                            energy is not None
                            and S_cur is not None
                            and dt_cur is not None
                        ):
                            thermo_stats.observe(
                                float(energy),
                                int(S_cur),
                                float(dt_cur),
                            )
                    except Exception:
                        # Stats are strictly best-effort; never impact dynamics.
                        continue

        # ------------------------------------------------------------------ #
        # Refresh deterministic RNG snapshot (for future STRICT replays)
        # ------------------------------------------------------------------ #
        if getattr(self.params, "deterministic", False):
            try:
                self._rng_saved = {
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                }
            except Exception:
                # Again, best-effort; we don't break the run on telemetry failure.
                pass

        # ------------------------------------------------------------------ #
        # Restore original shape
        # ------------------------------------------------------------------ #
        return cur.squeeze(0) if vector_in else cur

    # ----------------------------
    # EP nudged convergence
    # ----------------------------

    def converge_with_nudge(
        self,
        x_init: torch.Tensor,
        context_bundle: object,
        nudge_fn: Callable[[torch.Tensor], torch.Tensor],
        beta: float = 0.01,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run OSC convergence with an additional nudge energy term (EP Phase 2).

        Each step applies the standard operator-split update augmented by β·∇nudge_fn(x):
            x_half  = conservative_update(x)
            x_next  = x_half - dt·(∇φ(x_half) + β·∇nudge_fn(x_half))
            x_proj  = project(x_next)

        Used by EquilibriumPropagation to produce x*_nudge for the nudged phase.
        nudge_fn(x) must return a scalar tensor; its gradient is computed via autograd.
        context_bundle is accepted for API compatibility but not used internally.

        Args:
            x_init:        Starting state [d] or [B, d].
            context_bundle: Unused; retained for caller symmetry with converge().
            nudge_fn:      callable(x [B,d]) → scalar tensor with grad.
            beta:          Nudge strength (typically 0.001–0.1).
            n_steps:       Number of steps. Defaults to self._S.
        """
        total = int(n_steps) if n_steps is not None else int(self._S)
        vector_in = x_init.ndim == 1
        cur = x_init.unsqueeze(0) if vector_in else x_init
        cur = _to_device(cur, self.device, self.dtype)
        dt = float(self._dt)

        for _ in range(total):
            # --- Conservative half-step ---
            # Use no_grad (not inference_mode) so the output tensor supports requires_grad
            # for the nudge autograd computation below.
            with torch.no_grad():
                x_half = self.conservative_update(cur)

            # --- Dissipative step augmented with nudge gradient ---
            x_half_fp32 = x_half.float()

            # Analytic ICNN gradient (tape-free, no_grad is sufficient here)
            icnn = self.icnn
            if icnn is not None:
                with torch.no_grad():
                    g_icnn = icnn.grad(x_half).float()
            else:
                g_icnn = torch.zeros_like(x_half_fp32)

            # Nudge gradient via autograd
            x_req = x_half_fp32.detach().requires_grad_(True)
            nudge_val = nudge_fn(x_req)
            if nudge_val.requires_grad:
                (g_nudge,) = torch.autograd.grad(nudge_val, x_req)
            else:
                g_nudge = torch.zeros_like(x_req)

            # Combined update
            x_next = (x_half_fp32 - dt * (g_icnn + beta * g_nudge)).to(self.dtype)

            with torch.no_grad():
                cur = self.project(x_next)

        return cur.squeeze(0) if vector_in else cur

    # ----------------------------
    # Numerical failure handling
    # ----------------------------
    def _handle_numerical_failure(
        self, x_bad: torch.Tensor, trace_id: Optional[str]
    ) -> torch.Tensor:
        policy = self.params.clip_nan_policy
        logger.error(
            "Numerical failure detected (policy={}) trace_id={}", policy, trace_id
        )
        try:
            diag = self.debug_dump()
            os.makedirs("Runtime/Logs", exist_ok=True)
            with open(
                os.path.join("Runtime", "Logs", f"core_crash_{int(time.time())}.json"),
                "w",
            ) as f:
                json.dump(diag, f)
        except Exception:
            logger.debug("Failed to write crash diag")

        if policy == "raise":
            raise NumericalStabilityError("NaN/Inf detected in dynamics")
        elif policy == "clamp":
            x_clean = torch.nan_to_num(
                x_bad,
                nan=0.0,
                posinf=self.params.max_norm,
                neginf=-self.params.max_norm,
            )
            logger.warning("Clamped NaN/Inf to finite range")
            return x_clean
        elif policy == "rewind":
            if self._last_good_state is not None:
                logger.warning("Rewinding to last good state")
                return self._last_good_state.clone().to(
                    device=self.device, dtype=self.dtype
                )
            raise NumericalStabilityError("No last good state to rewind to")
        else:
            raise NumericalStabilityError("Unknown clip_nan_policy")

    # ----------------------------
    # Snapshot manifest + save/load
    # ----------------------------
    # Snapshot layout (directory):
    #   manifest.json  -> metadata + sha256(snapshot.bin) + signed flag
    #   snapshot.bin   -> torch.save bytes (possibly encrypted blob)
    #   snapshot.sig   -> raw signature bytes (if signed)
    #
    # manifest.json fields:
    #   version, created_at, step_count, state_dim, dt, spectral_est, snapshot_sha256, signed(bool), encrypted(bool)
    #
    def save_snapshot(
        self, dirpath: str, sign: bool = True, encrypt: bool = True
    ) -> None:
        """
        Save snapshot into a directory atomically (writes tmp files then rename).

        Canonical policy:
          - Sign the PLAINTEXT torch.save blob (if key present & sign=True).
          - Optionally encrypt the blob for snapshot.bin (if key present & encrypt=True).
          - snapshot_sha256 is computed over the EXACT BYTES written to snapshot.bin.
        """
        os.makedirs(dirpath, exist_ok=True)

        # 1) Build plaintext blob (torch.save payload produced by _snapshot_blob)
        plaintext = self._snapshot_blob(include_icnn=True)  # bytes

        # 2) Optional signature over PLAINTEXT
        signed_flag = False
        sig_bytes = None
        if sign and self.params.snapshot_signing_private_pem is not None:
            if not CRYPTO_AVAILABLE:
                raise RuntimeError("cryptography required for signing snapshots")
            sig_bytes = _sign_ed25519(
                self.params.snapshot_signing_private_pem, plaintext
            )
            signed_flag = True

        # 3) Optional encryption for snapshot.bin
        encrypted_flag = False
        out_bytes = plaintext
        if encrypt and self.params.snapshot_aes_key is not None:
            if not CRYPTO_AVAILABLE:
                raise RuntimeError("cryptography required for encryption")
            out_bytes = _aesgcm_encrypt(self.params.snapshot_aes_key, out_bytes)
            encrypted_flag = True

        # 4) Compute sha256 of EXACT BYTES to be written as snapshot.bin
        sha = hashlib.sha256(out_bytes).hexdigest()

        # 5) Optional git ref (best-effort)
        git_ref = None
        try:
            git_ref = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_ref = None

        # 6) Manifest (canonical fields)
        icnn_arch = (
            self.icnn.__class__.__name__
            if getattr(self, "icnn", None) is not None
            else "None"
        )
        icnn_included = bool(getattr(self, "icnn", None) is not None)
        manifest = {
            "version": "1.0",
            "created_at": time.time(),
            "step_count": int(self.step_count),
            "state_dim": int(self.params.state_dim),
            "dt": float(self.params.dt),
            "spectral_est": (
                float(self._spectral_est) if self._spectral_est is not None else 0.0
            ),
            "snapshot_sha256": sha,
            "signed": bool(signed_flag),
            "encrypted": bool(encrypted_flag),
            "icnn_included": icnn_included,
            "icnn_arch": icnn_arch,
        }
        if git_ref:
            manifest["engine_git_ref"] = git_ref

        # 7) Atomic writes
        tmp_bin = os.path.join(dirpath, "snapshot.bin.tmp")
        tmp_sig = os.path.join(dirpath, "snapshot.sig.tmp")
        tmp_manifest = os.path.join(dirpath, "manifest.json.tmp")
        final_bin = os.path.join(dirpath, "snapshot.bin")
        final_sig = os.path.join(dirpath, "snapshot.sig")
        final_manifest = os.path.join(dirpath, "manifest.json")

        with open(tmp_bin, "wb") as f:
            f.write(out_bytes)
        if sig_bytes is not None:
            with open(tmp_sig, "wb") as f:
                f.write(sig_bytes)
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

        os.replace(tmp_bin, final_bin)
        if sig_bytes is not None:
            os.replace(tmp_sig, final_sig)
        os.replace(tmp_manifest, final_manifest)

        logger.info(
            "Snapshot saved to {} (signed={}, encrypted={})",
            dirpath,
            signed_flag,
            encrypted_flag,
        )

    def load_snapshot(
        self, dirpath: str, verify: bool = True, decrypt: bool = True
    ) -> None:
        """
        Load snapshot from directory. Optionally verify signature and decrypt if keys/flags set.
        Restores state and preserves vector shape if the last state was saved as (d,).
        """
        manifest_path = os.path.join(dirpath, "manifest.json")
        bin_path = os.path.join(dirpath, "snapshot.bin")
        sig_path = os.path.join(dirpath, "snapshot.sig")

        if not (os.path.exists(manifest_path) and os.path.exists(bin_path)):
            raise FileNotFoundError("Snapshot manifest or bin missing")

        # --- read manifest & blob ---
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        with open(bin_path, "rb") as bf:
            raw: bytes = bf.read()

        # --- decrypt if needed ---
        if manifest.get("encrypted", False):
            if not decrypt or self.params.snapshot_aes_key is None:
                raise SnapshotSignatureError(
                    "Snapshot is encrypted but decrypt flag/key missing"
                )
            raw = _aesgcm_decrypt(self.params.snapshot_aes_key, raw)

        # --- verify signature if needed ---
        if manifest.get("signed", False):
            if not verify or self.params.snapshot_signing_public_pem is None:
                raise SnapshotSignatureError(
                    "Snapshot signed but verify flag/public key missing"
                )
            if not os.path.exists(sig_path):
                raise SnapshotSignatureError("Missing signature file for snapshot")
            with open(sig_path, "rb") as sf:
                sig = sf.read()
            if not _verify_ed25519(self.params.snapshot_signing_public_pem, raw, sig):
                raise SnapshotSignatureError("Snapshot signature invalid")

        # --- load torch object (safe mode: no NumPy objects inside) ---
        buf = io.BytesIO(raw)

        def _safe_load(b: io.BytesIO):
            return torch.load(b, map_location="cpu", weights_only=True)

        try:
            obj = _safe_load(buf)
        except Exception:
            # Allowlist NumPy internals that safe loader complains about in PyTorch 2.4+
            try:
                import numpy as _np
                import torch.serialization as _tser

                _np_core = getattr(_np, "core", None)
                _np_multi = getattr(_np_core, "multiarray", None) if _np_core else None
                _np_reconstruct = (
                    getattr(_np_multi, "_reconstruct", None) if _np_multi else None
                )
                if _np_reconstruct is not None:
                    _tser.add_safe_globals([_np_reconstruct, _np.dtype, _np.ufunc])
            except Exception:
                pass  # best-effort; continue

            # retry safe load after allowlisting
            buf.seek(0)
            try:
                obj = _safe_load(buf)
            except Exception:
                # final fallback: unsafe load, but only for local, just-created snapshots
                buf.seek(0)
                obj = torch.load(buf, map_location="cpu", weights_only=False)

            try:
                obj = torch.load(
                    buf, map_location="cpu", weights_only=True
                )  # safe-unpickler
            except TypeError:
                buf.seek(0)
                obj = torch.load(buf, map_location="cpu")

        if not isinstance(obj, dict):
            raise ValueError("Snapshot payload is not a dict")

        # --- restore to device/dtype ---
        with self._lock:
            # K (either dense or low-rank)
            if obj.get("K_matrix") is not None:
                self._K_matrix = obj["K_matrix"].to(
                    device=self.device, dtype=self.dtype
                )
                self._K_U = None
                self._K_V = None
            else:
                KU = obj.get("K_U")
                KV = obj.get("K_V")
                self._K_U = (
                    KU.to(device=self.device, dtype=self.dtype)
                    if KU is not None
                    else None
                )
                self._K_V = (
                    KV.to(device=self.device, dtype=self.dtype)
                    if KV is not None
                    else None
                )
                self._K_matrix = None

            # last state: prefer 'last_state', fallback to 'last_good_state'
            last = obj.get("last_state", None)
            if last is None:
                last = obj.get("last_good_state", None)
            if last is None:
                raise ValueError("Snapshot missing last_state/last_good_state")
            if last.ndim == 2 and last.shape[0] == 1:  # preserve vector shape
                last = last[0]
            self._last_good_state = last.to(device=self.device, dtype=self.dtype)

            # RNG states (best effort) — note: NumPy state is stored as pure-Python dict
            rng_state = obj.get("rng", {})

            try:
                cpu = rng_state.get("torch_cpu")
                if cpu is not None:
                    torch.set_rng_state(cpu)
            except Exception:
                logger.warning("Failed to set torch CPU RNG state during snapshot load")

            try:
                if torch.cuda.is_available():
                    cuda = rng_state.get("torch_cuda")
                    if cuda is not None:
                        torch.cuda.set_rng_state_all(cuda)
            except Exception:
                logger.warning(
                    "Failed to set torch CUDA RNG state during snapshot load"
                )

            try:
                py_state = rng_state.get("python")
                if py_state is not None:
                    random.setstate(py_state)
            except Exception:
                logger.warning("Failed to set Python RNG state during snapshot load")

            try:
                np_state = rng_state.get("numpy")
                if np_state is not None:
                    if isinstance(np_state, dict) and "keys" in np_state:
                        # reconstruct the canonical NumPy RNG state tuple
                        algo = np_state.get("algo", "MT19937")
                        keys = np.array(np_state["keys"], dtype=np.uint32)
                        pos = int(np_state.get("pos", 0))
                        has_gauss = bool(np_state.get("has_gauss", 0))
                        cached_gaussian = float(np_state.get("cached_gaussian", 0.0))
                        np.random.set_state(
                            (algo, keys, pos, has_gauss, cached_gaussian)
                        )
                    else:
                        # already a tuple (pure-Python) — rare if user serialized differently
                        np.random.set_state(np_state)
            except Exception:
                logger.warning("Failed to set NumPy RNG state during snapshot load")

            # Cache RNG pack for deterministic replay (per-engine, not global)
            self._rng_saved = {
                "torch_cpu": rng_state.get("torch_cpu"),
                "torch_cuda": rng_state.get("torch_cuda"),
                "python": rng_state.get("python"),
                "numpy": rng_state.get("numpy"),
            }

            # ICNN state if present
            icnn_state = obj.get("icnn_state")
            if icnn_state is not None:
                if self.params.icnn is not None and hasattr(
                    self.params.icnn, "load_state_dict"
                ):
                    try:
                        self.params.icnn.load_state_dict(icnn_state)
                    except Exception:
                        logger.warning(
                            "Failed to load icnn state into provided icnn module"
                        )
                else:
                    logger.warning(
                        "Snapshot contains icnn state but engine has no icnn module to load into"
                    )

            # step count (prefer manifest, fallback obj)
            self.step_count = int(
                manifest.get("step_count", obj.get("step_count", self.step_count))
            )

        logger.info("Snapshot loaded from %s (step_count=%s)", dirpath, self.step_count)

    def _snapshot_blob(self, include_icnn: bool = True) -> bytes:
        """
        Build torch.save blob containing:
          - meta
          - K_matrix / K_U / K_V
          - last_good_state
          - rng states (torch CPU, torch CUDA, Python, NumPy as pure-Python)
          - optional icnn state_dict
        """
        # encode NumPy RNG state without NumPy arrays (pure-Python)
        np_algo, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
        np_keys_arr = np.asarray(np_keys)
        numpy_rng_pure = {
            "algo": str(np_algo),
            "keys": np_keys_arr.tolist(),  # list of ints
            "pos": int(np_pos),
            "has_gauss": bool(np_has_gauss),
            "cached_gaussian": float(np_cached),
        }

        obj = {
            "meta": {
                "step_count": int(self.step_count),
                "state_dim": int(self.params.state_dim),
                "dt": float(self.params.dt),
                "created_at": time.time(),
            },
            "K_matrix": (
                self._K_matrix.detach().cpu() if self._K_matrix is not None else None
            ),
            "K_U": self._K_U.detach().cpu() if self._K_U is not None else None,
            "K_V": self._K_V.detach().cpu() if self._K_V is not None else None,
            "last_good_state": (
                self._last_good_state.detach().cpu()
                if self._last_good_state is not None
                else None
            ),
            "rng": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "python": random.getstate(),
                "numpy": numpy_rng_pure,
            },
        }
        if (
            include_icnn
            and getattr(self, "icnn", None) is not None
            and hasattr(self.icnn, "state_dict")
        ):
            try:
                obj["icnn_state"] = self.icnn.state_dict()  # type: ignore[attr-defined]
            except Exception:
                logger.warning("Failed to dump icnn state_dict in snapshot")

        buf = io.BytesIO()
        torch.save(obj, buf)
        return buf.getvalue()

    # ----------------------------
    # Determinism & RNG helpers
    # ----------------------------
    def set_seed(self, seed: int) -> None:
        """Set seed for torch & CUDA and record it for snapshotting."""
        with self._lock:
            self._seed = int(seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
            logger.info("Engine seeded with {}", seed)

    def enable_deterministic_algorithms(self, flag: bool = True) -> None:
        """
        Attempt to enable torch.use_deterministic_algorithms for strict determinism.
        Note: Some CUDA kernels are non-deterministic and will raise if enabled.
        """
        try:
            torch.use_deterministic_algorithms(flag, warn_only=True)
            logger.info("torch.use_deterministic_algorithms set to {}", flag)
        except Exception as e:
            logger.warning("Failed to set deterministic algorithms: {}", str(e))

    # ----------------------------
    # Device & compile helpers
    # ----------------------------
    def set_device(self, device: torch.device) -> None:
        """Move K factors, ICNN, and last_good_state to device."""
        with self._lock:
            self.device = device
            K_matrix = self._K_matrix
            if K_matrix is not None:
                self._K_matrix = K_matrix.to(device=self.device, dtype=self.dtype)
            K_U = self._K_U
            if K_U is not None:
                self._K_U = K_U.to(device=self.device, dtype=self.dtype)
            K_V = self._K_V
            if K_V is not None:
                self._K_V = K_V.to(device=self.device, dtype=self.dtype)
            if self._last_good_state is not None:
                self._last_good_state = self._last_good_state.to(
                    device=self.device, dtype=self.dtype
                )
            if getattr(self, "icnn", None) is not None:
                try:
                    self.icnn.to(device=self.device)  # type: ignore[attr-defined]
                except Exception:
                    logger.warning("ICNN device move failed")
            logger.info("Engine moved to device {}", device)

    def set_dtype(self, dtype) -> None:
        """
        Change engine dtype and move buffers/ICNN to that dtype on current device.

        Accepts either a torch.dtype or a string alias:
          "float32"/"fp32", "float16"/"fp16"/"half", "float64"/"fp64"/"double".
        """
        with self._lock:
            # Coerce dtype if user passed a string
            if isinstance(dtype, str):
                _map = {
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "half": torch.float16,
                    "float64": torch.float64,
                    "fp64": torch.float64,
                    "double": torch.float64,
                }
                new_dtype = _map.get(dtype.lower())
                if new_dtype is None:
                    raise ValueError(f"Unsupported dtype string: {dtype}")
                self.dtype = new_dtype
            elif isinstance(dtype, torch.dtype):
                self.dtype = dtype
            else:
                raise TypeError(f"dtype must be torch.dtype or str, got {type(dtype)}")

            # Ensure self.device is a torch.device (defensive)
            if not isinstance(self.device, torch.device):
                self.device = torch.device(self.device)

            # Move K representations
            K_matrix = self._K_matrix
            if K_matrix is not None:
                self._K_matrix = K_matrix.to(device=self.device, dtype=self.dtype)
            K_U = self._K_U
            if K_U is not None:
                self._K_U = K_U.to(device=self.device, dtype=self.dtype)
            K_V = self._K_V
            if K_V is not None:
                self._K_V = K_V.to(device=self.device, dtype=self.dtype)

            # Move last_good_state if present
            if self._last_good_state is not None:
                self._last_good_state = self._last_good_state.to(
                    device=self.device, dtype=self.dtype
                )

            # Move ICNN if attached
            if getattr(self, "icnn", None) is not None:
                try:
                    self.icnn.to(device=self.device, dtype=self.dtype)
                except Exception:
                    logger.warning("ICNN dtype move failed")

            logger.info("Engine dtype set to {}", self.dtype)

    def compile_jit(self) -> None:
        """Optional JIT compile of hot-path helpers (may not support dynamic shapes)."""
        # We provide simple JIT wrappers where possible.
        try:
            # Try to script the conservative_update and project if shapes stable
            # Note: torch.jit may fail for dynamic control; wrap in try/except
            # We will replace methods with scripted versions if successful.
            # Use small example shapes for tracing (state_dim).
            d = self.params.state_dim
            example = torch.zeros((1, d), dtype=self.dtype, device=self.device)
            # Trace-only to warm kernels; do not replace bound methods.
            _ = torch.jit.trace(self.conservative_update, example)  # noqa: F841
            _ = torch.jit.trace(self.project, example)  # noqa: F841
            self._jit_compiled = True
            logger.info("JIT trace warmed conservative_update & project")
        except Exception as e:
            logger.warning("JIT compile failed: {}", str(e))
            self._jit_compiled = False

    # ----------------------------
    # Profiling helpers
    # ----------------------------
    def profile_step(
        self,
        x: torch.Tensor,
        trace_id: Optional[str] = None,
        sim_graft: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run step with simple wall-clock profiling for subcomponents."""
        with self._lock:

            def _sync():
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            t1 = _now_ms()
            x1 = _to_device(x.clone(), self.device, self.dtype)
            _sync()
            t2 = _now_ms()

            with torch.inference_mode():
                x_half = self.conservative_update(x1)
            _sync()
            t3 = _now_ms()

            with torch.inference_mode():
                x_next = self.dissipative_prox(x_half)
            _sync()
            t4 = _now_ms()

            with torch.inference_mode():
                x_proj = self.project(x_next)
            _sync()
            t5 = _now_ms()

            metrics = {
                "fn": "profile_step",
                "alloc_ms": t2 - t1,
                "conservative_ms": t3 - t2,
                "dissipative_ms": t4 - t3,
                "project_ms": t5 - t4,
                "total_ms": t5 - t1,
            }
            try:
                self.telemetry.record(METRIC_PROFILING, metrics)
            except Exception:
                logger.debug("telemetry profiling failed")
            return x_proj


# END OperatorSplitEngine
# -------------------------------------------------------------------------------

# ---------------------------
# Small ICNN stub and test utilities (optional)
# ---------------------------
# If you don't have a production ICNN ready, you can use the following minimal convex network stub
# for tests: it implements grad(x) by computing a simple convex quadratic plus small nonlinearity.
#
# It is not a substitute for a properly trained ICNN but useful for fast tests + unit tests.


class ICNNStub(torch.nn.Module):
    """Minimal convex-ish ICNN stub providing tape-free grad(x) for tests."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.lin = torch.nn.Linear(d, hidden)  # W1, b1  with shapes [h,d], [h]
        self.out = torch.nn.Linear(hidden, d)  # W2, b2  with shapes [d,h], [d]
        # light init near zero
        torch.nn.init.uniform_(self.lin.weight, a=-0.01, b=0.01)
        torch.nn.init.uniform_(self.out.weight, a=-0.01, b=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Potential per sample:
          h = ReLU(W1 x + b1)
          y = W2 h + b2
          φ(x) = 0.5 * <y(x), x>
        (This path can be used in training for param grads if needed.)
        """
        h = torch.relu(self.lin(x))
        v = 0.5 * torch.sum(self.out(h) * x, dim=1)
        return v

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytic gradient (tape-free):
          ∇φ(x) = 0.5 * ( y(x) + J(x)^T x )
        where:
          y(x) = W2 ReLU(W1 x + b1) + b2
          J(x) = ∂y/∂x = W2 Diag(1_{W1 x + b1 > 0}) W1
        """
        with torch.inference_mode():
            was_1d = x.ndim == 1
            if was_1d:
                x = x.unsqueeze(0)  # [1, d]
            B, d = x.shape

            # Promote to fp32 for stable accumulations
            x32 = x.to(torch.float32)

            W1 = self.lin.weight.to(torch.float32)  # [h, d]
            b1 = (self.lin.bias or torch.zeros(W1.size(0), device=W1.device)).to(
                torch.float32
            )  # [h]
            W2 = self.out.weight.to(torch.float32)  # [d, h]
            b2 = (self.out.bias or torch.zeros(d, device=W2.device)).to(
                torch.float32
            )  # [d]

            # pre = W1 x + b1, mask = ReLU'(pre)
            pre = x32 @ W1.T + b1  # [B, h]
            mask = (pre > 0).to(x32.dtype)  # [B, h]
            h = pre.clamp_min_(0.0)  # ReLU

            # y = W2 h + b2
            y = h @ W2.T + b2  # [B, d]

            # J^T x = W1^T ( (W2^T x) ⊙ mask )
            WT_x = x32 @ W2  # [B, h]  (W2^T x)
            JT_x = (WT_x * mask) @ W1  # [B, d]

            g32 = 0.5 * (y + JT_x)  # [B, d]
            g = g32.to(x.dtype)
            return g[0] if was_1d else g


# ---------------------------
# Demo / Smoke-run function (not executed on import)
# ---------------------------


def demo_small_run():
    """
    Quick smoke demo: create engine with small dim, run a few steps.
    Use for manual verification only.
    """
    d = 32
    icnn = ICNNStub(d)
    params = EngineParams(state_dim=d, device=torch.device("cpu"), icnn=icnn)
    engine = OperatorSplitEngine(params)
    x0 = torch.randn((1, d))
    engine.set_seed(42)
    engine.enforce_spectral_cap()
    x1 = engine.step(x0)
    xN = engine.step_many(x1, 10)
    print("Demo final norm", _safe_norm(xN))
    # save snapshot
    os.makedirs("Runtime/Snapshots", exist_ok=True)
    engine.save_snapshot("Runtime/Snapshots/demo_snapshot", sign=False, encrypt=False)

    # end demo


# -------------------------------------------------------------------------------
# End of file
