# /home/sephi-asi/NOESIS-Σ/Core/Control/thermostat.py
# ─────────────────────────────────────────────────────────────────────────────
# NOESIS-Σ • Adaptive Thermostat for (S, Δt)
# Golden Edition: deterministic, minimal-overhead, no autograd tape.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import ceil, floor
from typing import Callable, Dict, Optional, Tuple

import torch

# Accept either a tensor or None (tooling/tests may pass None) and allow float or 0-D tensor return.
EnergyFn = Callable[[Optional[torch.Tensor]], torch.Tensor | float]

__all__ = ["ThermostatConfig", "Thermostat"]


@dataclass(frozen=True)
class ThermostatConfig:
    # Limits
    s_min: int = 8
    s_max: int = 256
    dt_min: float = 1e-4
    dt_max: float = 5e-2

    # Calm / transient bands (normalized deltas)
    lower_band: float = 1e-3
    upper_band: float = 5e-2

    # Control cadence + hysteresis
    window_M: int = 3  # calm hits needed before downshift
    check_interval: int = 4  # compute metrics every N inner steps
    warmup_checks: int = 2  # ignore the first K checks after reset

    # Adjust factors
    downscale_S: float = 0.5  # calm → shrink S
    upscale_S: float = 1.5  # transient → grow S
    up_dt_factor: float = 1.25  # calm → enlarge dt
    down_dt_factor: float = 0.75  # transient → shrink dt

    # Metrics
    max_ref_batch: int = 0  # 0 = use full x; >0 = use x[:max_ref_batch]
    eps: float = 1e-12
    energy_floor: float = 1e-3  # avoid divide-by-near-zero in dE normalization

    def validate(self) -> None:
        if self.s_min < 1:
            raise ValueError("s_min must be ≥ 1.")
        if self.s_max < self.s_min:
            raise ValueError("s_max must be ≥ s_min.")
        if not (self.dt_min > 0.0):
            raise ValueError("dt_min must be > 0.")
        if not (self.dt_max > self.dt_min):
            raise ValueError("dt_max must be > dt_min.")
        if not (0.0 <= self.lower_band < self.upper_band):
            raise ValueError("Expect 0 ≤ lower_band < upper_band.")
        if self.window_M < 1:
            raise ValueError("window_M must be ≥ 1.")
        if self.check_interval < 1:
            raise ValueError("check_interval must be ≥ 1.")
        if self.downscale_S <= 0.0:
            raise ValueError("downscale_S must be > 0.")
        if self.upscale_S <= 1.0:
            raise ValueError("upscale_S must be > 1.0.")
        if self.up_dt_factor <= 1.0:
            raise ValueError("up_dt_factor must be > 1.0.")
        if not (0.0 < self.down_dt_factor < 1.0):
            raise ValueError("down_dt_factor must be in (0, 1).")


class Thermostat:
    """
    Adaptive controller for (S, dt):
      • Reduce S when both normalized deltas (ΔE/E, Δμ/‖μ‖) stay below lower_band
        for M windows in a row.
      • Raise S only on transients (either metric exceeds upper_band).
      • dt expands on calm; shrinks on transients; always clamped.
      • Lightweight oscillation damper freezes S changes if the signal flips often.
    """

    def __init__(self, cfg: ThermostatConfig):
        cfg.validate()
        self.cfg = cfg
        self._prev_energy: Optional[torch.Tensor] = None  # scalar tensor
        self._prev_mu: Optional[torch.Tensor] = None  # vector tensor
        self._calm_streak: int = 0
        self._checks_seen: int = 0
        # Track last decisions to avoid ping-pong on borderline regimes
        self._sign_hist: deque[int] = deque(maxlen=8)  # -1=calm, 0=mid, +1=transient

    # ── public API ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def maybe_update(
        self,
        step_idx: int,
        x: Optional[torch.Tensor],
        S: int,
        dt: float,
        energy_fn: Optional[EnergyFn] = None,
    ) -> Tuple[int, float]:
        """
        Called cheaply every inner step; computes at check_interval.
        Returns (S_new, dt_new) — the engine may *apply* these at token boundaries.
        """
        if (step_idx + 1) % self.cfg.check_interval != 0:
            return S, dt

        dE_norm, dmu_norm = self._compute_metrics(x, energy_fn)
        self._checks_seen += 1
        if self._checks_seen <= self.cfg.warmup_checks:
            return S, dt  # warm-up: observe only

        # Decide regime
        transient = (dE_norm > self.cfg.upper_band) or (dmu_norm > self.cfg.upper_band)
        calm = (dE_norm < self.cfg.lower_band) and (dmu_norm < self.cfg.lower_band)

        # Oscillation damper: record coarse "sign"
        amp = max(dE_norm, dmu_norm)
        sign = (
            1 if amp > self.cfg.upper_band else (-1 if amp < self.cfg.lower_band else 0)
        )
        self._sign_hist.append(sign)
        freeze_S = self._detect_oscillation()

        # Update logic
        if transient:
            self._calm_streak = 0
            S_new = S if freeze_S else self._clamp_S_up(S)
            dt_new = self._clamp_dt_down(dt)
            return S_new, dt_new

        if calm:
            self._calm_streak += 1
            if self._calm_streak >= self.cfg.window_M:
                self._calm_streak = 0
                S_new = S if freeze_S else self._clamp_S_down(S)
                dt_new = self._clamp_dt_up(dt)
                return S_new, dt_new
            # Not enough calm windows yet → keep as-is
            return S, dt

        # mid band → neither calm nor transient: keep knobs steady
        self._calm_streak = 0
        return S, dt

    @torch.no_grad()
    def force_update(
        self,
        x: Optional[torch.Tensor],
        S: int,
        dt: float,
        energy_fn: Optional[EnergyFn] = None,
    ) -> Tuple[int, float]:
        """
        On-demand single evaluation (ignores cadence/warmup).
        """
        dE_norm, dmu_norm = self._compute_metrics(x, energy_fn)
        transient = (dE_norm > self.cfg.upper_band) or (dmu_norm > self.cfg.upper_band)
        calm = (dE_norm < self.cfg.lower_band) and (dmu_norm < self.cfg.lower_band)

        amp = max(dE_norm, dmu_norm)
        sign = (
            1 if amp > self.cfg.upper_band else (-1 if amp < self.cfg.lower_band else 0)
        )
        self._sign_hist.append(sign)
        freeze_S = self._detect_oscillation()

        if transient:
            self._calm_streak = 0
            S_new = S if freeze_S else self._clamp_S_up(S)
            dt_new = self._clamp_dt_down(dt)
            return S_new, dt_new

        if calm:
            self._calm_streak += 1
            if self._calm_streak >= self.cfg.window_M:
                self._calm_streak = 0
                S_new = S if freeze_S else self._clamp_S_down(S)
                dt_new = self._clamp_dt_up(dt)
                return S_new, dt_new

        self._calm_streak = 0
        return S, dt

    def reset(self) -> None:
        self._prev_energy = None
        self._prev_mu = None
        self._calm_streak = 0
        self._checks_seen = 0
        self._sign_hist.clear()

    def state_dict(self) -> Dict[str, object]:
        st: Dict[str, object] = {
            "cfg": asdict(self.cfg),
            "calm_streak": int(self._calm_streak),
            "checks_seen": int(self._checks_seen),
        }
        if self._prev_energy is not None:
            st["prev_energy"] = self._prev_energy.detach().cpu()
        if self._prev_mu is not None:
            st["prev_mu"] = self._prev_mu.detach().cpu()
        return st

    def load_state_dict(self, state: Dict[str, object]) -> None:
        calm = state.get("calm_streak", 0)
        checks = state.get("checks_seen", 0)
        self._calm_streak = int(calm) if isinstance(calm, (int, float, str)) else 0
        self._checks_seen = int(checks) if isinstance(checks, (int, float, str)) else 0
        pe = state.get("prev_energy", None)
        pm = state.get("prev_mu", None)
        self._prev_energy = pe if isinstance(pe, torch.Tensor) else None
        self._prev_mu = pm if isinstance(pm, torch.Tensor) else None

    # ── internals ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _compute_metrics(
        self, x: Optional[torch.Tensor], energy_fn: Optional[EnergyFn]
    ) -> Tuple[float, float]:
        """
        Returns:
          dE_norm: |E_t - E_{t-1}| / max(|E_{t-1}|, eps)  (with safe floor)
          dmu_norm: ||μ_t - μ_{t-1}|| / max(||μ_{t-1}||, eps)
        Energy is computed on the same sampled subset as μ. Supports x=None (energy-only).
        """
        eps = float(getattr(self.cfg, "eps", 1e-12))
        energy_floor = float(getattr(self.cfg, "energy_floor", 1e-3))
        lower_band = float(getattr(self.cfg, "lower_band", 1e-3))

        # ---- Energy-only mode (x=None): used by tests/tools ----
        if x is None:
            if energy_fn is None:
                return 0.0, 0.0
            E_raw = energy_fn(None)
            if isinstance(E_raw, torch.Tensor):
                e_val = float(E_raw.detach().cpu().item())
            else:
                e_val = float(E_raw)
            if self._prev_energy is None:
                self._prev_energy = torch.tensor(e_val, dtype=torch.float32)
                return 0.0, 0.0
            prev_val = float(self._prev_energy.item())
            dE_abs = abs(e_val - prev_val)
            # Tiny absolute change → calm regardless of normalization.
            if dE_abs < lower_band:
                dE_norm = 0.0
            else:
                denom = max(max(abs(prev_val), abs(e_val)), energy_floor)
                dE_norm = dE_abs / denom
            self._prev_energy = torch.tensor(e_val, dtype=torch.float32)
            return dE_norm, 0.0

        # ---- Tensor path ----
        device = x.device
        x_sample = (
            x[: self.cfg.max_ref_batch]
            if (self.cfg.max_ref_batch > 0 and x.shape[0] > self.cfg.max_ref_batch)
            else x
        )
        mu = x_sample.mean(dim=0, dtype=torch.float32)
        if energy_fn is None:
            E = (0.5 * (x_sample.float() * x_sample.float()).mean()).to(device=device)
        else:
            # Use the same sampled subset to keep cost consistent
            E_raw = energy_fn(x_sample)
            if isinstance(E_raw, torch.Tensor):
                if E_raw.numel() != 1:
                    raise ValueError(
                        "energy_fn(x) must return a scalar (0-dim or 1-elem) tensor or float."
                    )
                E = E_raw.to(device=device, dtype=torch.float32)
            else:
                E = torch.tensor(float(E_raw), device=device, dtype=torch.float32)

        # dE normalized (with tiny-ΔE calm override)
        if self._prev_energy is None:
            dE_norm_t = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            dE_abs_t = (E - self._prev_energy).abs()
            if float(dE_abs_t.item()) < lower_band:
                dE_norm_t = torch.tensor(0.0, device=device, dtype=torch.float32)
            else:
                denom_E = torch.maximum(
                    torch.maximum(self._prev_energy.abs(), E.abs()),
                    torch.tensor(energy_floor, device=device),
                )
                dE_norm_t = dE_abs_t / denom_E

        # dμ normalized
        if self._prev_mu is None:
            dmu_norm_t = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            mu_diff = mu - self._prev_mu
            prev_mu_norm = torch.maximum(
                self._prev_mu.norm(p=2), torch.tensor(eps, device=device)
            )
            dmu_norm_t = mu_diff.norm(p=2) / prev_mu_norm

        # Persist references
        self._prev_energy = E.detach().to(device=device, dtype=torch.float32)
        self._prev_mu = mu.detach().to(device=device, dtype=torch.float32)

        return float(dE_norm_t.item()), float(dmu_norm_t.item())

    def _detect_oscillation(self) -> bool:
        """
        Detect frequent flips (+1,-1,+1,-1,...) in recent signal;
        if detected, freeze S adjustments for this call.
        """
        if len(self._sign_hist) < 6:
            return False
        flips = 0
        last = None
        for s in self._sign_hist:
            if last is not None and (s * last) < 0:
                flips += 1
            if s != 0:
                last = s
        return flips >= 3

    # Clamp helpers
    def _clamp_S_up(self, S: int) -> int:
        S_up = int(ceil(float(S) * self.cfg.upscale_S))
        return max(self.cfg.s_min, min(self.cfg.s_max, S_up))

    def _clamp_S_down(self, S: int) -> int:
        S_dn = int(floor(float(S) * self.cfg.downscale_S))
        return max(self.cfg.s_min, min(self.cfg.s_max, S_dn))

    def _clamp_dt_up(self, dt: float) -> float:
        dt_new = dt * self.cfg.up_dt_factor
        return float(min(self.cfg.dt_max, dt_new))

    def _clamp_dt_down(self, dt: float) -> float:
        dt_new = dt * self.cfg.down_dt_factor
        return float(max(self.cfg.dt_min, dt_new))
