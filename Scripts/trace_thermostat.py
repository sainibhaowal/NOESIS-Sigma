
#!/usr/bin/env python3
"""
NOESIS-Σ — Thermostat trace (deterministic, CPU-only)

- Produces an identical 13-row CSV on every run (steps 0..12).
- Uses a fixed energy sequence, fixed seeds, deterministic toggles, and stable formatting.
- Writes to Runtime/Logs/thermo_trace_<UTC>.csv and prints the path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo = Path(__file__).resolve()
    for _ in range(6):
        if (repo / "pyproject.toml").exists():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

_ensure_repo_root()

import csv
import random
from datetime import datetime, timezone
from typing import Callable, Final

import numpy as np
import torch

# Determinism & environment toggles (set before importing thermostat)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if CUDA is used, keep matmul deterministic
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from Core.OSC.Control.thermostat import Thermostat, ThermostatConfig  # noqa: E402

# ---------------- Fixed energy sequence ----------------

E_SEQ: Final = [
    0.0000,
    0.0006,
    0.0011,
    0.0014,
    0.0018,
    0.0020,
    0.0021,  # calm region
    0.2521,  # spike / transient
    0.2522,
    0.2523,
    0.2524,
    0.2525,
    0.2526,  # calm again
]


def _fmt(v: float) -> str:
    """
    Stable float formatting for identical hashes across runs.
    """
    if isinstance(v, float):
        s = f"{v:.10f}"
        s = s.rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def main() -> int:
    # Thermostat config (must match tests)
    cfg = ThermostatConfig(
        s_min=8,
        s_max=256,
        dt_min=1e-4,
        dt_max=5e-2,
        lower_band=1e-3,
        upper_band=5e-2,
        check_interval=1,  # evaluate every call
        window_M=3,
        warmup_checks=0,
        downscale_S=0.5,
        upscale_S=1.5,
        up_dt_factor=1.25,
        down_dt_factor=0.75,
        max_ref_batch=0,  # unused in energy-only path
    )
    thermo = Thermostat(cfg)

    # Start knobs chosen to match earlier traces
    S = 64
    dt = 5e-3

    outdir = Path("Runtime/Logs")
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"thermo_trace_{ts}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "E", "S", "dt"])

        def _energy_fn_i(_x: object, e: float) -> float:
            return e

        def _energy_fn_bound(e: float) -> Callable[[object], float]:
            def _fn(_x: object) -> float:
                return _energy_fn_i(_x, e)

            return _fn

        for i, E in enumerate(E_SEQ):
            # energy-only mode: x=None, energy provided by closure
            S, dt = thermo.maybe_update(
                step_idx=i,
                x=None,
                S=S,
                dt=dt,
                energy_fn=_energy_fn_bound(E),
            )
            # Write AFTER update to reflect controller decision at step i
            writer.writerow([i, _fmt(E), int(S), _fmt(dt)])

    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
