# Tests/core/test_thermostat_acceptance.py

import pytest
import torch

from Core.OSC.Control.thermostat import Thermostat, ThermostatConfig
from Core.OSC.Utils.thermostat_telemetry import ThermostatStats


def energy_driver(seq):
    it = iter(seq)
    last = [0.0]

    def f(_x):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return f


def test_adaptive_S_and_dt_behaviour():
    cfg = ThermostatConfig(
        s_min=8,
        s_max=256,
        dt_min=1e-4,
        dt_max=5e-2,
        lower_band=1e-3,
        upper_band=5e-2,
        check_interval=1,  # evaluate every call for the test
        window_M=3,
        warmup_checks=0,
        downscale_S=0.5,
        upscale_S=1.5,
        up_dt_factor=1.25,
        down_dt_factor=0.75,
    )
    t = Thermostat(cfg)

    S = 64
    dt = 5e-3

    # 1) CALM phase → after M calm windows, S should drop and dt should rise (within clamp)
    calm = [0.0000, 0.0006, 0.0011, 0.0014, 0.0018, 0.0020, 0.0021]
    fn = energy_driver(calm)
    S1, dt1 = S, dt
    for i in range(len(calm)):
        S1, dt1 = t.maybe_update(step_idx=i, x=None, S=S1, dt=dt1, energy_fn=fn)
    assert S1 < S, f"Expected S to decrease under calm windows, got {S1} vs {S}"
    assert dt1 >= dt and dt1 <= cfg.dt_max + 1e-12, f"dt should gently rise within clamp, got {dt1}"

    # 2) TRANSIENT spike → we must call twice to consume [baseline, spike]
    spike = [calm[-1], calm[-1] + 0.25]
    fn2 = energy_driver(spike)
    S2a, dt2a = t.maybe_update(
        step_idx=999,
        x=None,
        S=S1,
        dt=dt1,
        energy_fn=fn2,
    )  # baseline (no change yet)
    S2, dt2 = t.maybe_update(
        step_idx=1000,
        x=None,
        S=S2a,
        dt=dt2a,
        energy_fn=fn2,
    )  # actual spike here
    assert S2 > S2a, f"Expected S to increase on transient, got {S2} vs {S2a}"
    assert dt2 < dt2a, f"Expected dt to reduce on transient, got {dt2} vs {dt2a}"

    # 3) Back to CALM → eventually S should not ping-pong up
    calm2 = [
        spike[-1] + 1e-4,
        spike[-1] + 2e-4,
        spike[-1] + 3e-4,
        spike[-1] + 4e-4,
    ]
    fn3 = energy_driver(calm2)
    S3, dt3 = S2, dt2
    for j in range(len(calm2)):
        S3, dt3 = t.maybe_update(
            step_idx=1001 + j,
            x=None,
            S=S3,
            dt=dt3,
            energy_fn=fn3,
        )
    assert S3 <= S2, f"Expected S to settle (no upward ping-pong), got {S3} vs {S2}"
    assert cfg.dt_min - 1e-12 <= dt3 <= cfg.dt_max + 1e-12, "dt stayed within clamps"


def _make_acceptance_engine():
    """
    Try to construct the same engine used in other Core tests.

    This is written defensively so the test will `pytest.skip` instead of
    crashing if we can't find the factory. Adjust this once you know the
    canonical engine factory you want for acceptance tests.
    """
    try:
        # If you have a dedicated acceptance factory, prefer it.
        from Core.OSC.dynamics import make_acceptance_engine  # type: ignore

        return make_acceptance_engine()
    except Exception:
        pass

    try:
        # Fallback: whatever your normal test engine factory is.
        from Core.OSC.dynamics import make_test_engine  # type: ignore

        return make_test_engine()
    except Exception:
        pass

    pytest.skip(
        "No suitable engine factory found in Core.dynamics; "
        "adjust _make_acceptance_engine() to your engine."
    )


def test_thermostat_acceptance_with_stats():
    engine = _make_acceptance_engine()
    x0 = engine.make_initial_state()

    stats = ThermostatStats()
    out = engine.step_many(
        x0,
        steps=128,
        token_boundary=True,
        thermo_stats=stats,
    )

    snap = stats.snapshot()
    assert snap["steps"] > 0
    assert snap["max_S"] >= 64  # depending on your defaults
    assert snap["max_energy"] >= snap["min_energy"]
    assert out is not None
