# Tests/ops/test_thermostat.py
import os

import pytest

from Core.Ops.thermostat import Thermostat


class _FakeTelemetry:
    def __init__(self):
        self.events = []

    def record(self, name, payload):
        self.events.append((name, payload))

class _FakeParams:
    def __init__(self, dt=0.01, max_step_time_ms=1.0):
        self.dt = dt
        self.max_step_time_ms = max_step_time_ms

class _FakeEngine:
    def __init__(self, snapshot_base):
        self.params = _FakeParams()
        self.telemetry = _FakeTelemetry()
        self._snapshots = []
        self._snapshot_base = snapshot_base

    def save_snapshot(self, dirpath, sign=False, encrypt=False):
        # Create a marker file to prove snapshot happened
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "SNAP_OK"), "w") as f:
            f.write("ok")

@pytest.mark.fast
def test_thermostat_reduces_dt_and_saves_snapshot(tmp_path):
    # Configure thermostat with very low threshold to trigger immediately
    alerts_log = tmp_path / "alerts.log"
    snap_base = tmp_path / "snaps"

    tstat = Thermostat(
        dt_min=1e-5,
        damping=0.9,
        warn_threshold=1,          # trigger on first warning
        window_sec=60,
        snapshot_on_trigger=True,
        snapshot_base=str(snap_base),
        alerts_log=str(alerts_log),
        proj_burst=10
    )
    eng = _FakeEngine(snapshot_base=str(snap_base))

    # Simulate a slow step: elapsed 100ms, max allowed 1ms => warning
    metrics = {
        "elapsed_ms": 100.0,
        "step_count": 1,
        "projection_count": 0,
        "trace_id": "test"
    }
    old_dt = eng.params.dt
    tstat(eng, metrics)

    # dt reduced
    assert eng.params.dt < old_dt
    assert eng.params.dt == pytest.approx(old_dt * 0.9)

    # Snapshot directory created with marker file
    entries = list(snap_base.iterdir())
    assert any(p.is_dir() and (p / "SNAP_OK").exists() for p in entries)

    # Alerts log written
    assert alerts_log.exists()
    with open(alerts_log, "r") as f:
        lines = f.readlines()
    assert any("thermostat_alert" in line for line in lines)
