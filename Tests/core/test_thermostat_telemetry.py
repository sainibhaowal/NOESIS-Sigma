import time

from Core.OSC.Utils.thermostat_telemetry import (
    AsyncThermostatSampler,
    ThermostatSample,
    ThermostatStats,
)


def test_thermostat_stats_basic():
    stats = ThermostatStats()
    stats.observe(0.0, 64, 0.005)
    stats.observe(0.001, 128, 0.0025)
    stats.observe(0.25, 256, 0.001)

    snap = stats.snapshot()
    assert snap["steps"] == 3
    assert snap["min_energy"] <= 0.0
    assert snap["max_energy"] >= 0.25
    assert snap["max_S"] == 256
    assert snap["min_dt"] <= 0.001
    # spike_count should be > 0 because we changed energy a bit
    assert snap["spike_count"] >= 1


def test_async_sampler_flushes_batches(tmp_path):
    out = []

    def writer(batch):
        out.append(list(batch))

    sampler = AsyncThermostatSampler(writer=writer, sample_every=1, max_queue=16)

    for i in range(10):
        sampler.record(i, float(i) * 0.1, 64 + i, 0.005)

    # Allow worker thread to run
    time.sleep(0.1)
    sampler.close(drain=True)

    flat = [s for batch in out for s in batch]
    assert len(flat) == 10
    assert isinstance(flat[0], ThermostatSample)
