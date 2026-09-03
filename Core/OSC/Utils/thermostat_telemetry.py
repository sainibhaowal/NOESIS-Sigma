"""
Core/Utils/thermostat_telemetry.py

Hot-path-safe telemetry helpers for the Thermostat and dynamics loop.

Goals:
- No Python logging calls inside inner step loops.
- O(1) work per step: just a few int/float ops.
- Optional sampled + async export of raw samples for diagnostics / tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Iterable, List, Optional


@dataclass(frozen=True)
class ThermostatSample:
    """
    Lightweight container for a single thermostat observation.
    Intended for off-thread export and trace scripts.
    """

    step: int
    energy: float
    S: int
    dt: float


class ThermostatStats:
    """
    Cheap aggregate statistics for a thermostat run.

    Safe to call from the hot path:
    - Only integer increments and a few float comparisons per step.
    - No logging, no allocations beyond attribute updates.
    """

    __slots__ = (
        "steps",
        "min_energy",
        "max_energy",
        "max_S",
        "min_dt",
        "_prev_energy",
        "spike_count",
    )

    def __init__(self) -> None:
        self.steps: int = 0
        self.min_energy: float = float("inf")
        self.max_energy: float = float("-inf")
        self.max_S: int = 0
        self.min_dt: float = float("inf")
        self._prev_energy: Optional[float] = None
        self.spike_count: int = 0

    def observe(self, energy: float, S: int, dt: float) -> None:
        """
        Update statistics with a single step observation.

        Call this from inside the inner loop if you want aggregate stats.
        """
        self.steps += 1

        if energy < self.min_energy:
            self.min_energy = energy
        if energy > self.max_energy:
            self.max_energy = energy
        if S > self.max_S:
            self.max_S = S
        if dt < self.min_dt:
            self.min_dt = dt

        if self._prev_energy is not None:
            # Simple, tunable spike heuristic; threshold can be adjusted.
            if abs(energy - self._prev_energy) > 1e-3:
                self.spike_count += 1

        self._prev_energy = energy

    def snapshot(self) -> dict:
        """
        Return a JSON-serialisable view of the current stats.
        Safe to log from outside the hot path.
        """
        if self.steps == 0:
            return {
                "steps": 0,
                "min_energy": 0.0,
                "max_energy": 0.0,
                "max_S": 0,
                "min_dt": 0.0,
                "spike_count": 0,
            }

        return {
            "steps": self.steps,
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
            "max_S": self.max_S,
            "min_dt": self.min_dt,
            "spike_count": self.spike_count,
        }


class AsyncThermostatSampler:
    """
    Optional sampled, async exporter for raw thermostat samples.

    Design:
    - Hot path only does: modulo, object construction, queue.put_nowait().
    - Background thread drains a queue and calls a user-supplied writer.
    - Intended for dev tooling (e.g. Scripts/trace_thermostat.py), not
      enabled by default in production runs.
    """

    def __init__(
        self,
        writer: Callable[[Iterable[ThermostatSample]], None],
        *,
        sample_every: int = 1,
        max_queue: int = 8192,
    ) -> None:
        if sample_every <= 0:
            raise ValueError("sample_every must be >= 1")

        self._writer = writer
        self._sample_every = int(sample_every)
        self._queue: "Queue[ThermostatSample]" = Queue(maxsize=max_queue)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="thermo_sampler", daemon=True)
        self._thread.start()

    def record(self, step: int, energy: float, S: int, dt: float) -> None:
        """
        Lightweight hot-path entrypoint.

        Only enqueues every `sample_every`-th step; drops samples if the queue
        is full instead of blocking.
        """
        if (step % self._sample_every) != 0:
            return

        sample = ThermostatSample(step=step, energy=energy, S=S, dt=dt)

        try:
            self._queue.put_nowait(sample)
        except Full:  # type: ignore[name-defined]
            # Backpressure: drop the sample silently rather than stalling
            # the dynamics loop.
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.5)
            except Empty:
                continue

            batch: List[ThermostatSample] = [first]
            try:
                while True:
                    batch.append(self._queue.get_nowait())
            except Empty:
                pass

            try:
                self._writer(batch)
            except Exception:
                # Intentionally swallow exceptions here; diagnostics must never
                # take down the core engine.
                continue

    def close(self, *, drain: bool = True) -> None:
        """
        Stop the worker thread. Optionally drain remaining samples first.
        """
        if drain:
            try:
                batch: List[ThermostatSample] = []
                while True:
                    batch.append(self._queue.get_nowait())
            except Empty:
                if batch:
                    try:
                        self._writer(batch)
                    except Exception:
                        pass

        self._stop.set()
        self._thread.join(timeout=1.0)
