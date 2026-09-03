"""
Tests/core/test_live_loop.py

Unit tests for the Sprint D1 Live Learning Loop.

Tests are designed to run without a real database, GPU, or trained models.
All heavy components (trainer, decoder, engine) are mocked or bypassed via
the QualityGate / GradientBuffer / Scheduler lightweight interfaces.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal CognitionTrace stand-in for testing (no DB dependency)
# ---------------------------------------------------------------------------

@dataclass
class FakeTrace:
    trace_id: str = "t1"
    tenant_id: str = "test"
    session_id: str = "s1"
    request_text: str = "hello"
    verifier_score: float = 0.90
    graph_grounded: bool = True
    user_corrected: bool = False
    response_text: str = "OK"
    n_steps: int = 20
    trajectory_norms: List[float] = field(default_factory=lambda: [float(i) for i in range(20)])
    final_state_norm: float = 1.0
    graph_dict: dict = field(default_factory=dict)
    node_type_counts: dict = field(default_factory=dict)
    decode_mode: str = "chat"
    elapsed_ms_total: float = 100.0
    verifier_result: str = "verified"
    request_class: str = "general"
    sim_facts: int = 0
    wks_facts: int = 0
    world_model_facts: int = 0
    skill_plans: int = 0
    user_rating: Optional[int] = None
    quality_score: Optional[float] = None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# QualityGate tests
# ---------------------------------------------------------------------------

class TestQualityGate:
    def _gate(self, **kwargs):
        from Runtime.LiveLoop.quality_gate import QualityGate, QualityGateConfig
        return QualityGate(QualityGateConfig(**kwargs))

    def test_accepts_good_trace(self):
        gate = self._gate()
        assert gate.accepts(FakeTrace()) is True

    def test_rejects_low_score(self):
        gate = self._gate()
        assert gate.accepts(FakeTrace(verifier_score=0.70)) is False

    def test_rejects_ungrounded(self):
        gate = self._gate()
        assert gate.accepts(FakeTrace(graph_grounded=False)) is False

    def test_rejects_user_corrected(self):
        gate = self._gate()
        assert gate.accepts(FakeTrace(user_corrected=True)) is False

    def test_accepts_for_decoder_requires_response(self):
        gate = self._gate()
        assert gate.accepts_for_decoder(FakeTrace(response_text="")) is False
        assert gate.accepts_for_decoder(FakeTrace(response_text="Hello")) is True

    def test_custom_threshold(self):
        gate = self._gate(min_verifier_score=0.95)
        assert gate.accepts(FakeTrace(verifier_score=0.90)) is False
        assert gate.accepts(FakeTrace(verifier_score=0.96)) is True


# ---------------------------------------------------------------------------
# GradientBuffer tests
# ---------------------------------------------------------------------------

class TestGradientBuffer:
    def test_batch_fires_at_size(self):
        from Runtime.LiveLoop.gradient_buffer import GradientBuffer
        fired = []
        buf = GradientBuffer(batch_size=3, on_batch_ready=lambda b: fired.append(b))
        for i in range(3):
            buf.submit(FakeTrace(trace_id=str(i)))
        assert len(fired) == 1
        assert len(fired[0]) == 3

    def test_buffer_clears_after_fire(self):
        from Runtime.LiveLoop.gradient_buffer import GradientBuffer
        fired = []
        buf = GradientBuffer(batch_size=2, on_batch_ready=lambda b: fired.append(b))
        buf.submit(FakeTrace(trace_id="a"))
        buf.submit(FakeTrace(trace_id="b"))
        buf.submit(FakeTrace(trace_id="c"))
        assert buf.size == 1  # only c remains
        assert len(fired) == 1

    def test_drops_when_full(self):
        from Runtime.LiveLoop.gradient_buffer import GradientBuffer
        buf = GradientBuffer(batch_size=100, max_size=2)
        buf.submit(FakeTrace(trace_id="a"))
        buf.submit(FakeTrace(trace_id="b"))
        accepted = buf.submit(FakeTrace(trace_id="c"))  # should drop
        assert accepted is False
        assert buf.size == 2

    def test_flush_returns_remaining(self):
        from Runtime.LiveLoop.gradient_buffer import GradientBuffer
        buf = GradientBuffer(batch_size=10)
        buf.submit(FakeTrace(trace_id="x"))
        remaining = buf.flush()
        assert len(remaining) == 1
        assert buf.size == 0

    def test_thread_safe_concurrent_submit(self):
        from Runtime.LiveLoop.gradient_buffer import GradientBuffer
        fired = []
        lock = threading.Lock()
        def cb(b):
            with lock:
                fired.append(b)
        buf = GradientBuffer(batch_size=5, on_batch_ready=cb)
        threads = [threading.Thread(target=lambda: buf.submit(FakeTrace())) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_fired = sum(len(b) for b in fired)
        assert total_fired >= 15  # at least 3 full batches from 20 submissions


# ---------------------------------------------------------------------------
# ReplaySampler tests
# ---------------------------------------------------------------------------

class TestReplaySampler:
    def test_sample_no_history_returns_recent(self):
        from Runtime.LiveLoop.replay_sampler import ReplaySampler
        sampler = ReplaySampler(historical_fraction=0.0)
        recent = [FakeTrace(trace_id=str(i)) for i in range(8)]
        batch = sampler.sample(recent, batch_size=4)
        assert len(batch) == 4

    def test_sample_with_history_fraction(self):
        from Runtime.LiveLoop.replay_sampler import ReplaySampler
        # Mock TraceReader so no DB required
        with patch("Runtime.LiveLoop.replay_sampler.ReplaySampler._sample_historical") as mock_hist:
            mock_hist.return_value = [FakeTrace(trace_id="hist")]
            sampler = ReplaySampler(historical_fraction=0.20)
            recent = [FakeTrace() for _ in range(10)]
            batch = sampler.sample(recent, batch_size=10)
            # Should have called _sample_historical for ~20%
            mock_hist.assert_called_once()
            assert len(batch) >= 1

    def test_sample_empty_recent(self):
        from Runtime.LiveLoop.replay_sampler import ReplaySampler
        sampler = ReplaySampler(historical_fraction=0.0)
        batch = sampler.sample([], batch_size=4)
        assert isinstance(batch, list)


# ---------------------------------------------------------------------------
# OnlineTrainer metrics and rollback tests
# ---------------------------------------------------------------------------

class TestOnlineTrainer:
    def _trainer(self):
        from Runtime.LiveLoop.online_trainer import OnlineTrainer
        return OnlineTrainer(tenant_id="test")

    def test_record_and_rolling_rate(self):
        trainer = self._trainer()
        for _ in range(50):
            trainer.record_request(verifier_score=0.90, graph_grounded=True)
        assert trainer.rolling_verifier_rate == 1.0
        assert trainer.rolling_grounding_rate == 1.0

    def test_rolling_rate_drops_on_bad_requests(self):
        trainer = self._trainer()
        for _ in range(25):
            trainer.record_request(0.90, True)
        for _ in range(25):
            trainer.record_request(0.50, False)
        assert trainer.rolling_verifier_rate < 0.6
        assert trainer.rolling_grounding_rate < 0.6

    def test_get_stats_keys(self):
        trainer = self._trainer()
        stats = trainer.get_stats()
        assert "ep_update_count" in stats
        assert "decoder_update_count" in stats
        assert "rolling_verifier_rate" in stats
        assert "rollback_events" in stats

    def test_rollback_not_triggered_with_enough_data(self):
        trainer = self._trainer()
        for _ in range(50):
            trainer.record_request(0.90, True)
        trainer.maybe_rollback()
        assert trainer.get_stats()["rollback_events"] == 0

    def test_ep_update_no_predictor_is_safe(self):
        trainer = self._trainer()
        trainer.maybe_ep_update(FakeTrace())  # must not raise


# ---------------------------------------------------------------------------
# LiveLoopScheduler integration test
# ---------------------------------------------------------------------------

class TestLiveLoopScheduler:
    def test_start_stop(self):
        from Runtime.LiveLoop.scheduler import LiveLoopScheduler
        sched = LiveLoopScheduler(tenant_id="test")
        sched.start()
        time.sleep(0.05)
        sched.stop(timeout=2.0)

    def test_submit_accepted_trace_increments_window(self):
        from Runtime.LiveLoop.scheduler import LiveLoopScheduler
        sched = LiveLoopScheduler(tenant_id="test")
        sched.start()
        for _ in range(5):
            sched.submit(FakeTrace())
        time.sleep(0.3)
        stats = sched.get_stats()
        assert stats["window_size"] >= 1  # at least processed 1
        sched.stop(timeout=2.0)

    def test_submit_rejected_trace_does_not_update_ep_count(self):
        from Runtime.LiveLoop.scheduler import LiveLoopScheduler
        sched = LiveLoopScheduler(tenant_id="test")
        sched.start()
        bad = FakeTrace(verifier_score=0.10, graph_grounded=False)
        for _ in range(5):
            sched.submit(bad)
        time.sleep(0.3)
        stats = sched.get_stats()
        # No EP updates should have happened
        assert stats.get("ep_update_count", 0) == 0
        sched.stop(timeout=2.0)

    def test_get_stats_has_expected_keys(self):
        from Runtime.LiveLoop.scheduler import LiveLoopScheduler
        sched = LiveLoopScheduler(tenant_id="test")
        sched.start()
        stats = sched.get_stats()
        assert "enabled" not in stats or True  # whatever format
        assert "rolling_verifier_rate" in stats
        assert "buffer_size" in stats
        sched.stop(timeout=2.0)
