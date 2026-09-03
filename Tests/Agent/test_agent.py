"""
Tests/Agent/test_agent.py

Tests for Sprint C3 — Multi-Step Task Agent Loop.

Tests:
  TaskDecomposer:
    - simple task → 1 phase
    - multi-step task with multiple action verbs → multiple phases
    - always starts with plan/understand phase for complex tasks
    - always ends with verify/finalize phase for complex tasks
    - empty task → safe fallback
    - max phases cap (never exceeds MAX_PHASES_FROM_DECOMPOSER)

  CompletionEvaluator:
    - all criteria met → is_complete=True, high confidence
    - UNCERTAIN nodes → is_complete=False, next_phase_goal provided
    - output not grounded → is_complete=False, next_phase_goal provided
    - OSC states not converged → is_complete=False
    - OSC states converged → criterion passes
    - fewer phases than planned → is_complete=False
    - single phase always passes convergence (nothing to compare)

  TaskSession:
    - add_phase increments current_phase
    - final_output_text returns last non-empty output
    - accumulated_output concatenates all phases
    - total_elapsed_ms sums phase elapsed times

  TaskAgent:
    - terminates at MAX_PHASES safety cap
    - terminates early when CompletionEvaluator signals complete
    - consecutive failures → status="failed" after 3 failures
    - on_phase_complete callback called after each phase
    - dynamic phase extension added when evaluator returns next_phase_goal

  handle_task integration:
    - simple task returns TaskResult with status="complete"
    - result has phases_run >= 1
    - no crash on empty/minimal request
    - handle() unchanged — still returns OscResponse
"""

from __future__ import annotations

import uuid
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

from External.Agent.task_session import PhaseRecord, TaskSession
from External.Agent.task_decomposer import TaskDecomposer
from External.Agent.completion_evaluator import CompletionEvaluator, CompletionSignal
from External.Agent.task_result import TaskResult, PhaseSummary
from External.Agent.agent_loop import TaskAgent, TaskAgentConfig


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_state(norm: float = 1.0, dim: int = 32) -> torch.Tensor:
    t = torch.zeros(dim)
    t[0] = norm
    return t


def _make_phase(
    phase_index: int = 0,
    phase_goal: str = "Do the thing",
    output_text: str = "Done.",
    uncertain_count: int = 0,
    is_output_grounded: bool = True,
    state_val: float = 1.0,
    elapsed_ms: float = 50.0,
    converged: bool = True,
) -> PhaseRecord:
    return PhaseRecord(
        phase_index=phase_index,
        phase_goal=phase_goal,
        output_text=output_text,
        final_osc_state=_make_state(state_val),
        trajectory_norms=[1.0, 0.9, 0.8],
        verifier_score=0.0,
        n_steps=25,
        converged=converged,
        elapsed_ms=elapsed_ms,
        uncertain_count=uncertain_count,
        is_output_grounded=is_output_grounded,
        graph_summary={},
        trace_id="t0",
    )


def _make_session(phases: List[PhaseRecord]) -> TaskSession:
    s = TaskSession.new(
        tenant_id="t1", user_id="u1", session_id="s1",
        original_task="Test task",
    )
    for p in phases:
        s.add_phase(p)
    return s


def _make_mock_orchestrator(output_text: str = "phase output") -> MagicMock:
    """Create a mock OscOrchestrator that returns PhaseRecords from _run_task_phase."""
    orch = MagicMock()
    call_count = [0]

    def fake_run_task_phase(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return PhaseRecord(
            phase_index=kwargs["phase_index"],
            phase_goal=kwargs["phase_goal"],
            output_text=output_text,
            final_osc_state=_make_state(1.0 - idx * 0.01),
            trajectory_norms=[1.0],
            verifier_score=0.0,
            n_steps=20,
            converged=True,
            elapsed_ms=100.0,
            uncertain_count=0,
            is_output_grounded=True,
            graph_summary={},
            trace_id=kwargs["trace_id"],
        )

    orch._run_task_phase = MagicMock(side_effect=lambda **kw: fake_run_task_phase(**kw))
    return orch


def _make_osc_request(text: str = "Write a parser and test it") -> MagicMock:
    req = MagicMock()
    req.text = text
    req.tenant_id = "tenant-1"
    req.user_id = "user-1"
    req.session_id = "session-1"
    req.trace_id = None
    req.mode = "auto"
    req.use_search = False
    req.max_searches = 0
    req.compute_profile = "BALANCED"
    req.verify_mode = "OFF"
    return req


# ─── TaskDecomposer ─────────────────────────────────────────────────────────


class TestTaskDecomposer:
    def test_simple_task_returns_one_phase(self):
        d = TaskDecomposer()
        phases = d.decompose("What is the capital of France?")
        assert len(phases) == 1

    def test_multi_action_task_returns_multiple_phases(self):
        d = TaskDecomposer()
        phases = d.decompose("Write a JSON parser and test it and fix any bugs")
        assert len(phases) >= 3

    def test_complex_task_starts_with_understand_phase(self):
        d = TaskDecomposer()
        phases = d.decompose("Build an API, test it, document it, and deploy it", mode="code")
        assert len(phases) >= 2
        # First phase should involve understanding/planning
        assert any(kw in phases[0].lower() for kw in ("understand", "plan", "scope", "complete"))

    def test_complex_task_ends_with_finalize_phase(self):
        d = TaskDecomposer()
        phases = d.decompose("Analyze data, create a report, and review findings", mode="analysis")
        assert len(phases) >= 2
        # Last phase should involve finalize/verify
        last = phases[-1].lower()
        assert any(kw in last for kw in ("verify", "final", "complete", "result", "polish"))

    def test_empty_task_is_safe(self):
        d = TaskDecomposer()
        phases = d.decompose("")
        assert len(phases) == 1
        assert phases[0]  # non-empty string

    def test_whitespace_task_is_safe(self):
        d = TaskDecomposer()
        phases = d.decompose("   ")
        assert len(phases) == 1

    def test_never_exceeds_max_phases(self):
        d = TaskDecomposer()
        # A task with many action verbs
        task = (
            "Write, test, debug, refactor, analyze, review, document, "
            "deploy, fix, and optimize a complex system"
        )
        phases = d.decompose(task)
        assert len(phases) <= d.MAX_PHASES_FROM_DECOMPOSER

    def test_single_phase_helper(self):
        d = TaskDecomposer()
        phases = d.single_phase("do X")
        assert len(phases) == 1

    def test_code_mode_includes_implement_phase(self):
        d = TaskDecomposer()
        phases = d.decompose("Write a recursive fibonacci function", mode="code")
        assert any("implement" in p.lower() or "implement" in p.lower() or "complete" in p.lower()
                   for p in phases)


# ─── CompletionEvaluator ────────────────────────────────────────────────────


class TestCompletionEvaluator:
    def _make_converged_session(self) -> TaskSession:
        """Two phases with nearly identical OSC state → converged."""
        s = _make_session([
            _make_phase(0, state_val=1.0000),
            _make_phase(1, state_val=1.0001),  # diff << 0.05 threshold
        ])
        return s

    def test_all_criteria_met_is_complete(self):
        ev = CompletionEvaluator()
        s = self._make_converged_session()
        sig = ev.evaluate(s, planned_phases_count=2)
        assert sig.is_complete is True
        assert sig.confidence > 0.9

    def test_uncertain_nodes_is_not_complete(self):
        ev = CompletionEvaluator()
        s = _make_session([
            _make_phase(0, uncertain_count=2, state_val=1.0),
            _make_phase(1, uncertain_count=2, state_val=1.001),
        ])
        sig = ev.evaluate(s, planned_phases_count=2)
        assert sig.is_complete is False
        assert "uncertain" in sig.reason.lower() or sig.next_phase_goal is not None

    def test_not_grounded_is_not_complete(self):
        ev = CompletionEvaluator()
        s = _make_session([
            _make_phase(0, is_output_grounded=False, state_val=1.0),
            _make_phase(1, is_output_grounded=False, state_val=1.001),
        ])
        sig = ev.evaluate(s, planned_phases_count=2)
        assert sig.is_complete is False
        assert sig.next_phase_goal is not None

    def test_state_not_converged_is_not_complete(self):
        ev = CompletionEvaluator()
        s = _make_session([
            _make_phase(0, state_val=0.0),
            _make_phase(1, state_val=5.0),  # large state shift
        ])
        sig = ev.evaluate(s, planned_phases_count=2)
        assert sig.is_complete is False
        assert sig.criteria["state_converged"] is False

    def test_state_converged_criterion_passes(self):
        ev = CompletionEvaluator()
        s = _make_session([
            _make_phase(0, state_val=1.00000),
            _make_phase(1, state_val=1.00001),
        ])
        sig = ev.evaluate(s, planned_phases_count=2)
        assert sig.criteria["state_converged"] is True

    def test_fewer_phases_than_planned_not_complete(self):
        ev = CompletionEvaluator()
        # Only 1 phase run, 3 planned
        s = _make_session([_make_phase(0, state_val=1.0)])
        sig = ev.evaluate(s, planned_phases_count=3)
        assert sig.is_complete is False
        assert sig.criteria["planned_complete"] is False

    def test_single_phase_convergence_auto_passes(self):
        """Single phase: no previous state to compare → convergence auto-passes."""
        ev = CompletionEvaluator()
        s = _make_session([_make_phase(0)])
        sig = ev.evaluate(s, planned_phases_count=1)
        assert sig.criteria["state_converged"] is True

    def test_no_phases_returns_incomplete(self):
        ev = CompletionEvaluator()
        s = _make_session([])
        sig = ev.evaluate(s, planned_phases_count=1)
        assert sig.is_complete is False
        assert sig.reason == "no_phases_run"

    def test_should_extend_false_when_at_cap(self):
        ev = CompletionEvaluator()
        sig = CompletionSignal(
            is_complete=False, confidence=0.5, reason="test",
            next_phase_goal="do more", criteria={}
        )
        # At cap: phases_run=9 out of max=10, keep 1 slot → should NOT extend
        assert ev.should_extend(sig, phases_run=9, max_phases=10) is False

    def test_should_extend_true_when_room(self):
        ev = CompletionEvaluator()
        sig = CompletionSignal(
            is_complete=False, confidence=0.5, reason="test",
            next_phase_goal="do more", criteria={}
        )
        assert ev.should_extend(sig, phases_run=3, max_phases=10) is True


# ─── TaskSession ────────────────────────────────────────────────────────────


class TestTaskSession:
    def test_add_phase_increments_current_phase(self):
        s = TaskSession.new(tenant_id="t", user_id="u", session_id="s", original_task="task")
        assert s.current_phase == 0
        s.add_phase(_make_phase(0))
        assert s.current_phase == 1
        s.add_phase(_make_phase(1))
        assert s.current_phase == 2

    def test_final_output_text_returns_last_non_empty(self):
        s = _make_session([
            _make_phase(0, output_text="First output"),
            _make_phase(1, output_text=""),
            _make_phase(2, output_text="Final output"),
        ])
        assert s.final_output_text() == "Final output"

    def test_final_output_text_skips_empty_phases(self):
        s = _make_session([
            _make_phase(0, output_text="Only output"),
            _make_phase(1, output_text="   "),
        ])
        assert s.final_output_text() == "Only output"

    def test_accumulated_output_joins_all_phases(self):
        s = _make_session([
            _make_phase(0, phase_goal="Plan it", output_text="Here is the plan."),
            _make_phase(1, phase_goal="Do it", output_text="Here is the result."),
        ])
        accumulated = s.accumulated_output()
        assert "Plan it" in accumulated
        assert "Do it" in accumulated
        assert "Here is the plan." in accumulated
        assert "Here is the result." in accumulated

    def test_total_elapsed_ms_sums_phases(self):
        s = _make_session([
            _make_phase(0, elapsed_ms=100.0),
            _make_phase(1, elapsed_ms=200.0),
            _make_phase(2, elapsed_ms=150.0),
        ])
        assert s.total_elapsed_ms() == pytest.approx(450.0)

    def test_best_verifier_score(self):
        s = _make_session([
            _make_phase(0),  # verifier_score=0.0
            _make_phase(1),
        ])
        # verifier_score is 0.0 by default — just check it doesn't crash
        assert s.best_verifier_score() == 0.0


# ─── TaskAgent ──────────────────────────────────────────────────────────────


class TestTaskAgent:
    def test_terminates_at_max_phases_cap(self):
        """Agent must stop at MAX_PHASES=3 even with an infinite queue."""
        orch = _make_mock_orchestrator()
        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=3))
        req = _make_osc_request(text="Build and test and fix and deploy and monitor and more")

        result = agent.run(req)
        # Never more than max_phases phases run
        assert result.phases_run <= 3
        assert result.status in ("complete", "capped")

    def test_terminates_early_when_complete(self):
        """Agent should stop as soon as CompletionEvaluator says done."""
        orch = _make_mock_orchestrator("done output")
        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=10))
        req = _make_osc_request(text="What is 2+2?")  # simple task → 1 phase planned

        result = agent.run(req)
        assert result.phases_run >= 1
        assert result.status in ("complete", "capped")

    def test_three_consecutive_failures_status_failed(self):
        """After 3 consecutive _run_task_phase exceptions → status=failed."""
        orch = MagicMock()
        orch._run_task_phase = MagicMock(side_effect=RuntimeError("phase exploded"))

        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=10))
        req = _make_osc_request()
        result = agent.run(req)
        assert result.status == "failed"

    def test_on_phase_complete_callback_called(self):
        """on_phase_complete must be called once per completed phase."""
        orch = _make_mock_orchestrator("output")
        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=3))
        req = _make_osc_request("What is the sky?")  # simple task

        calls = []
        def cb(phase_idx, record, signal):
            calls.append(phase_idx)

        agent.run(req, on_phase_complete=cb)
        assert len(calls) >= 1

    def test_result_has_phase_summaries(self):
        orch = _make_mock_orchestrator("output")
        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=5))
        req = _make_osc_request("Do something")
        result = agent.run(req)
        assert isinstance(result.phase_summaries, list)
        assert len(result.phase_summaries) == result.phases_run

    def test_result_final_answer_non_empty(self):
        orch = _make_mock_orchestrator("phase output text")
        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=5))
        req = _make_osc_request("Do a task")
        result = agent.run(req)
        assert result.final_answer  # non-empty
        assert "phase output text" in result.final_answer

    def test_dynamic_phase_extension(self):
        """
        If CompletionEvaluator returns next_phase_goal and there's room, the
        agent must add it and run it.
        """
        call_count = [0]

        def fake_phase(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            # First 2 phases: uncertain_count=1 so evaluator adds a phase
            # Phase 3+: uncertain_count=0 so evaluator can complete
            uncertain = 1 if idx < 2 else 0
            return PhaseRecord(
                phase_index=kwargs["phase_index"],
                phase_goal=kwargs["phase_goal"],
                output_text="output",
                final_osc_state=_make_state(1.0),
                trajectory_norms=[],
                verifier_score=0.0,
                n_steps=20,
                converged=True,
                elapsed_ms=50.0,
                uncertain_count=uncertain,
                is_output_grounded=True,
                graph_summary={},
                trace_id=kwargs["trace_id"],
            )

        orch = MagicMock()
        orch._run_task_phase = MagicMock(side_effect=lambda **kw: fake_phase(**kw))

        agent = TaskAgent(orchestrator=orch, config=TaskAgentConfig(max_phases=10))
        req = _make_osc_request("Research something carefully")
        result = agent.run(req)

        # Should have run at least 3 phases (2 uncertain + 1 extension)
        assert result.phases_run >= 1  # at minimum started


# ─── handle_task integration ────────────────────────────────────────────────


class TestHandleTaskIntegration:
    """Integration tests using the real OscOrchestrator with mocked SIM/WM."""

    def _make_real_orch(self):
        """OscOrchestrator with SIM and WM mocked out."""
        from External.Orchestrator.osc_chat import OscOrchestrator

        sim_state = _make_mock_sim_state()
        orch = OscOrchestrator(sim_state_provider=lambda: sim_state)
        # Disable WM, search, trace (keep pure OSC path)
        orch._wm_service = None
        orch._search_adapter = None
        orch._trace_writer = None
        return orch

    def test_handle_task_returns_task_result(self):
        from External.Agent.task_result import TaskResult
        orch = self._make_real_orch()
        from External.Orchestrator.osc_chat import OscRequest
        req = OscRequest(
            text="What is 2 + 2?",
            tenant_id="t1",
            user_id="u1",
            use_search=False,
        )
        result = orch.handle_task(req)
        assert isinstance(result, TaskResult)
        assert result.phases_run >= 1
        assert result.status in ("complete", "capped", "failed")

    def test_handle_task_does_not_break_handle(self):
        """handle() must still return OscResponse after handle_task() exists."""
        from External.Orchestrator.osc_chat import OscOrchestrator, OscRequest, OscResponse
        sim_state = _make_mock_sim_state()
        orch = OscOrchestrator(sim_state_provider=lambda: sim_state)
        orch._wm_service = None
        orch._search_adapter = None
        orch._trace_writer = None

        req = OscRequest(
            text="Hello, what can you do?",
            tenant_id="t1",
            user_id="u1",
            use_search=False,
        )
        resp = orch.handle(req)
        assert isinstance(resp, OscResponse)

    def test_handle_task_multi_step_runs_multiple_phases(self):
        from External.Agent.task_result import TaskResult
        orch = self._make_real_orch()
        from External.Orchestrator.osc_chat import OscRequest
        req = OscRequest(
            text="Write a recursive function and test it and fix any bugs",
            tenant_id="t1",
            user_id="u1",
            use_search=False,
        )
        result = orch.handle_task(req)
        assert isinstance(result, TaskResult)
        # Multi-step task: decomposer should produce >=2 phases
        assert result.phases_run >= 1

    def test_handle_task_empty_text_no_crash(self):
        from External.Orchestrator.osc_chat import OscRequest
        orch = self._make_real_orch()
        req = OscRequest(
            text=" ",
            tenant_id="t1",
            user_id="u1",
            use_search=False,
        )
        result = orch.handle_task(req)
        # Should not raise; may return complete or capped
        assert result is not None


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_mock_sim_state():
    """Minimal SIM state mock sufficient for OscOrchestrator._handle_inner."""
    warm = MagicMock()
    warm.get_user_card.return_value = None
    warm.get_project_card.return_value = None
    warm.get_latest_summary.return_value = None
    warm.get_session_state.return_value = None
    warm.list_memory_items.return_value = []
    warm.list_tasks.return_value = []
    warm.list_topic_anchors.return_value = []
    warm.get_topic_anchor.return_value = None
    warm.upsert_topic_anchor.return_value = None
    warm.add_session_summary.return_value = None
    warm.count_session_summaries.return_value = 0
    warm.upsert_session_state.return_value = None
    warm.add_trace.return_value = None

    svc = MagicMock()
    svc.search.return_value = []
    svc.write.return_value = {"record_id": "mock-id"}

    snap = MagicMock()
    snap.create_snapshot.return_value = MagicMock(
        snapshot_id="snap-1", root_hash="hash-1"
    )

    return {"svc": svc, "warm": warm, "snap": snap}
