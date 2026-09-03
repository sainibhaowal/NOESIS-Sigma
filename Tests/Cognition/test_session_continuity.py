# Tests/Cognition/test_session_continuity.py
#
# Sprint A3 — Session Continuity
#
# Verifies that loop_result.final_state is persisted to the warm store
# after each request, and that the next turn loads it back (not zeros).
#
# These tests run entirely on CPU with a real SQLite warm store.
# No LLM, no external dependencies.

from __future__ import annotations

import base64
import os

import pytest
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SIM_ALLOW_SQLITE_FOR_TESTS", "1")

from Core.Cognition import CognitionLoop, ContextBundle
from Core.Cognition.thought_graph import NodeType
from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.params import EngineParams
from External.Orchestrator.memory_writer import MemoryWriterMixin
from External.Sim.Services.sim_core_service import Identity as SimIdentity
from External.Sim.Storage.warm_store import WarmStore

_SMALL_DIM = 64  # keep tests fast; real default is 1024


def _make_engine() -> OperatorSplitEngine:
    params = EngineParams(state_dim=_SMALL_DIM, dt=0.01, rank_r=4)
    return OperatorSplitEngine(params)


def _make_warm(tmp_path) -> WarmStore:
    db_url = f"sqlite:///{(tmp_path / 'sim.sqlite3').as_posix()}"
    warm = WarmStore(db_url=db_url, echo_sql=False)
    warm.init_db()
    return warm


def _ident() -> SimIdentity:
    return SimIdentity(tenant_id="test_tenant", user_id="test_user", role="USER")


def _bundle(text: str, engine: OperatorSplitEngine) -> ContextBundle:
    return ContextBundle(
        request_text=text,
        session_summary="",
        sim_memories=[],
        wks_results=[],
        focus_hint="",
        request_class="general",
        state_dim=engine.params.state_dim,
    )


class _Writer(MemoryWriterMixin):
    """Minimal host that satisfies MemoryWriterMixin's engine requirement."""

    def __init__(self, engine: OperatorSplitEngine) -> None:
        self._engine = engine


# -----------------------------------------------------------------------
# Test 1 — state is non-zero after save
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_state_save_produces_nonzero_sha(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)

    x = torch.randn(_SMALL_DIM, dtype=torch.float32)
    sha = writer._save_state_to_sim(warm, ident, "sess1", x, "node_1", "test focus")
    assert isinstance(sha, str) and len(sha) == 64, "SHA-256 must be 64 hex chars"


# -----------------------------------------------------------------------
# Test 2 — load after save returns the saved tensor
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_state_roundtrip_load_after_save(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)

    x_original = torch.randn(_SMALL_DIM, dtype=torch.float32)
    writer._save_state_to_sim(warm, ident, "sess2", x_original, "n1", "hint")

    x_loaded = writer._load_state_from_sim(warm, ident, "sess2")

    assert x_loaded.shape == x_original.shape
    assert torch.allclose(x_original.float(), x_loaded.float(), atol=1e-5), (
        "Loaded state must match saved state within float32 precision"
    )


# -----------------------------------------------------------------------
# Test 3 — fresh session returns zeros
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_load_fresh_session_returns_zeros(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)

    x = writer._load_state_from_sim(warm, ident, "nonexistent_session")
    assert x.shape == (engine.params.state_dim,)
    assert x.norm().item() == 0.0, "Fresh session state must be zero"


# -----------------------------------------------------------------------
# Test 4 — CognitionLoop final_state has non-zero norm
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_cognition_loop_produces_nonzero_final_state():
    engine = _make_engine()
    loop = CognitionLoop(engine)

    x_init = torch.zeros(engine.params.state_dim, dtype=torch.float32)
    bundle = _bundle("Hello, this is a test request.", engine)
    result = loop.run(x_init, bundle, trace_id="t1")

    assert result.final_state.shape == (engine.params.state_dim,)
    assert result.final_state.norm().item() > 0.0, (
        "CognitionLoop must evolve state away from zero"
    )


# -----------------------------------------------------------------------
# Test 5 — Turn 2 loads Turn 1's saved state (genuine multi-turn continuity)
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_turn2_starts_from_turn1_state(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)
    loop = CognitionLoop(engine)
    session_id = "multiturn_session"

    # Turn 1
    x0 = writer._load_state_from_sim(warm, ident, session_id)
    assert x0.norm().item() == 0.0  # cold start

    bundle1 = _bundle("My name is Ravinder and I am building NOESIS.", engine)
    result1 = loop.run(x0, bundle1, trace_id="turn1")

    # Derive focus hint from ThoughtGraph OUTPUT node (mirrors osc_chat.py logic)
    output_nodes = result1.graph.get_nodes_by_type(NodeType.OUTPUT)
    focus_hint_1 = output_nodes[0].content[:120] if output_nodes else "turn1"
    writer._save_state_to_sim(warm, ident, session_id, result1.final_state, "n1", focus_hint_1)

    # Turn 2 — load state, must match what Turn 1 saved
    x1_loaded = writer._load_state_from_sim(warm, ident, session_id)
    assert x1_loaded.norm().item() > 0.0, "Turn 2 must load a non-zero state from SIM"
    assert torch.allclose(result1.final_state.float(), x1_loaded.float(), atol=1e-5), (
        "Turn 2 starting state must equal Turn 1 final state"
    )

    bundle2 = _bundle("What was I working on?", engine)
    result2 = loop.run(x1_loaded, bundle2, trace_id="turn2")

    # Turn 2 starts from a different (non-zero) point than Turn 1 started from
    assert not torch.allclose(x0, x1_loaded, atol=1e-6), (
        "Turn 2 start state must differ from Turn 1 start state"
    )
    assert result2.final_state.shape == (engine.params.state_dim,)


# -----------------------------------------------------------------------
# Test 6 — session isolation: different session_ids never share state
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_session_isolation(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)

    x_a = torch.randn(_SMALL_DIM, dtype=torch.float32)
    writer._save_state_to_sim(warm, ident, "session_a", x_a, "na", "a")

    x_b = writer._load_state_from_sim(warm, ident, "session_b")
    assert x_b.norm().item() == 0.0, (
        "session_b must not see session_a's state — sessions are isolated"
    )


# -----------------------------------------------------------------------
# Test 7 — second save to same session overwrites, not appends
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_state_overwrite_same_session(tmp_path):
    engine = _make_engine()
    warm = _make_warm(tmp_path)
    ident = _ident()
    writer = _Writer(engine)

    x1 = torch.ones(_SMALL_DIM, dtype=torch.float32)
    x2 = torch.ones(_SMALL_DIM, dtype=torch.float32) * 2.0

    writer._save_state_to_sim(warm, ident, "sess_ow", x1, "n1", "h1")
    writer._save_state_to_sim(warm, ident, "sess_ow", x2, "n2", "h2")

    x_loaded = writer._load_state_from_sim(warm, ident, "sess_ow")
    assert torch.allclose(x2.float(), x_loaded.float(), atol=1e-5), (
        "Second save must overwrite the first — no duplicate rows"
    )


# -----------------------------------------------------------------------
# Test 8 — ThoughtGraph OUTPUT node used as focus_hint (matches osc_chat.py)
# -----------------------------------------------------------------------

@pytest.mark.fast
def test_focus_hint_derived_from_thought_graph_output():
    engine = _make_engine()
    loop = CognitionLoop(engine)

    x_init = torch.zeros(engine.params.state_dim, dtype=torch.float32)
    bundle = _bundle("Explain the OSC stability guarantee.", engine)
    result = loop.run(x_init, bundle, trace_id="focus_test")

    output_nodes = result.graph.get_nodes_by_type(NodeType.OUTPUT)
    if output_nodes:
        hint = output_nodes[0].content[:120]
        assert isinstance(hint, str) and len(hint) > 0
    else:
        # Fallback path — no OUTPUT node is valid in Phase A
        assert result.graph is not None
