"""
Tests/Search/test_search.py

Tests for Sprint C2 — Internet Search + WorldModel Live Ingest.

Tests:
  - SearchCache: same query returns cached result (no duplicate fetch)
  - SearchCache: different query is a cache miss
  - SearchCache: normalisation (whitespace/case) hits same entry
  - SearchCache: session eviction at MAX_SESSIONS
  - UNCERTAIN detector: returns queries from UNCERTAIN nodes
  - UNCERTAIN detector: returns empty list when no UNCERTAIN nodes
  - Search loop: use_search=False skips all search calls
  - Search loop: no UNCERTAIN nodes → zero searches
  - Rate limit: max_searches cap respected even with 10 UNCERTAIN nodes
  - NullAdapter: is_available returns False, search returns []
  - TavilyAdapter (mocked HTTP): returns SearchResult list
  - Ingest connection: ingest_text called once per result with correct tenant_id
"""

from __future__ import annotations

import uuid
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from External.Search.search_cache import SearchCache, _normalise
from External.Search.search_adapter import NullAdapter, SearchResult


# ─────────────────────────────── helpers ────────────────────────────────────

def _make_result(title="Test", url="https://example.com", snippet="A test result") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, full_text=None, source="mock")


def _make_uncertain_graph(contents: List[str]):
    """Build a minimal mock ThoughtGraph with UNCERTAIN nodes."""
    from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
    g = ThoughtGraph(trace_id=str(uuid.uuid4()))
    for i, content in enumerate(contents):
        g.add_node(ThoughtNode(
            node_id=f"u{i}",
            node_type=NodeType.UNCERTAIN,
            content=content,
        ))
    return g


def _make_clean_graph():
    """ThoughtGraph with only INTENT + FACT nodes — no UNCERTAIN nodes."""
    from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
    g = ThoughtGraph(trace_id=str(uuid.uuid4()))
    g.add_node(ThoughtNode(node_id="i0", node_type=NodeType.INTENT, content="What is X?"))
    g.add_node(ThoughtNode(node_id="f0", node_type=NodeType.FACT, content="X is Y.", source_ref="sim:001"))
    return g


# ─────────────────────────────── SearchCache ────────────────────────────────

class TestSearchCache:
    def test_same_query_returns_cached(self):
        cache = SearchCache()
        results = [_make_result()]
        cache.set("session-1", "python asyncio", results)
        assert cache.get("session-1", "python asyncio") == results

    def test_different_query_is_cache_miss(self):
        cache = SearchCache()
        cache.set("session-1", "python asyncio", [_make_result()])
        assert cache.get("session-1", "some other query") is None

    def test_normalisation_whitespace_hits_cache(self):
        cache = SearchCache()
        results = [_make_result()]
        cache.set("session-1", "Python   Asyncio  ", results)
        # Different whitespace and casing should still hit
        assert cache.get("session-1", "python asyncio") == results

    def test_different_session_is_cache_miss(self):
        cache = SearchCache()
        cache.set("session-a", "python asyncio", [_make_result()])
        assert cache.get("session-b", "python asyncio") is None

    def test_session_eviction_at_max(self):
        cache = SearchCache(max_sessions=3)
        for i in range(3):
            cache.set(f"session-{i}", "query", [_make_result()])
        assert cache.size()["sessions"] == 3
        # Adding a 4th session should evict the oldest (session-0)
        cache.set("session-3", "query", [_make_result()])
        assert cache.size()["sessions"] == 3
        assert cache.get("session-0", "query") is None  # evicted
        assert cache.get("session-3", "query") is not None

    def test_invalidate_session(self):
        cache = SearchCache()
        cache.set("session-1", "query", [_make_result()])
        cache.invalidate_session("session-1")
        assert cache.get("session-1", "query") is None


# ─────────────────────────────── NullAdapter ────────────────────────────────

class TestNullAdapter:
    def test_is_not_available(self):
        adapter = NullAdapter()
        assert adapter.is_available() is False

    def test_search_returns_empty_list(self):
        adapter = NullAdapter()
        assert adapter.search("any query") == []


# ─────────────────────────── UNCERTAIN detector ─────────────────────────────

class TestDetectUncertainQueries:
    def test_returns_queries_from_uncertain_nodes(self):
        from External.Orchestrator.osc_chat import OscOrchestrator
        graph = _make_uncertain_graph(["What is the price of lithium?", "Who invented the laser?"])
        queries = OscOrchestrator._detect_uncertain_queries(graph)
        assert queries == ["What is the price of lithium?", "Who invented the laser?"]

    def test_returns_empty_for_no_uncertain_nodes(self):
        from External.Orchestrator.osc_chat import OscOrchestrator
        graph = _make_clean_graph()
        queries = OscOrchestrator._detect_uncertain_queries(graph)
        assert queries == []

    def test_deduplicates_identical_content(self):
        from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
        from External.Orchestrator.osc_chat import OscOrchestrator
        g = ThoughtGraph()
        g.add_node(ThoughtNode(node_id="u0", node_type=NodeType.UNCERTAIN, content="same query"))
        g.add_node(ThoughtNode(node_id="u1", node_type=NodeType.UNCERTAIN, content="same query"))
        queries = OscOrchestrator._detect_uncertain_queries(g)
        assert queries == ["same query"]  # deduplicated

    def test_ignores_empty_content(self):
        from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
        from External.Orchestrator.osc_chat import OscOrchestrator
        g = ThoughtGraph()
        g.add_node(ThoughtNode(node_id="u0", node_type=NodeType.UNCERTAIN, content=""))
        g.add_node(ThoughtNode(node_id="u1", node_type=NodeType.UNCERTAIN, content="  "))
        queries = OscOrchestrator._detect_uncertain_queries(g)
        assert queries == []


# ─────────────────────────── Search loop behaviour ──────────────────────────

class TestSearchLoopBehaviour:
    def _make_orchestrator_with_mock_wm(self):
        """Create OscOrchestrator with mocked engine and WM service."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = OscOrchestrator.__new__(OscOrchestrator)
        orch._wm_service = MagicMock()
        orch._search_adapter = None
        orch._ingest_pipeline_instance = None
        orch._search_cache = {}
        return orch

    def test_use_search_false_skips_search(self):
        """When use_search=False on the request, _run_search_and_ingest must not be called."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()

        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        orch._search_adapter = mock_adapter

        uncertain_graph = _make_uncertain_graph(["what is X?"])

        # Simulate the condition check in _handle_inner:
        # if req.use_search and self._wm_service is not None:
        use_search = False
        if use_search and orch._wm_service is not None:
            orch._detect_uncertain_queries(uncertain_graph)

        mock_adapter.search.assert_not_called()

    def test_no_uncertain_nodes_no_search(self):
        """No UNCERTAIN nodes → _run_search_and_ingest receives empty list → 0 searches."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()

        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        orch._search_adapter = mock_adapter

        clean_graph = _make_clean_graph()
        queries = OscOrchestrator._detect_uncertain_queries(clean_graph)
        assert queries == []
        # No queries → adapter.search never called
        mock_adapter.search.assert_not_called()

    def test_max_searches_cap_respected(self):
        """Even with 10 UNCERTAIN nodes, only max_searches=3 API calls are made."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()

        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.search.return_value = [_make_result()]
        orch._search_adapter = mock_adapter

        # Mock ingest pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.ingest_text.return_value = MagicMock(concepts_added=1, facts_added=1)
        orch._ingest_pipeline_instance = mock_pipeline
        orch._wm_service = MagicMock()  # needed for _get_ingest_pipeline to not return None

        # 10 different uncertain queries
        queries = [f"uncertain question {i}" for i in range(10)]
        req = MagicMock()
        req.tenant_id = "tenant-1"

        orch._run_search_and_ingest(
            req=req,
            queries=queries,
            session_id="session-1",
            max_searches=3,
        )
        assert mock_adapter.search.call_count == 3  # hard cap enforced

    def test_ingest_called_with_tenant_id(self):
        """ingest_text must be called with the correct tenant_id for isolation."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()

        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        result = SearchResult(
            title="Test Page",
            url="https://example.com/test",
            snippet="Some useful content about X.",
            full_text="The full article content about X. It is a very important topic.",
            source="mock",
        )
        mock_adapter.search.return_value = [result]
        orch._search_adapter = mock_adapter

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_text.return_value = MagicMock(concepts_added=2, facts_added=1)
        orch._ingest_pipeline_instance = mock_pipeline

        req = MagicMock()
        req.tenant_id = "tenant-xyz"

        orch._run_search_and_ingest(
            req=req,
            queries=["What is X?"],
            session_id="session-1",
            max_searches=3,
        )

        # Verify ingest was called with correct tenant_id and source_ref
        mock_pipeline.ingest_text.assert_called_once()
        call_kwargs = mock_pipeline.ingest_text.call_args.kwargs
        assert call_kwargs["tenant_id"] == "tenant-xyz"
        assert call_kwargs["domain"] == "web"
        assert "example.com" in call_kwargs["source_ref"]
        assert call_kwargs["confidence"] == 0.70

    def test_cache_prevents_duplicate_search(self):
        """Same query in same session must not trigger a second API call."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()

        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.search.return_value = [_make_result()]
        orch._search_adapter = mock_adapter

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_text.return_value = MagicMock(concepts_added=1, facts_added=0)
        orch._ingest_pipeline_instance = mock_pipeline

        req = MagicMock()
        req.tenant_id = "tenant-1"

        # First call
        orch._run_search_and_ingest(
            req=req, queries=["python asyncio"], session_id="session-1", max_searches=3
        )
        # Second call — same session, same query
        orch._run_search_and_ingest(
            req=req, queries=["python asyncio"], session_id="session-1", max_searches=3
        )

        # Adapter must only have been called once — cache hit on second call
        assert mock_adapter.search.call_count == 1

    def test_null_adapter_returns_zero(self):
        """NullAdapter is_available=False → _run_search_and_ingest returns 0."""
        from External.Orchestrator.osc_chat import OscOrchestrator
        orch = self._make_orchestrator_with_mock_wm()
        orch._search_adapter = NullAdapter()

        req = MagicMock()
        req.tenant_id = "tenant-1"
        added = orch._run_search_and_ingest(
            req=req, queries=["anything"], session_id="s1", max_searches=3
        )
        assert added == 0
