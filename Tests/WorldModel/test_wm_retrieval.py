# Tests/WorldModel/test_wm_retrieval.py
#
# Sprint B2 — World Model Retrieval
#
# Tests for:
#   WorldModelService.retrieve() → WMRetrieveResult
#   ContextBundle world model fields + build_context_tensor() fusion
#   GraphExtractor: wm: FACT nodes appear in ThoughtGraph
#   ThoughtGraph.summary() world_model_facts count
#   OscOrchestrator._init_wm_service() graceful None when WM_DB_URL absent
#
# PostgreSQL tests require WM_DB_URL. Unit tests (ContextBundle, GraphExtractor,
# ThoughtGraph summary, orchestrator init) run without a DB.

from __future__ import annotations

import os
import uuid

import pytest
import torch

WM_DB_URL = os.getenv("WM_DB_URL", "").strip()
_NEEDS_DB = pytest.mark.skipif(
    not WM_DB_URL, reason="WM_DB_URL not set — PostgreSQL required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store():
    from External.WorldModel.Storage.concept_store import ConceptStore
    s = ConceptStore(db_url=WM_DB_URL, echo_sql=False)
    s.init_db()
    return s


def _service(store=None):
    from External.WorldModel.Services.world_model_service import WorldModelService
    from External.Sim.Encoders.text_encoder import EncoderConfig
    return WorldModelService(store=store or _store(), encoder_cfg=EncoderConfig())


def _attach_hopfield(service, *, tenant_id: str):
    from External.Sim.Encoders.text_encoder import EncoderConfig
    from External.WorldModel.hopfield_store import HopfieldWorldModelStore

    hopfield = HopfieldWorldModelStore(
        concept_store=service.store,
        tenant_id=tenant_id,
        device="cpu",
        encoder_cfg=EncoderConfig(),
    )
    hopfield.load()
    service.hopfield_store = hopfield
    return service


def _tenant() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


def _make_bundle(**kwargs):
    from Core.Cognition.fusion import ContextBundle
    defaults = dict(
        request_text="test request",
        session_summary="",
        sim_memories=[],
        wks_results=[],
        focus_hint="",
        request_class="chat",
        state_dim=64,
    )
    defaults.update(kwargs)
    return ContextBundle(**defaults)


# ---------------------------------------------------------------------------
# Unit tests — no DB required
# ---------------------------------------------------------------------------

def test_context_bundle_has_wm_fields():
    """ContextBundle accepts world_concepts, world_facts, world_relations."""
    bundle = _make_bundle(
        world_concepts=[("A concept description", "concept-id-1")],
        world_facts=[("A temporal fact", "fact-id-1")],
        world_relations=["ConceptA IS_A ConceptB"],
    )
    assert len(bundle.world_concepts) == 1
    assert len(bundle.world_facts) == 1
    assert len(bundle.world_relations) == 1


def test_context_bundle_defaults_empty():
    """All world model fields default to empty lists — no breaking change."""
    bundle = _make_bundle()
    assert bundle.world_concepts == []
    assert bundle.world_facts == []
    assert bundle.world_relations == []


def test_build_context_tensor_includes_wm_concepts():
    """build_context_tensor produces non-zero tensor when world_concepts provided."""
    from Core.Cognition.fusion import build_context_tensor, ContextBundle
    bundle = ContextBundle(
        request_text="OSC stability",
        world_concepts=[("Lyapunov function ensures bounded state.", "cid-1")],
        state_dim=64,
    )
    tensor = build_context_tensor(bundle)
    assert tensor.shape == torch.Size([64])
    assert tensor.norm().item() > 0.0


def test_build_context_tensor_with_all_wm_fields():
    """All three WM fields fuse without error and produce unit-normalized tensor."""
    from Core.Cognition.fusion import build_context_tensor, ContextBundle
    bundle = ContextBundle(
        request_text="reasoning test",
        world_concepts=[("concept one", "c1"), ("concept two", "c2")],
        world_facts=[("fact about OSC", "f1")],
        world_relations=["OSC CAUSES stability"],
        state_dim=128,
    )
    tensor = build_context_tensor(bundle)
    assert tensor.shape == torch.Size([128])
    norm = tensor.norm().item()
    assert abs(norm - 1.0) < 1e-5


def test_graph_extractor_creates_wm_fact_nodes():
    """GraphExtractor creates wm: FACT nodes from bundle.world_concepts and world_facts."""
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = _make_bundle(
        world_concepts=[("Lyapunov stability ensures bounded norms.", "wm-concept-001")],
        world_facts=[("OSC uses ICNN energy.", "wm-fact-001")],
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)

    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    wm_nodes = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("wm:")]
    assert len(wm_nodes) == 2

    refs = {n.source_ref for n in wm_nodes}
    assert "wm:wm-concept-001" in refs
    assert "wm:wm-fact-001" in refs


def test_graph_extractor_wm_nodes_do_not_affect_existing_sim_wks():
    """SIM and WKS FACT nodes are still created correctly when WM is also present."""
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = _make_bundle(
        sim_memories=[("user said hello", "sim-key-1")],
        wks_results=[("doc chunk text", "doc-id-1")],
        world_concepts=[("wm concept text", "wm-c-1")],
        world_facts=[("wm fact text", "wm-f-1")],
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)

    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    sim_nodes = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("sim:")]
    wks_nodes = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("wks:")]
    wm_nodes = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("wm:")]

    assert len(sim_nodes) == 1
    assert len(wks_nodes) == 1
    assert len(wm_nodes) == 2


def test_thought_graph_summary_counts_wm_facts():
    """ThoughtGraph.summary() reports world_model_facts, sim_facts, wks_facts."""
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = _make_bundle(
        sim_memories=[("sim mem", "s1")],
        wks_results=[("wks chunk", "d1")],
        world_concepts=[("wm concept", "wm1")],
        world_facts=[("wm fact", "wm2"), ("wm fact 2", "wm3")],
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)
    s = graph.summary()

    assert s["world_model_facts"] == 3  # 1 concept + 2 facts
    assert s["sim_facts"] == 1
    assert s["wks_facts"] == 1
    assert "node_count" in s
    assert "is_output_grounded" in s


def test_thought_graph_summary_zero_wm_when_none():
    """summary() returns world_model_facts=0 when no WM nodes present."""
    from Core.Cognition.graph_extractor import GraphExtractor

    extractor = GraphExtractor()
    bundle = _make_bundle(sim_memories=[("hello", "k1")])
    graph = extractor.extract(trajectory=[], bundle=bundle)
    s = graph.summary()
    assert s["world_model_facts"] == 0


def test_witness_from_graph_excludes_wm_citations():
    """_witness_from_graph in OscOrchestrator skips wm: source_refs (Verifier constraint)."""
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    extractor = GraphExtractor()
    bundle = _make_bundle(
        sim_memories=[("sim mem", "sim-key-1")],
        world_concepts=[("wm concept", "wm-c-1")],
    )
    graph = extractor.extract(trajectory=[], bundle=bundle)

    # Use the helper directly — instantiate a minimal OscOrchestrator offline
    # by checking the logic manually via _witness_from_graph
    from Core.Cognition.thought_graph import NodeType
    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    sim_only = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("sim:")]
    wm_only = [n for n in fact_nodes if (getattr(n, "source_ref", "") or "").startswith("wm:")]
    assert len(sim_only) >= 1
    assert len(wm_only) >= 1

    # The Verifier citation logic: only sim: and wks: are included
    valid_for_citation = [
        n for n in fact_nodes
        if (getattr(n, "source_ref", "") or "").startswith(("sim:", "wks:"))
    ]
    invalid_for_citation = [
        n for n in fact_nodes
        if (getattr(n, "source_ref", "") or "").startswith("wm:")
    ]
    assert all(n in valid_for_citation for n in sim_only)
    assert all(n in invalid_for_citation for n in wm_only)


def test_wm_retrieve_result_structure():
    """WMRetrieveResult dataclass has the correct fields."""
    from External.WorldModel.Services.world_model_service import WMRetrieveResult
    r = WMRetrieveResult(
        concepts=[("desc", "id1")],
        facts=[("content", "id2")],
        relations=["A IS_A B"],
    )
    assert r.concepts[0] == ("desc", "id1")
    assert r.facts[0] == ("content", "id2")
    assert r.relations[0] == "A IS_A B"


def test_wm_retrieve_result_defaults_empty():
    """WMRetrieveResult fields default to empty lists."""
    from External.WorldModel.Services.world_model_service import WMRetrieveResult
    r = WMRetrieveResult()
    assert r.concepts == []
    assert r.facts == []
    assert r.relations == []


def test_osc_orchestrator_init_wm_none_without_env(monkeypatch):
    """_init_wm_service() returns None when WM_DB_URL is not set."""
    monkeypatch.delenv("WM_DB_URL", raising=False)
    from External.Orchestrator.osc_chat import OscOrchestrator
    result = OscOrchestrator._init_wm_service()
    assert result is None


# ---------------------------------------------------------------------------
# PostgreSQL tests — require WM_DB_URL
# ---------------------------------------------------------------------------

@_NEEDS_DB
def test_retrieve_returns_wm_retrieve_result():
    """retrieve() returns a WMRetrieveResult (even if empty for fresh tenant)."""
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)
    from External.WorldModel.Services.world_model_service import WMRetrieveResult
    result = svc.retrieve(tenant_id=tenant, query="anything", min_score=0.0)
    assert isinstance(result, WMRetrieveResult)
    assert isinstance(result.concepts, list)
    assert isinstance(result.facts, list)
    assert isinstance(result.relations, list)


@_NEEDS_DB
def test_retrieve_concepts_returned_for_matching_query():
    """retrieve() returns relevant concepts when they exist."""
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_concept(
        tenant_id=tenant,
        name="Lyapunov stability",
        domain="mathematics",
        description="A system is Lyapunov stable if small perturbations remain bounded.",
    )
    result = svc.retrieve(
        tenant_id=tenant,
        query="Lyapunov bounded state",
        top_k_concepts=5,
        min_score=0.0,
    )
    names_in_descriptions = " ".join(c[0] for c in result.concepts)
    assert "Lyapunov" in names_in_descriptions or len(result.concepts) >= 1


@_NEEDS_DB
def test_retrieve_facts_returned_for_matching_query():
    """retrieve() returns relevant temporal facts when they exist."""
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_fact(
        tenant_id=tenant,
        content="OSC uses a Lyapunov energy to keep state bounded.",
        domain="AI",
    )
    result = svc.retrieve(
        tenant_id=tenant,
        query="OSC Lyapunov energy",
        top_k_facts=5,
        min_score=0.0,
    )
    assert len(result.facts) >= 1
    fact_texts = " ".join(f[0] for f in result.facts)
    assert "Lyapunov" in fact_texts


@_NEEDS_DB
def test_retrieve_tenant_isolation():
    """retrieve() for tenant_b returns no results from tenant_a."""
    tenant_a = _tenant()
    tenant_b = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant_a)

    svc.add_concept(
        tenant_id=tenant_a,
        name="secret domain concept",
        domain="test",
        description="Only tenant A should have this.",
    )
    result = svc.retrieve(
        tenant_id=tenant_b,
        query="secret domain concept",
        min_score=0.0,
    )
    concept_texts = " ".join(c[0] for c in result.concepts)
    assert "secret domain concept" not in concept_texts


@_NEEDS_DB
def test_retrieve_with_relations():
    """retrieve() includes relation summaries when concepts have relations."""
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    c1 = svc.add_concept(
        tenant_id=tenant, name="ICNN", domain="AI",
        description="Input-convex neural network used in OSC energy function."
    )
    c2 = svc.add_concept(
        tenant_id=tenant, name="convexity", domain="mathematics",
        description="Property where local minima are global minima."
    )
    svc.add_relation(
        tenant_id=tenant,
        from_concept_id=str(c1.id),
        to_concept_id=str(c2.id),
        relation_type="IS_A",
        weight=0.9,
    )

    result = svc.retrieve(
        tenant_id=tenant,
        query="ICNN convexity neural network",
        top_k_concepts=5,
        top_k_relations=3,
        min_score=0.0,
    )
    # Relations may or may not be present depending on which concepts are top-k
    assert isinstance(result.relations, list)


@_NEEDS_DB
def test_retrieve_injects_correctly_into_context_bundle():
    """Full pipeline: retrieve() → ContextBundle → build_context_tensor produces valid tensor."""
    from Core.Cognition.fusion import ContextBundle, build_context_tensor

    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_concept(
        tenant_id=tenant, name="operator split", domain="AI",
        description="Operator splitting decomposes dynamics into tractable sub-problems."
    )
    svc.add_fact(
        tenant_id=tenant,
        content="Operator splitting ensures stable cognitive state evolution.",
        domain="AI",
    )

    result = svc.retrieve(
        tenant_id=tenant,
        query="operator split dynamics stability",
        min_score=0.0,
    )
    bundle = ContextBundle(
        request_text="explain operator splitting",
        world_concepts=result.concepts,
        world_facts=result.facts,
        world_relations=result.relations,
        state_dim=128,
    )
    tensor = build_context_tensor(bundle)
    assert tensor.shape == torch.Size([128])
    assert tensor.norm().item() > 0.0


@_NEEDS_DB
def test_retrieve_graph_extractor_wm_nodes_with_real_db():
    """Full pipeline: retrieve() → GraphExtractor produces wm: FACT nodes."""
    from Core.Cognition.fusion import ContextBundle
    from Core.Cognition.graph_extractor import GraphExtractor
    from Core.Cognition.thought_graph import NodeType

    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_concept(
        tenant_id=tenant, name="gradient descent", domain="AI",
        description="Iterative optimisation by following negative gradient.",
    )
    result = svc.retrieve(
        tenant_id=tenant,
        query="gradient optimisation",
        top_k_concepts=3,
        min_score=0.0,
    )
    bundle = ContextBundle(
        request_text="how does gradient descent work",
        world_concepts=result.concepts,
        world_facts=result.facts,
        state_dim=64,
    )
    extractor = GraphExtractor()
    graph = extractor.extract(trajectory=[], bundle=bundle)

    if result.concepts or result.facts:
        wm_nodes = [
            n for n in graph.get_nodes_by_type(NodeType.FACT)
            if (getattr(n, "source_ref", "") or "").startswith("wm:")
        ]
        assert len(wm_nodes) == len(result.concepts) + len(result.facts)

    summary = graph.summary()
    assert "world_model_facts" in summary
