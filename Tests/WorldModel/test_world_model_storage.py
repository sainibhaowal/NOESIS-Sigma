# Tests/WorldModel/test_world_model_storage.py
#
# Sprint B1 — World Model Storage
#
# These tests exercise the full storage stack:
#   ConceptStore → WMConcept / WMRelation / WMTemporalFact
#   WorldModelService (CRUD + search)
#   IngestPipeline (text → records)
#   Encryption round-trip
#   Tenant isolation
#
# PostgreSQL is REQUIRED. Tests are skipped if WM_DB_URL is not set.
# To run: WM_DB_URL=postgresql+psycopg://... pytest Tests/WorldModel/
#
# When WM_DB_URL is available, each test uses a unique tenant_id so
# tests never interfere with each other and a shared DB is safe.

from __future__ import annotations

import base64
import os
import time
import uuid

import pytest

# Require PostgreSQL — skip gracefully in CI without a DB
WM_DB_URL = os.getenv("WM_DB_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not WM_DB_URL,
    reason="WM_DB_URL not set — PostgreSQL required for World Model tests",
)


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


# -----------------------------------------------------------------------
# Test 1 — ConceptStore rejects SQLite
# -----------------------------------------------------------------------

def test_concept_store_rejects_sqlite():
    from External.WorldModel.Storage.concept_store import ConceptStore
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        ConceptStore(db_url="sqlite:///test.db")


# -----------------------------------------------------------------------
# Test 2 — add and retrieve a concept
# -----------------------------------------------------------------------

def test_add_and_get_concept():
    svc = _service()
    tenant = _tenant()

    row = svc.add_concept(
        tenant_id=tenant,
        name="Lyapunov stability",
        domain="mathematics",
        description="A system is Lyapunov stable if small perturbations remain bounded.",
        source_ref="test:001",
        confidence=0.99,
    )
    assert row.id is not None
    assert row.tenant_id == tenant
    assert row.name == "Lyapunov stability"

    fetched = svc.get_concept(concept_id=str(row.id), tenant_id=tenant)
    assert fetched is not None
    assert fetched.name == "Lyapunov stability"
    assert fetched.confidence == pytest.approx(0.99, abs=1e-4)


# -----------------------------------------------------------------------
# Test 3 — embedding stored and round-trippable
# -----------------------------------------------------------------------

def test_embedding_stored_non_empty():
    svc = _service()
    tenant = _tenant()

    row = svc.add_concept(
        tenant_id=tenant,
        name="ICNN convexity",
        domain="AI",
        description="Input-convex neural network guarantees a convex energy landscape.",
    )
    assert row.embedding_b64 != ""
    raw = base64.b64decode(row.embedding_b64.encode("ascii"))
    assert len(raw) == 128 * 4  # 128 float32 values


# -----------------------------------------------------------------------
# Test 4 — concept search returns relevant result
# -----------------------------------------------------------------------

def test_search_concepts_returns_match():
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_concept(
        tenant_id=tenant,
        name="operator split dynamics",
        domain="AI",
        description="OSC splits the dynamics operator to ensure stability.",
    )
    svc.add_concept(
        tenant_id=tenant,
        name="gradient descent",
        domain="AI",
        description="Iterative optimisation by following negative gradient.",
    )

    results = svc.search_concepts(
        tenant_id=tenant,
        query="OSC dynamics stability",
        limit=5,
        min_score=0.0,
    )
    assert len(results) >= 1
    names = [r.name for r in results]
    assert "operator split dynamics" in names


# -----------------------------------------------------------------------
# Test 5 — tenant isolation: concept from tenant A not visible to tenant B
# -----------------------------------------------------------------------

def test_tenant_isolation_concepts():
    tenant_a = _tenant()
    tenant_b = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant_a)

    row = svc.add_concept(
        tenant_id=tenant_a,
        name="secret concept",
        domain="test",
        description="Only tenant A should see this.",
    )

    # Tenant B cannot retrieve by id
    fetched = svc.get_concept(concept_id=str(row.id), tenant_id=tenant_b)
    assert fetched is None

    # Tenant B search returns nothing
    results = svc.search_concepts(
        tenant_id=tenant_b, query="secret concept", min_score=0.0
    )
    assert all(r.name != "secret concept" for r in results)


# -----------------------------------------------------------------------
# Test 6 — add relation and retrieve
# -----------------------------------------------------------------------

def test_add_and_get_relation():
    svc = _service()
    tenant = _tenant()

    c1 = svc.add_concept(tenant_id=tenant, name="ICNN", domain="AI", description="Input-convex NN.")
    c2 = svc.add_concept(tenant_id=tenant, name="convexity", domain="mathematics", description="Convex function.")

    rel = svc.add_relation(
        tenant_id=tenant,
        from_concept_id=str(c1.id),
        to_concept_id=str(c2.id),
        relation_type="IS_A",
        weight=0.9,
    )
    assert rel.id is not None

    rels = svc.get_relations(tenant_id=tenant, from_concept_id=str(c1.id))
    assert any(str(r.to_concept_id) == str(c2.id) and r.relation_type == "IS_A" for r in rels)


# -----------------------------------------------------------------------
# Test 7 — temporal fact: add and search
# -----------------------------------------------------------------------

def test_add_and_search_temporal_fact():
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)

    svc.add_fact(
        tenant_id=tenant,
        content="The OSC engine uses a Lyapunov function to guarantee bounded state norms.",
        domain="AI",
        source_ref="doc:osc_spec",
    )

    results = svc.search_facts(
        tenant_id=tenant,
        query="Lyapunov function bounded state",
        min_score=0.0,
    )
    assert len(results) >= 1
    assert any("Lyapunov" in r.content for r in results)


# -----------------------------------------------------------------------
# Test 8 — temporal fact validity window filtering
# -----------------------------------------------------------------------

def test_temporal_fact_validity_window():
    tenant = _tenant()
    svc = _attach_hopfield(_service(), tenant_id=tenant)
    now = time.time()

    # Fact valid in the past (expired)
    svc.add_fact(
        tenant_id=tenant,
        content="This fact expired yesterday.",
        domain="test",
        valid_from=now - 7200,
        valid_until=now - 3600,  # expired 1 hour ago
    )
    # Fact valid now
    svc.add_fact(
        tenant_id=tenant,
        content="This fact is currently valid.",
        domain="test",
        valid_from=now - 3600,
        valid_until=now + 3600,
    )

    results = svc.search_facts(
        tenant_id=tenant,
        query="fact",
        min_score=0.0,
        at_timestamp=now,
    )
    contents = [r.content for r in results]
    assert any("currently valid" in c for c in contents)
    assert not any("expired yesterday" in c for c in contents)


# -----------------------------------------------------------------------
# Test 9 — encryption round-trip
# -----------------------------------------------------------------------

def test_encryption_round_trip():
    master_key = base64.b64encode(b"\xAB" * 32).decode("ascii")
    os.environ["SIM_MASTER_KEY_B64"] = master_key

    from External.WorldModel.Security.encryption import seal, open_envelope
    tenant = _tenant()

    ct, nonce, dw, dn = seal(tenant_id=tenant, plaintext="hello world")
    recovered = open_envelope(
        tenant_id=tenant,
        ct_b64=ct,
        nonce_b64=nonce,
        dek_wrapped_b64=dw,
        dek_nonce_b64=dn,
    )
    assert recovered == "hello world"

    # Wrong tenant must fail
    with pytest.raises(Exception):
        open_envelope(
            tenant_id="wrong_tenant",
            ct_b64=ct,
            nonce_b64=nonce,
            dek_wrapped_b64=dw,
            dek_nonce_b64=dn,
        )


# -----------------------------------------------------------------------
# Test 10 — ingest pipeline: text → concepts + facts
# -----------------------------------------------------------------------

def test_ingest_pipeline_text():
    svc = _service()
    tenant = _tenant()

    from External.WorldModel.Ingest.ingest_pipeline import IngestPipeline
    pipeline = IngestPipeline(service=svc)

    text = (
        "Lyapunov Stability is a property of dynamical systems.\n"
        "The OSC Engine was developed to guarantee bounded state evolution.\n"
        "Gradient Descent is an optimisation algorithm.\n"
    )
    result = pipeline.ingest_text(
        tenant_id=tenant,
        text=text,
        domain="AI",
        source_ref="test:ingest_01",
    )
    assert result.errors == []
    assert result.concepts_added + result.facts_added > 0


# -----------------------------------------------------------------------
# Test 11 — ingest direct concept (bypasses text parsing)
# -----------------------------------------------------------------------

def test_ingest_direct_concept():
    svc = _service()
    tenant = _tenant()

    from External.WorldModel.Ingest.ingest_pipeline import IngestPipeline
    pipeline = IngestPipeline(service=svc)

    row = pipeline.ingest_concept(
        tenant_id=tenant,
        name="ThoughtGraph",
        domain="NOESIS",
        description="Structured intermediate representation built before any text is generated.",
        aliases=["thought graph", "TG"],
        source_ref="internal:spec",
        confidence=1.0,
    )
    assert row.name == "ThoughtGraph"
    assert row.tenant_id == tenant


# -----------------------------------------------------------------------
# Test 12 — WM_DB_URL config rejects SQLite
# -----------------------------------------------------------------------

def test_wm_config_rejects_sqlite(monkeypatch):
    monkeypatch.setenv("WM_DB_URL", "sqlite:///bad.db")
    from External.WorldModel.Config.wm_config import load_wm_config
    with pytest.raises(RuntimeError, match="SQLite"):
        load_wm_config()


# -----------------------------------------------------------------------
# Test 13 — WM_DB_URL config raises if missing
# -----------------------------------------------------------------------

def test_wm_config_raises_if_missing(monkeypatch):
    monkeypatch.delenv("WM_DB_URL", raising=False)
    from External.WorldModel.Config.wm_config import load_wm_config
    with pytest.raises(RuntimeError, match="WM_DB_URL"):
        load_wm_config()
