from __future__ import annotations

import torch


class FakeStore:
    def search_by_embedding(self, **kwargs):
        raise AssertionError("SQL search should not be called when Hopfield is loaded")

    def search_facts_by_embedding(self, **kwargs):
        raise AssertionError("SQL fact search should not be called when Hopfield is loaded")

    def get_relations(self, **kwargs):
        raise AssertionError("SQL relation lookup should not be called when Hopfield is loaded")


class FakeHopfieldStore:
    def is_loaded(self) -> bool:
        return True

    def retrieve(self, **kwargs):
        from External.WorldModel.Services.world_model_service import WMRetrieveResult

        return WMRetrieveResult(
            concepts=[("Paris is the capital of France.", "concept-1")],
            facts=[("France is in Europe.", "fact-1")],
            relations=["Paris located_in France"],
        )


def test_world_model_retrieve_prefers_hopfield(monkeypatch):
    monkeypatch.delenv("WM_ALLOW_SQL_RETRIEVAL", raising=False)

    from External.Sim.Encoders.text_encoder import EncoderConfig
    from External.WorldModel.Services.world_model_service import WorldModelService

    svc = WorldModelService(
        store=FakeStore(),
        encoder_cfg=EncoderConfig(),
        hopfield_store=FakeHopfieldStore(),
    )

    result = svc.retrieve(tenant_id="tenant-a", query="What is Paris?")

    assert result.concepts == [("Paris is the capital of France.", "concept-1")]
    assert result.facts == [("France is in Europe.", "fact-1")]
    assert result.relations == ["Paris located_in France"]


def test_world_model_reads_return_empty_without_hopfield(monkeypatch):
    monkeypatch.delenv("WM_ALLOW_SQL_RETRIEVAL", raising=False)

    from External.Sim.Encoders.text_encoder import EncoderConfig
    from External.WorldModel.Services.world_model_service import (
        WMRetrieveResult,
        WorldModelService,
    )

    svc = WorldModelService(
        store=FakeStore(),
        encoder_cfg=EncoderConfig(),
        hopfield_store=None,
    )

    assert svc.search_concepts(tenant_id="tenant-a", query="Paris", min_score=0.0) == []
    assert svc.search_facts(tenant_id="tenant-a", query="Paris", min_score=0.0) == []
    assert svc.retrieve(tenant_id="tenant-a", query="Paris") == WMRetrieveResult()
