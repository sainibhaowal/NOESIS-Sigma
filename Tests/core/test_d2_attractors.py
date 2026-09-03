"""
Tests/core/test_d2_attractors.py

Unit tests for Sprint D2 — Knowledge as Attractors.

All tests run without GPU, PostgreSQL, or a trained ICNN.
Heavy components (engine, ep_trainer, wm_service) are mocked or replaced
with minimal stubs that satisfy the interface contracts.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# ConceptStateEncoder tests
# ---------------------------------------------------------------------------

class TestConceptStateEncoder:
    def _enc(self, state_dim=64):
        from External.WorldModel.concept_state_encoder import ConceptStateEncoder
        return ConceptStateEncoder(state_dim=state_dim)

    def test_encode_concept_returns_correct_shape(self):
        enc = self._enc(64)
        x = enc.encode_concept("Paris", "Capital of France")
        assert x.shape == (64,)
        assert x.dtype == torch.float32

    def test_encode_concept_deterministic(self):
        enc = self._enc(64)
        x1 = enc.encode_concept("Paris", "Capital of France")
        x2 = enc.encode_concept("Paris", "Capital of France")
        assert torch.allclose(x1, x2)

    def test_different_concepts_differ(self):
        enc = self._enc(64)
        x1 = enc.encode_concept("Paris")
        x2 = enc.encode_concept("Berlin")
        assert not torch.allclose(x1, x2)

    def test_encode_query_same_as_concept_text(self):
        enc = self._enc(64)
        x_concept = enc.encode_concept("Python lists")
        x_query   = enc.encode_query("Python lists")
        # Should use the same pipeline → same result
        assert torch.allclose(x_concept, x_query)

    def test_encode_concept_obj(self):
        enc = self._enc(64)
        obj = MagicMock()
        obj.name = "Quantum"
        obj.description = "Physics branch"
        obj.description_plain = "Physics branch"
        x = enc.encode_concept_obj(obj)
        assert x.shape == (64,)

    def test_hash_fallback_deterministic(self):
        enc = self._enc(64)
        # Directly call the hash fallback — it must be deterministic
        x1 = enc._hash_fallback("some concept text")
        x2 = enc._hash_fallback("some concept text")
        assert torch.allclose(x1, x2)
        # Different inputs must differ
        x3 = enc._hash_fallback("different text")
        assert not torch.allclose(x1, x3)

    def test_encode_to_device(self):
        enc = self._enc(64)
        x = enc.encode_concept("Paris", device=torch.device("cpu"))
        assert x.device.type == "cpu"


# ---------------------------------------------------------------------------
# AttractorCarver tests (with stub engine + ep_trainer)
# ---------------------------------------------------------------------------

def _make_stub_engine(state_dim=64):
    """Minimal OperatorSplitEngine stub that satisfies AttractorCarver."""
    engine = MagicMock()
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.params = MagicMock()
    engine.params.state_dim = state_dim

    def project(x):
        # Normalise to unit norm
        x_f = x.float()
        n = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x_f / n).to(x.dtype)

    engine.project = project

    def step_many(x, n_steps=20, **kw):
        # Return a slightly perturbed version (simulates partial convergence)
        x_f = x.float()
        if x_f.dim() == 1:
            x_f = x_f.unsqueeze(0)
        n = x_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x_f / n)  # normalised, so close to target if target is normalised too

    engine.step_many = step_many
    engine.converge_with_nudge = MagicMock(return_value=torch.zeros(state_dim))
    return engine


def _make_stub_ep_trainer(engine):
    from Core.OSC.ep_trainer import EquilibriumPropagation
    from unittest.mock import patch, MagicMock
    from Core.OSC.icnn import ICNNDirectGrad

    icnn = ICNNDirectGrad(d=64, m=32)
    engine.icnn = icnn
    opt = torch.optim.AdamW(icnn.parameters(), lr=1e-4)
    # Patch isinstance check so it accepts our mock engine
    ep = MagicMock()
    ep.engine = engine
    ep.beta = 0.001
    ep.train_step_toward = MagicMock(return_value=0.01)
    return ep


class TestAttractorCarver:
    def _carver(self, state_dim=64):
        from Core.OSC.attractor_trainer import AttractorCarver
        from External.WorldModel.concept_state_encoder import ConceptStateEncoder
        engine = _make_stub_engine(state_dim)
        ep = _make_stub_ep_trainer(engine)
        enc = ConceptStateEncoder(state_dim=state_dim)
        return AttractorCarver(
            ep_trainer=ep,
            encoder=enc,
            steps_per_concept=3,
            beta=0.001,
        )

    def _concept(self, name="Paris", cid="c1"):
        c = MagicMock()
        c.concept_id = cid
        c.name = name
        c.description = f"{name} description"
        c.description_plain = f"{name} description"
        return c

    def test_carve_one_returns_record(self):
        carver = self._carver()
        rec = carver.carve_one(self._concept("Paris"))
        assert rec.concept_id == "c1"
        assert rec.concept_name == "Paris"
        assert 0.0 <= rec.basin_depth <= 1.0
        assert rec.ep_steps == 3

    def test_carve_one_stores_in_registry(self):
        carver = self._carver()
        carver.carve_one(self._concept("Paris", "c1"))
        assert "c1" in carver.registry

    def test_carve_batch_returns_metrics(self):
        carver = self._carver()
        concepts = [self._concept(f"C{i}", f"cid{i}") for i in range(4)]
        m = carver.carve_batch(concepts)
        assert "attempted" in m
        assert "carved" in m
        assert "avg_basin_depth" in m
        assert m["attempted"] == 4

    def test_is_carved_false_before_carving(self):
        carver = self._carver()
        assert not carver.is_carved(self._concept())

    def test_is_carved_true_after_carving_with_good_basin(self):
        carver = self._carver()
        carver._min_basin_depth = 0.0  # accept any depth for test
        carver.carve_one(self._concept())
        assert carver.is_carved(self._concept())

    def test_verify_attractor_tensor(self):
        carver = self._carver()
        x = torch.randn(64)
        depth = carver.verify_attractor_tensor(x)
        assert 0.0 <= depth <= 1.0

    def test_get_registry_summary(self):
        carver = self._carver()
        carver._min_basin_depth = 0.0
        carver.carve_one(self._concept("Paris", "c1"))
        s = carver.get_registry_summary()
        assert isinstance(s, list)
        assert s[0]["concept_id"] == "c1"

    def test_beta_restored_after_carve(self):
        carver = self._carver()
        original_beta = carver._ep.beta
        carver.carve_one(self._concept())
        assert carver._ep.beta == 0.001  # should be restored


# ---------------------------------------------------------------------------
# CarvingService lifecycle tests
# ---------------------------------------------------------------------------

class TestCarvingService:
    def _service(self, carver=None, wm=None):
        from External.WorldModel.carving_service import CarvingService
        return CarvingService(
            attractor_carver=carver,
            wm_service=wm,
            tenant_id="test",
            idle_gap_s=0.05,
            max_concepts_per_round=2,
        )

    def test_start_stop(self):
        svc = self._service()
        svc.start()
        time.sleep(0.1)
        svc.stop(timeout=2.0)

    def test_get_stats_keys(self):
        svc = self._service()
        s = svc.get_stats()
        assert "total_carved" in s
        assert "rounds" in s
        assert "registry_size" in s

    def test_no_crash_without_carver(self):
        svc = self._service(carver=None, wm=None)
        svc.start()
        time.sleep(0.15)
        svc.stop(timeout=2.0)
        assert svc.get_stats()["rounds"] == 0

    def test_busy_flag_prevents_carving(self):
        svc = self._service()
        svc.notify_request_start()
        assert svc._busy_flag.is_set()
        svc.notify_request_end()
        assert not svc._busy_flag.is_set()


# ---------------------------------------------------------------------------
# HopfieldStore.attach_attractor_retrieval tests
# ---------------------------------------------------------------------------

class TestHopfieldAttractorWiring:
    def test_attach_sets_attributes(self):
        from External.WorldModel.hopfield_store import HopfieldWorldModelStore
        store_mock = MagicMock()
        hs = HopfieldWorldModelStore.__new__(HopfieldWorldModelStore)
        hs._attractor_carver = None
        hs._concept_encoder = None
        hs._osc_engine = None

        carver = MagicMock()
        enc = MagicMock()
        engine = MagicMock()
        hs.attach_attractor_retrieval(carver, enc, engine)

        assert hs._attractor_carver is carver
        assert hs._concept_encoder is enc
        assert hs._osc_engine is engine

    def test_retrieve_falls_back_without_carver(self):
        """Retrieve with no carver attached behaves like pre-D2 (GPU matrix path)."""
        from External.WorldModel.hopfield_store import HopfieldWorldModelStore

        hs = HopfieldWorldModelStore.__new__(HopfieldWorldModelStore)
        hs._attractor_carver = None
        hs._concept_encoder = None
        hs._osc_engine = None
        hs._loaded = False
        hs._encoder_cfg = MagicMock()
        hs._device = torch.device("cpu")

        # Patch internal methods
        with patch.object(hs, "_query_concepts", return_value=[]) as mc, \
             patch.object(hs, "_query_facts", return_value=[]) as mf, \
             patch.object(hs, "_query_relations", return_value=[]) as mr, \
             patch("External.WorldModel.hopfield_store.encode_text", return_value=[0.0]*128):
            result = hs.retrieve("test query")
            mc.assert_called_once()   # GPU matrix path was used
