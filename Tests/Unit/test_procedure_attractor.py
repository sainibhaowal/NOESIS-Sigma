"""
Tests/Unit/test_procedure_attractor.py

Unit tests for Core/OSC/procedure_attractor.py

Tests:
1. ProcedureStep creation and serialization
2. ProcedureAttractor creation and serialization
3. ProcedureMetadata creation and serialization
4. ProcedureRegistry operations (register, get, find)
5. ProcedureStateEncoder operations
6. Backwards compatibility with existing code
"""

import sys
import time
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Core.Cognition.thought_graph import NodeType, EdgeType
from Core.OSC.procedure_attractor import (
    ProcedureAttractor,
    ProcedureMetadata,
    ProcedureRegistry,
    ProcedureStateEncoder,
    ProcedureStep,
    MAX_PROCEDURE_STEPS,
    MIN_PROCEDURE_STEPS,
)

pytestmark = pytest.mark.unit


class TestProcedureStep:
    """Tests for ProcedureStep dataclass."""

    def test_create_step(self):
        """Test creating a basic procedure step."""
        state = torch.randn(1024)
        step = ProcedureStep(
            step_index=0,
            state=state,
            description="Setup",
            energy_level=0.5,
            is_anchor=True,
        )

        assert step.step_index == 0
        assert step.description == "Setup"
        assert step.energy_level == 0.5
        assert step.is_anchor is True
        assert step.state.shape == (1024,)

    def test_step_serialization(self):
        """Test step to_dict and from_dict."""
        state = torch.randn(1024)
        step = ProcedureStep(
            step_index=1,
            state=state,
            description="Execute",
            energy_level=0.3,
            is_anchor=False,
        )

        # Serialize
        d = step.to_dict()
        assert d["step_index"] == 1
        assert d["description"] == "Execute"
        assert d["energy_level"] == 0.3
        assert d["is_anchor"] is False
        assert isinstance(d["state"], list)

        # Deserialize
        step2 = ProcedureStep.from_dict(d)
        assert step2.step_index == step.step_index
        assert step2.description == step.description
        assert torch.allclose(step2.state, step.state)

    def test_step_tensor_shape(self):
        """Test state tensor shape is preserved."""
        for dim in [256, 512, 1024, 2048]:
            state = torch.randn(dim)
            step = ProcedureStep(
                step_index=0,
                state=state,
                description="test",
                energy_level=0.5,
            )
            assert step.state.shape == (dim,)


class TestProcedureAttractor:
    """Tests for ProcedureAttractor dataclass."""

    def test_create_procedure(self):
        """Test creating a procedure attractor."""
        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.randn(1024),
                description=f"Step {i}",
                energy_level=0.5 - i * 0.1,
            )
            for i in range(5)
        ]

        proc = ProcedureAttractor(
            procedure_id="test_proc_1",
            name="Build Button",
            description="How to build a UI button",
            steps=steps,
        )

        assert proc.procedure_id == "test_proc_1"
        assert proc.name == "Build Button"
        assert proc.num_steps == 5
        assert proc.total_energy == 0.0  # Not computed yet

    def test_trajectory_property(self):
        """Test trajectory property returns states."""
        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.randn(1024) * (i + 1),
                description=f"Step {i}",
                energy_level=0.5,
            )
            for i in range(3)
        ]

        proc = ProcedureAttractor(
            procedure_id="test_traj",
            name="Test Trajectory",
            description="Test",
            steps=steps,
        )

        traj = proc.trajectory
        assert len(traj) == 3
        assert torch.allclose(traj[0], steps[0].state)
        assert torch.allclose(traj[1], steps[1].state)
        assert torch.allclose(traj[2], steps[2].state)

    def test_start_end_states(self):
        """Test start_state and end_state properties."""
        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.tensor([i * 1.0] * 1024),  # Uniform state for easy checking
                description=f"Step {i}",
                energy_level=0.5,
            )
            for i in range(4)
        ]

        proc = ProcedureAttractor(
            procedure_id="test_bounds",
            name="Test Bounds",
            description="Test",
            steps=steps,
        )

        start = proc.start_state
        end = proc.end_state
        assert torch.allclose(start, torch.zeros(1024))
        assert torch.allclose(end, torch.ones(1024) * 3)

    def test_serialization_roundtrip(self):
        """Test procedure serialization and deserialization."""
        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.randn(1024),
                description=f"Step {i}",
                energy_level=0.5 - i * 0.05,
            )
            for i in range(3)
        ]

        proc = ProcedureAttractor(
            procedure_id="test_roundtrip",
            name="Roundtrip Test",
            description="Test serialization",
            steps=steps,
            total_energy=1.5,
            generalization_score=0.8,
        )

        # Serialize
        d = proc.to_dict()
        assert d["procedure_id"] == "test_roundtrip"
        assert d["name"] == "Roundtrip Test"
        # num_steps is a property, check via len(steps)
        assert len(d["steps"]) == 3

        # Deserialize
        proc2 = ProcedureAttractor.from_dict(d)
        assert proc2.procedure_id == proc.procedure_id
        assert proc2.name == proc.name
        assert proc2.num_steps == proc.num_steps

    def test_empty_procedure(self):
        """Test procedure with no steps."""
        proc = ProcedureAttractor(
            procedure_id="empty",
            name="Empty",
            description="No steps",
            steps=[],
        )

        assert proc.num_steps == 0
        assert proc.start_state is None
        assert proc.end_state is None
        assert proc.trajectory == []


class TestProcedureMetadata:
    """Tests for ProcedureMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating procedure metadata."""
        meta = ProcedureMetadata(
            procedure_id="meta_test",
            name="Test Procedure",
            domain="coding",
            keywords=["button", "ui", "component"],
        )

        assert meta.procedure_id == "meta_test"
        assert meta.domain == "coding"
        assert "button" in meta.keywords

    def test_serialization(self):
        """Test metadata serialization."""
        meta = ProcedureMetadata(
            procedure_id="meta_serial",
            name="Serialization Test",
            domain="design",
            keywords=["ui", "ux"],
        )

        d = meta.to_dict()
        meta2 = ProcedureMetadata.from_dict(d)

        assert meta2.procedure_id == meta.procedure_id
        assert meta2.domain == meta.domain
        assert meta2.keywords == meta.keywords


class TestProcedureRegistry:
    """Tests for ProcedureRegistry singleton."""

    def setup_method(self):
        """Reset registry before each test."""
        ProcedureRegistry.reset()

    def test_singleton(self):
        """Test singleton behavior."""
        reg1 = ProcedureRegistry.get_instance()
        reg2 = ProcedureRegistry.get_instance()
        assert reg1 is reg2

    def test_register_and_get(self):
        """Test registering and retrieving procedures."""
        reg = ProcedureRegistry.get_instance()

        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.randn(1024),
                description=f"Step {i}",
                energy_level=0.5,
            )
            for i in range(2)
        ]

        proc = ProcedureAttractor(
            procedure_id="reg_test",
            name="Registration Test",
            description="Test",
            steps=steps,
        )

        reg.register(proc)

        retrieved = reg.get("reg_test")
        assert retrieved is not None
        assert retrieved.procedure_id == proc.procedure_id

    def test_get_by_name(self):
        """Test getting procedure by exact name."""
        reg = ProcedureRegistry.get_instance()

        proc = ProcedureAttractor(
            procedure_id="name_test",
            name="Build Website",
            description="Test",
            steps=[],
        )

        reg.register(proc)

        found = reg.get_by_name("Build Website")
        assert found is not None
        assert found.procedure_id == "name_test"

        # Case insensitive
        found2 = reg.get_by_name("build website")
        assert found2 is not None

    def test_find_by_domain(self):
        """Test finding procedures by domain."""
        reg = ProcedureRegistry.get_instance()

        for name, domain in [
            ("Button Builder", "coding"),
            ("Color Picker", "design"),
            ("API Builder", "coding"),
        ]:
            proc = ProcedureAttractor(
                procedure_id=f"domain_{name}",
                name=name,
                description="Test",
                steps=[],
            )
            meta = ProcedureMetadata(
                procedure_id=proc.procedure_id,
                name=name,
                domain=domain,
                keywords=[],
            )
            reg.register(proc, meta)

        coding_procs = reg.find_by_domain("coding")
        assert len(coding_procs) == 2
        assert all(p.name in ["Button Builder", "API Builder"] for p in coding_procs)

    def test_find_by_keywords(self):
        """Test finding procedures by keywords."""
        reg = ProcedureRegistry.get_instance()

        proc = ProcedureAttractor(
            procedure_id="kw_test",
            name="Build Button",
            description="Test",
            steps=[],
        )
        meta = ProcedureMetadata(
            procedure_id="kw_test",
            name="Build Button",
            domain="coding",
            keywords=["button", "ui", "component"],
        )
        reg.register(proc, meta)

        found = reg.find_by_keywords(["button"])
        assert len(found) == 1

        found2 = reg.find_by_keywords(["ui", "frontend"])
        assert len(found2) == 1

    def test_summary(self):
        """Test registry summary."""
        reg = ProcedureRegistry.get_instance()

        for name, domain in [
            ("Proc 1", "coding"),
            ("Proc 2", "coding"),
            ("Proc 3", "design"),
        ]:
            proc = ProcedureAttractor(
                procedure_id=f"sum_{name}",
                name=name,
                description="Test",
                steps=[],
            )
            meta = ProcedureMetadata(
                procedure_id=proc.procedure_id,
                name=name,
                domain=domain,
                keywords=[],
            )
            reg.register(proc, meta)

        summary = reg.summary()
        assert summary["total_procedures"] == 3
        assert summary["by_domain"]["coding"] == 2
        assert summary["by_domain"]["design"] == 1


class TestProcedureStateEncoder:
    """Tests for ProcedureStateEncoder."""

    def test_encode_procedure(self):
        """Test encoding a procedure."""
        encoder = ProcedureStateEncoder(state_dim=1024)

        steps = [
            ProcedureStep(
                step_index=i,
                state=torch.randn(1024),
                description=f"Step {i}",
                energy_level=0.5,
            )
            for i in range(3)
        ]

        proc = ProcedureAttractor(
            procedure_id="encode_test",
            name="Encode Test",
            description="Test",
            steps=steps,
        )

        encoded = encoder.encode_procedure(proc, device=torch.device("cpu"))
        assert len(encoded) == 3
        assert all(s.shape == (1024,) for s in encoded)

    def test_encode_from_description(self):
        """Test encoding from text description."""
        encoder = ProcedureStateEncoder(state_dim=1024)

        states = encoder.encode_from_description(
            "Build a button component",
            num_steps=5,
        )

        assert len(states) == 5
        assert all(s.shape == (1024,) for s in states)

    def test_encode_reproducibility(self):
        """Test that encoding same description produces same result."""
        encoder = ProcedureStateEncoder(state_dim=1024)

        desc = "Test description"
        states1 = encoder.encode_from_description(desc, num_steps=3)
        states2 = encoder.encode_from_description(desc, num_steps=3)

        for s1, s2 in zip(states1, states2):
            assert torch.allclose(s1, s2)

    def test_interpolate_states(self):
        """Test state interpolation."""
        encoder = ProcedureStateEncoder(state_dim=1024)

        state_a = torch.zeros(1024)
        state_b = torch.ones(1024) * 2

        interpolated = encoder.interpolate_states(state_a, state_b, num_intermediate=2)

        assert len(interpolated) == 4  # start + 2 intermediate + end
        assert torch.allclose(interpolated[0], state_a)
        assert torch.allclose(interpolated[-1], state_b)
        # Check intermediate values
        assert torch.allclose(interpolated[1], state_a * (2/3) + state_b * (1/3))
        assert torch.allclose(interpolated[2], state_a * (1/3) + state_b * (2/3))


class TestBackwardsCompatibility:
    """Tests to ensure no existing code breaks."""

    def test_thought_graph_node_types(self):
        """Test that existing NodeTypes still work."""
        assert NodeType.INTENT is not None
        assert NodeType.FACT is not None
        assert NodeType.REASONING is not None
        assert NodeType.PLAN is not None
        assert NodeType.OUTPUT is not None

    def test_new_node_types(self):
        """Test new node types exist."""
        assert NodeType.PROCEDURE is not None
        assert NodeType.IMPLEMENT is not None

    def test_new_edge_types(self):
        """Test new edge types exist."""
        assert EdgeType.STEPS_THROUGH is not None
        assert EdgeType.IMPLEMENTS is not None

    def test_constants(self):
        """Test constants are defined correctly."""
        assert MAX_PROCEDURE_STEPS >= 10
        assert MIN_PROCEDURE_STEPS >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])