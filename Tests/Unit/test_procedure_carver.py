"""
Tests/Unit/test_procedure_carver.py

Unit tests for Core/OSC/procedure_carver.py

Tests:
1. ProceduralCarver initialization
2. Carve procedure with target states
3. Carve from trajectory
4. Carve from example description
5. Verify procedure stability
6. Error handling (too few steps, etc.)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Core.OSC.procedure_attractor import (
    ProcedureRegistry,
    ProcedureStateEncoder,
)
from Core.OSC.procedure_carver import (
    CarverResult,
    ProceduralCarver,
    carve_coding_procedures,
    MAX_PROCEDURE_STEPS,
    MIN_PROCEDURE_STEPS,
)

pytestmark = pytest.mark.unit


class MockEPEngine:
    """Mock OSC engine for testing."""
    def __init__(self, state_dim=1024):
        self.params = MagicMock()
        self.params.state_dim = state_dim
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def project(self, x):
        # Handle both [d] and [1, d] inputs
        if x.dim() == 2:
            norm = x.norm(dim=-1, keepdim=True) + 1e-8
            return x / norm * torch.clamp(norm, max=16)
        else:
            norm = x.norm() + 1e-8
            return x / norm * min(norm, 16)

    def energy(self, x):
        # Handle both [d] and [1, d] inputs
        if x.dim() == 2:
            return (x ** 2).sum(dim=-1).unsqueeze(-1) * 0.5
        else:
            return (x ** 2).sum() * 0.5

    def step_many(self, x, n_steps=20):
        # Just return x for mock
        return x

    def step(self, x, **kwargs):
        return x.unsqueeze(0) if x.dim() == 1 else x


class MockEPTrainer:
    """Mock EP trainer for testing."""
    def __init__(self, engine):
        self.engine = engine
        self.beta = 0.001

    def train_step_toward(self, x_init, context_bundle, attractor):
        # Mock training loss - return scalar float
        return 0.1


class TestProceduralCarver:
    """Tests for ProceduralCarver class."""

    def setup_method(self):
        """Set up mock components for each test."""
        ProcedureRegistry.reset()
        self.engine = MockEPEngine(state_dim=1024)
        self.ep_trainer = MockEPTrainer(self.engine)
        self.carver = ProceduralCarver(
            self.ep_trainer,
            steps_per_anchor=5,
            trajectory_repetitions=1,
        )

    def test_init(self):
        """Test ProceduralCarver initialization."""
        assert self.carver._steps_per_anchor == 5
        assert self.carver._trajectory_reps == 1
        assert self.carver._beta_carve == 0.001

    def test_carve_procedure_basic(self):
        """Test carving a basic procedure."""
        # Create target states
        target_states = [
            torch.randn(1024) * 0.1 for _ in range(3)
        ]
        step_descriptions = ["Setup", "Execute", "Complete"]

        result = self.carver.carve_procedure(
            name="test_procedure",
            description="A test procedure",
            step_descriptions=step_descriptions,
            target_states=target_states,
            procedure_id="test_carve_1",
        )

        assert result.success is True
        assert result.avg_basin_depth >= 0.0
        assert result.ep_steps_used > 0

        # Check registry
        proc = ProcedureRegistry.get_instance().get("test_carve_1")
        assert proc is not None
        assert proc.name == "test_procedure"

    def test_carve_procedure_too_few_steps(self):
        """Test that carving fails with too few steps."""
        target_states = [torch.randn(1024)]  # Only 1 step

        result = self.carver.carve_procedure(
            name="too_few",
            description="Should fail",
            step_descriptions=["Only one"],
            target_states=target_states,
        )

        assert result.success is False
        assert "at least" in result.error

    def test_carve_procedure_too_many_steps(self):
        """Test that carving fails with too many steps."""
        target_states = [torch.randn(1024) for _ in range(MAX_PROCEDURE_STEPS + 10)]

        result = self.carver.carve_procedure(
            name="too_many",
            description="Should fail",
            step_descriptions=[f"step_{i}" for i in range(len(target_states))],
            target_states=target_states,
        )

        assert result.success is False
        assert "Exceeds max" in result.error

    def test_carve_from_trajectory(self):
        """Test carving from an existing trajectory."""
        trajectory = [torch.randn(1024) * 0.1 for _ in range(4)]

        result = self.carver.carve_from_trajectory(
            name="trajectory_test",
            description="From existing trajectory",
            trajectory=trajectory,
            step_descriptions=["A", "B", "C", "D"],
            procedure_id="traj_test",
        )

        assert result.success is True

    def test_carve_from_example(self):
        """Test carving from text description of example."""
        result = self.carver.carve_from_example(
            name="example_test",
            description="Test from example",
            example_input="Build a button",
            example_output="Button component code",
            num_steps=4,
            procedure_id="example_test",
        )

        assert result.success is True

        # Check procedure was registered
        proc = ProcedureRegistry.get_instance().get("example_test")
        assert proc is not None
        assert proc.num_steps == 4

    def test_verify_procedure(self):
        """Test procedure verification."""
        # First carve a procedure
        target_states = [torch.randn(1024) * 0.1 for _ in range(3)]
        self.carver.carve_procedure(
            name="verify_test",
            description="Verify test",
            step_descriptions=["S1", "S2", "S3"],
            target_states=target_states,
            procedure_id="verify_test",
        )

        # Verify
        score = self.carver.verify_procedure("verify_test")
        assert 0.0 <= score <= 1.0

    def test_verify_nonexistent_procedure(self):
        """Test verifying non-existent procedure."""
        score = self.carver.verify_procedure("does_not_exist")
        assert score == 0.0

    def test_get_registry(self):
        """Test getting the registry."""
        reg = self.carver.get_registry()
        assert reg is not None
        assert isinstance(reg, ProcedureRegistry)


class TestCarveCodingProcedures:
    """Tests for carve_coding_procedures helper."""

    def setup_method(self):
        ProcedureRegistry.reset()
        self.engine = MockEPEngine(state_dim=1024)
        self.ep_trainer = MockEPTrainer(self.engine)

    def test_carve_coding_procedures(self):
        """Test carving common coding procedures."""
        results = carve_coding_procedures(self.ep_trainer)

        assert len(results) == 3  # 3 procedures defined

        # All should succeed
        for result in results:
            assert result.success is True

        # Check registry has procedures
        reg = ProcedureRegistry.get_instance()
        summary = reg.summary()
        assert summary["total_procedures"] == 3


class TestCarverResult:
    """Tests for CarverResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = CarverResult(
            procedure_id="test",
            success=True,
            avg_basin_depth=0.8,
            min_basin_depth=0.6,
            total_energy=1.5,
            ep_steps_used=100,
            elapsed_seconds=0.5,
        )

        assert result.success is True
        assert result.avg_basin_depth == 0.8
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = CarverResult(
            procedure_id="test",
            success=False,
            avg_basin_depth=0.0,
            min_basin_depth=0.0,
            total_energy=0.0,
            ep_steps_used=0,
            elapsed_seconds=0.1,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])