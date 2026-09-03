"""
Core/OSC/procedure_attractor.py
NOESIS-Σ — Procedural Attractor System

Stores PROCEDURAL knowledge: how to do X, not just what X is.
Unlike concept attractors (single state), procedure attractors store
TRAJECTORIES (sequence of states representing causal flow).

Example:
  - Concept "button": single state encoding "this is a button"
  - Procedure "build_button": trajectory [state:setup → state:style → state:render]

Key difference from concept attractors:
  - Concept attractor: learn once, recognize pattern
  - Procedure attractor: learn mechanism, construct new

Mathematical basis:
  Procedure stored as sequence of states x₀, x₁, ..., xₙ
  Each xᵢ is a stable fixed point (attractor basin)
  Trajectory connects them via PRECEDES edges
  OSC flows through trajectory to construct solution

Safety constraints:
  - All operations use EP (not autograd through forward pass)
  - Lyapunov guarantee via project() after every step
  - Trajectory length bounded (max_procedure_steps)
  - Background carving only — never on request path
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PROCEDURE_STEPS = 50  # Maximum steps in a procedure trajectory
MIN_PROCEDURE_STEPS = 2   # Minimum steps (start → end)
DEFAULT_PROCEDURE_STEPS = 5


@dataclass
class ProcedureStep:
    """Single step in a procedure trajectory."""
    step_index: int          # Position in procedure (0, 1, 2, ...)
    state: torch.Tensor      # [state_dim] - the state at this step
    description: str         # Human-readable description of this step
    energy_level: float     # Energy at this state (for basin depth)
    is_anchor: bool = False # Anchor steps are carved, intermediate are learned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "description": self.description,
            "energy_level": self.energy_level,
            "is_anchor": self.is_anchor,
            # State stored as list for serialization
            "state": self.state.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcedureStep":
        return cls(
            step_index=d["step_index"],
            state=torch.tensor(d["state"], dtype=torch.float32),
            description=d["description"],
            energy_level=d["energy_level"],
            is_anchor=d.get("is_anchor", False),
        )


@dataclass
class ProcedureAttractor:
    """
    A procedural attractor: stores the mechanism of HOW to do something.

    Unlike concept attractors (single state), this stores a TRAJECTORY
    of states representing the causal flow to accomplish a task.

    Attributes:
        procedure_id: Unique identifier
        name: Human-readable name (e.g., "build_website", "solve_equation")
        description: What this procedure accomplishes
        steps: Ordered list of ProcedureStep objects
        total_energy: Sum of energies (lower = more stable)
        carved_at: Timestamp when carved
        ep_steps: Number of EP steps used to carve
        generalization_score: How well this generalizes to new inputs (0-1)
    """
    procedure_id: str
    name: str
    description: str
    steps: List[ProcedureStep] = field(default_factory=list)
    total_energy: float = 0.0
    carved_at: float = field(default_factory=time.time)
    ep_steps: int = 0
    generalization_score: float = 0.0  # Will be computed after verification

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def trajectory(self) -> List[torch.Tensor]:
        """Get just the state tensors in order."""
        return [s.state for s in self.steps]

    @property
    def start_state(self) -> Optional[torch.Tensor]:
        """First state in trajectory (initial state for procedure)."""
        return self.steps[0].state if self.steps else None

    @property
    def end_state(self) -> Optional[torch.Tensor]:
        """Last state in trajectory (final state after procedure)."""
        return self.steps[-1].state if self.steps else None

    def step_descriptions(self) -> List[str]:
        """Get list of step descriptions."""
        return [s.description for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "total_energy": self.total_energy,
            "carved_at": self.carved_at,
            "ep_steps": self.ep_steps,
            "generalization_score": self.generalization_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcedureAttractor":
        return cls(
            procedure_id=d["procedure_id"],
            name=d["name"],
            description=d["description"],
            steps=[ProcedureStep.from_dict(s) for s in d["steps"]],
            total_energy=d.get("total_energy", 0.0),
            carved_at=d.get("carved_at", time.time()),
            ep_steps=d.get("ep_steps", 0),
            generalization_score=d.get("generalization_score", 0.0),
        )


@dataclass
class ProcedureMetadata:
    """Metadata for a procedure for composition and retrieval."""
    procedure_id: str
    name: str
    domain: str = "general"                      # e.g., "coding", "math", "design"
    input_types: List[str] = field(default_factory=list)           # What inputs this procedure expects
    output_type: str = "unknown"                  # What this procedure produces
    prerequisite_procedures: List[str] = field(default_factory=list)  # Procedure IDs this depends on
    complexity: float = 0.5                 # 0-1, complexity of procedure
    keywords: List[str] = field(default_factory=list)               # For retrieval matching

    def to_dict(self) -> Dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "domain": self.domain,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "prerequisite_procedures": self.prerequisite_procedures,
            "complexity": self.complexity,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcedureMetadata":
        return cls(
            procedure_id=d["procedure_id"],
            name=d["name"],
            domain=d.get("domain", "general"),
            input_types=d.get("input_types", []),
            output_type=d.get("output_type", "unknown"),
            prerequisite_procedures=d.get("prerequisite_procedures", []),
            complexity=d.get("complexity", 0.5),
            keywords=d.get("keywords", []),
        )


class ProcedureRegistry:
    """
    Registry of all procedure attractors in the system.

    Provides:
    - Registration of new procedures
    - Retrieval by ID, name, domain, keywords
    - Composition support (find compatible procedures)
    """
    _instance: Optional["ProcedureRegistry"] = None

    def __init__(self):
        self._procedures: Dict[str, ProcedureAttractor] = {}
        self._metadata: Dict[str, ProcedureMetadata] = {}

    @classmethod
    def get_instance(cls) -> "ProcedureRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(
        self,
        procedure: ProcedureAttractor,
        metadata: Optional[ProcedureMetadata] = None,
    ) -> None:
        """Register a new procedure attractor."""
        self._procedures[procedure.procedure_id] = procedure
        if metadata:
            self._metadata[procedure.procedure_id] = metadata
        logger.debug("Registered procedure: %s (%s)", procedure.name, procedure.procedure_id)

    def get(self, procedure_id: str) -> Optional[ProcedureAttractor]:
        """Get procedure by ID."""
        return self._procedures.get(procedure_id)

    def get_by_name(self, name: str) -> Optional[ProcedureAttractor]:
        """Get procedure by exact name match."""
        for proc in self._procedures.values():
            if proc.name.lower() == name.lower():
                return proc
        return None

    def find_by_domain(self, domain: str) -> List[ProcedureAttractor]:
        """Find all procedures in a domain."""
        results = []
        for meta in self._metadata.values():
            if meta.domain.lower() == domain.lower():
                proc = self._procedures.get(meta.procedure_id)
                if proc:
                    results.append(proc)
        return results

    def find_by_keywords(self, keywords: List[str]) -> List[ProcedureAttractor]:
        """Find procedures matching keywords (any match)."""
        results = []
        keywords_lower = [k.lower() for k in keywords]
        for meta in self._metadata.values():
            if any(kw in meta.keywords for kw in keywords_lower):
                proc = self._procedures.get(meta.procedure_id)
                if proc:
                    results.append(proc)
        return results

    def find_composable(self, procedure_ids: List[str]) -> List[ProcedureAttractor]:
        """Get procedures and ensure they're composable (prerequisites met)."""
        results = []
        for pid in procedure_ids:
            proc = self._procedures.get(pid)
            if proc:
                meta = self._metadata.get(pid)
                if meta:
                    # Check prerequisites
                    prereqs_met = all(p in self._procedures for p in meta.prerequisite_procedures)
                    if prereqs_met:
                        results.append(proc)
        return results

    def list_all(self) -> List[ProcedureAttractor]:
        """List all registered procedures."""
        return list(self._procedures.values())

    def summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        domains: Dict[str, int] = {}
        for meta in self._metadata.values():
            domains[meta.domain] = domains.get(meta.domain, 0) + 1
        return {
            "total_procedures": len(self._procedures),
            "by_domain": domains,
            "total_steps": sum(p.num_steps for p in self._procedures.values()),
        }


# ---------------------------------------------------------------------------
# Procedure State Encoder
# ---------------------------------------------------------------------------

class ProcedureStateEncoder:
    """
    Encodes procedure information into OSC state space.

    Converts a procedure description into a target state trajectory.
    """

    def __init__(self, state_dim: int = 1024):
        self.state_dim = state_dim

    def encode_procedure(
        self,
        procedure: ProcedureAttractor,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Encode procedure into list of target states.

        Args:
            procedure: The procedure to encode
            device: Target device

        Returns:
            List of [state_dim] tensors, one per step
        """
        # For now, we use the stored states directly
        # Future: could add encoding transformations
        return [s.state.to(device) for s in procedure.steps]

    def encode_from_description(
        self,
        description: str,
        num_steps: int = DEFAULT_PROCEDURE_STEPS,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Encode a procedure from text description.

        This creates a synthetic trajectory for a procedure described in text.
        Real trajectories come from observing actual procedures being performed.

        Args:
            description: Text description of procedure
            num_steps: Number of steps to create
            device: Target device

        Returns:
            List of [state_dim] tensors
        """
        # Generate anchor states using hash of description
        # In real system, this would come from actual procedure observations
        rng = torch.Generator(device=device)
        # Seed RNG from description for reproducibility
        seed = hash(description) % (2**31)
        rng.manual_seed(seed)

        states = []
        for i in range(num_steps):
            # Create state with position encoding
            state = torch.randn(self.state_dim, generator=rng, device=device) * 0.1
            # Add step encoding
            step_weight = i / max(num_steps - 1, 1)
            state += torch.full_like(state, step_weight * 0.5)
            states.append(state)

        return states

    def interpolate_states(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        num_intermediate: int = 2,
    ) -> List[torch.Tensor]:
        """
        Create intermediate states between two anchor states.

        Used to expand sparse trajectories.

        Args:
            state_a: Start state [state_dim]
            state_b: End state [state_dim]
            num_intermediate: Number of intermediate states

        Returns:
            List including start, intermediates, end
        """
        result = [state_a]
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            intermediate = state_a * (1 - alpha) + state_b * alpha
            result.append(intermediate)
        result.append(state_b)
        return result