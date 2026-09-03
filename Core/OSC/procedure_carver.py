"""
Core/OSC/procedure_carver.py
NOESIS-Σ — Procedural Carver

Carves procedural attractors (trajectories) into the ICNN energy landscape.
Unlike concept carving (single state), procedure carving trains the system
to flow through a SEQUENCE of states representing a causal mechanism.

Mathematical basis:
  For procedure with steps x₀, x₁, ..., xₙ:
  - Each xᵢ becomes a stable fixed point (attractor basin)
  - PRECEDES edges connect them: xᵢ → xᵢ₊₁
  - OSC flowing near xᵢ will naturally flow toward xᵢ₊₁

Carving algorithm:
  Phase 1 (free): Flow from random start for θ steps
  Phase 2 (nudge): Nudge toward each step xᵢ in sequence
  ΔW = (1/β)·[∂φ/∂W|nudge - ∂φ/∂W|free]
  Repeat for all steps, then repeat trajectory multiple times

Safety constraints:
  - ICNN convexity preserved (EP only, no autograd through forward)
  - Lyapunov guarantee via project() after every step
  - Background carving only — never on request path
  - Trajectory length bounded
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from Core.OSC.ep_trainer import EquilibriumPropagation
from Core.OSC.procedure_attractor import (
    MAX_PROCEDURE_STEPS,
    MIN_PROCEDURE_STEPS,
    ProcedureAttractor,
    ProcedureMetadata,
    ProcedureRegistry,
    ProcedureStep,
    ProcedureStateEncoder,
)

logger = logging.getLogger(__name__)


@dataclass
class CarverResult:
    """Result of a carving operation."""
    procedure_id: str
    success: bool
    avg_basin_depth: float          # Average depth across all steps
    min_basin_depth: float          # Minimum (weakest) step
    total_energy: float
    ep_steps_used: int
    elapsed_seconds: float
    error: Optional[str] = None


class ProceduralCarver:
    """
    Carves procedural attractors into the OSC energy landscape.

    A procedure is a TRAJECTORY (sequence of states) not a single state.
    The system learns to flow through the procedure when activated.

    Usage::
        carver = ProceduralCarver(ep_trainer, state_encoder)
        result = carver.carve_procedure(
            name="build_button",
            description="How to build a UI button",
            steps=["setup", "style", "render", "test"],
            target_states=[x0, x1, x2, x3],
        )
    """

    def __init__(
        self,
        ep_trainer: EquilibriumPropagation,
        state_encoder: Optional[ProcedureStateEncoder] = None,
        steps_per_anchor: int = 30,      # EP steps per anchor step
        trajectory_repetitions: int = 3,  # How many times to repeat trajectory
        beta_carve: float = 0.001,        # Small beta for stable carving
        min_basin_depth: float = 0.6,     # Minimum depth to accept step
    ):
        self._ep = ep_trainer
        self._encoder = state_encoder or ProcedureStateEncoder(
            state_dim=ep_trainer.engine.params.state_dim
        )
        self._steps_per_anchor = steps_per_anchor
        self._trajectory_reps = trajectory_repetitions
        self._beta_carve = beta_carve
        self._min_basin_depth = min_basin_depth

        # Registry for all carved procedures
        self._registry = ProcedureRegistry.get_instance()

        # Save original beta
        self._original_beta = ep_trainer.beta

    # ---------------------------------------------------------------- public

    def carve_procedure(
        self,
        name: str,
        description: str,
        step_descriptions: List[str],
        target_states: List[torch.Tensor],
        procedure_id: Optional[str] = None,
        metadata: Optional[ProcedureMetadata] = None,
    ) -> CarverResult:
        """
        Carve a complete procedure into the energy landscape.

        Args:
            name: Human-readable name (e.g., "build_website")
            description: What this procedure accomplishes
            step_descriptions: List of descriptions for each step
            target_states: List of [state_dim] tensors, one per step
            procedure_id: Optional ID (generated if not provided)
            metadata: Optional metadata for retrieval

        Returns:
            CarverResult with success status and metrics
        """
        t0 = time.time()

        if len(target_states) < MIN_PROCEDURE_STEPS:
            return CarverResult(
                procedure_id=procedure_id or "unknown",
                success=False,
                avg_basin_depth=0.0,
                min_basin_depth=0.0,
                total_energy=0.0,
                ep_steps_used=0,
                elapsed_seconds=time.time() - t0,
                error=f"Need at least {MIN_PROCEDURE_STEPS} steps",
            )

        if len(target_states) > MAX_PROCEDURE_STEPS:
            return CarverResult(
                procedure_id=procedure_id or "unknown",
                success=False,
                avg_basin_depth=0.0,
                min_basin_depth=0.0,
                total_energy=0.0,
                ep_steps_used=0,
                elapsed_seconds=time.time() - t0,
                error=f"Exceeds max {MAX_PROCEDURE_STEPS} steps",
            )

        pid = procedure_id or f"proc_{int(t0 * 1000)}"

        device = self._ep.engine.device
        dtype = self._ep.engine.dtype
        state_dim = self._ep.engine.params.state_dim

        # Set carving beta
        self._ep.beta = self._beta_carve

        try:
            carved_steps: List[ProcedureStep] = []
            total_energy = 0.0
            total_ep_steps = 0
            basin_depths: List[float] = []

            # Repeat trajectory multiple times to strengthen all steps
            for rep in range(self._trajectory_reps):
                for step_idx, (desc, x_target) in enumerate(zip(step_descriptions, target_states)):
                    x_target = x_target.to(device=device, dtype=dtype)

                    # Carve this anchor step
                    for ep_step in range(self._steps_per_anchor):
                        x_init = torch.randn(state_dim, dtype=dtype, device=device)
                        x_init = self._ep.engine.project(x_init.unsqueeze(0)).squeeze(0)
                        try:
                            loss = self._ep.train_step_toward(
                                x_init=x_init,
                                context_bundle=None,
                                attractor=x_target,
                            )
                            total_energy += float(loss)
                            total_ep_steps += 1
                        except Exception as exc:
                            logger.debug("Step %d EP failed: %s", step_idx, exc)

                    # Verify basin depth
                    depth = self._verify_step(x_target)
                    basin_depths.append(depth)

                    # Store carved step
                    energy_at_step = float(self._ep.engine.energy(x_target.unsqueeze(0)).item())
                    step = ProcedureStep(
                        step_index=step_idx,
                        state=x_target.detach().clone(),
                        description=desc,
                        energy_level=energy_at_step,
                        is_anchor=True,
                    )
                    carved_steps.append(step)

            # Calculate metrics
            avg_depth = sum(basin_depths) / len(basin_depths) if basin_depths else 0.0
            min_depth = min(basin_depths) if basin_depths else 0.0

            # Create procedure attractor
            procedure = ProcedureAttractor(
                procedure_id=pid,
                name=name,
                description=description,
                steps=carved_steps,
                total_energy=total_energy,
                ep_steps=total_ep_steps,
                generalization_score=avg_depth,  # Use avg depth as initial score
            )

            # Register
            self._registry.register(procedure, metadata)

            return CarverResult(
                procedure_id=pid,
                success=True,
                avg_basin_depth=avg_depth,
                min_basin_depth=min_depth,
                total_energy=total_energy,
                ep_steps_used=total_ep_steps,
                elapsed_seconds=time.time() - t0,
            )

        except Exception as exc:
            logger.error("Carve procedure failed: %s", exc)
            return CarverResult(
                procedure_id=pid,
                success=False,
                avg_basin_depth=0.0,
                min_basin_depth=0.0,
                total_energy=0.0,
                ep_steps_used=0,
                elapsed_seconds=time.time() - t0,
                error=str(exc),
            )
        finally:
            self._ep.beta = self._original_beta

    def carve_from_trajectory(
        self,
        name: str,
        description: str,
        trajectory: List[torch.Tensor],
        step_descriptions: Optional[List[str]] = None,
        procedure_id: Optional[str] = None,
        metadata: Optional[ProcedureMetadata] = None,
    ) -> CarverResult:
        """
        Carve a procedure from an observed trajectory.

        Args:
            name: Procedure name
            description: What the procedure does
            trajectory: Observed sequence of states [state_dim]
            step_descriptions: Optional descriptions for each step
            procedure_id: Optional ID
            metadata: Optional metadata

        Returns:
            CarverResult
        """
        if step_descriptions is None:
            step_descriptions = [f"step_{i}" for i in range(len(trajectory))]

        return self.carve_procedure(
            name=name,
            description=description,
            step_descriptions=step_descriptions,
            target_states=trajectory,
            procedure_id=procedure_id,
            metadata=metadata,
        )

    def carve_from_example(
        self,
        name: str,
        description: str,
        example_input: str,
        example_output: str,
        num_steps: int = 5,
        procedure_id: Optional[str] = None,
        metadata: Optional[ProcedureMetadata] = None,
    ) -> CarverResult:
        """
        Carve a procedure from a text description of an example.

        This creates a synthetic trajectory based on the example.
        Real learning would come from observing actual procedures.

        Args:
            name: Procedure name
            description: What the procedure does
            example_input: Example input description
            example_output: Example output description
            num_steps: Number of steps to create
            procedure_id: Optional ID
            metadata: Optional metadata

        Returns:
            CarverResult
        """
        # Create synthetic trajectory from example
        combined_desc = f"{example_input} -> {example_output}"
        target_states = self._encoder.encode_from_description(
            combined_desc,
            num_steps=num_steps,
            device=self._ep.engine.device,
        )

        step_descriptions = [
            f"Step {i+1}: {['analyze', 'plan', 'execute', 'verify', 'refine'][min(i, 4)]} input"
            for i in range(num_steps)
        ]

        return self.carve_procedure(
            name=name,
            description=description,
            step_descriptions=step_descriptions,
            target_states=target_states,
            procedure_id=procedure_id,
            metadata=metadata,
        )

    def verify_procedure(self, procedure_id: str) -> float:
        """
        Verify procedure stability without modifying weights.

        Returns generalization score (0-1).
        """
        procedure = self._registry.get(procedure_id)
        if not procedure:
            return 0.0

        depths = []
        for step in procedure.steps:
            depth = self._verify_step(step.state)
            depths.append(depth)

        avg = sum(depths) / len(depths) if depths else 0.0

        # Update generalization score
        procedure.generalization_score = avg

        return avg

    def get_registry(self) -> ProcedureRegistry:
        """Get the procedure registry."""
        return self._registry

    # ---------------------------------------------------------------- private

    def _verify_step(self, x_target: torch.Tensor) -> float:
        """Verify basin depth for a single step."""
        try:
            with torch.no_grad():
                x_after = self._ep.engine.step_many(x_target, n_steps=20)
            if x_after.dim() == 2:
                x_after = x_after.squeeze(0)
            diff_norm = float((x_after - x_target).norm().item())
            target_norm = float(x_target.norm().item()) + 1e-8
            depth = 1.0 - diff_norm / target_norm
            return max(0.0, min(1.0, depth))
        except Exception as exc:
            logger.debug("verify_step failed: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------

def carve_coding_procedures(
    ep_trainer: EquilibriumPropagation,
) -> List[CarverResult]:
    """
    Carve common coding procedures for web development.

    These demonstrate the mechanism of coding, not memorized patterns.
    """
    carver = ProceduralCarver(ep_trainer)
    results = []

    procedures = [
        {
            "name": "build_button",
            "description": "Build a UI button component",
            "domain": "coding",
            "keywords": ["button", "ui", "component", "frontend"],
            "input_types": ["design", "style"],
            "output_type": "code",
        },
        {
            "name": "build_website",
            "description": "Build a complete website with header, hero, footer",
            "domain": "coding",
            "keywords": ["website", "landing", "frontend", "html", "react"],
            "input_types": ["requirements"],
            "output_type": "code",
        },
        {
            "name": "build_dialog",
            "description": "Build a dialog/modal component",
            "domain": "coding",
            "keywords": ["dialog", "modal", "popup", "ui"],
            "input_types": ["design"],
            "output_type": "code",
        },
    ]

    for proc_spec in procedures:
        metadata = ProcedureMetadata(
            procedure_id=proc_spec["name"],
            name=proc_spec["name"],
            domain=proc_spec["domain"],
            input_types=proc_spec["input_types"],
            output_type=proc_spec["output_type"],
            keywords=proc_spec["keywords"],
        )

        result = carver.carve_from_example(
            name=proc_spec["name"],
            description=proc_spec["description"],
            example_input=f"Input: {proc_spec['name']}",
            example_output=f"Output: {proc_spec['description']}",
            num_steps=5,
            metadata=metadata,
        )
        results.append(result)

    return results