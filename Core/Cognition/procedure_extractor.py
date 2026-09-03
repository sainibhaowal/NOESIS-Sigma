"""
Core/Cognition/procedure_extractor.py
NOESIS-Σ — Procedure Graph Extractor

Extracts PROCEDURE nodes from OSC trajectories and composes them
into ThoughtGraph. This is the bridge between dynamics and understanding.

Key difference from rule-based GraphExtractor:
  - Rule-based: Extracts nodes based on heuristics (inflection points, etc.)
  - Procedure-aware: Understands causal chains (procedures) and can compose them

How it works:
  1. Analyze trajectory for causal transitions (procedure steps)
  2. Identify procedure patterns (start → intermediate → end)
  3. Extract PROCEDURE nodes with STEPS_THROUGH edges
  4. If composition requested, add IMPLEMENT nodes

Safety constraints:
  - DAG enforcement maintained
  - No cycles in edge construction
  - All existing NodeTypes preserved
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from Core.Cognition.fusion import ContextBundle
from Core.Cognition.thought_graph import (
    EdgeType,
    NodeType,
    ThoughtGraph,
    ThoughtNode,
    make_edge,
    make_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCEDURE_MIN_STEPS = 3          # Minimum steps to be a procedure
CAUSAL_TRANSITION_THRESHOLD = 0.1  # Energy change threshold for causal transition
MAX_PROCEDURE_NODES = 10        # Maximum PROCEDURE nodes per graph


class ProcedureGraphExtractor:
    """
    Extracts PROCEDURE nodes from OSC trajectories.

    Adds capability to understand causal chains and compose them
    into solutions. This is what enables "general intelligence":
    understanding mechanisms, not just patterns.

    Usage::
        extractor = ProcedureGraphExtractor()
        graph = extractor.extract(
            trajectory=trajectory,
            bundle=bundle,
            include_procedures=True,
            compose_requests=["build website"],
        )
    """

    def __init__(
        self,
        causal_threshold: float = CAUSAL_TRANSITION_THRESHOLD,
        min_procedure_steps: int = PROCEDURE_MIN_STEPS,
    ):
        self._causal_threshold = causal_threshold
        self._min_steps = min_procedure_steps

    def extract(
        self,
        trajectory: List[torch.Tensor],
        bundle: ContextBundle,
        trace_id: Optional[str] = None,
        final_state: Optional[torch.Tensor] = None,
        include_procedures: bool = True,
        compose_requests: Optional[List[str]] = None,
    ) -> ThoughtGraph:
        """
        Extract ThoughtGraph including PROCEDURE nodes.

        Args:
            trajectory: OSC state trajectory
            bundle: Context bundle with request and evidence
            trace_id: Trace identifier
            final_state: Final state after convergence
            include_procedures: Whether to extract PROCEDURE nodes
            compose_requests: Requests to compose procedures for

        Returns:
            ThoughtGraph with PROCEDURE and IMPLEMENT nodes
        """
        from Core.Cognition.graph_extractor import GraphExtractor

        # Start with rule-based extraction
        base_extractor = GraphExtractor()
        graph = base_extractor.extract(
            trajectory=trajectory,
            bundle=bundle,
            trace_id=trace_id,
            final_state=final_state,
        )

        if not include_procedures:
            return graph

        # Add PROCEDURE nodes from trajectory
        if len(trajectory) >= self._min_steps:
            self._extract_procedure_nodes(graph, trajectory, bundle)

        # Add IMPLEMENT nodes for composed solutions
        if compose_requests:
            self._extract_implement_nodes(graph, compose_requests, bundle)

        return graph

    # ---------------------------------------------------------------- private

    def _extract_procedure_nodes(
        self,
        graph: ThoughtGraph,
        trajectory: List[torch.Tensor],
        bundle: ContextBundle,
    ) -> None:
        """
        Extract PROCEDURE nodes from trajectory.

        A procedure is identified by:
        1. Sufficient length (>= min_procedure_steps)
        2. Causal transitions (energy changes above threshold)
        3. Coherent flow from start to end
        """
        # Find causal transitions in trajectory
        transitions = self._find_causal_transitions(trajectory)

        if len(transitions) < 2:
            # Not enough transitions for a procedure
            return

        # Build procedure structure
        procedure_node = self._build_procedure_node(
            transitions=transitions,
            bundle=bundle,
            graph=graph,
        )

        # Add step nodes connected via STEPS_THROUGH edges
        self._add_procedure_steps(
            graph=graph,
            procedure_node=procedure_node,
            transitions=transitions,
        )

    def _find_causal_transitions(
        self,
        trajectory: List[torch.Tensor],
    ) -> List[Tuple[int, torch.Tensor, float]]:
        """
        Find causal transitions in trajectory.

        A causal transition is where the state meaningfully changes,
        indicating a step in some procedure.

        Returns:
            List of (step_index, state, energy_change) tuples
        """
        if len(trajectory) < 2:
            return []

        transitions = []
        prev_energy = float(trajectory[0].norm())

        for i in range(1, len(trajectory)):
            curr_energy = float(trajectory[i].norm())
            delta = abs(curr_energy - prev_energy)

            if delta > self._causal_threshold:
                transitions.append((i, trajectory[i], delta))

            prev_energy = curr_energy

        # Always include start and end as anchors
        anchors = [(0, trajectory[0], 0.0)]
        if transitions:
            anchors.extend(transitions)
            # Add final state
            anchors.append((len(trajectory) - 1, trajectory[-1], 0.0))

        return anchors

    def _build_procedure_node(
        self,
        transitions: List[Tuple[int, torch.Tensor, float]],
        bundle: ContextBundle,
        graph: ThoughtGraph,
    ) -> ThoughtNode:
        """Build a PROCEDURE node."""
        # Use request text to describe procedure
        request_text = bundle.request_text or "Unknown procedure"
        n_steps = len(transitions)

        procedure_node = make_node(
            node_type=NodeType.PROCEDURE,
            content=f"Procedure: {request_text[:100]} ({n_steps} steps)",
            confidence=0.85,
            metadata={
                "n_steps": n_steps,
                "is_composed": False,
                "transitions": [t[0] for t in transitions],
            },
        )

        graph.add_node(procedure_node)

        # Connect to INTENT
        intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
        if intent_nodes:
            try:
                graph.add_edge(make_edge(intent_nodes[0], procedure_node, EdgeType.DECOMPOSES_INTO))
            except ValueError:
                pass

        return procedure_node

    def _add_procedure_steps(
        self,
        graph: ThoughtGraph,
        procedure_node: ThoughtNode,
        transitions: List[Tuple[int, torch.Tensor, float]],
    ) -> None:
        """Add step nodes with STEPS_THROUGH edges."""
        prev_step_node = None

        for idx, (step_idx, state, energy_delta) in enumerate(transitions):
            # Create step node
            step_node = make_node(
                node_type=NodeType.PROCEDURE,
                content=f"Step {idx + 1} at trajectory position {step_idx}",
                confidence=0.8,
                metadata={
                    "step_number": idx + 1,
                    "trajectory_index": step_idx,
                    "energy_delta": energy_delta,
                    "is_anchor": idx == 0 or idx == len(transitions) - 1,
                },
            )

            graph.add_node(step_node)

            # Connect to procedure
            try:
                graph.add_edge(make_edge(procedure_node, step_node, EdgeType.STEPS_THROUGH))
            except ValueError:
                pass

            # Chain steps together
            if prev_step_node:
                try:
                    graph.add_edge(make_edge(prev_step_node, step_node, EdgeType.PRECEDES))
                except ValueError:
                    pass

            prev_step_node = step_node

    def _extract_implement_nodes(
        self,
        graph: ThoughtGraph,
        compose_requests: List[str],
        bundle: ContextBundle,
    ) -> None:
        """
        Add IMPLEMENT nodes for composed solutions.

        IMPLEMENT nodes represent outputs constructed from procedures.
        """
        # Find any existing PROCEDURE nodes
        proc_nodes = graph.get_nodes_by_type(NodeType.PROCEDURE)

        for request in compose_requests:
            # Create IMPLEMENT node
            implement_node = make_node(
                node_type=NodeType.IMPLEMENT,
                content=f"Constructed solution for: {request[:100]}",
                confidence=0.9,
                metadata={
                    "request": request,
                    "procedures_used": [p.node_id for p in proc_nodes],
                    "composition_type": "constructed",
                },
            )

            graph.add_node(implement_node)

            # Connect to OUTPUT
            output_nodes = graph.get_nodes_by_type(NodeType.OUTPUT)
            for out_node in output_nodes:
                try:
                    graph.add_edge(make_edge(implement_node, out_node, EdgeType.IMPLEMENTS))
                except ValueError:
                    pass

            # Connect to PROCEDURE nodes
            for proc_node in proc_nodes:
                try:
                    graph.add_edge(make_edge(proc_node, implement_node, EdgeType.IMPLIES))
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Hybrid Extractor (combines rule-based + procedure-aware)
# ---------------------------------------------------------------------------

class HybridProcedureExtractor:
    """
    Combines rule-based GraphExtractor with procedure awareness.

    This is the production extractor that:
    1. Uses rule-based extraction for baseline nodes
    2. Adds procedure extraction for causal understanding
    3. Handles composition for novel requests

    Drop-in replacement for GraphExtractor with identical interface.
    """

    def __init__(self):
        self._base = GraphExtractor()
        self._procedure = ProcedureGraphExtractor()

    def extract(
        self,
        trajectory: List[torch.Tensor],
        bundle: ContextBundle,
        trace_id: Optional[str] = None,
        final_state: Optional[torch.Tensor] = None,
    ) -> ThoughtGraph:
        """
        Extract graph with both rule-based and procedure-aware nodes.
        """
        return self._procedure.extract(
            trajectory=trajectory,
            bundle=bundle,
            trace_id=trace_id,
            final_state=final_state,
            include_procedures=True,
            compose_requests=[bundle.request_text] if bundle.request_text else None,
        )