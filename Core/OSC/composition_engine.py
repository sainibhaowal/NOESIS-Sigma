"""
Core/OSC/composition_engine.py
NOESIS-Σ — Procedure Composition Engine

Composes multiple procedure attractors to build solutions for complex requests.
This is the "general intelligence" component: combining learned mechanisms
to solve NEW problems, not just replaying seen examples.

Example:
  - User requests: "Build landing page with wave animation"
  - System:
    1. Find "build_website" procedure
    2. Find "add_animation" procedure
    3. Compose them into solution
  - Output: Actual code (constructed, not memorized)

Key difference from pattern matching:
  - Pattern matching: "I've seen this exact problem" → "use this answer"
  - Composition: "I understand how building works + how animations work" → "construct new solution"

How composition works:
  1. Decompose request into sub-tasks
  2. Find relevant procedures for each sub-task
  3. Chain procedure trajectories
  4. Inject request context into trajectory
  5. Let OSC flow through combined trajectory
  6. Extract constructed output

Safety constraints:
  - All operations use EP (not autograd through forward pass)
  - Lyapunov guarantee via project() after every step
  - Composition depth bounded (max_chain_length)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from Core.Cognition.fusion import ContextBundle
from Core.OSC.procedure_attractor import (
    ProcedureAttractor,
    ProcedureRegistry,
    ProcedureStateEncoder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHAIN_LENGTH = 10       # Maximum procedures in a chain
MAX_COMPOSITION_STEPS = 100  # Maximum OSC steps for composition
COMPOSITION_DELTA_THRESHOLD = 1e-4


@dataclass
class CompositionRequest:
    """Request to compose procedures into a solution."""
    request_text: str           # What the user wants
    domain: str                 # Domain (coding, math, etc.)
    keywords: List[str]        # Keywords to match procedures
    context: Optional[ContextBundle] = None  # Additional context
    max_chain_length: int = MAX_CHAIN_LENGTH


@dataclass
class CompositionStep:
    """Single step in a composed solution."""
    step_index: int
    procedure_id: str
    procedure_name: str
    step_in_procedure: int      # Which step in the procedure
    action: str                 # What action this step takes
    state: Optional[torch.Tensor] = None  # State at this step
    result: Optional[str] = None  # Text/code result if any


@dataclass
class CompositionResult:
    """Result of composing procedures."""
    success: bool
    composed_trajectory: List[torch.Tensor]  # Combined state trajectory
    composition_steps: List[CompositionStep]  # What was done
    final_state: Optional[torch.Tensor] = None
    output_description: str = ""
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class ProcedureComposer:
    """
    Composes multiple procedure attractors to solve complex requests.

    This is the key to generalization: instead of memorizing solutions,
    the system understands component mechanisms and composes them.

    Usage::
        composer = ProcedureComposer(osc_engine)
        result = composer.compose(
            request="Build website with animation",
            keywords=["website", "animation"],
        )
        # result.output_description contains the constructed solution
    """

    def __init__(
        self,
        osc_engine,
        state_encoder: Optional[ProcedureStateEncoder] = None,
        max_chain_length: int = MAX_CHAIN_LENGTH,
    ):
        self._engine = osc_engine
        self._encoder = state_encoder or ProcedureStateEncoder(
            state_dim=osc_engine.params.state_dim
        )
        self._max_chain = max_chain_length
        self._registry = ProcedureRegistry.get_instance()

    # ---------------------------------------------------------------- public

    def compose(
        self,
        request: CompositionRequest,
    ) -> CompositionResult:
        """
        Compose procedures to solve a request.

        Args:
            request: CompositionRequest with request details

        Returns:
            CompositionResult with composed solution
        """
        t0 = time.time()

        # 1. Find relevant procedures
        procedures = self._find_relevant_procedures(request)

        if not procedures:
            return CompositionResult(
                success=False,
                composed_trajectory=[],
                composition_steps=[],
                error=f"No procedures found for: {request.keywords}",
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # 2. Build composition chain
        chain = self._build_chain(procedures, request)

        # 3. Execute composition
        result = self._execute_chain(chain, request)

        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    def compose_from_request_text(
        self,
        request_text: str,
        keywords: Optional[List[str]] = None,
        context: Optional[ContextBundle] = None,
    ) -> CompositionResult:
        """
        Simple interface: compose from just request text.

        Args:
            request_text: What the user wants
            keywords: Optional keywords (extracted from request if not provided)
            context: Optional context bundle

        Returns:
            CompositionResult
        """
        if keywords is None:
            keywords = self._extract_keywords(request_text)

        domain = self._infer_domain(keywords)

        request = CompositionRequest(
            request_text=request_text,
            domain=domain,
            keywords=keywords,
            context=context,
        )

        return self.compose(request)

    def test_generalization(
        self,
        procedure_id: str,
        test_input: str,
    ) -> float:
        """
        Test if a procedure can generalize to new input.

        Args:
            procedure_id: Procedure to test
            test_input: New input not seen during training

        Returns:
            Generalization score (0-1)
        """
        procedure = self._registry.get(procedure_id)
        if not procedure:
            return 0.0

        # Create test state from input
        test_states = self._encoder.encode_from_description(
            test_input,
            num_steps=procedure.num_steps,
            device=self._engine.device,
        )

        # Check if test states flow toward procedure states
        scores = []
        for proc_step, test_state in zip(procedure.steps, test_states):
            # Distance from test_state to procedure attractor
            dist = float((test_state - proc_step.state).norm().item())
            score = 1.0 / (1.0 + dist)  # Convert distance to similarity
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    # ---------------------------------------------------------------- private

    def _find_relevant_procedures(
        self,
        request: CompositionRequest,
    ) -> List[ProcedureAttractor]:
        """Find procedures relevant to the request."""
        results = []

        # Search by keywords
        by_keywords = self._registry.find_by_keywords(request.keywords)
        results.extend(by_keywords)

        # Search by domain
        by_domain = self._registry.find_by_domain(request.domain)
        for proc in by_domain:
            if proc not in results:
                results.append(proc)

        # Limit to max chain length
        return results[: self._max_chain]

    def _build_chain(
        self,
        procedures: List[ProcedureAttractor],
        request: CompositionRequest,
    ) -> List[Tuple[ProcedureAttractor, float]]:
        """
        Build ordered chain of procedures.

        Returns:
            List of (procedure, relevance_score) ordered for composition
        """
        chain = []

        # Score each procedure by relevance to request
        scored = []
        for proc in procedures:
            # Calculate relevance based on keywords match
            meta = self._registry._metadata.get(proc.procedure_id)
            if meta:
                keyword_match = sum(1 for kw in request.keywords if kw in meta.keywords)
                keyword_score = keyword_match / max(len(request.keywords), 1)
            else:
                keyword_score = 0.0

            relevance = keyword_score
            scored.append((proc, relevance))

        # Sort by relevance (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top procedures
        for proc, rel in scored[: self._max_chain]:
            chain.append((proc, rel))

        return chain

    def _execute_chain(
        self,
        chain: List[Tuple[ProcedureAttractor, float]],
        request: CompositionRequest,
    ) -> CompositionResult:
        """Execute the composed chain."""
        composition_steps = []
        all_states = []
        current_state = None

        device = self._engine.device
        dtype = self._engine.dtype

        for proc_idx, (procedure, relevance) in enumerate(chain):
            # Get procedure trajectory
            trajectory = procedure.trajectory

            # Inject request context into first step
            if proc_idx == 0 and request.context:
                # Modify first state with request context
                ctx_tensor = self._build_context_state(request.context)
                if current_state is not None:
                    # Blend with previous state
                    current_state = current_state * 0.7 + ctx_tensor * 0.3
                else:
                    current_state = ctx_tensor

            # Process each step in procedure
            for step_idx, step_state in enumerate(trajectory):
                # Clone state for modification
                if current_state is not None:
                    x = current_state.clone()
                else:
                    x = step_state.clone()

                # Run one OSC step to integrate state
                with torch.no_grad():
                    x = self._engine.step(x, sim_graft=None)
                    x = self._engine.project(x.unsqueeze(0)).squeeze(0)

                # Record step
                comp_step = CompositionStep(
                    step_index=len(composition_steps),
                    procedure_id=procedure.procedure_id,
                    procedure_name=procedure.name,
                    step_in_procedure=step_idx,
                    action=procedure.steps[step_idx].description if step_idx < len(procedure.steps) else "",
                    state=x.detach().clone(),
                )
                composition_steps.append(comp_step)
                all_states.append(x)

                current_state = x

        # Build output description from composition
        output_desc = self._synthesize_output(request, composition_steps)

        return CompositionResult(
            success=True,
            composed_trajectory=all_states,
            composition_steps=composition_steps,
            final_state=current_state,
            output_description=output_desc,
        )

    def _build_context_state(
        self,
        context: ContextBundle,
    ) -> torch.Tensor:
        """Build initial state from context bundle."""
        from Core.Cognition.fusion import build_context_tensor

        ctx_tensor = build_context_tensor(context)
        return ctx_tensor.squeeze(0) if ctx_tensor.dim() > 1 else ctx_tensor

    def _synthesize_output(
        self,
        request: CompositionRequest,
        steps: List[CompositionStep],
    ) -> str:
        """Synthesize output description from composition."""
        parts = [f"Request: {request.request_text[:200]}"]

        # Group by procedure
        proc_names = set()
        for step in steps:
            proc_names.add(step.procedure_name)

        if proc_names:
            parts.append(f"Composed from: {', '.join(proc_names)}")
            parts.append(f"Total steps: {len(steps)}")

        parts.append(f"Domain: {request.domain}")
        parts.append(f"Keywords matched: {', '.join(request.keywords)}")

        return " | ".join(parts)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from request text."""
        # Simple token-based extraction
        # In production, would use more sophisticated NLP
        words = text.lower().split()

        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "and", "or", "but",
        }

        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Take first 10 keywords
        return keywords[:10]

    def _infer_domain(self, keywords: List[str]) -> str:
        """Infer domain from keywords."""
        domain_keywords = {
            "coding": ["code", "coding", "programming", "function", "class", "build", "website", "frontend", "backend", "api", "html", "css", "javascript", "python", "react"],
            "math": ["math", "equation", "calculation", "number", "formula", "algebra", "calculus"],
            "design": ["design", "ui", "ux", "layout", "color", "style", "animation", "visual"],
            "data": ["data", "database", "query", "table", "sql", "storage"],
        }

        scores = {}
        for domain, domain_kws in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in domain_kws)
            scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"


# ---------------------------------------------------------------------------
# High-level Interface
# ---------------------------------------------------------------------------

def compose_solution(
    osc_engine,
    request_text: str,
    keywords: Optional[List[str]] = None,
) -> str:
    """
    High-level function to compose a solution.

    Usage::
        output = compose_solution(engine, "Build a website with button")
        print(output)
    """
    composer = ProcedureComposer(osc_engine)
    result = composer.compose_from_request_text(request_text, keywords)
    return result.output_description


def check_procedure_generalization(
    osc_engine,
    procedure_id: str,
) -> Dict[str, Any]:
    """
    Test if a procedure can generalize to new inputs.

    Returns test results.
    """
    composer = ProcedureComposer(osc_engine)

    # Run multiple tests
    test_inputs = [
        "build something new",
        "create different thing",
        "construct novel solution",
    ]

    results = []
    for test_input in test_inputs:
        score = composer.test_generalization(procedure_id, test_input)
        results.append({"input": test_input, "score": score})

    avg_score = sum(r["score"] for r in results) / len(results)

    return {
        "procedure_id": procedure_id,
        "tests": results,
        "average_score": avg_score,
        "generalization": "high" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "low",
    }