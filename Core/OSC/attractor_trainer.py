"""
Core/OSC/attractor_trainer.py
NOESIS-Σ — Sprint D2: Attractor Carving via Equilibrium Propagation.

Trains φ(x) in ICNNDirectGrad so that high-value WorldModel concepts become
local energy minima (attractor basins).

Mathematical basis:
  Each concept c has a target state x*_c = ConceptStateEncoder.encode_concept(c).
  EP carving makes x*_c a stable attractor by running:
    Phase 1 (free):  x*_free  = OSC.step_many(x_rand)
    Phase 2 (nudged): x*_nudge = OSC.converge_with_nudge(x*_free, nudge=MSE→x*_c, β)
    ΔW = (1/β)·[∂φ/∂W|x*_nudge − ∂φ/∂W|x*_free]

After enough steps, ∇_x φ(x*_c) ≈ 0 — i.e., x*_c is a fixed point.

Storage capacity (Ramsauer 2021 Modern Hopfield theory):
  Modern Hopfield capacity = O(exp(d/2)).  At d=1024 → far more than 1000 concepts.

Safety constraints enforced here:
  - ICNN convexity is PRESERVED: ep_trainer updates only via EP (param_grad_at),
    never via autograd through the forward pass.
  - Lyapunov guarantee holds: project() is called after every step.
  - Carving is BACKGROUND only — never on the request path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from Core.OSC.ep_trainer import EquilibriumPropagation
from External.WorldModel.concept_state_encoder import ConceptStateEncoder

logger = logging.getLogger(__name__)


@dataclass
class AttractorRecord:
    concept_id: str
    concept_name: str
    basin_depth: float        # 0..1, higher = stronger attractor
    carved_at: float = field(default_factory=time.time)
    ep_steps: int = 0


class AttractorCarver:
    """
    Carves attractor basins for WorldModel concepts into OSC φ(x).

    Usage::
        carver = AttractorCarver(ep_trainer, encoder, steps_per_concept=50)
        for concept in top_concepts:
            record = carver.carve_one(concept)
            if record.basin_depth >= 0.7:
                carver.registry[concept.concept_id] = record

    carve_batch() processes a list and returns aggregate metrics.
    verify_attractor() checks basin depth without modifying weights.
    """

    def __init__(
        self,
        ep_trainer: EquilibriumPropagation,
        encoder: ConceptStateEncoder,
        steps_per_concept: int = 50,
        min_basin_depth: float = 0.7,
        beta: float = 0.001,   # small for carving (stability)
    ) -> None:
        self._ep = ep_trainer
        self._encoder = encoder
        self._steps_per_concept = steps_per_concept
        self._min_basin_depth = min_basin_depth
        self._beta_carve = beta

        # Registry: concept_id → AttractorRecord
        self.registry: Dict[str, AttractorRecord] = {}

        # Save the original beta and restore after carving
        self._original_beta = ep_trainer.beta

    # ---------------------------------------------------------------- public

    def carve_one(self, concept) -> AttractorRecord:
        """
        Carve an attractor for one concept.  Runs steps_per_concept EP steps.

        Args:
            concept: Any object with .concept_id, .name, and optional .description.

        Returns:
            AttractorRecord with measured basin_depth.
        """
        concept_id = getattr(concept, "concept_id", "") or str(concept)
        concept_name = getattr(concept, "name", concept_id)

        device = self._ep.engine.device
        dtype = self._ep.engine.dtype
        state_dim = self._ep.engine.params.state_dim

        x_target = self._encoder.encode_concept_obj(concept, device=device).to(dtype)

        # Save and temporarily override beta for carving
        self._ep.beta = self._beta_carve

        try:
            total_loss = 0.0
            for _ in range(self._steps_per_concept):
                # Random initialisation per step — teaches the basin to be wide
                x_init = torch.randn(state_dim, dtype=dtype, device=device)
                x_init = self._ep.engine.project(x_init.unsqueeze(0)).squeeze(0)
                try:
                    loss = self._ep.train_step_toward(
                        x_init=x_init,
                        context_bundle=None,
                        attractor=x_target,
                    )
                    total_loss += float(loss)
                except Exception as exc:
                    logger.debug("carve_one step failed: %s", exc)

            basin_depth = self.verify_attractor_tensor(x_target)
        finally:
            self._ep.beta = self._original_beta

        record = AttractorRecord(
            concept_id=concept_id,
            concept_name=concept_name,
            basin_depth=basin_depth,
            ep_steps=self._steps_per_concept,
        )
        self.registry[concept_id] = record
        logger.debug(
            "AttractorCarver: carved '%s'  basin_depth=%.3f",
            concept_name, basin_depth,
        )
        return record

    def carve_batch(self, concepts: List) -> Dict[str, Any]:
        """
        Carve attractors for a batch of concepts.
        Returns aggregate metrics dict.
        """
        depths = []
        carved = 0
        t0 = time.time()

        for concept in concepts:
            try:
                rec = self.carve_one(concept)
                depths.append(rec.basin_depth)
                if rec.basin_depth >= self._min_basin_depth:
                    carved += 1
            except Exception as exc:
                logger.warning("carve_batch: concept failed: %s", exc)

        elapsed = time.time() - t0
        return {
            "attempted":    len(concepts),
            "carved":       carved,
            "avg_basin_depth": float(sum(depths) / len(depths)) if depths else 0.0,
            "elapsed_s":    elapsed,
        }

    def verify_attractor(self, concept) -> float:
        """
        Verify basin depth for a concept without modifying weights.

        Basin depth = 1 - ||OSC.step_many(x_target) - x_target|| / ||x_target||

        Returns float in [0, 1]. Higher = stronger attractor.
        """
        device = self._ep.engine.device
        dtype = self._ep.engine.dtype
        x_target = self._encoder.encode_concept_obj(concept, device=device).to(dtype)
        return self.verify_attractor_tensor(x_target)

    def verify_attractor_tensor(self, x_target: torch.Tensor) -> float:
        """Basin depth given raw target tensor (no concept object needed)."""
        try:
            with torch.no_grad():
                x_after = self._ep.engine.step_many(x_target, n_steps=20)
            if x_after.dim() == 2:
                x_after = x_after.squeeze(0)
            x_t = x_target.to(x_after)
            diff_norm = float((x_after - x_t).norm().item())
            target_norm = float(x_t.norm().item()) + 1e-8
            depth = 1.0 - diff_norm / target_norm
            return max(0.0, min(1.0, depth))
        except Exception as exc:
            logger.debug("verify_attractor_tensor failed: %s", exc)
            return 0.0

    def is_carved(self, concept) -> bool:
        cid = getattr(concept, "concept_id", "") or str(concept)
        rec = self.registry.get(cid)
        return rec is not None and rec.basin_depth >= self._min_basin_depth

    def get_registry_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "concept_id":  r.concept_id,
                "concept_name": r.concept_name,
                "basin_depth":  r.basin_depth,
                "ep_steps":     r.ep_steps,
                "carved_at":    r.carved_at,
            }
            for r in self.registry.values()
        ]
