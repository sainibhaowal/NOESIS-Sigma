"""
NOESIS-Σ :: Core/Cognition/cognitive_loop.py

CognitionLoop — drives N OSC steps, injects context via sim_graft, detects
convergence, and returns a ThoughtGraph + final state.

Architectural rules enforced here:
  Rule 1: OSC runs N=20–100 steps BEFORE any output is generated
  Rule 4: Lyapunov stability — OSC engine ensures ||x(t)|| bounded; we assert it
  Rule 5: No O(n²) — loop is O(N × d²) independent of input length

Context injection schedule:
  - Steps 0..(inject_window-1): inject u_t as sim_graft (OSC absorbs context)
  - Steps inject_window..N:     free evolution (convergence / crystallization)

Convergence criterion:
  ||x(t+1) - x(t)||₂ < delta_threshold for `convergence_window` consecutive steps

Usage:
    loop = CognitionLoop(engine)
    graph, x_final, traj = loop.run(x_init, bundle, trace_id="abc")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from Core.Cognition.fusion import ContextBundle, build_context_tensor
from Core.Cognition.graph_extractor import GraphExtractor, TrainedGraphExtractor
from Core.Cognition.thought_graph import ThoughtGraph

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ config


@dataclass
class LoopConfig:
    n_min: int = 20             # minimum steps regardless of convergence
    n_max: int = 100            # maximum steps (safety cap)
    inject_window: int = 8      # steps during which sim_graft is applied
    delta_threshold: float = 1e-4  # convergence: ||x_new - x_old|| < this
    convergence_window: int = 5    # consecutive steps below threshold needed
    max_norm_assert: float = 64.0  # Lyapunov guard: warn if norm exceeds this
    record_trajectory: bool = True


# ------------------------------------------------------------------ result


@dataclass
class LoopResult:
    graph: ThoughtGraph
    final_state: torch.Tensor          # [state_dim]
    trajectory: List[torch.Tensor]     # [state_dim] per step (may be empty if not recorded)
    n_steps_taken: int
    converged: bool
    elapsed_ms: float
    trace_id: str


# ------------------------------------------------------------------ loop


class CognitionLoop:
    """
    Drives the OSC engine for N steps, then delegates to GraphExtractor
    to build a ThoughtGraph from the resulting state trajectory.

    The engine is injected (not owned) so the same engine instance is reused
    across requests — matching the existing osc_chat.py pattern.
    """

    def __init__(
        self,
        engine,                       # Core.OSC.dynamics.OperatorSplitEngine
        cfg: Optional[LoopConfig] = None,
        extractor: Optional[GraphExtractor] = None,
        predictor=None,               # Optional NextStatePredictor — None = Phase A/B behavior
    ):
        self._engine = engine
        self.cfg = cfg or LoopConfig()

        # Extractor: trained > rule-based
        if extractor is not None:
            self._extractor = extractor
        elif TrainedGraphExtractor.is_available():
            self._extractor = TrainedGraphExtractor()
        else:
            self._extractor = GraphExtractor()

        # Next-State Predictor (Sprint C6): load from weights if available and not injected
        if predictor is not None:
            self._predictor = predictor
        else:
            self._predictor = _load_predictor_if_available(engine)

    # ---------------------------------------------------------------- run

    def run(
        self,
        x_init: torch.Tensor,
        bundle: ContextBundle,
        *,
        trace_id: Optional[str] = None,
        n_override: Optional[int] = None,
    ) -> LoopResult:
        """
        Run the cognitive loop.

        Args:
            x_init:     Initial OSC state [state_dim]. Typically loaded from SIM.
            bundle:     Context evidence bundle (request + memories + WKS).
            trace_id:   Optional trace identifier for telemetry.
            n_override: If set, overrides n_max from config (useful for testing).

        Returns:
            LoopResult with graph, final state, trajectory, and metadata.
        """
        trace_id = trace_id or str(uuid.uuid4())
        t_start = time.monotonic()
        cfg = self.cfg
        n_max = n_override if n_override is not None else cfg.n_max

        # Build the context injection tensor once (deterministic for this bundle)
        u_t = build_context_tensor(bundle)
        u_t = u_t.to(dtype=x_init.dtype, device=x_init.device)

        x = x_init.clone()
        if x.dim() == 2:
            x = x.squeeze(0)  # keep [state_dim] for single-request path

        trajectory: List[torch.Tensor] = []
        consecutive_below = 0
        converged = False

        for step_idx in range(n_max):
            # Inject context during inject_window, free-evolve after
            graft = u_t if step_idx < cfg.inject_window else None

            x_prev = x
            with torch.no_grad():
                x = self._engine.step(
                    x,
                    trace_id=trace_id,
                    sim_graft=graft,
                    telemetry_tag=f"cloop_{step_idx}",
                )

            # OperatorSplitEngine.step returns [1, d] when input is [d]
            if x.dim() == 2:
                x = x.squeeze(0)

            # Next-State Predictor guidance (Sprint C6)
            # Applied AFTER OSC step so Lyapunov guarantee is preserved.
            # δx is small and re-projected onto the norm ball.
            if self._predictor is not None:
                try:
                    step_norm = step_idx / max(n_max - 1, 1)
                    with torch.no_grad():
                        dx = self._predictor(x, u_t, step_norm)
                    x = x + dx
                    x = self._engine.project(x.unsqueeze(0)).squeeze(0)
                except Exception as _pred_exc:
                    logger.debug("predictor guidance skipped: %s", _pred_exc)

            if cfg.record_trajectory:
                trajectory.append(x.detach().clone())

            # Lyapunov guard
            norm = float(x.norm())
            if norm > cfg.max_norm_assert:
                logger.warning(
                    "CognitionLoop[%s] step %d: ||x||=%.2f exceeds guard %.2f",
                    trace_id, step_idx, norm, cfg.max_norm_assert,
                )

            # Convergence check (only after n_min steps)
            if step_idx >= cfg.n_min - 1:
                delta = float((x - x_prev).norm())
                if delta < cfg.delta_threshold:
                    consecutive_below += 1
                else:
                    consecutive_below = 0

                if consecutive_below >= cfg.convergence_window:
                    converged = True
                    logger.debug(
                        "CognitionLoop[%s] converged at step %d (delta=%.2e)",
                        trace_id, step_idx, delta,
                    )
                    break

        n_steps = len(trajectory) if trajectory else (
            min(n_max, cfg.n_min)
        )

        # Build ThoughtGraph from trajectory + evidence
        graph = self._extractor.extract(
            trajectory=trajectory,
            bundle=bundle,
            trace_id=trace_id,
            final_state=x,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        return LoopResult(
            graph=graph,
            final_state=x.detach(),
            trajectory=trajectory if cfg.record_trajectory else [],
            n_steps_taken=n_steps,
            converged=converged,
            elapsed_ms=elapsed_ms,
            trace_id=trace_id,
        )

    # ---------------------------------------------------------------- helpers

    def warm_up(self, state_dim: int = 1024, n_steps: int = 5) -> None:
        """
        Run a few no-op steps to warm JIT / CUDA kernels before first request.
        Safe to call at service startup.
        """
        x = torch.zeros(state_dim, dtype=torch.float32)
        bundle = ContextBundle(request_text="warmup", state_dim=state_dim)
        self.run(
            x, bundle,
            trace_id="warmup",
            n_override=n_steps,
        )
        logger.info("CognitionLoop warm-up done (%d steps)", n_steps)


# ---------------------------------------------------------------------------
# Auto-load helper — Sprint C6
# ---------------------------------------------------------------------------

_PREDICTOR_WEIGHTS_DIR = Path(__file__).parents[2] / "Core" / "OSC" / "predictor_weights"


def _load_predictor_if_available(engine):
    """
    Load NextStatePredictor from Core/OSC/predictor_weights/ if weights exist.
    Returns None (Phase A/B behavior) if weights are missing or load fails.
    engine is used to set max_guidance_abs from engine.params.max_norm.
    """
    d = _PREDICTOR_WEIGHTS_DIR
    cfg_path = d / "predictor_config.json"
    wts_path = d / "predictor_weights.pt"
    if not cfg_path.exists() or not wts_path.exists():
        return None

    try:
        import sys
        _ROOT = Path(__file__).parents[2]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from Runtime.Models.predictor_training.model import (
            NextStatePredictorConfig,
            NextStatePredictor,
        )
        cfg   = NextStatePredictorConfig.load(cfg_path)
        model = NextStatePredictor(cfg)
        model.load_state_dict(
            torch.load(str(wts_path), map_location="cpu"),
            strict=True,
        )
        model.eval()

        # Set magnitude cap from engine params when available
        try:
            max_norm = float(getattr(engine.params, "max_norm", 16.0))
            model.set_max_guidance(cfg.max_guidance * max_norm)
        except Exception:
            pass

        # Move to same device as engine
        try:
            device = engine.device
            model  = model.to(device)
        except Exception:
            pass

        logger.info("NextStatePredictor loaded: %d params", model.param_count())
        return model

    except Exception as exc:
        logger.warning("NextStatePredictor: failed to load (%s), running without guidance", exc)
        return None
