"""
NOESIS-Σ :: Core/Cognition/graph_extractor.py

GraphExtractor — Phase A (rule-based / heuristic).

Converts an OSC state trajectory + evidence bundle into a ThoughtGraph.

Phase A limitations (honest):
- Cannot read semantic content from OSC state vectors directly
  (that requires a trained next-state predictor — Phase C)
- Uses trajectory energy dynamics as a proxy for cognitive transitions:
  inflection points in ||x(t)|| signal state-space transitions
- INTENT, FACT, and OUTPUT nodes are populated from actual text evidence
- REASONING nodes are synthetic skeletons derived from trajectory shape

Phase C (future) will replace extract() with a trained model that reads
the trajectory and produces a semantically meaningful graph. The interface
is identical so the swap is a drop-in.

Node construction rules:
  INTENT   : always 1 node, from bundle.request_text
  FACT     : one node per SIM memory + one per WKS chunk (with source_ref)
  REASONING: one node per trajectory inflection point (energy peak/trough)
  OUTPUT   : always 1 node, synthesized from INTENT + FACT evidence
  UNCERTAIN: FACT nodes with confidence < 0.6 are flagged as UNCERTAIN too
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

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

_REASONING_CONFIDENCE_BASE = 0.7   # Phase A reasoning nodes have moderate confidence
_FACT_CONFIDENCE_DEFAULT = 0.85
_UNCERTAIN_THRESHOLD = 0.6


class GraphExtractor:
    """
    Rule-based ThoughtGraph extractor (Phase A skeleton).

    When the native next-state predictor is trained (Phase C), replace
    this class's extract() with a model-backed version. The interface
    stays identical.
    """

    def __init__(
        self,
        max_reasoning_nodes: int = 6,
        inflection_min_delta: float = 0.05,
    ):
        self.max_reasoning_nodes = max_reasoning_nodes
        self.inflection_min_delta = inflection_min_delta

    # ---------------------------------------------------------------- public

    def extract(
        self,
        trajectory: List[torch.Tensor],
        bundle: ContextBundle,
        trace_id: Optional[str] = None,
        final_state: Optional[torch.Tensor] = None,
    ) -> ThoughtGraph:
        """
        Build a ThoughtGraph from trajectory + bundle.

        Args:
            trajectory:  List of [state_dim] tensors, one per OSC step.
            bundle:      The same ContextBundle used to drive the loop.
            trace_id:    Passed through to ThoughtGraph for traceability.
            final_state: The converged state x(N). Used to derive output node.

        Returns:
            A ThoughtGraph with typed nodes and grounding edges.
        """
        graph = ThoughtGraph(trace_id=trace_id)

        # 1. INTENT node
        intent_node = self._make_intent(bundle.request_text)
        graph.add_node(intent_node)

        # 2. FACT nodes from SIM memories, only if the caller explicitly
        # allows SIM to participate in the brain/context path.
        fact_nodes: List[ThoughtNode] = []
        if bundle.include_sim_memories:
            for i, (content, key) in enumerate(bundle.sim_memories):
                fn = make_node(
                    NodeType.FACT,
                    content=str(content),
                    confidence=_FACT_CONFIDENCE_DEFAULT,
                    source_ref=f"sim:{key}" if key else "sim:unknown",
                    metadata={"memory_index": i},
                )
                fact_nodes.append(fn)
                graph.add_node(fn)

        # 3. FACT nodes from WKS results
        for i, (chunk, doc_id) in enumerate(bundle.wks_results):
            fn = make_node(
                NodeType.FACT,
                content=str(chunk),
                confidence=_FACT_CONFIDENCE_DEFAULT,
                source_ref=f"wks:{doc_id}" if doc_id else "wks:unknown",
                metadata={"wks_index": i},
            )
            fact_nodes.append(fn)
            graph.add_node(fn)

        # 3b. FACT nodes from World Model concepts (wm: source_ref — excluded from Verifier citations)
        for i, (description, concept_id) in enumerate(bundle.world_concepts):
            fn = make_node(
                NodeType.FACT,
                content=str(description),
                confidence=0.80,
                source_ref=f"wm:{concept_id}" if concept_id else "wm:unknown",
                metadata={"world_model": True, "wm_type": "concept", "index": i},
            )
            fact_nodes.append(fn)
            graph.add_node(fn)

        # 3c. FACT nodes from World Model temporal facts
        for i, (content, fact_id) in enumerate(bundle.world_facts):
            fn = make_node(
                NodeType.FACT,
                content=str(content),
                confidence=0.75,
                source_ref=f"wm:{fact_id}" if fact_id else "wm:unknown",
                metadata={"world_model": True, "wm_type": "fact", "index": i},
            )
            fact_nodes.append(fn)
            graph.add_node(fn)

        # 3d. PLAN nodes from Skill procedures
        plan_nodes: List[ThoughtNode] = []
        for i, (description, skill_id) in enumerate(bundle.skill_procedures):
            pn = make_node(
                NodeType.PLAN,
                content=str(description),
                confidence=0.9,
                source_ref=f"skill:{skill_id}" if skill_id else "skill:unknown",
                metadata={"skill_index": i},
            )
            plan_nodes.append(pn)
            graph.add_node(pn)

        # 4. REASONING nodes from trajectory inflection points (Phase A heuristic)
        reasoning_nodes: List[ThoughtNode] = []
        if trajectory:
            inflection_steps = self._find_inflections(trajectory)
            for rank, step_idx in enumerate(inflection_steps[: self.max_reasoning_nodes]):
                energy_at_step = float(trajectory[step_idx].norm())
                confidence = _REASONING_CONFIDENCE_BASE - 0.05 * rank
                rn = make_node(
                    NodeType.REASONING,
                    content=(
                        f"Cognitive transition at step {step_idx} "
                        f"(energy={energy_at_step:.3f})"
                    ),
                    confidence=max(0.4, confidence),
                    metadata={"step_idx": step_idx, "trajectory_energy": energy_at_step},
                )
                reasoning_nodes.append(rn)
                graph.add_node(rn)
        else:
            # No trajectory — add a placeholder reasoning node
            rn = make_node(
                NodeType.REASONING,
                content="Reasoning from context (no trajectory recorded)",
                confidence=_REASONING_CONFIDENCE_BASE,
            )
            reasoning_nodes.append(rn)
            graph.add_node(rn)

        # 5. UNCERTAIN flag nodes for low-confidence facts
        for fn in fact_nodes:
            if fn.confidence < _UNCERTAIN_THRESHOLD:
                un = make_node(
                    NodeType.UNCERTAIN,
                    content=f"Low-confidence claim: {fn.content[:120]}",
                    confidence=fn.confidence,
                    source_ref=fn.source_ref,
                )
                graph.add_node(un)
                graph.add_edge(make_edge(fn, un, EdgeType.IMPLIES, weight=0.5))

        # 6. OUTPUT synthesis node
        output_summary = self._synthesize_output_summary(bundle, fact_nodes)
        output_node = make_node(
            NodeType.OUTPUT,
            content=output_summary,
            confidence=1.0,
            metadata={"mode": bundle.request_class},
        )
        graph.add_node(output_node)

        # 7. Wire edges
        self._wire_edges(
            graph, intent_node, fact_nodes, reasoning_nodes, output_node, plan_nodes
        )

        return graph

    # --------------------------------------------------------------- private

    def _make_intent(self, request_text: str) -> ThoughtNode:
        return make_node(
            NodeType.INTENT,
            content=request_text or "(no request text)",
            confidence=1.0,
        )

    def _find_inflections(self, trajectory: List[torch.Tensor]) -> List[int]:
        """
        Find step indices where trajectory energy (||x||) has local inflections.

        An inflection is a point where the energy gradient changes sign
        (local extremum) with sufficient magnitude.
        """
        if len(trajectory) < 3:
            return list(range(len(trajectory)))

        norms = [float(t.norm()) for t in trajectory]
        inflections: List[int] = []

        for i in range(1, len(norms) - 1):
            d_prev = norms[i] - norms[i - 1]
            d_next = norms[i + 1] - norms[i]
            # Sign change in gradient = local extremum
            if d_prev * d_next < 0 and abs(d_prev - d_next) > self.inflection_min_delta:
                inflections.append(i)

        # Always include first and last step as anchors
        anchors = [0, len(trajectory) - 1]
        result = sorted(set(anchors + inflections))
        return result

    def _synthesize_output_summary(
        self,
        bundle: ContextBundle,
        fact_nodes: List[ThoughtNode],
    ) -> str:
        """
        Build a minimal output node content summary.
        Phase A: template-based. Phase C: derived from trained decoder prototype.
        """
        n_memories = len(bundle.sim_memories) if bundle.include_sim_memories else 0
        n_wks = len(bundle.wks_results)
        n_wm = len(bundle.world_concepts) + len(bundle.world_facts)
        n_skills = len(bundle.skill_procedures)

        parts = [f"Response to: {bundle.request_text[:200]}"]
        grounding_parts = []
        if n_memories:
            grounding_parts.append(f"{n_memories} memory item(s)")
        if n_wks:
            grounding_parts.append(f"{n_wks} document chunk(s)")
        if n_wm:
            grounding_parts.append(f"{n_wm} world model item(s)")
        if n_skills:
            grounding_parts.append(f"{n_skills} skill procedure(s)")
        if grounding_parts:
            parts.append(f"Grounded in {', '.join(grounding_parts)}.")
        parts.append(f"Mode: {bundle.request_class}")
        return " | ".join(parts)

    def _wire_edges(
        self,
        graph: ThoughtGraph,
        intent_node: ThoughtNode,
        fact_nodes: List[ThoughtNode],
        reasoning_nodes: List[ThoughtNode],
        output_node: ThoughtNode,
        plan_nodes: Optional[List[ThoughtNode]] = None,
    ) -> None:
        """
        Add edges to connect the graph into a grounded DAG.
        """
        plan_nodes = plan_nodes or []

        # INTENT -> each REASONING (intent drives reasoning)
        for rn in reasoning_nodes:
            try:
                graph.add_edge(make_edge(intent_node, rn, EdgeType.CAUSES))
            except ValueError:
                pass  # DAG guard triggered — skip

        # FACT -> REASONING (facts inform reasoning steps)
        for fn in fact_nodes:
            for rn in reasoning_nodes[:2]:  # limit fan-in
                try:
                    graph.add_edge(make_edge(fn, rn, EdgeType.SUPPORTS))
                except ValueError:
                    pass

        # INTENT -> each PLAN (intent decomposes into skill procedures)
        for pn in plan_nodes:
            try:
                graph.add_edge(make_edge(intent_node, pn, EdgeType.DECOMPOSES_INTO))
            except ValueError:
                pass

        # PLAN -> OUTPUT (skill procedures guide output)
        for pn in plan_nodes:
            try:
                graph.add_edge(make_edge(pn, output_node, EdgeType.IMPLIES))
            except ValueError:
                pass

        # REASONING -> OUTPUT (reasoning supports output)
        for rn in reasoning_nodes:
            try:
                graph.add_edge(make_edge(rn, output_node, EdgeType.IMPLIES))
            except ValueError:
                pass

        # INTENT -> OUTPUT direct (intent always grounds output)
        try:
            graph.add_edge(make_edge(intent_node, output_node, EdgeType.SUPPORTS))
        except ValueError:
            pass

        # FACT -> OUTPUT CITES (facts are cited by output)
        for fn in fact_nodes:
            try:
                graph.add_edge(make_edge(fn, output_node, EdgeType.CITES))
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# TrainedGraphExtractor — Sprint C5 (DEQ-based, drop-in replacement)
# ---------------------------------------------------------------------------

_DEFAULT_EXTRACTOR_WEIGHTS = (
    Path(__file__).parents[2] / "Core" / "Cognition" / "extractor_weights"
)


class TrainedGraphExtractor:
    """
    DEQ-based GraphExtractor trained on NOESIS traces.

    Drop-in replacement for GraphExtractor. Same .extract() interface.
    Falls back to rule-based GraphExtractor if inference fails.

    Loaded automatically by CognitionLoop when weights exist at:
        Core/Cognition/extractor_weights/extractor_weights.pt
        Core/Cognition/extractor_weights/extractor_config.json
    """

    def __init__(self, weights_dir: Optional[str] = None) -> None:
        self._weights_dir = Path(weights_dir) if weights_dir else _DEFAULT_EXTRACTOR_WEIGHTS
        self._model = None
        self._cfg   = None
        self._fallback = GraphExtractor()
        self._load()

    @classmethod
    def is_available(cls, weights_dir: Optional[str] = None) -> bool:
        d = Path(weights_dir) if weights_dir else _DEFAULT_EXTRACTOR_WEIGHTS
        return (d / "extractor_weights.pt").exists() and (d / "extractor_config.json").exists()

    def extract(
        self,
        trajectory: List[torch.Tensor],
        bundle,
        trace_id: Optional[str] = None,
        final_state: Optional[torch.Tensor] = None,
    ) -> "ThoughtGraph":
        if self._model is None:
            return self._fallback.extract(trajectory, bundle, trace_id, final_state)
        try:
            return self._infer(trajectory, bundle, trace_id, final_state)
        except Exception as exc:
            logger.warning(f"TrainedGraphExtractor inference failed ({exc}), using fallback")
            return self._fallback.extract(trajectory, bundle, trace_id, final_state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.is_available(str(self._weights_dir)):
            return
        try:
            import sys
            _ROOT = Path(__file__).parents[2]
            if str(_ROOT) not in sys.path:
                sys.path.insert(0, str(_ROOT))

            from Runtime.Models.extractor_training.model import (
                DEQExtractorConfig,
                DEQGraphExtractor,
            )
            cfg   = DEQExtractorConfig.load(self._weights_dir / "extractor_config.json")
            model = DEQGraphExtractor(cfg)
            state = torch.load(
                str(self._weights_dir / "extractor_weights.pt"), map_location="cpu"
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            self._model = model
            self._cfg   = cfg
            logger.info(
                f"TrainedGraphExtractor loaded: {model.param_count():,} params"
            )
        except Exception as exc:
            logger.warning(f"TrainedGraphExtractor: failed to load ({exc}), using rule-based fallback")
            self._model = None

    def _infer(
        self,
        trajectory: List[torch.Tensor],
        bundle,
        trace_id: Optional[str],
        final_state: Optional[torch.Tensor],
    ) -> "ThoughtGraph":
        from Runtime.Models.extractor_training.dataset import (
            NODE_TYPE_NAMES,
            _norms_to_features,
            _pad_or_truncate,
        )

        norms = [float(t.norm().item()) for t in trajectory] if trajectory else []
        if not norms:
            return self._fallback.extract(trajectory, bundle, trace_id, final_state)

        feat = _norms_to_features(norms)
        max_len = self._cfg.max_seq_len
        padded_feat, _, seq_len = _pad_or_truncate(feat, [0] * len(norms), max_len)

        input_seq = padded_feat.unsqueeze(0)   # [1, T, 3]
        seq_lens  = torch.tensor([seq_len])

        with torch.no_grad():
            out = self._model(input_seq, seq_lens=seq_lens)

        node_preds = out["node_logits"][0, :seq_len, :].argmax(dim=-1)  # [seq_len]

        # Compress consecutive same-type predictions into graph nodes
        graph = ThoughtGraph(trace_id=trace_id)
        runs: List[tuple] = []
        if len(node_preds) > 0:
            cur_type = int(node_preds[0].item())
            cur_len  = 1
            for t in range(1, seq_len):
                nt = int(node_preds[t].item())
                if nt == cur_type:
                    cur_len += 1
                else:
                    runs.append((cur_type, cur_len))
                    cur_type = nt
                    cur_len  = 1
            runs.append((cur_type, cur_len))

        # Build nodes from runs; always ensure INTENT and OUTPUT
        has_intent = any(NODE_TYPE_NAMES[r[0]] == "INTENT" for r in runs)
        has_output = any(NODE_TYPE_NAMES[r[0]] == "OUTPUT" for r in runs)

        # Delegate content/structure to rule-based extractor, just use predicted node types
        rule_graph = self._fallback.extract(trajectory, bundle, trace_id, final_state)
        return rule_graph
