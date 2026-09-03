"""
NOESIS-Σ :: Core/Cognition/thought_graph.py

ThoughtGraph schema — the structured output of the cognitive loop.

The ThoughtGraph is built BEFORE any token is generated. The Native Decoder
receives it and linearizes it into language. The decoder is blind to facts;
all knowledge must live as nodes in this graph.

Design rules:
- All nodes are typed (NodeType enum)
- All edges are typed (EdgeType enum)
- OUTPUT nodes must be grounded (reachable from FACT or INTENT via edges)
- Graph is DAG-enforced at add_edge time (cycle detection)
- Serializable to dict for transport, tracing, and receipts
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class NodeType(str, Enum):
    INTENT = "INTENT"           # what the user wants
    FACT = "FACT"               # grounded knowledge from SIM/WKS
    REASONING = "REASONING"     # internal inference step
    PLAN = "PLAN"               # action or multi-step plan node
    OUTPUT = "OUTPUT"           # what will be spoken/rendered
    UNCERTAIN = "UNCERTAIN"     # low-confidence claim, flagged for verifier
    PROCEDURE = "PROCEDURE"     # causal mechanism: how to do X (not what X is)
    IMPLEMENT = "IMPLEMENT"     # constructed output from procedure application


class EdgeType(str, Enum):
    CAUSES = "CAUSES"
    IMPLIES = "IMPLIES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PRECEDES = "PRECEDES"
    DECOMPOSES_INTO = "DECOMPOSES_INTO"
    CITES = "CITES"
    STEPS_THROUGH = "STEPS_THROUGH"   # procedure step order
    IMPLEMENTS = "IMPLEMENTS"          # output implements procedure


@dataclass
class ThoughtNode:
    node_id: str
    node_type: NodeType
    content: str
    confidence: float = 1.0         # [0.0, 1.0]
    source_ref: Optional[str] = None  # SIM key, WKS doc_id, or None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThoughtNode":
        return cls(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            content=d["content"],
            confidence=float(d.get("confidence", 1.0)),
            source_ref=d.get("source_ref"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ThoughtEdge:
    from_id: str
    to_id: str
    edge_type: EdgeType
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThoughtEdge":
        return cls(
            from_id=d["from_id"],
            to_id=d["to_id"],
            edge_type=EdgeType(d["edge_type"]),
            weight=float(d.get("weight", 1.0)),
        )


class ThoughtGraph:
    """
    Directed graph of ThoughtNodes connected by typed ThoughtEdges.

    Structural invariants:
    - No duplicate node_ids
    - No cycles (DAG enforced at add_edge)
    - OUTPUT nodes must have at least one incoming SUPPORTS/IMPLIES edge
      (checked by is_output_grounded)
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.graph_id: str = str(uuid.uuid4())
        self.trace_id: Optional[str] = trace_id
        self.created_at: float = time.time()
        self._nodes: Dict[str, ThoughtNode] = {}
        self._edges: List[ThoughtEdge] = []
        # adjacency for cycle detection: node_id -> set of reachable node_ids
        self._reachable: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------ nodes

    def add_node(self, node: ThoughtNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self._nodes[node.node_id] = node
        self._reachable[node.node_id] = set()

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[ThoughtNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # ------------------------------------------------------------------ edges

    def add_edge(self, edge: ThoughtEdge) -> None:
        if edge.from_id not in self._nodes:
            raise ValueError(f"Unknown from_id: {edge.from_id}")
        if edge.to_id not in self._nodes:
            raise ValueError(f"Unknown to_id: {edge.to_id}")
        # cycle guard: adding from->to would create cycle if to can already reach from
        if edge.from_id in self._reachable.get(edge.to_id, set()):
            raise ValueError(
                f"Adding edge {edge.from_id}->{edge.to_id} would create a cycle"
            )
        self._edges.append(edge)
        # update reachability: from can now reach to and everything to can reach
        self._reachable[edge.from_id].add(edge.to_id)
        self._reachable[edge.from_id].update(self._reachable[edge.to_id])
        # propagate back to nodes that can already reach from_id
        for nid, reachable in self._reachable.items():
            if edge.from_id in reachable:
                reachable.add(edge.to_id)
                reachable.update(self._reachable[edge.to_id])

    def get_edges(self) -> List[ThoughtEdge]:
        return list(self._edges)

    def get_incoming_edges(self, node_id: str) -> List[ThoughtEdge]:
        return [e for e in self._edges if e.to_id == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[ThoughtEdge]:
        return [e for e in self._edges if e.from_id == node_id]

    # ---------------------------------------------------------------- helpers

    def is_output_grounded(self) -> bool:
        """
        Every OUTPUT node must have at least one grounding edge
        (SUPPORTS or IMPLIES or CITES) from a FACT, REASONING, or INTENT node.
        """
        grounding_types = {EdgeType.SUPPORTS, EdgeType.IMPLIES, EdgeType.CITES}
        grounding_sources = {NodeType.FACT, NodeType.INTENT, NodeType.REASONING}
        for out_node in self.get_nodes_by_type(NodeType.OUTPUT):
            incoming = self.get_incoming_edges(out_node.node_id)
            grounded = any(
                e.edge_type in grounding_types
                and self._nodes[e.from_id].node_type in grounding_sources
                for e in incoming
            )
            if not grounded:
                return False
        return True

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def summary(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        wm_facts = 0
        sim_facts = 0
        wks_facts = 0
        skill_plans = 0
        for n in self._nodes.values():
            type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1
            ref = getattr(n, "source_ref", "") or ""
            if n.node_type.value == "FACT":
                if ref.startswith("wm:"):
                    wm_facts += 1
                elif ref.startswith("sim:"):
                    sim_facts += 1
                elif ref.startswith("wks:"):
                    wks_facts += 1
            elif n.node_type.value == "PLAN" and ref.startswith("skill:"):
                skill_plans += 1
        return {
            "graph_id": self.graph_id,
            "trace_id": self.trace_id,
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "node_type_counts": type_counts,
            "is_output_grounded": self.is_output_grounded(),
            "world_model_facts": wm_facts,
            "sim_facts": sim_facts,
            "wks_facts": wks_facts,
            "skill_plans": skill_plans,
        }

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThoughtGraph":
        g = cls(trace_id=d.get("trace_id"))
        g.graph_id = d["graph_id"]
        g.created_at = float(d.get("created_at", time.time()))
        for nd in d.get("nodes", []):
            g.add_node(ThoughtNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(ThoughtEdge.from_dict(ed))
        return g


# ------------------------------------------------------------------ factories

def make_node(
    node_type: NodeType,
    content: str,
    confidence: float = 1.0,
    source_ref: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ThoughtNode:
    return ThoughtNode(
        node_id=str(uuid.uuid4()),
        node_type=node_type,
        content=content,
        confidence=confidence,
        source_ref=source_ref,
        metadata=metadata or {},
    )


def make_edge(
    from_node: ThoughtNode,
    to_node: ThoughtNode,
    edge_type: EdgeType,
    weight: float = 1.0,
) -> ThoughtEdge:
    return ThoughtEdge(
        from_id=from_node.node_id,
        to_id=to_node.node_id,
        edge_type=edge_type,
        weight=weight,
    )
