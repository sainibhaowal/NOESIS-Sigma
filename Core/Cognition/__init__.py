"""
NOESIS-Σ :: Core/Cognition

Sprint A1 — Cognition skeleton.

Public surface:
    ThoughtGraph, ThoughtNode, ThoughtEdge, NodeType, EdgeType
    ContextBundle, build_context_tensor, infer_mode
    CognitionLoop, LoopConfig, LoopResult
    GraphExtractor
    make_node, make_edge
"""

from Core.Cognition.thought_graph import (
    EdgeType,
    NodeType,
    ThoughtEdge,
    ThoughtGraph,
    ThoughtNode,
    make_edge,
    make_node,
)
from Core.Cognition.fusion import (
    ContextBundle,
    build_context_tensor,
    infer_mode,
)
from Core.Cognition.cognitive_loop import (
    CognitionLoop,
    LoopConfig,
    LoopResult,
)
from Core.Cognition.graph_extractor import GraphExtractor, TrainedGraphExtractor

__all__ = [
    # thought graph schema
    "ThoughtGraph",
    "ThoughtNode",
    "ThoughtEdge",
    "NodeType",
    "EdgeType",
    "make_node",
    "make_edge",
    # fusion
    "ContextBundle",
    "build_context_tensor",
    "infer_mode",
    # loop
    "CognitionLoop",
    "LoopConfig",
    "LoopResult",
    # extractor
    "GraphExtractor",
    "TrainedGraphExtractor",
]
