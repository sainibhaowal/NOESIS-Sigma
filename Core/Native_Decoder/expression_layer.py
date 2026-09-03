"""Stateful non-transformer expression layer for ThoughtGraph rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from Core.Cognition.thought_graph import NodeType, ThoughtGraph


@dataclass
class ExpressionState:
    trace_id: str
    mode: str
    summary: Dict[str, Any]
    output: str


class StatefulExpressionLayer:
    """Deterministic, stateful, non-transformer mouth for ThoughtGraph output."""

    def __init__(self, history_size: int = 32) -> None:
        self._history: Dict[str, Deque[ExpressionState]] = {}
        self._history_size = history_size

    def render(
        self,
        graph: ThoughtGraph,
        mode: str = "chat",
        *,
        trace_id: Optional[str] = None,
        graph_summary: Optional[Dict[str, Any]] = None,
        trajectory_norms: Optional[List[float]] = None,
        runtime_telemetry: Optional[Dict[str, Any]] = None,
    ) -> str:
        summary = graph_summary or graph.summary()
        trace_key = trace_id or summary.get("trace_id") or graph.trace_id or "anonymous"

        intent = self._first_content(graph, NodeType.INTENT)
        output = self._first_content(graph, NodeType.OUTPUT)
        facts = self._collect(graph, NodeType.FACT)
        plans = self._collect(graph, NodeType.PLAN)
        reasoning = self._collect(graph, NodeType.REASONING)
        uncertain = self._collect(graph, NodeType.UNCERTAIN)

        prior = self._last_output(trace_key)
        parts: list[str] = []

        if mode == "plan":
            if intent:
                parts.append(f"Goal: {intent}")
            if plans:
                parts.extend(f"{i + 1}. {step}" for i, step in enumerate(plans))
            elif reasoning:
                parts.extend(f"{i + 1}. {step}" for i, step in enumerate(reasoning))
            if output:
                parts.append(f"Result: {output}")
        elif mode == "analysis":
            if intent:
                parts.append(f"Query: {intent}")
            if facts:
                parts.append("Evidence:")
                parts.extend(f"- {fact}" for fact in facts[:6])
            if reasoning:
                parts.append("Reasoning:")
                parts.extend(f"- {step}" for step in reasoning[:4])
            if output:
                parts.append(f"Conclusion: {output}")
        elif mode == "code":
            if output:
                body = output
            elif reasoning:
                body = "\n".join(reasoning[:4])
            else:
                body = intent or ""
            parts.append("```")
            parts.append(body)
            parts.append("```")
        else:
            if output:
                parts.append(output)
            elif intent:
                parts.append(intent)
            if facts:
                parts.append("Facts: " + "; ".join(facts[:4]))
            if reasoning:
                parts.append("Reasoning: " + " | ".join(reasoning[:3]))

        if uncertain:
            parts.append(f"Uncertain: {len(uncertain)}")

        if trajectory_norms:
            parts.append(f"Trajectory steps: {len(trajectory_norms)}")

        if runtime_telemetry:
            mode_hint = runtime_telemetry.get("mode") or runtime_telemetry.get("lane")
            if mode_hint:
                parts.append(f"Telemetry: {mode_hint}")

        text = "\n".join(part for part in parts if part).strip()
        if not text:
            text = prior or "(no output)"

        self._remember(trace_key, mode, summary, text)
        return text

    def _collect(self, graph: ThoughtGraph, node_type: NodeType) -> list[str]:
        return [node.content.strip() for node in graph.get_nodes_by_type(node_type) if node.content.strip()]

    def _first_content(self, graph: ThoughtGraph, node_type: NodeType) -> str:
        items = self._collect(graph, node_type)
        return items[0] if items else ""

    def _last_output(self, trace_key: str) -> str:
        bucket = self._history.get(trace_key)
        if not bucket:
            return ""
        return bucket[-1].output

    def _remember(self, trace_key: str, mode: str, summary: Dict[str, Any], output: str) -> None:
        bucket = self._history.setdefault(trace_key, deque(maxlen=self._history_size))
        bucket.append(
            ExpressionState(
                trace_id=trace_key,
                mode=mode,
                summary=summary,
                output=output,
            )
        )
