"""
Deterministic ThoughtGraph → text renderer. No model, no VRAM.
Every word in the output comes directly from a graph node.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from Core.Cognition.thought_graph import NodeType, ThoughtGraph

if TYPE_CHECKING:
    pass


def _section(header: str, lines: list[str]) -> str:
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"**{header}**\n{body}"


def render_chat(graph: ThoughtGraph) -> str:
    intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    reason_nodes = graph.get_nodes_by_type(NodeType.REASONING)
    output_nodes = graph.get_nodes_by_type(NodeType.OUTPUT)

    if output_nodes:
        return output_nodes[0].content.strip()

    parts: list[str] = []
    if intent_nodes:
        parts.append(intent_nodes[0].content.strip())
    if fact_nodes:
        fact_lines = [f"- {n.content.strip()}" for n in fact_nodes if n.content]
        if fact_lines:
            parts.append(_section("Facts", fact_lines))
    if reason_nodes:
        parts.append(reason_nodes[-1].content.strip())
    return "\n\n".join(p for p in parts if p) or "(no output)"


def render_code(graph: ThoughtGraph) -> str:
    output_nodes = graph.get_nodes_by_type(NodeType.OUTPUT)
    if output_nodes:
        content = output_nodes[0].content.strip()
        if content.startswith("```"):
            return content
        return f"```\n{content}\n```"

    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    reason_nodes = graph.get_nodes_by_type(NodeType.REASONING)
    lines: list[str] = []
    for n in fact_nodes:
        lines.append(f"# {n.content.strip()}")
    for n in reason_nodes:
        lines.append(n.content.strip())
    body = "\n".join(lines) or "# (no code generated)"
    return f"```\n{body}\n```"


def render_plan(graph: ThoughtGraph) -> str:
    intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
    output_nodes = graph.get_nodes_by_type(NodeType.OUTPUT)
    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    reason_nodes = graph.get_nodes_by_type(NodeType.REASONING)

    parts: list[str] = []
    if intent_nodes:
        parts.append(f"**Goal:** {intent_nodes[0].content.strip()}")

    steps: list[str] = []
    for i, n in enumerate(output_nodes or reason_nodes, 1):
        steps.append(f"{i}. {n.content.strip()}")
    if steps:
        parts.append("\n".join(steps))
    elif fact_nodes:
        for i, n in enumerate(fact_nodes, 1):
            parts.append(f"{i}. {n.content.strip()}")

    return "\n\n".join(p for p in parts if p) or "(no plan generated)"


def render_analysis(graph: ThoughtGraph) -> str:
    intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
    fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
    reason_nodes = graph.get_nodes_by_type(NodeType.REASONING)
    output_nodes = graph.get_nodes_by_type(NodeType.OUTPUT)

    parts: list[str] = []
    if intent_nodes:
        parts.append(f"**Query:** {intent_nodes[0].content.strip()}")
    if fact_nodes:
        fact_lines = [f"- {n.content.strip()}" for n in fact_nodes if n.content]
        if fact_lines:
            parts.append(_section("Evidence", fact_lines))
    if reason_nodes:
        reason_lines = [n.content.strip() for n in reason_nodes if n.content]
        if reason_lines:
            parts.append(_section("Reasoning", reason_lines))
    if output_nodes:
        parts.append(_section("Conclusion", [output_nodes[0].content.strip()]))

    return "\n\n".join(p for p in parts if p) or "(no analysis generated)"


_RENDERERS = {
    "chat": render_chat,
    "code": render_code,
    "plan": render_plan,
    "analysis": render_analysis,
}


def render(graph: ThoughtGraph, mode: str) -> str:
    fn = _RENDERERS.get(mode, render_chat)
    return fn(graph)
