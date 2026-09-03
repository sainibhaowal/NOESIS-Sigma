"""
Fixtures for acceptance suite tests.

Provides realistic sample CognitionTrace objects that can be compiled into SkillEpisodes.
"""

import json
from Runtime.Traces.trace_schema import CognitionTrace


def sample_verified_trace(
    trace_id: str = "trace_verified_001",
    tenant_id: str = "tenant_alpha",
    session_id: str = "session_001",
    request_text: str = "What is the capital of France?",
    response_text: str = "Paris is the capital of France.",
    verifier_score: float = 0.95,
) -> CognitionTrace:
    """Create a sample verified trace with high confidence."""
    return CognitionTrace(
        trace_id=trace_id,
        tenant_id=tenant_id,
        session_id=session_id,
        request_text=request_text,
        n_steps=42,
        trajectory_norms=[0.1, 0.2, 0.35, 0.5, 0.62, 0.71, 0.78, 0.83],
        final_state_norm=0.85,
        graph_dict={
            "nodes": [
                {"id": "n1", "type": "entity", "label": "Paris"},
                {"id": "n2", "type": "entity", "label": "France"},
                {"id": "n3", "type": "relation", "label": "capital_of"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "capital_of"},
            ],
        },
        graph_grounded=True,
        node_type_counts={"entity": 2, "relation": 1},
        response_text=response_text,
        decode_mode="native",
        elapsed_ms_total=1250,
        verifier_result="verified",
        verifier_score=verifier_score,
        request_class="factual_qa",
        sim_facts=3,
        wks_facts=1,
        world_model_facts=2,
        skill_plans=0,
    )


def sample_unverified_trace(
    trace_id: str = "trace_unverified_001",
    tenant_id: str = "tenant_alpha",
    session_id: str = "session_002",
    request_text: str = "What is 2+2?",
    response_text: str = "2+2 equals 4.",
    verifier_score: float = 0.3,
) -> CognitionTrace:
    """Create a sample unverified trace with low confidence."""
    return CognitionTrace(
        trace_id=trace_id,
        tenant_id=tenant_id,
        session_id=session_id,
        request_text=request_text,
        n_steps=15,
        trajectory_norms=[0.1, 0.15, 0.25, 0.35, 0.42],
        final_state_norm=0.45,
        graph_dict={
            "nodes": [
                {"id": "n1", "type": "operation", "label": "add"},
                {"id": "n2", "type": "value", "label": "2"},
                {"id": "n3", "type": "value", "label": "4"},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "operand"},
                {"source": "n1", "target": "n3", "type": "result"},
            ],
        },
        graph_grounded=False,
        node_type_counts={"operation": 1, "value": 2},
        response_text=response_text,
        decode_mode="expression",
        elapsed_ms_total=850,
        verifier_result="unverified",
        verifier_score=verifier_score,
        request_class="math_qa",
        sim_facts=0,
        wks_facts=0,
        world_model_facts=1,
        skill_plans=1,
    )


def sample_repair_trace(
    trace_id: str = "trace_repair_001",
    tenant_id: str = "tenant_beta",
    session_id: str = "session_003",
    request_text: str = "How do I fix a syntax error?",
    response_text: str = "Check your code for missing punctuation.",
) -> CognitionTrace:
    """Create a sample trace that was repaired."""
    return CognitionTrace(
        trace_id=trace_id,
        tenant_id=tenant_id,
        session_id=session_id,
        request_text=request_text,
        n_steps=28,
        trajectory_norms=[0.05, 0.12, 0.22, 0.35, 0.48, 0.58, 0.65],
        final_state_norm=0.72,
        graph_dict={
            "nodes": [
                {"id": "n1", "type": "problem", "label": "syntax_error"},
                {"id": "n2", "type": "solution", "label": "punctuation_check"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "resolves"},
            ],
        },
        graph_grounded=True,
        node_type_counts={"problem": 1, "solution": 1},
        response_text=response_text,
        decode_mode="native",
        elapsed_ms_total=2100,
        verifier_result="verified",
        verifier_score=0.88,
        request_class="coding_help",
        sim_facts=2,
        wks_facts=3,
        world_model_facts=1,
        skill_plans=2,
    )


def sample_tenant_alpha_trace_1() -> CognitionTrace:
    """Sample trace for tenant_alpha - for tenant isolation testing."""
    return CognitionTrace(
        trace_id="tenant_isolation_alpha_1",
        tenant_id="tenant_alpha",
        session_id="session_alpha_1",
        request_text="Question for alpha tenant",
        n_steps=20,
        trajectory_norms=[0.1 * i for i in range(1, 9)],
        final_state_norm=0.8,
        graph_dict={"nodes": [], "edges": []},
        graph_grounded=False,
        node_type_counts={},
        response_text="Response for alpha",
        decode_mode="native",
        elapsed_ms_total=1000,
        verifier_result="verified",
        verifier_score=0.9,
    )


def sample_tenant_beta_trace_1() -> CognitionTrace:
    """Sample trace for tenant_beta - for tenant isolation testing."""
    return CognitionTrace(
        trace_id="tenant_isolation_beta_1",
        tenant_id="tenant_beta",
        session_id="session_beta_1",
        request_text="Question for beta tenant",
        n_steps=25,
        trajectory_norms=[0.08 * i for i in range(1, 10)],
        final_state_norm=0.75,
        graph_dict={"nodes": [], "edges": []},
        graph_grounded=False,
        node_type_counts={},
        response_text="Response for beta",
        decode_mode="expression",
        elapsed_ms_total=1200,
        verifier_result="verified",
        verifier_score=0.85,
    )


def sample_code_task_trace() -> CognitionTrace:
    """Sample trace for code task testing."""
    return CognitionTrace(
        trace_id="code_task_001",
        tenant_id="tenant_alpha",
        session_id="session_code_1",
        request_text="Write a function to sum two numbers",
        n_steps=35,
        trajectory_norms=[0.05 * i for i in range(1, 12)],
        final_state_norm=0.55,
        graph_dict={
            "nodes": [
                {"id": "n1", "type": "task", "label": "sum_function"},
                {"id": "n2", "type": "implementation", "label": "code"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "implements"},
            ],
        },
        graph_grounded=True,
        node_type_counts={"task": 1, "implementation": 1},
        response_text="def sum_two(a, b):\n    return a + b",
        decode_mode="native",
        elapsed_ms_total=3200,
        verifier_result="verified",
        verifier_score=0.92,
        request_class="code_generation",
    )
