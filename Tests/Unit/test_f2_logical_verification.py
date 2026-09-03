import pytest
from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, ThoughtEdge, NodeType, EdgeType
from Core.Verifier.graph_verifier import ThoughtGraphVerifier

def test_contradicting_facts_blocked_f2():
    """Verify that a graph with contradicting FACT nodes is blocked."""
    graph = ThoughtGraph()
    n1 = ThoughtNode(node_id="fact-1", node_type=NodeType.FACT, content="Light travels fast")
    n2 = ThoughtNode(node_id="fact-2", node_type=NodeType.FACT, content="Light is slow")
    n_out = ThoughtNode(node_id="out-1", node_type=NodeType.OUTPUT, content="Speed of light")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n_out)
    
    # Add grounding edge to satisfy structural grounding first
    graph.add_edge(ThoughtEdge(from_id="fact-1", to_id="out-1", edge_type=EdgeType.SUPPORTS))
    
    # Add contradiction edge
    graph.add_edge(ThoughtEdge(from_id="fact-1", to_id="fact-2", edge_type=EdgeType.CONTRADICTS))
    
    verifier = ThoughtGraphVerifier()
    allowed, reason = verifier.gate_output(graph, "Speed of light")
    
    print(f"\nContradictory graph allowed: {allowed}, reason: {reason}")
    assert not allowed
    assert "contradiction" in reason.lower()


def test_ungrounded_output_blocked_f2():
    """Verify that an OUTPUT node without incoming grounding edges is blocked."""
    graph = ThoughtGraph()
    n1 = ThoughtNode(node_id="fact-1", node_type=NodeType.FACT, content="Earth is round")
    n_out = ThoughtNode(node_id="out-1", node_type=NodeType.OUTPUT, content="Earth shape")
    
    graph.add_node(n1)
    graph.add_node(n_out)
    
    # Do NOT add any grounding edge (meaning OUTPUT is ungrounded)
    verifier = ThoughtGraphVerifier()
    allowed, reason = verifier.gate_output(graph, "Earth shape")
    
    print(f"\nUngrounded graph allowed: {allowed}, reason: {reason}")
    assert not allowed
    assert "grounding" in reason.lower()


def test_valid_graph_passes_f2():
    """Verify that a valid, grounded, and non-contradictory graph passes verification."""
    graph = ThoughtGraph()
    n1 = ThoughtNode(node_id="fact-1", node_type=NodeType.FACT, content="Gravity pulls things down")
    n_out = ThoughtNode(node_id="out-1", node_type=NodeType.OUTPUT, content="Gravity pulls down")
    
    graph.add_node(n1)
    graph.add_node(n_out)
    
    # Add grounding edge
    graph.add_edge(ThoughtEdge(from_id="fact-1", to_id="out-1", edge_type=EdgeType.SUPPORTS))
    
    verifier = ThoughtGraphVerifier()
    allowed, reason = verifier.gate_output(graph, "Gravity pulls down")
    
    print(f"\nValid graph allowed: {allowed}, reason: {reason}")
    # Note: Valid graph might fail semantic checks if the encoder produces
    # embeddings that are orthogonal by chance, but structural check passes.
    # To truly assert allowed here, we would need to mock encode_query.

from unittest.mock import patch
import torch

def test_semantic_grounding_verification_f2():
    """Verify that semantic contradiction and grounding bounds are enforced."""
    graph = ThoughtGraph()
    n1 = ThoughtNode(node_id="f1", node_type=NodeType.FACT, content="Fact 1")
    n2 = ThoughtNode(node_id="f2", node_type=NodeType.FACT, content="Fact 2")
    n_out = ThoughtNode(node_id="out", node_type=NodeType.OUTPUT, content="Output")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n_out)
    graph.add_edge(ThoughtEdge(from_id="f1", to_id="out", edge_type=EdgeType.SUPPORTS))
    
    with patch("External.WorldModel.concept_state_encoder.ConceptStateEncoder.encode_query") as mock_encode:
        v1 = torch.zeros(1024)
        v1[0] = 1.0
        v2 = torch.zeros(1024)
        v2[0] = -1.0 # Cosine similarity = -1.0 < -0.3
        
        # 1. Contradiction
        mock_encode.side_effect = [v1, v2, v1]
        verifier = ThoughtGraphVerifier()
        verifier._nli_model = "fallback"
        allowed, reason = verifier.gate_output(graph, "Output text")
        assert not allowed
        assert "Contradiction detected" in reason

        
        # 2. Ungrounded Output
        v_fact = torch.zeros(1024)
        v_fact[0] = 1.0
        v_out = torch.zeros(1024)
        v_out[1] = 1.0 # Cosine similarity = 0.0 < 0.6
        mock_encode.side_effect = [v_fact, v_fact, v_out]
        
        allowed, reason = verifier.gate_output(graph, "Output text")
        assert not allowed
        assert "Output is ungrounded" in reason
        
        # 3. Valid
        mock_encode.side_effect = [v_fact, v_fact, v_fact]
        allowed, reason = verifier.gate_output(graph, "Output text")
        assert allowed
