from __future__ import annotations
import re
from typing import Tuple, Set
from Core.Cognition.thought_graph import ThoughtGraph, NodeType, EdgeType

class ThoughtGraphVerifier:
    """Hard logical verification of converged ThoughtGraph nodes and output text.
    
    Checks:
    1. FACT consistency: no two nodes with a CONTRADICTS edge between them.
    2. OUTPUT grounding: every OUTPUT node has a path from a FACT/INTENT/REASONING node.
    3. Confidence gating: block if any ancestor of an OUTPUT node has confidence < threshold.
    4. Semantic Grounding verification: generated text content embeddings must be semantically grounded to the graph node centroids.
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        # Basic functional words and punctuation to ignore during grounding checks
        self.common_words: Set[str] = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "once",
            "here", "there", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "should", "now", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their", "this", "that", "these", "those",
            "is_output_grounded"
        }
        self._nli_model = None

    def _get_ancestors(self, graph: ThoughtGraph, node_id: str) -> Set[str]:
        """Traverse the graph backwards to find all ancestor node IDs."""
        ancestors = set()
        queue = [node_id]
        while queue:
            curr = queue.pop(0)
            # Find incoming edges where edge.to_id == curr
            incoming = [e for e in getattr(graph, "_edges", []) if e.to_id == curr]
            for edge in incoming:
                if edge.from_id not in ancestors:
                    ancestors.add(edge.from_id)
                    queue.append(edge.from_id)
        return ancestors

    def verify_graph(self, graph: ThoughtGraph) -> tuple[bool, str]:
        """Check the graph structural consistency and constraints.
        
        Returns:
            (allowed, reason)
        """
        # 1. FACT consistency check: no CONTRADICTS edges
        for edge in getattr(graph, "_edges", []):
            if edge.edge_type == EdgeType.CONTRADICTS:
                return False, f"Contradiction detected: edge {edge.from_id} CONTRADICTS {edge.to_id}"

        # 2. OUTPUT grounding check
        if not graph.is_output_grounded():
            return False, "Output grounding check failed: not all OUTPUT nodes are grounded"

        # 3. Confidence gating
        for out_node in graph.get_nodes_by_type(NodeType.OUTPUT):
            ancestors = self._get_ancestors(graph, out_node.node_id)
            for anc_id in ancestors:
                anc_node = graph.get_node(anc_id)
                if anc_node and anc_node.confidence < self.confidence_threshold:
                    return False, (
                        f"OUTPUT depends on low-confidence node {anc_id} "
                        f"(confidence={anc_node.confidence} < {self.confidence_threshold})"
                    )

        return True, "Success"

    def gate_output(self, graph: ThoughtGraph, text: str) -> tuple[bool, str]:
        """Verifies both the graph structure and semantic grounding of the generated output text.
        
        Returns:
            (allowed, reason)
        """
        # First verify the graph itself
        allowed, reason = self.verify_graph(graph)
        if not allowed:
            return False, reason

        # Semantic Grounding Verification
        if not text or len(text.strip()) < 3:
            return True, "Success"  # Let standard fallback handle empty/short outputs
            
        import torch
        import torch.nn.functional as F
        from External.WorldModel.concept_state_encoder import ConceptStateEncoder
        
        fact_nodes = graph.get_nodes_by_type(NodeType.FACT)
        if not fact_nodes:
            return True, "Success"
            
        facts = [node.content for node in fact_nodes if node.content]
        if not facts:
            return True, "Success"
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ConceptStateEncoder(state_dim=1024)
        
        # Calculate fact_embeds
        with torch.no_grad():
            fact_embeds = []
            for f in facts:
                # encode_query handles tensor creation
                embed = encoder.encode_query(f, device=device)
                if not isinstance(embed, torch.Tensor):
                    embed = torch.tensor(embed, dtype=torch.float32)
                embed = embed.to(device=device, dtype=torch.float32)
                fact_embeds.append(embed)
            
            fact_tensor = torch.stack(fact_embeds) # [N, 1024]
        
            # Check contradiction via pairwise NLI instead of naive cosine similarity
            # High cosine similarity could just mean they are about the same topic, 
            # but they could still be contradictory (e.g. "is X" vs "is not X").
            N = len(fact_tensor)
            if N > 1:
                if self._nli_model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                        # Using a lightweight NLI model to check entailment/contradiction
                        self._nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small', device=device)
                    except ImportError:
                        self._nli_model = "fallback"
                
                if self._nli_model != "fallback":
                    # nli-deberta-v3-small outputs: [Contradiction, Entailment, Neutral]
                    # We check all pairs of facts
                    pairs = []
                    for i in range(N):
                        for j in range(i+1, N):
                            pairs.append((facts[i], facts[j]))
                    
                    scores = self._nli_model.predict(pairs)
                    # scores shape: (num_pairs, 3)
                    # We look for high contradiction score (index 0)
                    for idx, score in enumerate(scores):
                        if score[0] > score[1] and score[0] > score[2] and score[0] > 1.0: # Strongly predicted as contradiction
                            f1, f2 = pairs[idx]
                            return False, f"Semantic Grounding Verification failed: NLI Contradiction detected between '{f1}' and '{f2}'"
                else:
                    # Fallback to tightened cosine similarity if NLI not available
                    norm_facts = F.normalize(fact_tensor, p=2, dim=1)
                    sim_matrix = torch.matmul(norm_facts, norm_facts.T)
                    
                    mask = ~torch.eye(N, dtype=torch.bool, device=device)
                    min_sim = torch.min(sim_matrix[mask]).item()
                    
                    # -0.3 is too loose. Since we rely on ConceptStateEncoder which uses contrastive learning 
                    # to push anomalies away, we tighten this to 0.2.
                    theta_contra = 0.2
                    if min_sim < theta_contra:
                        return False, f"Semantic Grounding Verification failed: Anomaly/Contradiction detected (similarity {min_sim:.4f} < {theta_contra})"
                    
            # Check grounding via cosine similarity between output embedding and fact_embeds centroid
            output_embed = encoder.encode_query(text, device=device)
            if not isinstance(output_embed, torch.Tensor):
                output_embed = torch.tensor(output_embed, dtype=torch.float32)
            output_embed = output_embed.to(device=device, dtype=torch.float32)
            
            centroid = torch.mean(fact_tensor, dim=0)
            
            sim_out_centroid = F.cosine_similarity(output_embed.unsqueeze(0), centroid.unsqueeze(0)).item()
            
            theta_ground = 0.6
            if sim_out_centroid < theta_ground:
                return False, f"Semantic Grounding Verification failed: Output is ungrounded (similarity {sim_out_centroid:.4f} < {theta_ground})"
                
        return True, "Success"
