import torch
import pytest
from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
from Core.Native_Decoder.sigma_native_emitter import SigmaEmitterConfig, SigmaNativeEmitter

class SimpleTokenizer:
    def __init__(self):
        # Extremely simple vocabulary mapping words to IDs
        self.vocab = {
            "[PAD]": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "jupiter": 3,
            "is": 4,
            "the": 5,
            "largest": 6,
            "planet": 7,
            "saturn": 8,
            "small": 9,
            " ": 10,
            ".": 11,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def token_to_id(self, text):
        return self.vocab.get(text.lower().strip(), 3)
        
    def encode(self, text):
        # Simple word-level splitter
        words = text.lower().strip().split()
        ids = []
        for w in words:
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                # split punctuation
                if w.endswith("."):
                    base = w[:-1]
                    if base in self.vocab:
                        ids.extend([self.vocab[base], self.vocab["."]])
        class Enc:
            def __init__(self, ids):
                self.ids = ids
        return Enc(ids)

    def decode(self, token_ids):
        return " ".join(self.inv_vocab.get(t, "[UNK]") for t in token_ids)


def test_constrained_decoder_masking_f3():
    """Verify that every generated content word traces back to a FACT/OUTPUT node in the graph."""
    tokenizer = SimpleTokenizer()
    
    # 1. Construct graph with Jupiter fact
    graph = ThoughtGraph()
    node = ThoughtNode(
        node_id="fact-1",
        node_type=NodeType.FACT,
        content="Jupiter is the largest planet"
    )
    graph.add_node(node)
    
    # 2. Build allowed tokens
    allowed_tokens = set()
    for n in graph._nodes.values():
        allowed_tokens.update(tokenizer.encode(n.content).ids)
        
    # Add grammar words
    grammar = ["is", "the", ".", " "]
    for g in grammar:
        allowed_tokens.update(tokenizer.encode(g).ids)
        
    # Always allow BOS/EOS/PAD
    allowed_tokens.update([0, 1, 2])
    
    # Verify that disallowed words (e.g., saturn=8, small=9) are not in allowed_tokens
    assert 8 not in allowed_tokens
    assert 9 not in allowed_tokens
    
    # 3. Setup emitter
    cfg = SigmaEmitterConfig(
        state_dim=64,
        d_model=32,
        vocab_size=len(tokenizer.vocab),
        prefix_len=4,
    )
    emitter = SigmaNativeEmitter(cfg)
    
    # Let's mock LM head projection weight so that the logits for "saturn" (8) are extremely high
    # to see if the constraint successfully masks it out.
    emitter.lm_head.weight.data.fill_(0.0)
    # Give 'saturn' a huge positive logit by default
    emitter.lm_head.weight.data[8, :] = 100.0
    # Give 'jupiter' a small logit
    emitter.lm_head.weight.data[3, :] = 1.0
    
    # Run generate with allowed_token_ids
    x_final = torch.randn(1, 64)
    text = emitter.generate(
        x_final,
        tokenizer,
        max_new_tokens=10,
        allowed_token_ids=allowed_tokens
    )
    
    print(f"\nGenerated output under constraint: {text}")
    
    # The output MUST NOT contain "saturn" even though its logit was heavily boosted!
    assert "saturn" not in text.lower()
    # It must contain allowed tokens
    words = text.lower().split()
    for w in words:
        if w not in ("[pad]", "<bos>", "<eos>"):
            assert w in ["jupiter", "is", "the", "largest", "planet", "."]
