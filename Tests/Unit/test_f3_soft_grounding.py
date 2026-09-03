import pytest
import torch
import torch.nn.functional as F
from Core.Native_Decoder.sigma_native_emitter import SigmaNativeEmitter, SigmaEmitterConfig, StructuredPrefixProjector

def test_f3_structured_prefix_projector():
    cfg = SigmaEmitterConfig.small()
    cfg.state_dim = 1024
    cfg.d_model = 512
    cfg.prefix_len = 24
    
    projector = StructuredPrefixProjector(cfg.state_dim, cfg.d_model)
    x_final = torch.randn(2, 1024)
    prefix = projector(x_final)
    
    # Assert dimensions: [B, 24, d_model]
    assert prefix.shape == (2, 24, 512), "StructuredPrefixProjector must output exactly 24 prefix vectors."

def test_f3_emitter_soft_grounding_loss():
    cfg = SigmaEmitterConfig.small()
    cfg.state_dim = 1024
    cfg.d_model = 128
    cfg.vocab_size = 100
    cfg.prefix_len = 24
    
    emitter = SigmaNativeEmitter(cfg)
    
    B, L = 2, 5
    x_final = torch.randn(B, 1024)
    input_ids = torch.randint(0, 100, (B, L))
    labels = torch.randint(0, 100, (B, L))
    target_nodes = torch.randn(B, L, 128)
    
    out = emitter(x_final, input_ids, labels=labels, target_nodes=target_nodes)
    
    assert "loss" in out
    assert out["loss"] is not None
    
    # Since CE loss is positive and l_ground is between 0 and 2
    # The loss should be a valid scalar > 0
    assert out["loss"].item() > 0
