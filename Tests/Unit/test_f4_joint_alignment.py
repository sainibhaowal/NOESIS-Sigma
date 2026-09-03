import pytest
import torch
import torch.nn as nn
from Runtime.Models.native_decoder_training.joint_train import UnifiedTrainingOrchestrator
from Core.Native_Decoder.sigma_native_emitter import SigmaNativeEmitter, SigmaEmitterConfig

class MockICNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1024, 1024))
    def forward(self, x):
        return x @ self.w

class MockDEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1024, 64))
    def forward(self, x):
        return x @ self.w

def test_f4_joint_alignment_gradients():
    """Verify that the UnifiedTrainingOrchestrator allows gradients to flow backwards through all 3 stages."""
    icnn = MockICNN()
    deq = MockDEQ()
    
    cfg = SigmaEmitterConfig.small()
    cfg.state_dim = 1024
    cfg.d_model = 128
    cfg.vocab_size = 100
    cfg.prefix_len = 12
    emitter = SigmaNativeEmitter(cfg)
    
    orchestrator = UnifiedTrainingOrchestrator(icnn, deq, emitter)
    
    B, L = 2, 5
    x_initial = torch.randn(B, 1024)
    target_nodes = torch.randn(B, 64)
    input_ids = torch.randint(0, 100, (B, L))
    labels = torch.randint(0, 100, (B, L))
    
    l_total, l_ep, l_deq, l_emitter = orchestrator.compute_joint_loss(
        x_initial, target_nodes, input_ids, labels
    )
    
    # Assert losses are valid
    assert l_total.item() > 0
    
    # Backpropagate
    l_total.backward()
    
    # Assert gradients DO NOT flow to frozen components
    assert icnn.w.grad is None, "Gradients must NOT flow to ICNN."
    assert deq.w.grad is None, "Gradients must NOT flow to DEQ."
    
    # Check that at least some parameter in emitter has gradients
    has_grad = False
    for p in emitter.parameters():
        if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            has_grad = True
            break
    assert has_grad, "Gradients must flow to Emitter."
