import torch
import pytest
from Runtime.Models.extractor_training.model import DEQCore
from Core.OSC.dynamics import EngineParams, OperatorSplitEngine
from Core.Native_Decoder.sigma_native_emitter import SigmaEmitterConfig, SigmaNativeEmitter

class MockTokenizer:
    def token_to_id(self, text):
        if text in ("<BOS>", "<s>", "[BOS]", "<bos>"):
            return 1
        if text in ("<EOS>", "</s>", "[EOS]", "<eos>"):
            return 2
        return 3
    
    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


def test_deq_contraction_f5():
    """Verify contraction property: ||f(z1) - f(z2)|| / ||z1 - z2|| < 1.0"""
    hidden_dim = 128
    cell = DEQCore(hidden_dim=hidden_dim, lip_target=0.9)
    
    B, T = 4, 10
    u = torch.randn(B, T, hidden_dim)
    z1 = torch.randn(B, T, hidden_dim)
    z2 = torch.randn(B, T, hidden_dim)
    
    out1 = cell(z1, u)
    out2 = cell(z2, u)
    
    diff_in = torch.linalg.norm(z1 - z2)
    diff_out = torch.linalg.norm(out1 - out2)
    
    lip_est = float(diff_out / (diff_in + 1e-12))
    print(f"\nDEQ Lipschitz estimate: {lip_est}")
    assert lip_est < 1.0, f"DEQ is not a contraction: Lipschitz = {lip_est}"


def test_retrograde_feedback_stability_f1():
    """Verify 100-token generation with retrograde ON, ensuring ||x(t)|| <= max_norm"""
    state_dim = 128
    max_norm = 16.0
    
    params = EngineParams(
        state_dim=state_dim,
        dt=0.005,
        max_norm=max_norm,
        implicit_iters=3,
        implicit_tol=1e-5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    engine = OperatorSplitEngine(params)
    
    cfg = SigmaEmitterConfig(
        state_dim=state_dim,
        d_model=64,
        n_layers=2,
        kernel_size=3,
        prefix_len=12,
        vocab_size=100,
        max_seq_len=50,
        retro_alpha=0.1,
        retro_decay=0.95,
        retro_micro_steps=3,
        retro_u_max=1.0,
    )
    emitter = SigmaNativeEmitter(cfg)
    
    # project initial state
    x_final = torch.randn(1, state_dim)
    x_final = engine.project(x_final)
    
    tokenizer = MockTokenizer()
    original_step_many = engine.step_many
    max_observed_norm = 0.0
    
    def tracked_step_many(*args, **kwargs):
        nonlocal max_observed_norm
        res = original_step_many(*args, **kwargs)
        flat = res.reshape(res.shape[0], -1)
        norms = torch.linalg.norm(flat, dim=1)
        max_observed_norm = max(max_observed_norm, float(norms.max().item()))
        return res
        
    engine.step_many = tracked_step_many
    
    # Generate 100 tokens with retrograde feedback
    text = emitter.generate(
        x_final,
        tokenizer,
        max_new_tokens=100,
        engine=engine,
    )
    
    print(f"\nGenerated tokens sequence: {text}")
    print(f"Max observed state norm: {max_observed_norm}")
    assert max_observed_norm <= max_norm + 1e-5, f"State norm exceeded limit: {max_observed_norm}"
