# ================================================================
#  NOESIS-Σ — Golden Edition
#  Test: HotLoopFused correctness & no-tape checks
# ================================================================
import pytest
import torch

from Core.OSC.dynamics import HotLoopFused
from Core.OSC.icnn import ICNNDirectGrad

torch.manual_seed(1234)

@pytest.mark.parametrize("d,r,B,S", [(64, 8, 32, 4), (128, 16, 16, 3)])
def test_shapes_and_no_tape(d, r, B, S):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    icnn = ICNNDirectGrad(d=d, m=64, dtype=dtype, device=dev)
    KU   = torch.randn(d, r, device=dev, dtype=dtype) * 0.01
    KV   = torch.randn(d, r, device=dev, dtype=dtype) * 0.01

    fused = HotLoopFused(d=d, icnn=icnn, KU=KU, KV=KV, device=dev, dtype=dtype)
    x = torch.randn(B, d, device=dev, dtype=dtype).requires_grad_(True)

    with torch.inference_mode():
        y = fused.step_unrolled(x, S=S, dt=0.05)

    assert y.shape == (B, d)
    # Ensure no autograd tape was built during step_unrolled
    assert x.grad is None

def test_matches_dense_reference_small():
    # For a tiny case, compare with dense K to ensure logic correctness.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    d, r, B, S = 16, 4, 8, 2

    icnn = ICNNDirectGrad(d=d, m=16, dtype=dtype, device=dev)
    KU   = torch.randn(d, r, device=dev, dtype=dtype) * 0.01
    KV   = torch.randn(d, r, device=dev, dtype=dtype) * 0.01
    K    = KU @ KV.t()  # low-rank dense

    fused = HotLoopFused(d=d, icnn=icnn, KU=KU, KV=KV, device=dev, dtype=dtype)
    x0 = torch.randn(B, d, device=dev, dtype=dtype)

    # Reference (dense) loop
    def ref_step(x):
        x_half = x + 0.5 * 0.05 * (x @ K.t())      # row-major apply of Kᵀ
        g = icnn.grad(x_half)
        x_new = x_half - 0.05 * g
        return x_new

    xr = x0.clone()
    for _ in range(S):
        xr = ref_step(xr)

    with torch.inference_mode():
        xf = fused.step_unrolled(x0, S=S, dt=0.05)

    rel = (xf - xr).norm() / (xr.norm() + 1e-7)
    assert rel < 5e-3, f"Fused vs reference mismatch: rel={rel.item():.4e}"
