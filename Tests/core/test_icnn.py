# ================================================================
#  NOESIS-Σ — Golden Edition
#  Test: ICNNDirectGrad surface and state roundtrip
#  (Pure tape-free; no autograd)
# ================================================================
import pytest
import torch

from Core.OSC.icnn import ICNNDirectGrad


@pytest.mark.fast
def test_icnn_direct_grad_shape_and_finite():
    d, m, B = 16, 64, 3
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    icnn = ICNNDirectGrad(d=d, m=m, device=dev, dtype=dtype)
    x = torch.randn((B, d), device=dev, dtype=dtype)
    with torch.inference_mode():
        g = icnn.grad(x)
    assert g.shape == (B, d)
    assert torch.isfinite(g).all()

@pytest.mark.fast
def test_icnn_direct_state_dict_roundtrip():
    d, m = 12, 48
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    icnn1 = ICNNDirectGrad(d=d, m=m, device=dev, dtype=dtype)
    x = torch.randn((2, d), device=dev, dtype=dtype)

    g1 = icnn1.grad(x)
    sd = icnn1.export_state()

    icnn2 = ICNNDirectGrad(d=d, m=m, device=dev, dtype=dtype)
    icnn2.load_state(sd)
    g2 = icnn2.grad(x)

    assert torch.allclose(g1, g2, atol=1e-5, rtol=1e-5)
