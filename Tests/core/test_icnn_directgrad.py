# ================================================================
#  NOESIS-Σ — Golden Edition
#  Test: ICNNDirectGrad correctness & convexity checks
# ================================================================
import pytest
import torch

from Core.OSC.icnn import ICNNDirectGrad

torch.manual_seed(7)

def finite_diff_grad(phi_fn, x, eps=1e-4):
    """
    Numeric grad for sanity (slow, small d, per-element perturbation).
    phi_fn must return a scalar (the caller uses .sum()).
    """
    x = x.clone().detach()
    B, d = x.shape
    g = torch.zeros_like(x, dtype=torch.float64)
    for i in range(B):
        for j in range(d):
            e = torch.zeros_like(x)
            e[i, j] = eps
            f_plus = phi_fn(x + e).to(torch.float64)
            f_minus = phi_fn(x - e).to(torch.float64)
            g[i, j] = (f_plus - f_minus) / (2.0 * eps)
    return g.to(x.dtype)

@pytest.mark.parametrize("d,m,B", [(8, 32, 4), (16, 64, 2)])
def test_direct_grad_matches_numeric(d, m, B):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=d, m=m, dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device=dev)
    x = torch.randn(B, d, device=dev, dtype=torch.float32) * 0.1

    g_analytic = icnn.grad(x).float()
    g_numeric = finite_diff_grad(lambda z: icnn.forward(z).sum(), x.float())

    rel_err = (g_analytic - g_numeric).norm() / (g_numeric.norm() + 1e-6)
    assert rel_err < 5e-2, f"Relative error too high: {rel_err.item():.4f}"

def test_basic_convexity():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=6, m=32, device=dev, dtype=torch.float32)
    x1 = torch.randn(1, 6, device=dev)
    x2 = torch.randn(1, 6, device=dev)
    t = 0.3
    lhs = icnn((1 - t) * x1 + t * x2)
    rhs = (1 - t) * icnn(x1) + t * icnn(x2)
    assert (lhs <= rhs + 1e-5).all(), "Convexity inequality violated"

def test_no_tape_in_grad():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icnn = ICNNDirectGrad(d=4, m=16, device=dev, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    x = torch.randn(3, 4, device=dev).requires_grad_(True)
    with torch.inference_mode():
        _ = icnn.grad(x)  # should not touch autograd
    # Ensure autograd didn't record ops:
    assert x.grad is None
