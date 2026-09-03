import pytest
import torch

from Core.OSC.kernels.placeholder import apply_K_dense, apply_K_lowrank


@pytest.mark.fast
def test_lowrank_matches_dense_small():
    torch.manual_seed(0)
    B, d, r = 5, 16, 4
    x = torch.randn(B, d)
    U = torch.randn(d, r)
    V = torch.randn(d, r)
    K = U @ V.T
    y_lr = apply_K_lowrank(U, V, x)
    y_dn = apply_K_dense(K, x)
    assert y_lr.shape == (B, d)
    assert torch.allclose(y_lr, y_dn, atol=1e-6, rtol=1e-5)

@pytest.mark.fast
def test_vector_input_supported():
    torch.manual_seed(0)
    d, r = 8, 3
    x = torch.randn(d)             # 1D input
    U = torch.randn(d, r)
    V = torch.randn(d, r)
    K = U @ V.T
    y_lr = apply_K_lowrank(U, V, x)
    y_dn = apply_K_dense(K, x)
    assert y_lr.shape == (d,)
    assert y_dn.shape == (d,)
    assert torch.allclose(y_lr, y_dn, atol=1e-6, rtol=1e-5)

@pytest.mark.fast
def test_shape_checks():
    B, d, r = 2, 8, 2
    x = torch.randn(B, d)
    U = torch.randn(d, r)
    V = torch.randn(d + 1, r)   # wrong d
    with pytest.raises(ValueError):
        _ = apply_K_lowrank(U, V, x)
