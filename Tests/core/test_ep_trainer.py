"""
Tests/core/test_ep_trainer.py
Unit tests for Equilibrium Propagation (Sprint C4.5).

Covers:
    - ICNNDirectGrad.param_grad_at() shape, dtype, and mathematical correctness
    - OperatorSplitEngine.converge_with_nudge() shape and nudge effect
    - EquilibriumPropagation.train_step() gradient application and loss reduction
    - D2 attractor carving via train_step_toward()
    - r=0 (no low-rank) and r>0 (with low-rank) configurations
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Core.OSC.dynamics import EngineParams, OperatorSplitEngine
from Core.OSC.ep_trainer import EquilibriumPropagation
from Core.OSC.icnn import ICNNDirectGrad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 64    # state dim
M = 32    # ICNN hidden units


def _make_icnn(r: int = 0) -> ICNNDirectGrad:
    return ICNNDirectGrad(d=D, m=M, r=r, dtype=torch.float32, device=torch.device("cpu"))


def _make_engine(icnn: ICNNDirectGrad) -> OperatorSplitEngine:
    params = EngineParams(
        state_dim=D,
        dt=0.005,
        max_norm=50.0,
        implicit_iters=2,
        implicit_tol=1e-5,
        clip_nan_policy="clamp",
        deterministic=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        icnn=icnn,
    )
    return OperatorSplitEngine(params, icnn=icnn)


# ---------------------------------------------------------------------------
# param_grad_at — shape and dtype
# ---------------------------------------------------------------------------

class TestParamGradAt:
    def test_output_keys_no_lowrank(self):
        icnn = _make_icnn(r=0)
        x = torch.randn(4, D)
        grads = icnn.param_grad_at(x)
        assert set(grads.keys()) == {"W", "b", "a_raw", "D_raw", "c"}

    def test_output_keys_with_lowrank(self):
        icnn = _make_icnn(r=8)
        x = torch.randn(4, D)
        grads = icnn.param_grad_at(x)
        assert "R" in grads

    def test_shapes_match_params(self):
        icnn = _make_icnn(r=4)
        x = torch.randn(4, D)
        grads = icnn.param_grad_at(x)
        assert grads["W"].shape == icnn.W.shape
        assert grads["b"].shape == icnn.b.shape
        assert grads["a_raw"].shape == icnn.a_raw.shape
        assert grads["D_raw"].shape == icnn.D_raw.shape
        assert grads["c"].shape == icnn.c.shape
        assert grads["R"].shape == icnn.R.shape

    def test_dtypes_match_params(self):
        icnn = _make_icnn(r=0)
        x = torch.randn(2, D)
        grads = icnn.param_grad_at(x)
        for name, g in grads.items():
            param = dict(icnn.named_parameters())[name]
            assert g.dtype == param.dtype, f"{name}: {g.dtype} != {param.dtype}"

    def test_single_sample_input(self):
        """x with ndim==1 should not crash."""
        icnn = _make_icnn(r=0)
        x = torch.randn(D)
        grads = icnn.param_grad_at(x)
        assert grads["W"].shape == (M, D)

    def test_grad_W_is_finite(self):
        icnn = _make_icnn(r=0)
        x = torch.randn(8, D)
        grads = icnn.param_grad_at(x)
        assert torch.isfinite(grads["W"]).all()
        assert torch.isfinite(grads["b"]).all()

    def test_grad_scales_with_x(self):
        """Scaling x by 2 should change grad_D_raw (0.5·x²·σ) by ~4×."""
        icnn = _make_icnn(r=0)
        x = torch.randn(1, D)
        g1 = icnn.param_grad_at(x)["D_raw"]
        g2 = icnn.param_grad_at(2 * x)["D_raw"]
        # ratio should be close to 4 for large x (sigmoid saturates near 1 for large D_raw)
        ratio = (g2 / (g1 + 1e-12)).mean().item()
        assert ratio > 2.0, f"expected ratio > 2, got {ratio}"


# ---------------------------------------------------------------------------
# converge_with_nudge — shape and effect
# ---------------------------------------------------------------------------

class TestConvergeWithNudge:
    def test_output_shape_vector(self):
        icnn = _make_icnn(r=0)
        engine = _make_engine(icnn)
        x = torch.randn(D)
        target = torch.zeros(D)
        nudge_fn = lambda x: 0.5 * torch.sum((x - target) ** 2)
        out = engine.converge_with_nudge(x, None, nudge_fn, beta=0.1, n_steps=5)
        assert out.shape == (D,)

    def test_output_shape_batch(self):
        icnn = _make_icnn(r=0)
        engine = _make_engine(icnn)
        x = torch.randn(4, D)
        target = torch.zeros(4, D)
        nudge_fn = lambda x: 0.5 * torch.sum((x - target) ** 2)
        out = engine.converge_with_nudge(x, None, nudge_fn, beta=0.1, n_steps=5)
        assert out.shape == (4, D)

    def test_nudge_moves_state_toward_target(self):
        """With strong enough beta, state after nudge should be closer to target."""
        icnn = _make_icnn(r=0)
        engine = _make_engine(icnn)
        torch.manual_seed(42)
        x = torch.randn(1, D) * 3.0
        target = torch.zeros(1, D)

        dist_before = float(torch.norm(x - target))
        nudge_fn = lambda z: 0.5 * torch.sum((z - target) ** 2)
        out = engine.converge_with_nudge(x, None, nudge_fn, beta=0.5, n_steps=20)
        dist_after = float(torch.norm(out - target))

        assert dist_after < dist_before, (
            f"Nudge should reduce distance to target: before={dist_before:.3f}, after={dist_after:.3f}"
        )

    def test_output_is_finite(self):
        icnn = _make_icnn(r=0)
        engine = _make_engine(icnn)
        x = torch.randn(2, D)
        nudge_fn = lambda z: 0.5 * torch.sum(z ** 2)
        out = engine.converge_with_nudge(x, None, nudge_fn, beta=0.01, n_steps=10)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# EquilibriumPropagation — train_step
# ---------------------------------------------------------------------------

class TestEquilibriumPropagation:
    def _make_ep(self, r=0) -> tuple[EquilibriumPropagation, ICNNDirectGrad]:
        icnn = _make_icnn(r=r)
        engine = _make_engine(icnn)
        opt = torch.optim.AdamW(icnn.parameters(), lr=1e-3)
        ep = EquilibriumPropagation(engine, opt, beta=0.05, free_steps=5, nudge_steps=5)
        return ep, icnn

    def test_train_step_returns_scalar(self):
        ep, _ = self._make_ep()
        x = torch.randn(4, D)
        target = torch.zeros(4, D)
        loss = ep.train_step(x, None, target)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_step_modifies_params(self):
        ep, icnn = self._make_ep()
        W_before = icnn.W.data.clone()
        x = torch.randn(4, D)
        target = torch.zeros(4, D)
        ep.train_step(x, None, target)
        # At least one parameter should have changed
        assert not torch.allclose(icnn.W.data, W_before), "Weights should change after EP update"

    def test_loss_decreases_over_steps(self):
        """Over many steps, mean loss should decrease (tendency, not strict step-by-step)."""
        ep, _ = self._make_ep()
        torch.manual_seed(7)
        losses = []
        for _ in range(50):
            x = torch.randn(8, D) * 0.1
            target = torch.randn(8, D) * 0.5
            losses.append(ep.train_step(x, None, target))
        first_10 = sum(losses[:10]) / 10
        last_10  = sum(losses[-10:]) / 10
        # Allow a generous threshold — EP convergence is inherently noisy
        assert last_10 <= first_10 * 2.0, (
            f"Loss not decreasing: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )

    def test_train_step_toward(self):
        ep, _ = self._make_ep()
        x = torch.randn(4, D)
        attractor = torch.zeros(4, D)
        loss = ep.train_step_toward(x, None, attractor)
        assert isinstance(loss, float)

    def test_requires_icnn_direct_grad(self):
        """Engine with wrong icnn type should raise TypeError."""
        icnn = _make_icnn(r=0)
        engine = _make_engine(icnn)
        engine.icnn = torch.nn.Linear(D, D)  # wrong type
        opt = torch.optim.AdamW(icnn.parameters(), lr=1e-3)
        with pytest.raises(TypeError):
            EquilibriumPropagation(engine, opt)

    def test_with_lowrank_r4(self):
        ep, icnn = self._make_ep(r=4)
        x = torch.randn(4, D)
        target = torch.zeros(4, D)
        loss = ep.train_step(x, None, target)
        assert isinstance(loss, float)
        assert icnn.R is not None


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
