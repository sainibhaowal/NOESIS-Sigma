"""
Core/OSC/ep_trainer.py
NOESIS-Σ — Equilibrium Propagation trainer for ICNNDirectGrad / OperatorSplitEngine.

Algorithm (Scellier & Bengio 2017, biologically-inspired weight update):
    Phase 1 (free):  x*_free  = converge(x_init, context)
    Phase 2 (nudged): x*_nudge = converge_with_nudge(x_init, context, nudge_fn, β)
    Update:          ΔW = (1/β) · [ ∂φ/∂W|x*_nudge − ∂φ/∂W|x*_free ]

Memory: O(state_dim) — no BPTT, no unrolled graph.

Public API:
    ep = EquilibriumPropagation(engine, optimizer, beta=0.01)
    loss = ep.train_step(x_init, context_bundle, target)
    ep.train_step_toward(x_init, context_bundle, attractor)   # D2 attractor carving
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.icnn import ICNNDirectGrad


class EquilibriumPropagation:
    """
    Equilibrium Propagation training loop for NOESIS-Σ OSC core.

    The optimizer must be pre-constructed by the caller with the ICNN parameters.
    Typical usage::

        icnn = ICNNDirectGrad(d=512, m=256)
        engine = OperatorSplitEngine(params, icnn=icnn)
        opt = torch.optim.AdamW(icnn.parameters(), lr=1e-4)
        ep = EquilibriumPropagation(engine, opt, beta=0.01)

        for x_init, context, target in loader:
            loss = ep.train_step(x_init, context, target)
    """

    def __init__(
        self,
        engine: OperatorSplitEngine,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.01,
        free_steps: Optional[int] = None,
        nudge_steps: Optional[int] = None,
    ) -> None:
        """
        Args:
            engine:       OperatorSplitEngine with ICNNDirectGrad attached.
            optimizer:    Optimizer for ICNN parameters (caller owns lr schedule).
            beta:         Nudge strength. Typical: 0.001–0.1. Smaller = more stable.
            free_steps:   Steps for free-phase convergence. Defaults to engine._S.
            nudge_steps:  Steps for nudged-phase convergence. Defaults to free_steps.
        """
        if not hasattr(engine, "icnn") or not isinstance(engine.icnn, ICNNDirectGrad):
            raise TypeError(
                "EquilibriumPropagation requires engine.icnn to be ICNNDirectGrad"
            )
        self.engine      = engine
        self.icnn: ICNNDirectGrad = engine.icnn  # type: ignore[assignment]
        self.optimizer   = optimizer
        self.beta        = float(beta)
        self.free_steps  = free_steps
        self.nudge_steps = nudge_steps or free_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(
        self,
        x_init: torch.Tensor,
        context_bundle: object,
        target: torch.Tensor,
        nudge_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> float:
        """
        One EP training step toward target.

        Args:
            x_init:        Initial state [d] or [B, d].
            context_bundle: Passed through to engine (may be None).
            target:        Desired equilibrium [d] or [B, d].
            nudge_fn:      Optional override. If None, uses MSE toward target.

        Returns:
            Scalar loss (mean squared distance ||x*_nudge − target||²).
        """
        if nudge_fn is None:
            tgt = target.to(x_init.device, dtype=torch.float32)
            nudge_fn = self._build_nudge(tgt)

        # Phase 1: free convergence
        x_free = self.engine.step_many(
            x_init,
            n_steps=self.free_steps,
            token_boundary=False,
        ).detach()

        # Phase 2: nudged convergence
        x_nudge = self.engine.converge_with_nudge(
            x_free,
            context_bundle,
            nudge_fn,
            beta=self.beta,
            n_steps=self.nudge_steps,
        ).detach()

        # EP weight update
        self._ep_update(x_free, x_nudge)

        loss_val = float(torch.mean((x_nudge - target.to(x_nudge)) ** 2).item())
        return loss_val

    def train_step_toward(
        self,
        x_init: torch.Tensor,
        context_bundle: object,
        attractor: torch.Tensor,
    ) -> float:
        """
        D2 attractor carving — nudge toward a desired attractor state.
        Convenience wrapper around train_step with MSE nudge toward attractor.
        """
        return self.train_step(x_init, context_bundle, attractor)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nudge(
        target: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a nudge_fn that computes 0.5·||x − target||² (scalar)."""
        def nudge_fn(x: torch.Tensor) -> torch.Tensor:
            diff = x.float() - target.float()
            return 0.5 * torch.sum(diff * diff)
        return nudge_fn

    def _ep_update(self, x_free: torch.Tensor, x_nudge: torch.Tensor) -> None:
        """
        Apply EP weight update: ΔW = (1/β) · [∂φ/∂W|nudge − ∂φ/∂W|free].

        Uses param_grad_at() on both states; difference is the EP gradient.
        """
        grads_free  = self.icnn.param_grad_at(x_free)
        grads_nudge = self.icnn.param_grad_at(x_nudge)

        self.optimizer.zero_grad()

        for name, param in self.icnn.named_parameters():
            if name not in grads_free:
                continue
            ep_grad = (grads_nudge[name] - grads_free[name]) / self.beta
            if param.grad is None:
                param.grad = ep_grad.to(param.dtype)
            else:
                param.grad.copy_(ep_grad.to(param.dtype))

        self.optimizer.step()
