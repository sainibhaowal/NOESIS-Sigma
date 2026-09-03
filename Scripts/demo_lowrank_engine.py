
import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo = Path(__file__).resolve()
    for _ in range(6):
        if (repo / 'pyproject.toml').exists():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

_ensure_repo_root()

# Scripts/demo_lowrank_engine.py
# Tiny sanity demo to SEE the low-rank K + projector + engine steps.

import torch

from Core.OSC.dynamics import EngineParams, OperatorSplitEngine


def main() -> None:
    # 1) Choose device + small dimension for a human-readable demo
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 64
    B = 8

    params = EngineParams(
        state_dim=d,
        dt=0.01,
        max_norm=4.0,       # projector radius L
        spectral_cap=1.0,   # ||K||_2 cap
        rank_r=8,           # low-rank K
        dtype=torch.float32,
        device=dev,
    )

    engine = OperatorSplitEngine(params)

    # 2) Inspect the low-rank K operator
    k_op = getattr(engine, "_k_operator", None)
    print("Device:", dev)
    print("State dim d:", d)
    if k_op is None:
        print("WARNING: engine._k_operator is None (fallback K path in use).")
    else:
        print("LowRankK operator:", k_op)
        est = k_op.spectral_norm_estimate(n_power_iters=10, batch_size=4, seed=123)
        print(f"Estimated ||K||_2 ≈ {est:.4f}  (cap = {k_op.lambda_cap})")

    # 3) Start from a random batch of states and watch norms over steps
    x = torch.randn(B, d, device=dev, dtype=params.dtype)
    print("\nInitial norms per sample:")
    print(x.norm(dim=-1))

    n_steps = 10
    for step_idx in range(1, n_steps + 1):
        x = engine.step(x)
        norms = x.norm(dim=-1)
        print(f"Step {step_idx:02d} norms:", norms)

    print("\nFinal batch mean norm:", x.norm(dim=-1).mean().item())


if __name__ == "__main__":
    main()
