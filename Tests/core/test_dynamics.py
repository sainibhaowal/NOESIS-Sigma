# Tests/core/test_dynamics.py
# NOESIS-Σ — OperatorSplitEngine acceptance tests (fast + nightly)
#
# Covers:
#  - CPU stepping with small dims using tape-free ICNNDirectGrad
#  - Projection keeps norms within cap (with small epsilon)
#  - Deterministic snapshot round-trip (RNG + state restored)
#  - Nightly long-run stability skeleton (opt-in via marker)
#
# Marks:
#   @pytest.mark.fast     → runs in PR/CI quick job
#   @pytest.mark.nightly  → runs in scheduled/nightly job (heavier)
#
# Notes:
# - Tests run on CPU; no CUDA required.
# - Snapshot test writes to tmp_path; signing/encryption disabled for unit speed.

import json
import os
from pathlib import Path

import pytest
import torch

from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.icnn import ICNNDirectGrad
from Core.OSC.params import load_params

# Keep unit tests deterministic and lightweight on CI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.set_num_threads(1)

FAST_STEPS = 200
TOL = 1e-6  # numeric tolerance for projection/round-trip in float32


@pytest.mark.fast
def test_step_runs_and_projects_cpu(tmp_path):
    """
    Build a tiny engine on CPU (tape-free ICNN), run several steps and confirm:
      - output shape preserved
      - norm after step is within the configured max_norm + eps
    """
    d = 16

    params = load_params()
    params.device = torch.device("cpu")
    params.state_dim = d
    params.max_norm = 1.0
    params.proj_eps = 1e-6
    params.implicit_iters = 1
    params.clip_nan_policy = "raise"
    params.deterministic = True

    # Tape-free ICNN (hybrid + diag quad + low rank + softplus stack, direct grad)
    params.icnn = ICNNDirectGrad(d=d, m=64, device=torch.device("cpu"), dtype=torch.float32)

    eng = OperatorSplitEngine(params)
    eng.set_seed(123)

    # Start with a large-norm state to trigger projection
    x = torch.randn(d) * 10.0
    x_out = eng.step_many(x, steps=FAST_STEPS)

    assert tuple(x_out.shape) == (d,)
    norm_after = torch.linalg.vector_norm(x_out).item()
    assert norm_after <= params.max_norm + 1e-4  # allow tiny numeric slop


@pytest.mark.fast
def test_snapshot_roundtrip_is_deterministic(tmp_path):
    """
    Save a snapshot mid-run, load it twice into fresh engines, and ensure
    both continuations produce identical trajectories (deterministic replay).
    """
    d = 24

    params = load_params()
    params.device = torch.device("cpu")
    params.state_dim = d
    params.max_norm = 2.5
    params.implicit_iters = 1
    params.clip_nan_policy = "raise"
    params.deterministic = True
    params.enable_jit = False

    # Direct-grad ICNN on CPU (tape-free)
    params.icnn = ICNNDirectGrad(d=d, m=64, device=torch.device("cpu"), dtype=torch.float32)

    # Produce initial run and snapshot
    engine0 = OperatorSplitEngine(params)
    engine0.set_seed(7)
    x0 = torch.randn(d)  # initial state
    x_mid = engine0.step_many(x0, 100)
    assert x_mid.shape == (d,)

    snap_dir = tmp_path / "snap_a"
    engine0.save_snapshot(str(snap_dir), sign=False, encrypt=False)

    # Load same snapshot into two fresh engines and continue
    engine_a = OperatorSplitEngine(params)
    engine_b = OperatorSplitEngine(params)
    engine_a.load_snapshot(str(snap_dir))
    engine_b.load_snapshot(str(snap_dir))

    assert engine_a._last_good_state is not None
    assert engine_b._last_good_state is not None
    xa = engine_a.step_many(engine_a._last_good_state.clone(), 50)
    xb = engine_b.step_many(engine_b._last_good_state.clone(), 50)

    assert xa.shape == xb.shape == (d,)
    assert torch.allclose(xa, xb, atol=TOL, rtol=1e-5)

    # Manifest existence & basic fields
    manifest_path = Path(snap_dir) / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("state_dim") == d
    assert "snapshot_sha256" in manifest


@pytest.mark.nightly
def test_long_run_stability_nightly(tmp_path):
    """
    Nightly skeleton: longer run to catch slow drifts.
    Keep limits modest; CI runner can adjust via env:
      NOESIS_NIGHTLY_STEPS (default 2000)
      NOESIS_NIGHTLY_DIM   (default 64)
    """
    steps = int(os.getenv("NOESIS_NIGHTLY_STEPS", "2000"))
    d = int(os.getenv("NOESIS_NIGHTLY_DIM", "64"))

    params = load_params()
    params.device = torch.device("cpu")
    params.state_dim = d
    params.max_norm = 4.0
    params.implicit_iters = 1
    params.clip_nan_policy = "raise"
    params.deterministic = True

    # Direct-grad ICNN on CPU (tape-free)
    params.icnn = ICNNDirectGrad(d=d, m=96, device=torch.device("cpu"), dtype=torch.float32)

    eng = OperatorSplitEngine(params)
    eng.set_seed(2025)

    x = torch.randn(d)
    x_out = eng.step_many(x, steps=steps)

    # Sanity checks
    assert x_out.shape == (d,)
    assert torch.isfinite(x_out).all()
    assert torch.linalg.vector_norm(x_out).item() <= params.max_norm + 1e-3
