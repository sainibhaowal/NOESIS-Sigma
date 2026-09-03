from __future__ import annotations

import random

from External.Sim.Algorithms.ot_transport import OTParams, update_particles_soft


def _rand_vec(n: int) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(n)]


def test_ot_transport_stability_deterministic():
    random.seed(123)
    xs = [_rand_vec(16) for _ in range(32)]
    ws = [random.random() for _ in range(32)]
    y = _rand_vec(16)

    params = OTParams(tau=0.35, eta=0.25, lam=0.35, novelty_thresh=0.05, w_max=10.0)
    xs1, ws1, _ = update_particles_soft(xs=xs, ws=ws, y=y, v=1.0, params=params, max_particles=64)
    xs2, ws2, _ = update_particles_soft(xs=xs, ws=ws, y=y, v=1.0, params=params, max_particles=64)

    assert xs1 == xs2
    assert ws1 == ws2

    for row in xs1:
        for v in row:
            assert v == v  # not NaN
            assert abs(v) < 1e6

    s = sum(ws1)
    assert s == s
    assert 0.1 <= s <= 1e6


def test_ot_transport_lipschitzish():
    random.seed(7)
    xs = [_rand_vec(8) for _ in range(8)]
    ws = [0.5 for _ in range(8)]
    y = _rand_vec(8)
    y2 = [v + 1e-4 for v in y]

    params = OTParams(tau=0.4, eta=0.2, lam=0.3, novelty_thresh=0.01, w_max=5.0)
    xs1, _, _ = update_particles_soft(xs=xs, ws=ws, y=y, v=1.0, params=params, max_particles=16)
    xs2, _, _ = update_particles_soft(xs=xs, ws=ws, y=y2, v=1.0, params=params, max_particles=16)

    # small perturbation should not explode
    max_delta = 0.0
    for a, b in zip(xs1, xs2):
        for va, vb in zip(a, b):
            max_delta = max(max_delta, abs(va - vb))
    assert max_delta < 1.0
