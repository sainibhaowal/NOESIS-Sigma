from __future__ import annotations

import random

from External.Sim.Algorithms.ot_barycenter import CompactParams, compact_hot_particles
from External.Sim.Models.measures import HotParticle


def _mk_particle(i: int) -> HotParticle:
    random.seed(i + 100)
    x = [random.uniform(-1.0, 1.0) for _ in range(12)]
    return HotParticle.create(
        tenant_id="t",
        user_id="u",
        memory_type="episodic",
        payload=f"p{i}",
        x=x,
        w=random.uniform(0.1, 2.0),
        tags={},
    )


def test_compaction_mass_preserved():
    parts = [_mk_particle(i) for i in range(64)]
    w_before = sum(p.w for p in parts)

    params = CompactParams(bucket_dims=8, target_n=16)
    out = compact_hot_particles(parts, params)
    w_after = sum(p.w for p in out)

    if w_before > 0:
        rel_err = abs(w_before - w_after) / w_before
        assert rel_err <= 1e-6
