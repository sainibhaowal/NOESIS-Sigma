from typing import cast

from Core.Reconstruction.reconstruct import build_working_scene
from Core.Reconstruction.scene_models import SceneBudget


class FakeSIM:
    def __init__(self):
        self._rows = [
            {"record_id": "r1", "span_start": 0, "span_end": 5, "sha256": "a", "snippet": "A", "score": 0.9, "ts_ms": 1},
            {"record_id": "r2", "span_start": 0, "span_end": 5, "sha256": "b", "snippet": "B", "score": 0.8, "ts_ms": 2},
        ]

    def query_spans(self, **kwargs):
        snap = kwargs.get("snapshot_id")
        seed = int(kwargs.get("seed", 0))
        if snap == "snapA":
            rows = list(self._rows)
        else:
            rows = list(reversed(self._rows))
        if seed % 2 == 1:
            rows = [dict(r, score=float(cast(float, r["score"])) + 0.001) for r in rows]
        return rows


class FakeWKS:
    def query_shards(self, **kwargs):
        return [
            {"pack_id": "p", "shard_id": "s", "license_id": "cc", "sha256": "w", "snippet": "W", "score": 0.7}
        ]


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_c1_same_snapshot_same_seed_same_hash():
    b = SceneBudget(top_k_sim=8, top_k_wks=6, pointer_byte_budget=1000)
    s1 = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=42,
        budget=b,
        snapshot_id="snapA",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )
    s2 = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=42,
        budget=b,
        snapshot_id="snapA",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )
    assert s1.scene_hash == s2.scene_hash


def test_c1_change_snapshot_or_seed_changes_hash():
    b = SceneBudget(top_k_sim=8, top_k_wks=6, pointer_byte_budget=1000)
    base = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=42,
        budget=b,
        snapshot_id="snapA",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )
    diff_seed = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=43,
        budget=b,
        snapshot_id="snapA",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )
    diff_snap = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=42,
        budget=b,
        snapshot_id="snapB",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )
    assert base.scene_hash != diff_seed.scene_hash
    assert base.scene_hash != diff_snap.scene_hash
