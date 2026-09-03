from Core.Reconstruction.reconstruct import build_working_scene
from Core.Reconstruction.scene_models import SceneBudget


class FakeSIM:
    def query_spans(self, **kwargs):
        return []


class FakeWKS:
    def query_shards(self, **kwargs):
        return [
            {"pack_id": "p", "shard_id": "s1", "license_id": "blocked", "sha256": "x", "snippet": "X", "score": 1.0},
            {"pack_id": "p", "shard_id": "s2", "license_id": "ok", "sha256": "y", "snippet": "Y", "score": 0.9},
        ]


class Policy:
    def allow(self, license_id: str) -> bool:
        return license_id != "blocked"


def test_c1_license_blocks_wks_items():
    s = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="sess",
        text="q",
        profile="STRICT",
        seed=0,
        budget=SceneBudget(pointer_byte_budget=1000, top_k_sim=1, top_k_wks=10),
        snapshot_id="snap",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=Policy(),
    )
    assert len(s.evidence) == 1
    assert s.evidence[0].source_kind == "WKS"
    assert s.evidence[0].license_id == "ok"
