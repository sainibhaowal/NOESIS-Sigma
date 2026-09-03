from Core.Reconstruction.reconstruct import build_working_scene
from Core.Reconstruction.scene_models import SceneBudget


class FakeSIM:
    def query_spans(self, **kwargs):
        out = []
        for i in range(10):
            out.append(
                {
                    "record_id": f"r{i}",
                    "span_start": 0,
                    "span_end": 10,
                    "sha256": f"h{i}",
                    "snippet": ("x" * 198) + f"{i:02d}",
                    "score": 1.0 - (i * 0.01),
                    "ts_ms": 1000 + i,
                }
            )
        return out


class FakeWKS:
    def query_shards(self, **kwargs):
        return []


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_c1_budgeted_snippets_deterministic_prefix():
    budget = SceneBudget(pointer_byte_budget=600, top_k_sim=10, top_k_wks=0)
    s = build_working_scene(
        tenant_id="t",
        user_id="u",
        session_id="s",
        text="q",
        profile="STRICT",
        seed=0,
        budget=budget,
        snapshot_id="snap",
        sim_client=FakeSIM(),
        wks_client=FakeWKS(),
        wks_license_policy=AllowAll(),
    )

    assert len(s.evidence) == 3
    assert s.evidence[0].record_id == "r0"
    assert s.evidence[1].record_id == "r1"
    assert s.evidence[2].record_id == "r2"
