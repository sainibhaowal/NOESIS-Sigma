from Core.Reconstruction.canonicalize import scene_sha256
from Core.Reconstruction.scene_models import SceneEvidence, WorkingScene


def test_c1_scene_hash_stable_even_if_evidence_order_changes():
    base = WorkingScene(
        tenant_id="t",
        user_id="u",
        session_id="s",
        profile="STRICT",
        text="hello",
        seed=1,
        snapshot_id="snapA",
        goals=["g1"],
        constraints=["c1"],
        evidence=[
            SceneEvidence(source_kind="SIM", record_id="r2", span_start=0, span_end=3, sha256="b", score=0.2, snippet="bbb"),
            SceneEvidence(source_kind="SIM", record_id="r1", span_start=0, span_end=3, sha256="a", score=0.9, snippet="aaa"),
        ],
    )

    h1 = scene_sha256(base)
    swapped = base.model_copy(update={"evidence": list(reversed(base.evidence))})
    h2 = scene_sha256(swapped)

    assert h1 == h2
