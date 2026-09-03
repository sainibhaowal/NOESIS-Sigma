import json
from pathlib import Path

from External.Routing import RouterInput, decide


class FakeSimProbe:
    def query_spans(self, **kwargs):
        text = str(kwargs.get("text", "")).lower()
        if "[sim]" in text or "[merge]" in text or "my" in text:
            return [{"score": 0.9}]
        return []


class FakeWksProbe:
    def query_shards(self, **kwargs):
        text = str(kwargs.get("text", "")).lower()
        if "[no_wks]" in text:
            return []
        if "[wks]" in text or "[merge]" in text or "what is" in text:
            return [{"license_id": "ok", "score": 0.9}]
        return []


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_router_labeled_accuracy():
    path = Path("Tests/Fixtures/router_labeled.jsonl")
    assert path.exists(), "router_labeled.jsonl fixture missing"

    total = 0
    correct = 0

    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inp = RouterInput(
            user_request=row["request"],
            policy_mode=row.get("policy_mode", "BALANCED"),
        )
        out = decide(
            inp,
            sim_probe_client=FakeSimProbe(),
            wks_probe_client=FakeWksProbe(),
            wks_license_policy=AllowAll(),
        )
        total += 1
        if out.route_kind == row["expected_route_kind"]:
            correct += 1

    acc = correct / max(1, total)
    assert acc >= 0.95, f"route accuracy {acc:.3f} < 0.95"
