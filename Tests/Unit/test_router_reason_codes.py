from External.Routing import RouterInput, decide
from External.Routing.reason_codes import ALL_REASON_CODES


class FakeSimProbe:
    def query_spans(self, **kwargs):
        return [{"score": 0.9}]


class FakeWksProbe:
    def query_shards(self, **kwargs):
        return [{"license_id": "ok", "score": 0.8}]


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_router_reason_codes_valid_and_sorted():
    inp = RouterInput(user_request="what is the capital of france", policy_mode="BALANCED")
    out = decide(
        inp,
        sim_probe_client=FakeSimProbe(),
        wks_probe_client=FakeWksProbe(),
        wks_license_policy=AllowAll(),
    )
    assert all(rc in ALL_REASON_CODES for rc in out.reason_codes)
    assert out.reason_codes == sorted(out.reason_codes)
