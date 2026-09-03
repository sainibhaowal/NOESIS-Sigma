from External.Routing import RouterInput, decide
from External.Routing.reason_codes import RC_BLOCK_NO_CITATIONS


class FakeSimProbe:
    def query_spans(self, **kwargs):
        return []


class FakeWksProbe:
    def query_shards(self, **kwargs):
        return []


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_router_strict_blocks_without_citeable_wks():
    inp = RouterInput(user_request="what is the capital of france", policy_mode="STRICT")
    out = decide(
        inp,
        sim_probe_client=FakeSimProbe(),
        wks_probe_client=FakeWksProbe(),
        wks_license_policy=AllowAll(),
    )
    assert out.route_kind == "BLOCK"
    assert RC_BLOCK_NO_CITATIONS in out.reason_codes
