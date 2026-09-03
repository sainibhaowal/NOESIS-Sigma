from External.Routing import RouterInput, decide


class FakeSimProbe:
    def query_spans(self, **kwargs):
        return [{"score": 0.9}]


class FakeWksProbe:
    def query_shards(self, **kwargs):
        return [{"license_id": "ok", "score": 0.9}]


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_router_merge_shortlist_order():
    inp = RouterInput(
        user_request="my project and what is the capital of france",
        policy_mode="BALANCED",
    )
    out = decide(
        inp,
        sim_probe_client=FakeSimProbe(),
        wks_probe_client=FakeWksProbe(),
        wks_license_policy=AllowAll(),
    )
    assert out.route_kind == "MERGE"
    assert len(out.shortlist) == 2
    assert out.shortlist[0].source_kind == "SIM"
    assert out.shortlist[1].source_kind == "WKS"
    assert out.shortlist[0].priority < out.shortlist[1].priority
