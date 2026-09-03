from External.Routing import RouterInput, decide


class FakeSimProbe:
    def query_spans(self, **kwargs):
        return [{"score": 0.7}]


class FakeWksProbe:
    def query_shards(self, **kwargs):
        return [{"license_id": "ok", "score": 0.9}]


class AllowAll:
    def allow(self, license_id: str) -> bool:
        return True


def test_router_hash_deterministic_same_input():
    inp = RouterInput(
        user_request="what is the capital of france",
        policy_mode="BALANCED",
        tenant_id="t",
        user_id="u",
        session_id="s",
        turn_id=1,
        trace_id="trace",
    )
    r1 = decide(inp, sim_probe_client=FakeSimProbe(), wks_probe_client=FakeWksProbe(), wks_license_policy=AllowAll())
    r2 = decide(inp, sim_probe_client=FakeSimProbe(), wks_probe_client=FakeWksProbe(), wks_license_policy=AllowAll())

    assert r1.routing_trace_hash == r2.routing_trace_hash
    assert r1.route_kind == r2.route_kind
