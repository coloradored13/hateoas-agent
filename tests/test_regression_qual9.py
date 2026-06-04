"""Regression test for QUAL-9 (production-readiness audit, 2026-06-04).

Resource.get_gateway() mutated the class-level shared GatewayDef in place
(``gw_def.handler = method``), so two instances of the same Resource subclass
clobbered each other's bound handler (last-writer-wins). It now returns a
per-instance copy.
"""

from hateoas_agent import Resource
from hateoas_agent import gateway as gw_decorator


class TestQual9GetGatewayCopy:
    @staticmethod
    def _make_class():
        class Res(Resource):
            name = "res"

            @gw_decorator(name="enter", description="Enter")
            def enter(self, **kwargs):
                return {"ok": True, "_state": "active"}

        return Res

    def test_two_instances_keep_independent_handlers(self):
        Res = self._make_class()
        a = Res()
        b = Res()

        gw_a = a.get_gateway()
        gw_b = b.get_gateway()

        assert gw_a.handler.__self__ is a
        assert gw_b.handler.__self__ is b
        # Fetching b's gateway must not have rebound a's handler (the old bug).
        assert a.get_gateway().handler.__self__ is a

    def test_does_not_mutate_shared_class_level_def(self):
        Res = self._make_class()
        a = Res()
        shared = Res.enter._hateoas_gateway
        a.get_gateway()
        assert shared.handler is None
        assert a.get_gateway() is not shared
