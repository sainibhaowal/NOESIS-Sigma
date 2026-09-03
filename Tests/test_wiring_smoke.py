from __future__ import annotations

from External.Tools.diagnostics import collect_wiring_diagnostics


def test_wiring_diag_shape_and_cleanup_flags() -> None:
    diag = collect_wiring_diagnostics()
    assert diag["ok"] is True
    assert isinstance(diag.get("modules"), dict)
    assert isinstance(diag.get("wiring"), dict)
    assert isinstance(diag.get("runtime"), dict)
    assert isinstance(diag.get("keys"), dict)
    assert diag["wiring"]["ops_module_present"] is False
    assert diag["wiring"]["routing_module_present"] is False

