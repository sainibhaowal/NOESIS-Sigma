# Tests/core/test_params.py
import pytest

from Core.OSC.params import DEFAULTS, load_params


def test_default_implicit_tol_and_state_dim():
    p = load_params()  # uses defaults and env overrides (none in CI)
    assert p.implicit_tol == pytest.approx(DEFAULTS["implicit_tol"])
    assert p.state_dim == DEFAULTS["state_dim"]
