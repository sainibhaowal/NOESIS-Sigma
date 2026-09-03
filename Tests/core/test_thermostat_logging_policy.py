# Tests/core/test_thermostat_logging_policy.py

import inspect
import io
import logging
import tokenize

import pytest

from Core.OSC.dynamics import OperatorSplitEngine


def _ensure_no_forbidden_logging(func) -> None:
    """
    Static guard: fail if step_many contains direct logging/print calls.

    We scan tokens (not raw text), so comments and docstrings are ignored.
    Forbids patterns like:
      logger.debug(...)
      logging.debug(...)
      loguru.logger.info(...)
      print(...)
    """
    src = inspect.getsource(func)
    tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))

    forbidden_names = ("logger", "logging", "loguru", "print")

    for idx, tok in enumerate(tokens):
        if tok.type != tokenize.NAME:
            continue
        if tok.string not in forbidden_names:
            continue

        # Look ahead to the next “real” token (skip whitespace/indent/newline)
        j = idx + 1
        while j < len(tokens) and tokens[j].type in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            j += 1
        if j >= len(tokens):
            continue

        nxt = tokens[j]

        if tok.string == "print":
            # print(
            if nxt.string == "(":
                pytest.fail(
                    f"Forbidden logging-style call 'print(' found in {func.__qualname__}"
                )
        else:
            # logger.debug / logging.info / loguru.logger.error etc.
            if nxt.string == ".":
                pytest.fail(
                    f"Forbidden logging-style call '{tok.string}.' found in {func.__qualname__}"
                )


def test_step_many_has_no_logging_calls():
    _ensure_no_forbidden_logging(OperatorSplitEngine.step_many)


def _make_engine():
    """
    Reuse the same factory logic as thermostat_acceptance:
    try acceptance engine, then test engine, else skip.
    """
    try:
        from Core.OSC.dynamics import make_acceptance_engine  # type: ignore

        return make_acceptance_engine()
    except Exception:
        pass

    try:
        from Core.OSC.dynamics import make_test_engine  # type: ignore

        return make_test_engine()
    except Exception:
        pass

    pytest.skip(
        "No suitable engine factory found in Core.dynamics; "
        "adjust _make_engine() to your engine factory."
    )


@pytest.mark.parametrize("inner_steps", [32, 128])
def test_step_many_emits_no_logs(caplog, inner_steps):
    """
    Dynamic guard: calling step_many directly must not emit logs
    from the engine's own module (no per-step logging).
    """
    engine = _make_engine()
    x0 = engine.make_initial_state()

    caplog.set_level(logging.DEBUG)
    caplog.clear()

    hot_prefix = engine.__class__.__module__

    with caplog.at_level(logging.DEBUG):
        engine.step_many(
            x0,
            steps=inner_steps,
            token_boundary=True,
        )

    offending = [rec for rec in caplog.records if rec.name.startswith(hot_prefix)]
    assert not offending, (
        f"Expected no logs from '{hot_prefix}' during step_many, "
        f"but saw {len(offending)} records"
    )
