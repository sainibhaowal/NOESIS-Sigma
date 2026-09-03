# Verifier/checker.py
# NOESIS-S -- Witness Checker (D4)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from Core.Verifier.dsl_ast import compile_expr
from Core.Verifier.errors import (
    E_DSL_001,
    E_HASH_409,
    E_LIM_413,
    E_MATH_002,
    E_REF_404,
    VerifierError,
)
from Core.Verifier.models import VerifyResult, Witness
from Core.Verifier.span_store import SpanLimits, SpanStore, safe_fetch_span


@dataclass(frozen=True)
class CheckLimits:
    max_statements: int = 128
    max_eval_ops: int = 50_000
    span_limits: SpanLimits = SpanLimits()


class _SpanLimitExceeded(Exception):
    pass


def _mk_functions(
    witness: Witness,
    store: SpanStore,
    limits: CheckLimits,
) -> dict[str, Any]:
    span_cache: dict[tuple[str, str, int, int], str] = {}

    def SPAN(span_id: str, start: int, end: int, source: str = "SIM") -> str:
        key = (source, span_id, int(start), int(end))
        if key in span_cache:
            return span_cache[key]
        if len(span_cache) >= limits.span_limits.max_spans:
            raise _SpanLimitExceeded(f"max_spans={limits.span_limits.max_spans}")
        st, err = safe_fetch_span(
            store,
            source=source,
            span_id=span_id,
            start=int(start),
            end=int(end),
            snapshot_id=witness.snapshot_id,
            limits=limits.span_limits,
        )
        if err or st is None:
            raise KeyError(err.detail if err else "missing span")
        span_cache[key] = st.text
        return st.text

    def CITE(span_id: str, source: str = "SIM") -> str:
        return f"{source}:{span_id}"

    def COUNT(term: str, span_id: str, source: str = "SIM") -> int:
        txt = SPAN(span_id, 0, limits.span_limits.max_bytes, source=source)
        return txt.count(term)

    def ABS(x: float) -> float:
        return abs(float(x))

    def ROUND(x: float, nd: int = 0) -> float:
        return round(float(x), int(nd))

    def SUM(*xs: float) -> float:
        return float(sum(float(v) for v in xs))

    def MEAN(*xs: float) -> float:
        vals = [float(v) for v in xs]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    return {
        "SPAN": SPAN,
        "CITE": CITE,
        "COUNT": COUNT,
        "ABS": ABS,
        "ROUND": ROUND,
        "SUM": SUM,
        "MEAN": MEAN,
        "math": math,
    }


def verify_witness(
    witness: Witness, store: SpanStore, limits: CheckLimits
) -> VerifyResult:
    errors: list[dict] = []
    details: dict[str, Any] = {"checked": 0, "failed": 0, "eval_ops": 0}

    if len(witness.citations) > limits.span_limits.max_spans:
        return VerifyResult(
            verdict="fail",
            verifier_score=0.0,
            errors=[
                VerifierError(
                    code=E_LIM_413,
                    message="Too many citations",
                    detail=f"n={len(witness.citations)} max={limits.span_limits.max_spans}",
                ).as_dict()
            ],
        )

    # Snapshot root hash check (if provided and store supports it)
    if (
        witness.snapshot_id
        and witness.snapshot_root_hash
        and hasattr(store, "get_snapshot_root_hash")
    ):
        try:
            root = store.get_snapshot_root_hash(snapshot_id=witness.snapshot_id)  # type: ignore[arg-type]
            if root and root != witness.snapshot_root_hash:
                errors.append(
                    VerifierError(
                        code=E_HASH_409,
                        message="Snapshot root hash mismatch",
                        detail=f"expected {witness.snapshot_root_hash} got {root}",
                    ).as_dict()
                )
        except Exception as e:
            errors.append(
                VerifierError(
                    code=E_HASH_409,
                    message="Snapshot root hash lookup failed",
                    detail=str(e),
                ).as_dict()
            )

    # Citation hash checks
    cited_hashes: list[str] = []
    for c in witness.citations:
        span = c.span
        if witness.mode == "strict" and not span.sha256:
            errors.append(
                VerifierError(
                    code=E_HASH_409,
                    message="Missing citation hash in strict mode",
                    detail=f"{span.id}",
                ).as_dict()
            )
        st, err = safe_fetch_span(
            store,
            source=span.source,
            span_id=span.id,
            start=span.start,
            end=span.end,
            snapshot_id=witness.snapshot_id,
            limits=limits.span_limits,
        )
        if err or st is None:
            errors.append(
                err.as_dict() if err else {"code": E_REF_404, "message": "Missing span"}
            )
            continue
        if span.sha256:
            if st.sha256 and st.sha256 != span.sha256:
                errors.append(
                    VerifierError(
                        code=E_HASH_409,
                        message="Citation hash mismatch",
                        detail=f"{span.id}",
                    ).as_dict()
                )
        if span.sha256:
            cited_hashes.append(span.sha256)

    # Evidence hash list consistency (if provided)
    ev_hashes = witness.vars.get("evidence_hashes") if witness.vars else None
    if isinstance(ev_hashes, list) and ev_hashes:
        if sorted(set(ev_hashes)) != sorted(set(cited_hashes)):
            errors.append(
                VerifierError(
                    code=E_HASH_409,
                    message="Evidence hashes mismatch",
                    detail="evidence_hashes does not match citations",
                ).as_dict()
            )

    if len(witness.statements) > limits.max_statements:
        return VerifyResult(
            verdict="fail",
            verifier_score=0.0,
            errors=[
                {
                    "code": "E-LIM-413",
                    "message": "Too many statements",
                    "detail": f"n={len(witness.statements)}",
                }
            ],
        )

    env: dict[str, Any] = {}
    env.update(witness.vars or {})
    env.update(_mk_functions(witness, store, limits))

    passed = 0
    failed = 0
    eval_ops_used = 0

    for stmt in witness.statements:
        stmt = (stmt or "").strip()
        if not stmt:
            continue

        compiled, err = compile_expr(stmt)
        if err or compiled is None:
            if err is None:
                errors.append(
                    VerifierError(
                        code=E_DSL_001, message="DSL compile failed", detail=stmt
                    ).as_dict()
                )
            else:
                errors.append(err.as_dict())
            failed += 1
            continue

        eval_ops_used += int(getattr(compiled, "node_count", 1))
        details["eval_ops"] = eval_ops_used
        if eval_ops_used > limits.max_eval_ops:
            failed += 1
            errors.append(
                VerifierError(
                    code=E_LIM_413,
                    message="Evaluation operation budget exceeded",
                    detail=f"eval_ops={eval_ops_used} max={limits.max_eval_ops}",
                ).as_dict()
            )
            break

        try:
            val = eval(compiled.code, {"__builtins__": {}}, env)
            ok = bool(val)
            details["checked"] += 1
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append(
                    {
                        "code": "E-CLAIM-FAIL",
                        "message": "Claim evaluated to false",
                        "detail": stmt,
                    }
                )
        except ZeroDivisionError as e:
            failed += 1
            errors.append(
                VerifierError(
                    code=E_MATH_002, message="Math domain error", detail=str(e)
                ).as_dict()
            )
        except KeyError as e:
            failed += 1
            errors.append(
                VerifierError(
                    code=E_REF_404, message="Missing span/ref", detail=str(e)
                ).as_dict()
            )
        except _SpanLimitExceeded as e:
            failed += 1
            errors.append(
                VerifierError(
                    code=E_LIM_413, message="Span lookup limit exceeded", detail=str(e)
                ).as_dict()
            )
        except Exception as e:
            failed += 1
            errors.append(
                VerifierError(
                    code=E_MATH_002, message="Evaluation error", detail=str(e)
                ).as_dict()
            )

    details["failed"] = failed
    total = max(1, passed + failed)
    score = float(passed / total)

    verdict: Literal["pass", "fail", "unsure"]
    if failed == 0 and passed > 0 and not errors:
        verdict = "pass"
    elif passed == 0:
        verdict = "unsure"
    else:
        verdict = "fail"
    return VerifyResult(
        verdict=verdict, verifier_score=score, errors=errors, details=details
    )
