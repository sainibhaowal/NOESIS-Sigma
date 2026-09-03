# Verifier/service.py
# NOESIS-S -- Verifier Service (D4)

from __future__ import annotations

from Core.Verifier.checker import CheckLimits, verify_witness
from Core.Verifier.models import VerifyResult, Witness
from Core.Verifier.policy import policy_for
from Core.Verifier.rules_engine import apply_rules
from Core.Verifier.span_store import SpanStore


def verify(
    witness: Witness, store: SpanStore, *, rules_path: str | None = None
) -> VerifyResult:
    pol = policy_for(witness.mode)
    limits = CheckLimits(max_statements=pol.max_statements)
    res = verify_witness(witness, store, limits)

    if pol.require_spans and not witness.citations:
        res.verdict = "fail"
        res.verifier_score = 0.0
        res.errors.append(
            {
                "code": "E-REF-400",
                "message": "Citations required by policy",
                "detail": "no citations provided",
            }
        )

    if witness.mode == "strict":
        if res.verdict != "pass" or res.verifier_score < pol.min_score:
            res.verdict = "fail"
    elif witness.mode == "balanced":
        if res.verdict == "pass" and res.verifier_score < pol.min_score:
            res.verdict = "unsure"

    return apply_rules(witness, res, rules_path=rules_path)
