from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from Core.Verifier.models import VerifyResult, Witness

DEFAULT_RULES_PATH = "Core/Verifier/rules/core_rules.json"


@lru_cache(maxsize=8)
def _load_rules_cached(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"version": 0, "notes": "rules file missing", "rules": []}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 0, "notes": "rules parse error", "rules": []}
    if not isinstance(obj, dict):
        return {"version": 0, "notes": "rules object invalid", "rules": []}
    rules = obj.get("rules")
    if not isinstance(rules, list):
        rules = []
    return {
        "version": int(obj.get("version", 0) or 0),
        "notes": str(obj.get("notes", "")),
        "rules": rules,
    }


def _mode_allowed(rule: dict[str, Any], mode: str) -> bool:
    modes = rule.get("modes")
    if modes is None:
        return True
    if not isinstance(modes, list):
        return False
    return str(mode).lower() in {str(m).lower() for m in modes}


def _rule_error(rule_id: str, message: str, detail: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "code": "E-RULE-001",
        "message": message,
        "rule_id": rule_id,
    }
    if detail:
        out["detail"] = detail
    return out


def apply_rules(
    witness: Witness,
    result: VerifyResult,
    *,
    rules_path: str | None = None,
) -> VerifyResult:
    path = rules_path or (os.getenv("NOESIS_VERIFIER_RULES_PATH") or DEFAULT_RULES_PATH)
    pack = _load_rules_cached(path)
    rules = [r for r in pack.get("rules", []) if isinstance(r, dict)]

    applied = 0
    rule_errors: list[dict[str, Any]] = []

    for rule in rules:
        enabled = bool(rule.get("enabled", True))
        if not enabled:
            continue
        if not _mode_allowed(rule, witness.mode):
            continue

        kind = str(rule.get("kind") or "").strip().lower()
        rule_id = str(rule.get("id") or kind or "unnamed_rule")
        applied += 1

        if kind == "min_citations":
            min_citations = int(rule.get("min", 1) or 1)
            if len(witness.citations) < min_citations:
                rule_errors.append(
                    _rule_error(
                        rule_id,
                        "Minimum citation count not met",
                        detail=f"have={len(witness.citations)} min={min_citations}",
                    )
                )
        elif kind == "require_snapshot":
            if not witness.snapshot_id or not witness.snapshot_root_hash:
                rule_errors.append(
                    _rule_error(
                        rule_id,
                        "Snapshot fields required by rule",
                        detail="snapshot_id/root_hash missing",
                    )
                )
        elif kind == "require_evidence_hashes":
            ev = witness.vars.get("evidence_hashes") if witness.vars else None
            if not isinstance(ev, list) or not ev:
                rule_errors.append(
                    _rule_error(
                        rule_id,
                        "Evidence hashes required by rule",
                        detail="vars.evidence_hashes missing/empty",
                    )
                )

    if rule_errors:
        result.errors.extend(rule_errors)
        result.verdict = "fail"
        result.verifier_score = 0.0

    result.details["rule_pack"] = {
        "path": path,
        "version": int(pack.get("version", 0)),
        "rules_total": len(rules),
        "rules_applied": applied,
        "rule_errors": len(rule_errors),
    }
    return result

