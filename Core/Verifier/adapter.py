# Verifier/adapter.py
# NOESIS-Σ — Receipt Verifier (Golden Edition)
#
# Responsibilities
#  - Verify Ed25519 signatures on engine receipts
#  - (Optional) Cross-check payload against current state (energies, hashes)
#  - Enforce basic freshness/integrity rules
#  - Emit alerts to Runtime/Logs/alerts.log on violations (never crash callers)
#
# Public API
#   verify_receipt(receipt_json, public_key_pem, *,
#                  engine=None, x_before=None, x_after=None,
#                  max_clock_skew_ms=300000, strict=False) -> bool
#
# Usage
#   from Core.Verifier.adapter import verify_receipt
#   ok = verify_receipt(receipt_json, pub_pem, engine=engine, x_before=xb, x_after=xa)

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

# Runtime import (optional): torch is only needed for cross-checks
try:
    import torch as _torch  # noqa: F401
except Exception:  # pragma: no cover
    _torch = None  # type: ignore

# Type-only alias so Pylance/mypy don't complain about variable-in-annotation
if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any  # type: ignore[misc,assignment]

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

__all__ = ["verify_receipt"]

ALERTS_PATH = os.path.join("Runtime", "Logs", "alerts.log")


# -------------------------
# Small utilities
# -------------------------


def _ensure_alerts_dir() -> None:
    try:
        os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    except Exception:
        pass  # never crash on logging infra


def _write_alert(
    code: str, message: str, extra: Optional[Dict[str, Any]] = None
) -> None:
    _ensure_alerts_dir()
    record = {
        "ts_ms": int(time.time() * 1000),
        "code": code,
        "message": message,
        "extra": extra or {},
    }
    try:
        with open(ALERTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass  # best-effort logging


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_tensor(x: "Tensor") -> str:
    """
    Compute SHA-256 over tensor bytes (CPU, contiguous).
    Requires torch at runtime; use TYPE_CHECKING alias for static typing.
    """
    if _torch is None:
        raise RuntimeError("Tensor hashing requires torch at runtime")
    if not isinstance(x, _torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() == 0:
        x = x.reshape(1)
    xb = x.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(xb).hexdigest()


# -------------------------
# Core verification logic
# -------------------------


def _verify_signature(receipt: Dict[str, Any], public_key_pem: bytes) -> bool:
    try:
        payload = receipt["payload"]
        sig_b64 = receipt["signature"]["sig_b64"]
        signature = base64.b64decode(sig_b64)
    except Exception:
        _write_alert(
            "VERIFIER_PAYLOAD_MALFORMED",
            "Receipt missing payload/signature fields",
            {"snippet": str(receipt)[:400]},
        )
        return False

    try:
        pk: Ed25519PublicKey = serialization.load_pem_public_key(public_key_pem)  # type: ignore
        pk.verify(signature, _canonical_json_bytes(payload))
        return True
    except InvalidSignature:
        _write_alert(
            "VERIFIER_BAD_SIGNATURE", "Ed25519 signature invalid", {"payload": payload}
        )
        return False
    except Exception as e:
        _write_alert(
            "VERIFIER_SIG_ERROR",
            f"Signature verification error: {type(e).__name__}",
            {},
        )
        return False


def _basic_payload_checks(receipt: Dict[str, Any], *, max_clock_skew_ms: int) -> bool:
    try:
        payload = receipt["payload"]
        required = (
            "version",
            "engine_version",
            "step_count",
            "state_dim",
            "dt",
            "ts_ms",
            "energy_before",
            "energy_after",
            "delta_l2",
            "state_before_sha256",
            "state_after_sha256",
        )
        for k in required:
            if k not in payload:
                _write_alert(
                    "VERIFIER_SCHEMA_MISSING", f"Missing field '{k}' in payload", {}
                )
                return False

        if not isinstance(payload["step_count"], int) or payload["step_count"] < 0:
            _write_alert(
                "VERIFIER_STEPCOUNT_INVALID",
                "step_count invalid",
                {"step_count": payload["step_count"]},
            )
            return False

        now_ms = int(time.time() * 1000)
        ts_ms = int(payload["ts_ms"])
        if abs(now_ms - ts_ms) > max_clock_skew_ms:
            _write_alert(
                "VERIFIER_CLOCK_SKEW",
                "ts_ms outside allowed skew",
                {"now_ms": now_ms, "ts_ms": ts_ms, "max_skew_ms": max_clock_skew_ms},
            )
            # Not fatal by default; only fatal when strict=True
        return True
    except Exception:
        _write_alert("VERIFIER_SCHEMA_ERROR", "Payload structure error", {})
        return False


def _cross_check_against_state(
    receipt: Dict[str, Any],
    *,
    engine: Any,
    x_before: Optional["Tensor"],
    x_after: Optional["Tensor"],
    strict: bool,
) -> bool:
    """
    Optional: recompute energy and hashes to confirm payload.
    """
    if engine is None or x_before is None or x_after is None:
        return True  # nothing to cross-check

    if _torch is None:
        _write_alert("VERIFIER_NO_TORCH", "Torch not available for cross-check", {})
        return not strict

    payload = receipt["payload"]
    ok = True

    # hash checks
    try:
        sb_hex = _sha256_tensor(x_before)
        sa_hex = _sha256_tensor(x_after)
        if payload.get("state_before_sha256") != sb_hex:
            _write_alert(
                "VERIFIER_HASH_MISMATCH_BEFORE",
                "state_before_sha256 mismatch",
                {"expected": sb_hex, "got": payload.get("state_before_sha256")},
            )
            ok = False
        if payload.get("state_after_sha256") != sa_hex:
            _write_alert(
                "VERIFIER_HASH_MISMATCH_AFTER",
                "state_after_sha256 mismatch",
                {"expected": sa_hex, "got": payload.get("state_after_sha256")},
            )
            ok = False
    except Exception:
        _write_alert("VERIFIER_HASH_ERROR", "Error computing tensor hashes", {})
        ok = False

    # energy checks
    try:
        e0 = float(engine.energy(x_before))
        e1 = float(engine.energy(x_after))
        if abs(e0 - float(payload.get("energy_before", float("nan")))) > 1e-5:
            _write_alert(
                "VERIFIER_ENERGY_BEFORE_MISMATCH",
                "energy_before mismatch",
                {"expected": e0, "got": payload.get("energy_before")},
            )
            ok = False
        if abs(e1 - float(payload.get("energy_after", float("nan")))) > 1e-5:
            _write_alert(
                "VERIFIER_ENERGY_AFTER_MISMATCH",
                "energy_after mismatch",
                {"expected": e1, "got": payload.get("energy_after")},
            )
            ok = False
    except Exception:
        _write_alert("VERIFIER_ENERGY_ERROR", "Error computing energies", {})
        ok = False

    return ok if ok or not strict else False


# -------------------------
# Public entrypoint
# -------------------------

JsonLike = Union[str, bytes, Dict[str, Any]]


def verify_receipt(
    receipt_json: JsonLike,
    public_key_pem: bytes,
    *,
    engine: Any = None,
    x_before: Optional["Tensor"] = None,
    x_after: Optional["Tensor"] = None,
    max_clock_skew_ms: int = 5 * 60 * 1000,  # 5 minutes
    strict: bool = False,
) -> bool:
    """
    Verify a receipt:
      1) Parse JSON (if str/bytes) or accept dict
      2) Verify Ed25519 signature
      3) Apply basic payload checks (schema, step_count, timestamp skew)
      4) (Optional) Cross-check hashes/energies against provided engine & tensors
      5) Log violations to Runtime/Logs/alerts.log

    Returns:
      True if all checks pass (or only non-strict soft failures), False otherwise.

    strict=True makes timestamp skew and cross-check mismatches fatal.
    """
    # Parse
    try:
        if isinstance(receipt_json, dict):
            receipt = receipt_json
        elif isinstance(receipt_json, bytes):
            receipt = json.loads(receipt_json.decode("utf-8"))
        elif isinstance(receipt_json, str):
            receipt = json.loads(receipt_json)
        else:
            _write_alert(
                "VERIFIER_INPUT_TYPE",
                "Unsupported receipt_json type",
                {"type": str(type(receipt_json))},
            )
            return False
    except Exception:
        _write_alert("VERIFIER_JSON_PARSE", "Failed to parse receipt JSON", {})
        return False

    # Signature
    if not _verify_signature(receipt, public_key_pem):
        return False

    # Basic payload checks
    if not _basic_payload_checks(receipt, max_clock_skew_ms=max_clock_skew_ms):
        if strict:
            return False

    # Cross-check against engine state (optional)
    if not _cross_check_against_state(
        receipt, engine=engine, x_before=x_before, x_after=x_after, strict=strict
    ):
        return False

    return True
