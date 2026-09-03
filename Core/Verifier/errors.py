# Verifier/errors.py
# NOESIS-S -- Verifier Errors (D4)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerifierError:
    code: str
    message: str
    detail: str | None = None

    def as_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out


E_DSL_001 = "E-DSL-001"  # syntax / unsupported AST
E_REF_404 = "E-REF-404"  # missing span/source
E_MATH_002 = "E-MATH-002"  # math domain/zero div
E_UNIT_003 = "E-UNIT-003"  # unit mismatch (stub hook)
E_POL_401 = "E-POL-401"  # policy/auth/consent violation
E_LIM_413 = "E-LIM-413"  # limits exceeded (bytes/steps/time)
E_HASH_409 = "E-HASH-409"  # hash/manifest mismatch
