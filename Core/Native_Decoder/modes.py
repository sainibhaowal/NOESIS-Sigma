from __future__ import annotations

CHAT = "chat"
CODE = "code"
PLAN = "plan"
ANALYSIS = "analysis"

VALID_MODES = frozenset({CHAT, CODE, PLAN, ANALYSIS})


def normalize_mode(mode: str) -> str:
    m = (mode or "chat").strip().lower()
    return m if m in VALID_MODES else CHAT
