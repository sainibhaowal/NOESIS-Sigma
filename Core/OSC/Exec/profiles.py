# Core/Exec/profiles.py
import os
from dataclasses import dataclass

# --- add near top of profiles.py ---
from pathlib import Path

import torch


def _load_dotenv():
    env = Path(__file__).resolve().parents[2] / "Runtime" / "Config" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# load .env before reading NOESIS_PROFILE
_load_dotenv()

# existing code that reads env:
_profile = os.getenv("NOESIS_PROFILE", "AUTO").upper()


# (optional) handy helper so you can switch in notebooks/REPL:
def set_active(name: str):
    os.environ["NOESIS_PROFILE"] = name.upper()
    # no importlib reload needed if you read os.getenv() every call,
    # but if you cache config, reload your cached values here.


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    icnn_ws: torch.dtype
    k_ws: torch.dtype
    main_dtype: torch.dtype
    tf32: bool
    deterministic: bool


PROFILES = {
    "FAST": ProfileConfig(
        "FAST", torch.float16, torch.float16, torch.float16, True, False
    ),
    "BALANCED": ProfileConfig(
        "BALANCED", torch.float32, torch.float16, torch.float16, True, False
    ),
    "STRICT": ProfileConfig(
        "STRICT", torch.float32, torch.float32, torch.float32, False, True
    ),
}


def _auto_policy() -> str:
    det = os.getenv("NOESIS_DETERMINISTIC", "0") == "1"
    if det:
        return "STRICT"
    safety = os.getenv("NOESIS_NUMERIC_SAFETY", "low").lower()
    if safety in ("high", "strict", "audit"):
        return "BALANCED"
    try:
        tgt_us = int(os.getenv("NOESIS_LATENCY_US_TARGET", "0"))
        if tgt_us and tgt_us <= 60:
            return "FAST"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "FAST"
    return "BALANCED"


def resolve_profile(name: str | None) -> ProfileConfig:
    raw = name if name is not None else os.getenv("NOESIS_PROFILE", "AUTO")
    target = raw.upper()
    if target == "AUTO":
        target = _auto_policy()
    if target not in PROFILES:
        raise ValueError(f"Unknown profile: {target}")
    return PROFILES[target]


def apply_profile(cfg: ProfileConfig) -> ProfileConfig:
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.use_deterministic_algorithms(bool(cfg.deterministic))
    return cfg


# --- small helpers used by your REPL/snippets ---
def active_name() -> str:
    name = os.getenv("NOESIS_PROFILE", "AUTO").upper()
    if name == "AUTO":
        name = _auto_policy()
    return name


def active_config() -> dict:
    cfg = apply_profile(resolve_profile(os.getenv("NOESIS_PROFILE", "AUTO")))
    return {
        "name": cfg.name,
        "dtype": cfg.main_dtype,  # expose as "dtype" for scripts/snippets
        "icnn_ws": cfg.icnn_ws,
        "k_ws": cfg.k_ws,
        "tf32": cfg.tf32,
        "deterministic": cfg.deterministic,
    }
