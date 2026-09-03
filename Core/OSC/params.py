# Core/params.py
# NOESIS-Σ — EngineParams loader & defaults (Golden Edition)
# Produces EngineParams used by Core/dynamics.OperatorSplitEngine
#
# Responsibilities:
#  - Provide canonical DEFAULTS and locked numeric tolerances
#  - Load params from YAML/JSON file (if provided) and then override from environment NOESIS_* vars
#  - Convert/validate types and return an EngineParams instance
#  - Provide save_params(path) to write reproducible YAML
#  - Load optional snapshot key files (AES key, PEMs) when env paths provided
#
# Usage:
#   from Core.OSC.params import load_params
#   params = load_params()                    # use defaults + .env overrides
#   params = load_params("Core/params.yml")   # load YAML then env overrides
#
# Notes on tolerances (recommended):
#  - For float32 (default): IMPLICIT_TOL = 1e-6, PROJ_EPS = 1e-6
#  - For float16 (mixed precision): IMPLICIT_TOL = 1e-4, PROJ_EPS = 1e-4

from __future__ import annotations

import json
import logging
import os
import typing as t
from dataclasses import asdict
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from Core.OSC.Exec import profiles as _profiles

# Local import of EngineParams (dataclass defined in dynamics)
from .dynamics import EngineParams

# optional .env loader (dev convenience)
_load_dotenv: t.Optional[t.Callable[..., bool]]
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    _load_dotenv = None

load_dotenv_fn: t.Optional[t.Callable[..., bool]] = t.cast(
    t.Optional[t.Callable[..., bool]], _load_dotenv
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical defaults (locked numeric tolerances)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, t.Any] = {
    "state_dim": 1024,
    "dt": 0.005,
    "max_norm": 16.0,
    "spectral_cap": 1.0,
    "implicit_iters": 1,
    "implicit_tol": 1e-6,  # float32 default; use 1e-4 for FP16 runs
    "rank_r": 32,
    "proj_eps": 1e-6,
    "clip_nan_policy": "raise",  # 'raise' | 'clamp' | 'rewind'
    "max_step_time_ms": 200.0,
    "deterministic": True,
    "dtype": "float32",  # string, converted later to torch.dtype
    "device": "cpu",  # string, converted later to torch.device
    "enable_jit": False,
    "enable_profiling": False,
    "ENABLE_TRITON": False,  # env/YAML knob; mapped to EngineParams.enable_triton
    "allow_mixed_precision": False,
}

# Environment variable prefix (NOESIS_)
ENV_PREFIX = "NOESIS"

# Optional env keys for snapshot keys (paths)
ENV_SNAPSHOT_AES_KEY_PATH = f"{ENV_PREFIX}_SNAPSHOT_AES_KEY_PATH"
ENV_SNAPSHOT_SIGN_PRIVATE_PEM_PATH = f"{ENV_PREFIX}_SNAPSHOT_SIGN_PRIVATE_PEM_PATH"
ENV_SNAPSHOT_SIGN_PUBLIC_PEM_PATH = f"{ENV_PREFIX}_SNAPSHOT_SIGN_PUBLIC_PEM_PATH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_value(key: str, value: t.Any):
    """
    Convert string values from env/YAML into typed python values expected by EngineParams.
    Supported keys: ints, floats, bools, dtype, device.
    """
    if value is None:
        return None

    # Booleans (include both lower/upper for convenience)
    if key in (
        "deterministic",
        "enable_jit",
        "enable_profiling",
        "allow_mixed_precision",
        "enable_triton",
        "ENABLE_TRITON",
    ):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        return s in ("1", "true", "yes", "on")

    # Integers
    if key in ("state_dim", "implicit_iters", "rank_r"):
        return int(value)

    # Floats
    if key in (
        "dt",
        "max_norm",
        "spectral_cap",
        "implicit_tol",
        "proj_eps",
        "max_step_time_ms",
    ):
        return float(value)

    # Enum-like policy
    if key == "clip_nan_policy":
        s = str(value).strip().lower()
        if s not in ("raise", "clamp", "rewind"):
            raise ValueError(f"invalid clip_nan_policy: {value}")
        return s

    # Dtype (lazy import torch)
    if key == "dtype":
        s = str(value).strip().lower()
        import torch

        if s in ("float32", "fp32"):
            return torch.float32
        if s in ("float16", "fp16", "half"):
            return torch.float16
        if s in ("bfloat16", "bf16"):
            return torch.bfloat16
        if s in ("float64", "fp64", "double"):
            return torch.float64
        raise ValueError(f"unsupported dtype: {value}")

    # Device (lazy import torch)
    if key == "device":
        import torch

        return torch.device(str(value))

    # Fallback: return unchanged
    return value


def _read_key_file(path: t.Union[str, Path]) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"key file not found: {p}")
    return p.read_bytes()


def _merge_dicts(base: dict, override: dict | None) -> dict:
    """Shallow merge with override taking precedence."""
    out = base.copy()
    if override:
        out.update(override)
    return out


# Simple env-bool helper (used if you want to read a fresh env flag)
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_params(
    path: t.Optional[str] = None, env_prefix: str = ENV_PREFIX
) -> EngineParams:
    """
    Load EngineParams with the following precedence (lowest → highest):
      1) DEFAULTS (in-repo)
      2) Profile selected by NOESIS_PROFILE (if env not ignored) — but profiles may NOT change state_dim
      3) YAML/JSON file at `path` (if provided)
      4) Environment variables with prefix `env_prefix` (e.g., NOESIS_DT)
         (skipped entirely when NOESIS_IGNORE_ENV=1)

    Tests can set NOESIS_IGNORE_ENV=1 to freeze defaults (and file) deterministically.
    """
    ignore_env = os.getenv("NOESIS_IGNORE_ENV", "0") in ("1", "true", "True")

    # start from defaults as a mutable config dict
    cfg: dict[str, t.Any] = DEFAULTS.copy()

    # 1) (optional) profile — ONLY if env is not ignored
    if not ignore_env:
        profile_name = os.getenv("NOESIS_PROFILE", "").upper().strip()
        if profile_name:
            try:
                # Support either dict-like module or PROFILES mapping
                if hasattr(_profiles, "get"):
                    prof_cfg = dict(_profiles.get(profile_name, {}))
                elif hasattr(_profiles, "PROFILES"):
                    prof_cfg = dict(
                        getattr(_profiles, "PROFILES", {}).get(profile_name, {})
                    )
                else:
                    prof_cfg = {}
            except Exception:
                prof_cfg = {}
            # Never allow profiles to override canonical default of state_dim
            prof_cfg.pop("state_dim", None)
            # Shallow merge: DEFAULTS <- profile
            tmp = cfg.copy()
            tmp.update(prof_cfg)
            cfg = tmp

    # 2) optional file (YAML first, then JSON)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"params file not found: {path}")
        text = p.read_text()
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                cfg = _merge_dicts(cfg, loaded)
        except Exception as e:
            logger.warning("YAML parse failed for %s: %s; trying JSON", path, e)
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    cfg = _merge_dicts(cfg, loaded)
            except Exception as e2:
                raise ValueError(
                    f"params file not valid YAML/JSON: {path}; errors: {e}, {e2}"
                )

    # 3) environment (optional)
    if not ignore_env:
        # load project .env (dev convenience); does nothing if absent
        if load_dotenv_fn is not None:
            load_dotenv_fn("Runtime/Config/.env", override=False)

        # prefix-based overrides e.g. NOESIS_STATE_DIM=2048
        for k in list(cfg.keys()):
            env_name = f"{env_prefix}_{k.upper()}"
            if env_name in os.environ:
                raw = os.environ[env_name]
                try:
                    cfg[k] = _coerce_value(k, raw)
                except Exception as e:
                    logger.warning("Failed to coerce env %s=%s: %s", env_name, raw, e)

        # explicit opt-in for state_dim via dedicated env
        sd_env = os.getenv(f"{env_prefix}_STATE_DIM")
        if sd_env is not None:
            try:
                cfg["state_dim"] = int(sd_env)
            except Exception as e:
                logger.warning(
                    "Failed to parse %s_STATE_DIM=%r: %s", env_prefix, sd_env, e
                )

    # normalize dtype/device if strings (can come from file/env)
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = _coerce_value("dtype", cfg["dtype"])
    if isinstance(cfg.get("device"), str):
        cfg["device"] = _coerce_value("device", cfg["device"])

    # handle triton enabling from either key spelling
    enable_triton_val = cfg.get("enable_triton", cfg.get("ENABLE_TRITON", False))
    enable_triton = bool(_coerce_value("enable_triton", enable_triton_val))

    # build the EngineParams
    params = EngineParams(
        state_dim=int(cfg["state_dim"]),
        dt=float(cfg["dt"]),
        max_norm=float(cfg["max_norm"]),
        spectral_cap=float(cfg["spectral_cap"]),
        implicit_iters=int(cfg["implicit_iters"]),
        implicit_tol=float(cfg["implicit_tol"]),
        rank_r=int(cfg["rank_r"]),
        proj_eps=float(cfg["proj_eps"]),
        clip_nan_policy=str(cfg["clip_nan_policy"]),
        max_step_time_ms=float(cfg["max_step_time_ms"]),
        deterministic=bool(cfg["deterministic"]),
        dtype=cfg["dtype"],
        device=cfg["device"],
        enable_triton=enable_triton,
        enable_jit=bool(cfg.get("enable_jit", False)),
        enable_profiling=bool(cfg.get("enable_profiling", False)),
        allow_mixed_precision=bool(cfg.get("allow_mixed_precision", False)),
    )

    # optional snapshot keys via env (best-effort)
    if not ignore_env:
        aes_key_path = os.getenv(ENV_SNAPSHOT_AES_KEY_PATH)
        priv_pem_path = os.getenv(ENV_SNAPSHOT_SIGN_PRIVATE_PEM_PATH)
        pub_pem_path = os.getenv(ENV_SNAPSHOT_SIGN_PUBLIC_PEM_PATH)

        def _read_key_file(pth: str) -> t.Optional[bytes]:
            try:
                return Path(pth).read_bytes()
            except Exception as e:
                logger.warning("Failed to read key %s: %s", pth, e)
                return None

        if aes_key_path:
            params.snapshot_aes_key = _read_key_file(aes_key_path)
        if priv_pem_path:
            params.snapshot_signing_private_pem = _read_key_file(priv_pem_path)
        if pub_pem_path:
            params.snapshot_signing_public_pem = _read_key_file(pub_pem_path)

    return params


def save_params(params: EngineParams, path: str) -> None:
    """
    Save the EngineParams to a YAML file (human-readable).
    Note: keys containing binary data (e.g., snapshot_aes_key) are not written raw to disk; instead
    file-path placeholders are expected to be used (see Runtime/Config/.env.template).
    """
    d = {
        "state_dim": params.state_dim,
        "dt": params.dt,
        "max_norm": params.max_norm,
        "spectral_cap": params.spectral_cap,
        "implicit_iters": params.implicit_iters,
        "implicit_tol": params.implicit_tol,
        "rank_r": params.rank_r,
        "proj_eps": params.proj_eps,
        "enable_triton": getattr(params, "enable_triton", False),
        "clip_nan_policy": params.clip_nan_policy,
        "max_step_time_ms": params.max_step_time_ms,
        "deterministic": params.deterministic,
        "dtype": str(params.dtype).split(".")[-1] if params.dtype is not None else None,
        "device": str(params.device) if params.device is not None else None,
        "enable_jit": getattr(params, "enable_jit", False),
        "enable_profiling": getattr(params, "enable_profiling", False),
        "allow_mixed_precision": getattr(params, "allow_mixed_precision", False),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(d, f)
    logger.info("Engine params saved to %s", path)


# Convenience CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NOESIS-Σ EngineParams loader/saver")
    ap.add_argument("--out", "-o", help="Write current params to YAML", required=False)
    ap.add_argument(
        "--file", "-f", help="Load params from YAML/JSON file", required=False
    )
    args = ap.parse_args()
    params = load_params(path=args.file if args.file else None)
    print("Loaded EngineParams:")
    print(asdict(params) if hasattr(params, "__dataclass_fields__") else params)
    if args.out:
        save_params(params, args.out)
        print("Saved params to", args.out)
